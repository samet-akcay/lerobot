#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Iterative runner: action chunk emerges from N denoising / flow-matching steps.

Used by Diffusion Policy and similar policies that start from noise and refine
``x_t`` over a fixed number of inference steps. Schedulers (DDPM/DDIM/Euler)
are pure-numpy and parameterised by the manifest's ``model.runner`` block.

Example::

    from lerobot.export import load_exported_policy

    policy = load_exported_policy("diffusion_package", backend="onnx")
    actions = policy.predict_action_chunk(observation, num_steps=10)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ..interfaces import BackendSession
from ..protocols import is_exportable
from ..schedulers import create_scheduler
from .base import ExportModule, build_dynamic_axes, build_normalizer, get_output_by_names, register_runner
from .single_pass import policy_as_exportable

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


@register_runner
class IterativeRunner:
    type: ClassVar[str] = "iterative"

    def __init__(self, manifest: dict[str, Any], artifacts_dir: Path, adapter: BackendSession):
        self._manifest = manifest
        self._adapter = adapter
        self._normalizer = build_normalizer(manifest, artifacts_dir.parent)
        runner_cfg = manifest["model"]["runner"]

        self._num_steps: int = runner_cfg.get("num_inference_steps", 10)
        self._action_dim: int = runner_cfg["action_dim"]
        self._chunk_size: int = runner_cfg.get("horizon", runner_cfg.get("chunk_size", 16))
        self._diffusion_scheduler = create_scheduler(runner_cfg)

    @classmethod
    def matches(cls, policy: object) -> bool:
        return is_exportable(policy) and policy.get_inference_type() == cls.type

    @classmethod
    def export(
        cls,
        policy: object,
        example_batch: dict[str, Tensor],
    ) -> tuple[list[ExportModule], dict[str, Any]]:
        exportable = policy_as_exportable(policy)
        policy_obj: Any = exportable
        export_config = exportable.get_export_config()
        modules = exportable.get_export_modules()
        stage = exportable.prepare_inputs(example_batch)["model"]
        export_module = ExportModule(
            name="model",
            wrapper=modules["model"],
            example_inputs=stage.tensors,
            input_names=stage.input_names,
            output_names=stage.output_names,
            dynamic_axes=build_dynamic_axes(stage.input_names, stage.output_names),
        )

        config = policy_obj.config
        runner_cfg: dict[str, Any] = {
            "horizon": export_config.horizon,
            "n_action_steps": getattr(config, "n_action_steps", export_config.horizon),
            "action_dim": export_config.action_dim,
            "num_inference_steps": export_config.num_inference_steps,
        }

        is_diffusion = hasattr(policy_obj, "diffusion") and hasattr(policy_obj.diffusion, "noise_scheduler")
        if is_diffusion:
            runner_cfg["scheduler"] = config.noise_scheduler_type.lower()
            runner_cfg["timestep_spacing"] = "leading"
            runner_cfg["timestep_range"] = [config.num_train_timesteps - 1, 0]
            runner_cfg["num_train_timesteps"] = config.num_train_timesteps
            runner_cfg["beta_start"] = config.beta_start
            runner_cfg["beta_end"] = config.beta_end
            runner_cfg["beta_schedule"] = config.beta_schedule
            runner_cfg["prediction_type"] = config.prediction_type
            runner_cfg["clip_sample"] = config.clip_sample
            runner_cfg["clip_sample_range"] = config.clip_sample_range
        else:
            runner_cfg["scheduler"] = "euler"
            runner_cfg["timestep_range"] = [1.0, 0.0]

        return [export_module], runner_cfg

    @classmethod
    def load(
        cls,
        manifest: dict[str, Any],
        artifacts_dir: Path,
        backend_session: BackendSession,
    ) -> IterativeRunner:
        return cls(manifest, artifacts_dir, backend_session)

    def predict_action_chunk(
        self,
        batch: dict[str, np.ndarray],
        num_steps: int | None = None,
        noise: np.ndarray | None = None,
        generator: np.random.Generator | None = None,
    ) -> np.ndarray:
        num_steps = num_steps or self._num_steps

        obs = self._normalizer.normalize_inputs(batch) if self._normalizer else batch
        obs = {k: v.astype(np.float32) for k, v in obs.items()}

        first_obs = next(iter(obs.values()))
        batch_size = first_obs.shape[0] if first_obs.ndim > 1 else 1

        action_shape = (batch_size, self._chunk_size, self._action_dim)
        if noise is not None:
            x_t = noise.astype(np.float32)
        elif generator is not None:
            x_t = generator.standard_normal(action_shape).astype(np.float32)
        else:
            x_t = np.random.randn(*action_shape).astype(np.float32)

        if self._diffusion_scheduler is not None:
            x_t = self._run_diffusion_loop(x_t, obs, num_steps, generator)
        else:
            x_t = self._run_euler_loop(x_t, obs, num_steps, batch_size)

        action = self._normalizer.denormalize_outputs(x_t, key="action") if self._normalizer else x_t

        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]

        return action

    def _run_euler_loop(
        self,
        x_t: np.ndarray,
        obs: dict[str, np.ndarray],
        num_steps: int,
        batch_size: int,
    ) -> np.ndarray:
        runner_cfg = self._manifest["model"]["runner"]
        timestep_range = runner_cfg.get("timestep_range", [1.0, 0.0])
        t_start, t_end = timestep_range
        dt = (t_end - t_start) / num_steps

        for step in range(num_steps):
            t = t_start + step * dt
            timestep = np.full((batch_size,), t, dtype=np.float32)

            inputs = {"x_t": x_t, "timestep": timestep, **obs}
            outputs = self._adapter.run("model", inputs)

            v_t = get_output_by_names(
                outputs,
                primary_name="v_t",
                fallback_names=["velocity"],
                context="IterativeRunner.euler",
            )

            x_t = x_t + dt * v_t

        return x_t

    def _run_diffusion_loop(
        self,
        x_t: np.ndarray,
        obs: dict[str, np.ndarray],
        num_steps: int,
        generator: np.random.Generator | None = None,
    ) -> np.ndarray:
        scheduler = self._diffusion_scheduler
        timesteps = scheduler.set_timesteps(num_steps)

        for t in timesteps:
            timestep_array = np.array([t], dtype=np.float32)

            inputs = {"x_t": x_t, "timestep": timestep_array, **obs}
            outputs = self._adapter.run("model", inputs)

            model_output = get_output_by_names(
                outputs,
                primary_name="v_t",
                fallback_names=["velocity", "noise_pred"],
                context="IterativeRunner.diffusion",
            )

            x_t = scheduler.step(model_output, int(t), x_t, generator=generator)

        return x_t

    def reset(self) -> None:
        return None
