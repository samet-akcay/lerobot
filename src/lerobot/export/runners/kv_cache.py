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
"""KV-cache runner: encode-once, then iteratively denoise with cached attention.

Used by VLA policies (PI0, PI05, SmolVLA). The exported package contains two
stages: ``encoder`` runs once per observation to produce ``past_*`` KV tensors
and a ``prefix_pad_mask``; ``denoise`` then runs N flow-matching steps with the
cached prefix and an evolving ``x_t``.

Example::

    from lerobot.export import load_exported_policy

    policy = load_exported_policy("pi0_package", backend="onnx")
    actions = policy.predict_action_chunk(observation, num_steps=10)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from ..interfaces import BackendSession
from ..protocols import is_exportable
from .base import ExportModule, build_dynamic_axes, build_normalizer, get_output_by_names, register_runner
from .single_pass import policy_as_exportable

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


@register_runner
class KVCacheRunner:
    type: ClassVar[str] = "kv_cache"

    def __init__(self, manifest: dict[str, Any], artifacts_dir: Path, sessions: BackendSession):
        self._manifest = manifest
        self._backend_session = sessions
        self._normalizer = build_normalizer(manifest, artifacts_dir.parent)

        runner_cfg = manifest["model"]["runner"]
        self._num_steps: int = runner_cfg.get("num_inference_steps", 10)
        self._action_dim: int = runner_cfg["action_dim"]
        self._chunk_size: int = runner_cfg.get("chunk_size", 50)
        self._state_dim: int | None = runner_cfg.get("state_dim")
        self._input_mapping: dict[str, str] = runner_cfg.get("input_mapping", {})

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
        inputs_by_stage = exportable.prepare_inputs(example_batch)
        encoder_stage = inputs_by_stage["encoder"]
        encoder_wrapper = modules["encoder"]

        with torch.no_grad():
            encoder_outputs = encoder_wrapper(*encoder_stage.tensors)
            prefix_len = encoder_outputs[0].shape[1]

        device = next(policy_obj.parameters()).device
        denoise_stage = exportable.prepare_runtime_inputs(
            "denoise",
            {"prefix_len": prefix_len, "device": device},
        )

        encoder_module = ExportModule(
            name="encoder",
            wrapper=encoder_wrapper,
            example_inputs=encoder_stage.tensors,
            input_names=encoder_stage.input_names,
            output_names=encoder_stage.output_names,
            dynamic_axes=build_dynamic_axes(encoder_stage.input_names, encoder_stage.output_names),
        )
        denoise_module = ExportModule(
            name="denoise",
            wrapper=modules["denoise"],
            example_inputs=denoise_stage.tensors,
            input_names=denoise_stage.input_names,
            output_names=denoise_stage.output_names,
            dynamic_axes=build_dynamic_axes(denoise_stage.input_names, denoise_stage.output_names),
            hints={
                "onnx_fixups": ["scatter_gather_dtypes", "double_to_float"],
                "executorch_io_spec_extras": {"stage": "denoise"},
            },
        )

        runner_cfg = {
            "num_inference_steps": export_config.num_steps,
            "scheduler": "euler",
            "action_dim": export_config.action_dim,
            "chunk_size": export_config.chunk_size,
            "n_action_steps": export_config.chunk_size,
            "num_layers": export_config.num_layers,
            "num_kv_heads": export_config.num_kv_heads,
            "head_dim": export_config.head_dim,
            "input_mapping": encoder_stage.metadata["input_mapping"],
            "state_dim": export_config.state_dim,
        }
        return [encoder_module, denoise_module], runner_cfg

    @classmethod
    def load(
        cls,
        manifest: dict[str, Any],
        artifacts_dir: Path,
        backend_session: BackendSession,
    ) -> KVCacheRunner:
        return cls(manifest, artifacts_dir, backend_session)

    def predict_action_chunk(
        self,
        batch: dict[str, np.ndarray],
        num_steps: int | None = None,
        noise: np.ndarray | None = None,
        generator: np.random.Generator | None = None,
    ) -> np.ndarray:
        num_steps = num_steps or self._num_steps

        obs = self._normalizer.normalize_inputs(batch) if self._normalizer else dict(batch)

        if self._input_mapping:
            mapped: dict[str, np.ndarray] = {}
            for obs_key, value in obs.items():
                onnx_key = self._input_mapping.get(obs_key, obs_key)
                mapped[onnx_key] = value
            obs = mapped

        for key in list(obs.keys()):
            obs[key] = obs[key].astype(np.float32)

        first_obs = next(iter(obs.values()))
        batch_size = first_obs.shape[0] if first_obs.ndim > 1 else 1

        num_images = sum(1 for k in obs if k.startswith("image_"))
        for i in range(num_images):
            mask_key = f"img_mask_{i}"
            if mask_key not in obs:
                obs[mask_key] = np.ones((batch_size,), dtype=np.float32)

        if self._state_dim is not None and "state" in obs:
            state = obs["state"]
            current_dim = state.shape[-1]
            if current_dim < self._state_dim:
                padding = np.zeros((*state.shape[:-1], self._state_dim - current_dim), dtype=state.dtype)
                obs["state"] = np.concatenate([state, padding], axis=-1)

        encoder_outputs = self._backend_session.run("encoder", obs)

        prefix_pad_mask = encoder_outputs.get("prefix_pad_mask")
        if prefix_pad_mask is None:
            prefix_pad_mask = next(iter(encoder_outputs.values()))

        kv_cache = {k: v for k, v in encoder_outputs.items() if k.startswith("past_")}

        action_shape = (batch_size, self._chunk_size, self._action_dim)
        if noise is not None:
            x_t = noise.astype(np.float32)
        elif generator is not None:
            x_t = generator.standard_normal(action_shape).astype(np.float32)
        else:
            x_t = np.random.randn(*action_shape).astype(np.float32)

        dt = -1.0 / num_steps

        for step in range(num_steps):
            t = 1.0 + step * dt
            timestep = np.full((batch_size,), t, dtype=np.float32)

            denoise_inputs: dict[str, np.ndarray] = {
                "x_t": x_t,
                "timestep": timestep,
                "prefix_pad_mask": prefix_pad_mask,
                **kv_cache,
            }

            if "state" in obs:
                denoise_inputs["state"] = obs["state"]

            outputs = self._backend_session.run("denoise", denoise_inputs)

            v_t = get_output_by_names(
                outputs,
                primary_name="v_t",
                fallback_names=["velocity"],
                context="KVCacheRunner.denoise",
            )

            x_t = x_t + dt * v_t

        action = self._normalizer.denormalize_outputs(x_t, key="action") if self._normalizer else x_t

        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]

        return action

    def reset(self) -> None:
        return None
