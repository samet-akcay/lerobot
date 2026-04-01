#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Runner implementations for executing exported policies.

Runners orchestrate the inference loop for a given policy type:

- :class:`SinglePassRunner` — single forward pass (ACT, VQ-BeT)
- :class:`IterativeRunner` — multi-step denoising / flow-matching (Diffusion)
- :class:`TwoPhaseRunner` — encode once + iterative denoise (PI0, SmolVLA)
- :class:`ActionChunkingWrapper` — single-action queue on top of any runner
"""

from __future__ import annotations

import logging
import warnings
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .backends import RuntimeAdapter, get_runtime_adapter
from .manifest import Manifest
from .normalize import Normalizer
from .schedulers import create_scheduler

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_output_by_names(
    outputs: dict[str, NDArray[np.floating]],
    primary_name: str,
    fallback_names: list[str],
    context: str,
) -> NDArray[np.floating]:
    """Extract an output tensor by name with fallbacks."""
    if primary_name in outputs:
        return outputs[primary_name]

    for name in fallback_names:
        if name in outputs:
            logger.debug("%s: using fallback output '%s' instead of '%s'", context, name, primary_name)
            return outputs[name]

    if len(outputs) == 1:
        actual_name = next(iter(outputs.keys()))
        warnings.warn(
            f"{context}: Expected output '{primary_name}' not found. "
            f"Using only available output '{actual_name}'.",
            stacklevel=3,
        )
        return outputs[actual_name]

    raise KeyError(
        f"{context}: Expected output '{primary_name}' not found. Available: {list(outputs.keys())}."
    )


def _detect_backend(manifest: Manifest) -> str:
    """Auto-detect backend from the first artifact file extension."""
    first_artifact = next(iter(manifest.model.artifacts.values()))
    if first_artifact.endswith(".onnx"):
        return "onnx"
    elif first_artifact.endswith(".xml"):
        return "openvino"
    else:
        return "onnx"


def _build_normalizer(manifest: Manifest, package_path: Path) -> Normalizer | None:
    """Build a Normalizer from manifest preprocessor/postprocessor specs."""
    return Normalizer.from_specs(
        manifest.model.preprocessors,
        manifest.model.postprocessors,
        package_path,
    )


# ---------------------------------------------------------------------------
# InferenceRunner protocol
# ---------------------------------------------------------------------------


class InferenceRunner(Protocol):
    """Interface for running exported policies."""

    @property
    def manifest(self) -> Manifest:
        """The loaded manifest."""
        ...

    def predict_action_chunk(self, observation: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Run inference and return an action chunk."""
        ...

    def reset(self) -> None:
        """Reset internal state (call on episode boundary)."""
        ...


# ---------------------------------------------------------------------------
# SinglePassRunner
# ---------------------------------------------------------------------------


class SinglePassRunner:
    """Runner for single-pass policies (ACT, VQ-BeT).

    Produces an action chunk in one forward pass through the model.
    """

    def __init__(
        self,
        package_path: Path,
        manifest: Manifest,
        backend: str,
        device: str = "cpu",
    ) -> None:
        self._manifest = manifest
        self._package_path = Path(package_path)

        artifact_path = manifest.model.artifacts.get("model")
        if artifact_path is None:
            raise ValueError("No 'model' artifact found in manifest.model.artifacts")

        model_path = self._package_path / artifact_path
        self._adapter: RuntimeAdapter = get_runtime_adapter(backend, model_path, device)
        self._normalizer = _build_normalizer(manifest, self._package_path)

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def predict_action_chunk(self, observation: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        obs = self._normalizer.normalize_inputs(observation) if self._normalizer else observation
        obs = {k: v.astype(np.float32) for k, v in obs.items()}

        outputs = self._adapter.run(obs)

        action = _get_output_by_names(
            outputs,
            primary_name="action",
            fallback_names=[],
            context="SinglePassRunner",
        )

        if self._normalizer:
            action = self._normalizer.denormalize_outputs(action, key="action")

        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]

        return action

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# IterativeRunner
# ---------------------------------------------------------------------------


class IterativeRunner:
    """Runner for iterative policies (flow-matching, diffusion).

    Supports Euler (flow-matching), DDPM, and DDIM schedulers.
    """

    def __init__(
        self,
        package_path: Path,
        manifest: Manifest,
        backend: str,
        device: str = "cpu",
    ) -> None:
        self._manifest = manifest
        self._package_path = Path(package_path)
        runner_cfg = manifest.model.runner

        artifact_path = manifest.model.artifacts.get("model")
        if artifact_path is None:
            raise ValueError("No 'model' artifact found in manifest.model.artifacts")

        model_path = self._package_path / artifact_path
        self._adapter: RuntimeAdapter = get_runtime_adapter(backend, model_path, device)
        self._normalizer = _build_normalizer(manifest, self._package_path)

        self._num_steps: int = runner_cfg.get("num_inference_steps", 10)
        self._scheduler_type: str = runner_cfg.get("scheduler", "euler").lower()
        self._action_dim: int = runner_cfg["action_dim"]
        self._chunk_size: int = runner_cfg.get("horizon", runner_cfg.get("chunk_size", 16))

        self._diffusion_scheduler = create_scheduler(runner_cfg)

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def predict_action_chunk(
        self,
        observation: dict[str, NDArray[np.floating]],
        num_steps: int | None = None,
        noise: NDArray[np.floating] | None = None,
        generator: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        num_steps = num_steps or self._num_steps

        obs = self._normalizer.normalize_inputs(observation) if self._normalizer else observation
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
        x_t: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        num_steps: int,
        batch_size: int,
    ) -> NDArray[np.floating]:
        runner_cfg = self._manifest.model.runner
        timestep_range = runner_cfg.get("timestep_range", [1.0, 0.0])
        t_start, t_end = timestep_range
        dt = (t_end - t_start) / num_steps

        for step in range(num_steps):
            t = t_start + step * dt
            timestep = np.full((batch_size,), t, dtype=np.float32)

            inputs = {"x_t": x_t, "timestep": timestep, **obs}
            outputs = self._adapter.run(inputs)

            v_t = _get_output_by_names(
                outputs,
                primary_name="v_t",
                fallback_names=["velocity"],
                context="IterativeRunner.euler",
            )

            x_t = x_t + dt * v_t

        return x_t

    def _run_diffusion_loop(
        self,
        x_t: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        num_steps: int,
        generator: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        scheduler = self._diffusion_scheduler
        timesteps = scheduler.set_timesteps(num_steps)

        for t in timesteps:
            timestep_array = np.array([t], dtype=np.float32)

            inputs = {"x_t": x_t, "timestep": timestep_array, **obs}
            outputs = self._adapter.run(inputs)

            model_output = _get_output_by_names(
                outputs,
                primary_name="v_t",
                fallback_names=["velocity", "noise_pred"],
                context="IterativeRunner.diffusion",
            )

            x_t = scheduler.step(model_output, int(t), x_t, generator=generator)

        return x_t

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# TwoPhaseRunner
# ---------------------------------------------------------------------------


class TwoPhaseRunner:
    """Runner for two-phase VLA policies (PI0, SmolVLA).

    Phase 1 (encode): process images/language/state → KV cache (run once).
    Phase 2 (denoise): iterative denoising using cached KV values (run N times).
    """

    def __init__(
        self,
        package_path: Path,
        manifest: Manifest,
        backend: str,
        device: str = "cpu",
    ) -> None:
        self._manifest = manifest
        self._package_path = Path(package_path)
        runner_cfg = manifest.model.runner

        encoder_artifact = manifest.model.artifacts.get("encoder")
        denoise_artifact = manifest.model.artifacts.get("denoise")
        if encoder_artifact is None or denoise_artifact is None:
            raise ValueError("Two-phase runner requires 'encoder' and 'denoise' in model.artifacts")

        encoder_path = self._package_path / encoder_artifact
        denoise_path = self._package_path / denoise_artifact

        self._encoder_adapter: RuntimeAdapter = get_runtime_adapter(backend, encoder_path, device)
        self._denoise_adapter: RuntimeAdapter = get_runtime_adapter(backend, denoise_path, device)

        self._normalizer = _build_normalizer(manifest, self._package_path)

        self._num_steps: int = runner_cfg.get("num_inference_steps", 10)
        self._num_layers: int = runner_cfg.get("num_layers", 18)
        self._action_dim: int = runner_cfg["action_dim"]
        self._chunk_size: int = runner_cfg.get("chunk_size", 50)
        self._state_dim: int | None = runner_cfg.get("state_dim")
        self._input_mapping: dict[str, str] = runner_cfg.get("input_mapping", {})

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def predict_action_chunk(
        self,
        observation: dict[str, NDArray[np.floating]],
        num_steps: int | None = None,
        noise: NDArray[np.floating] | None = None,
        generator: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        num_steps = num_steps or self._num_steps

        obs = self._normalizer.normalize_inputs(observation) if self._normalizer else dict(observation)

        # Apply input mapping (e.g. observation.state -> state)
        if self._input_mapping:
            mapped: dict[str, NDArray[np.floating]] = {}
            for obs_key, value in obs.items():
                onnx_key = self._input_mapping.get(obs_key, obs_key)
                mapped[onnx_key] = value
            obs = mapped

        # Cast dtypes based on encoder adapter input metadata
        for key in list(obs.keys()):
            obs[key] = obs[key].astype(np.float32)

        first_obs = next(iter(obs.values()))
        batch_size = first_obs.shape[0] if first_obs.ndim > 1 else 1

        # Auto-add image masks if missing
        num_images = sum(1 for k in obs if k.startswith("image_"))
        for i in range(num_images):
            mask_key = f"img_mask_{i}"
            if mask_key not in obs:
                obs[mask_key] = np.ones((batch_size,), dtype=np.float32)

        # Pad state if needed
        if self._state_dim is not None and "state" in obs:
            state = obs["state"]
            current_dim = state.shape[-1]
            if current_dim < self._state_dim:
                padding = np.zeros((*state.shape[:-1], self._state_dim - current_dim), dtype=state.dtype)
                obs["state"] = np.concatenate([state, padding], axis=-1)

        # Phase 1: Encode
        encoder_outputs = self._encoder_adapter.run(obs)

        prefix_pad_mask = encoder_outputs.get("prefix_pad_mask")
        if prefix_pad_mask is None:
            prefix_pad_mask = next(iter(encoder_outputs.values()))

        kv_cache = {k: v for k, v in encoder_outputs.items() if k.startswith("past_")}

        # Phase 2: Iterative denoise
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

            denoise_inputs: dict[str, NDArray[np.floating]] = {
                "x_t": x_t,
                "timestep": timestep,
                "prefix_pad_mask": prefix_pad_mask,
                **kv_cache,
            }

            if "state" in obs:
                denoise_inputs["state"] = obs["state"]

            outputs = self._denoise_adapter.run(denoise_inputs)

            v_t = _get_output_by_names(
                outputs,
                primary_name="v_t",
                fallback_names=["velocity"],
                context="TwoPhaseRunner.denoise",
            )

            x_t = x_t + dt * v_t

        action = self._normalizer.denormalize_outputs(x_t, key="action") if self._normalizer else x_t

        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]

        return action

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# ActionChunkingWrapper
# ---------------------------------------------------------------------------


class ActionChunkingWrapper:
    """Wraps any runner to provide a single-action interface.

    Manages an action queue internally and dispenses one action per
    call to :meth:`select_action`, matching the semantics of
    ``Policy.select_action()`` from eager inference.
    """

    def __init__(self, runner: InferenceRunner) -> None:
        self.runner = runner
        self._queue: deque[NDArray[np.floating]] = deque()

    def reset(self) -> None:
        """Clear action queue and reset inner runner."""
        self._queue.clear()
        self.runner.reset()

    def select_action(self, observation: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Return a single action, managing the chunk queue internally."""
        if len(self._queue) == 0:
            chunk = self.runner.predict_action_chunk(observation)
            n_steps = self.runner.manifest.model.runner.get("n_action_steps", len(chunk))

            if chunk.ndim == 3:
                chunk = chunk[0]

            for i in range(min(n_steps, len(chunk))):
                self._queue.append(chunk[i])

        return self._queue.popleft()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_runner(
    package_path: Path | str,
    backend: str | None = None,
    device: str = "cpu",
) -> InferenceRunner:
    """Create the appropriate runner for an exported policy package.

    Args:
        package_path: Path to the policy package directory.
        backend: Runtime backend (``"onnx"`` or ``"openvino"``).
            Auto-detected from artifacts if ``None``.
        device: Device for inference.

    Returns:
        An :class:`InferenceRunner` instance.
    """
    package_path = Path(package_path)
    manifest_path = package_path / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in {package_path}")

    manifest = Manifest.load(manifest_path)

    if backend is None:
        backend = _detect_backend(manifest)

    runner_type = manifest.runner_type

    if runner_type == "action_chunking":
        return SinglePassRunner(package_path, manifest, backend, device)
    elif runner_type == "iterative":
        return IterativeRunner(package_path, manifest, backend, device)
    elif runner_type == "two_phase":
        return TwoPhaseRunner(package_path, manifest, backend, device)
    else:
        raise ValueError(f"Unknown runner type: {runner_type!r}")
