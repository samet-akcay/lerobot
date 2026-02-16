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
"""Runner implementations for executing exported policies."""

from __future__ import annotations

import logging
import warnings
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .backends import RuntimeAdapter, get_runtime_adapter
from .manifest import IterativeConfig, Manifest, TwoPhaseConfig
from .normalize import Normalizer
from .schedulers import create_scheduler

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _get_output_by_names(
    outputs: dict[str, NDArray[np.floating]],
    primary_name: str,
    fallback_names: list[str],
    context: str,
) -> NDArray[np.floating]:
    """Extract output by name with fallbacks and validation.

    Args:
        outputs: Dict of model outputs.
        primary_name: Primary expected output name (from manifest).
        fallback_names: Alternative names to try.
        context: Description for warning messages.

    Returns:
        The output array.

    Raises:
        KeyError: If no matching output found.
    """
    if primary_name in outputs:
        return outputs[primary_name]

    for name in fallback_names:
        if name in outputs:
            logger.debug(
                "%s: Expected output '%s' not found, using fallback '%s'",
                context,
                primary_name,
                name,
            )
            return outputs[name]

    if len(outputs) == 1:
        actual_name = next(iter(outputs.keys()))
        warnings.warn(
            f"{context}: Expected output '{primary_name}' not found. "
            f"Using only available output '{actual_name}'. "
            f"This may indicate a manifest/model mismatch.",
            stacklevel=3,
        )
        return outputs[actual_name]

    raise KeyError(
        f"{context}: Expected output '{primary_name}' not found. "
        f"Available outputs: {list(outputs.keys())}. "
        f"Check that manifest matches the exported model."
    )


class InferenceRunner(Protocol):
    """Interface for running exported policies."""

    @property
    def manifest(self) -> Manifest:
        """The loaded manifest."""
        ...

    def predict_action_chunk(self, observation: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Run inference and return action chunk.

        Args:
            observation: Dict mapping input names to numpy arrays.

        Returns:
            Action chunk with shape (chunk_size, action_dim) or (B, chunk_size, action_dim).
        """
        ...

    def reset(self) -> None:
        """Reset internal state (call on episode boundary)."""
        ...


class SinglePassRunner:
    """Runner for single-pass policies like ACT.

    Single-pass policies produce an action chunk in one forward pass.
    """

    def __init__(
        self,
        package_path: Path,
        manifest: Manifest,
        adapter_name: str,
        device: str = "cpu",
    ):
        """Initialize the single-pass runner.

        Args:
            package_path: Path to the PolicyPackage directory.
            manifest: Loaded manifest.
            adapter_name: Runtime adapter to use ("onnx" or "openvino").
            device: Device for inference.
        """
        self._manifest = manifest
        self._package_path = Path(package_path)
        self._device = device

        # Load runtime adapter
        artifact_path = manifest.artifacts.get(adapter_name)
        if artifact_path is None:
            raise ValueError(f"No artifact found for runtime adapter: {adapter_name}")

        model_path = self._package_path / artifact_path
        self._adapter: RuntimeAdapter = get_runtime_adapter(adapter_name, model_path, device)

        # Load normalizer if available
        self._normalizer: Normalizer | None = None
        if manifest.normalization:
            stats_path = self._package_path / manifest.normalization.artifact
            if stats_path.exists():
                self._normalizer = Normalizer.from_safetensors(stats_path, manifest.normalization)

    @property
    def manifest(self) -> Manifest:
        """The loaded manifest."""
        return self._manifest

    def predict_action_chunk(self, observation: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Run single-pass inference and return action chunk.

        Args:
            observation: Dict mapping input names to numpy arrays.

        Returns:
            Action chunk with shape (chunk_size, action_dim) or (B, chunk_size, action_dim).
        """
        obs_normalized = self._normalizer.normalize_inputs(observation) if self._normalizer else observation
        obs_normalized = {k: v.astype(np.float32) for k, v in obs_normalized.items()}

        outputs = self._adapter.run(obs_normalized)

        action_key = self._manifest.io.outputs[0].name
        action = _get_output_by_names(
            outputs,
            primary_name=action_key,
            fallback_names=["action"],
            context="SinglePassRunner.predict_action_chunk",
        )

        if self._normalizer:
            action = self._normalizer.denormalize_outputs(action, key="action")

        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]

        return action

    def reset(self) -> None:
        """Reset internal state. No-op for single-pass policies."""
        pass


class IterativeRunner:
    """Runner for iterative policies (flow matching, diffusion).

    Supports Euler (flow matching), DDPM, and DDIM schedulers.
    """

    def __init__(
        self,
        package_path: Path,
        manifest: Manifest,
        adapter_name: str,
        device: str = "cpu",
    ):
        self._manifest = manifest
        self._package_path = Path(package_path)
        self._device = device

        artifact_path = manifest.artifacts.get(adapter_name)
        if artifact_path is None and adapter_name == "openvino":
            artifact_path = manifest.artifacts.get("onnx")
        if artifact_path is None:
            raise ValueError(f"No artifact found for runtime adapter: {adapter_name}")

        model_path = self._package_path / artifact_path
        actual_adapter = "onnx" if adapter_name.startswith("onnx") else adapter_name
        self._adapter: RuntimeAdapter = get_runtime_adapter(actual_adapter, model_path, device)

        self._normalizer: Normalizer | None = None
        if manifest.normalization:
            stats_path = self._package_path / manifest.normalization.artifact
            if stats_path.exists():
                self._normalizer = Normalizer.from_safetensors(stats_path, manifest.normalization)

        if not manifest.is_iterative or not isinstance(manifest.inference, IterativeConfig):
            raise ValueError("IterativeConfig is required for iterative policies")

        self._iterative_config = manifest.inference
        self._num_steps = manifest.inference.num_steps
        self._scheduler_type = manifest.inference.scheduler.lower()
        self._timestep_range = manifest.inference.timestep_range

        self._action_dim = manifest.action.dim
        self._chunk_size = manifest.action.chunk_size

        self._diffusion_scheduler = create_scheduler(manifest.inference)

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

        obs_normalized = self._normalizer.normalize_inputs(observation) if self._normalizer else observation

        obs_normalized = {k: v.astype(np.float32) for k, v in obs_normalized.items()}

        first_obs = next(iter(obs_normalized.values()))
        batch_size = first_obs.shape[0] if first_obs.ndim > 1 else 1

        action_shape = (batch_size, self._chunk_size, self._action_dim)
        if noise is not None:
            x_t = noise.astype(np.float32)
        else:
            if generator is not None:
                x_t = generator.standard_normal(action_shape).astype(np.float32)
            else:
                x_t = np.random.randn(*action_shape).astype(np.float32)

        if self._diffusion_scheduler is not None:
            x_t = self._run_diffusion_loop(x_t, obs_normalized, num_steps, generator)
        else:
            x_t = self._run_euler_loop(x_t, obs_normalized, num_steps, batch_size)

        action = self._normalizer.denormalize_outputs(x_t, key="action") if self._normalizer else x_t

        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]

        return action

    def _run_euler_loop(
        self,
        x_t: NDArray[np.floating],
        obs_normalized: dict[str, NDArray[np.floating]],
        num_steps: int,
        batch_size: int,
    ) -> NDArray[np.floating]:
        t_start, t_end = self._timestep_range
        dt = (t_end - t_start) / num_steps

        for step in range(num_steps):
            t = t_start + step * dt
            timestep = np.full((batch_size,), t, dtype=np.float32)

            inputs = {"x_t": x_t, "timestep": timestep, **obs_normalized}
            outputs = self._adapter.run(inputs)

            v_t = _get_output_by_names(
                outputs,
                primary_name="v_t",
                fallback_names=["velocity"],
                context="IterativeRunner._run_euler_loop",
            )

            x_t = x_t + dt * v_t

        return x_t

    def _run_diffusion_loop(
        self,
        x_t: NDArray[np.floating],
        obs_normalized: dict[str, NDArray[np.floating]],
        num_steps: int,
        generator: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        scheduler = self._diffusion_scheduler
        timesteps = scheduler.set_timesteps(num_steps)

        for t in timesteps:
            timestep_array = np.array([t], dtype=np.int64)

            inputs = {"x_t": x_t, "timestep": timestep_array.astype(np.float32), **obs_normalized}
            outputs = self._adapter.run(inputs)

            model_output = _get_output_by_names(
                outputs,
                primary_name="v_t",
                fallback_names=["velocity", "noise_pred"],
                context="IterativeRunner._run_diffusion_loop",
            )

            x_t = scheduler.step(model_output, int(t), x_t, generator=generator)

        return x_t

    def reset(self) -> None:
        pass


class TwoPhaseRunner:
    """Runner for two-phase VLA policies (PI0, SmolVLA).

    Two-phase policies have:
    1. Encoder: Encodes images/language/state → KV cache (run once)
    2. Denoise: Iterative denoising using cached KV values (run N times)
    """

    def __init__(
        self,
        package_path: Path,
        manifest: Manifest,
        adapter_name: str,
        device: str = "cpu",
    ):
        self._manifest = manifest
        self._package_path = Path(package_path)
        self._device = device

        if not manifest.is_two_phase or not isinstance(manifest.inference, TwoPhaseConfig):
            raise ValueError("TwoPhaseConfig is required for two-phase policies")

        self._two_phase_config = manifest.inference

        encoder_artifact = manifest.artifacts.get(manifest.inference.encoder_artifact)
        denoise_artifact = manifest.artifacts.get(manifest.inference.denoise_artifact)

        if encoder_artifact is None or denoise_artifact is None:
            raise ValueError("Both encoder and denoise artifacts are required")

        encoder_path = self._package_path / encoder_artifact
        denoise_path = self._package_path / denoise_artifact

        actual_adapter = "onnx" if adapter_name.startswith("onnx") else adapter_name
        self._encoder_adapter: RuntimeAdapter = get_runtime_adapter(actual_adapter, encoder_path, device)
        self._denoise_adapter: RuntimeAdapter = get_runtime_adapter(actual_adapter, denoise_path, device)

        self._normalizer: Normalizer | None = None
        if manifest.normalization:
            stats_path = self._package_path / manifest.normalization.artifact
            if stats_path.exists():
                self._normalizer = Normalizer.from_safetensors(stats_path, manifest.normalization)

        self._num_steps = manifest.inference.num_steps
        self._num_layers = manifest.inference.num_layers
        self._action_dim = manifest.action.dim
        self._chunk_size = manifest.action.chunk_size

        self._state_dim: int | None = None
        self._input_dtypes: dict[str, str] = {}
        for spec in manifest.io.inputs:
            if spec.name == "state":
                shape = spec.shape
                self._state_dim = shape[-1] if isinstance(shape[-1], int) else None
            self._input_dtypes[spec.name] = spec.dtype

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

        if self._normalizer:
            obs_normalized = self._normalizer.normalize_inputs(observation)
        else:
            obs_normalized = dict(observation)

        if self._two_phase_config.input_mapping:
            obs_for_encoder = {}
            for obs_key, value in obs_normalized.items():
                onnx_key = self._two_phase_config.input_mapping.get(obs_key, obs_key)
                obs_for_encoder[onnx_key] = value
            obs_normalized = obs_for_encoder

        for key, value in obs_normalized.items():
            expected_dtype = self._input_dtypes.get(key, "float32")
            if expected_dtype == "int64":
                obs_normalized[key] = value.astype(np.int64)
            elif expected_dtype == "bool":
                obs_normalized[key] = value.astype(np.bool_)
            else:
                obs_normalized[key] = value.astype(np.float32)

        first_obs = next(iter(obs_normalized.values()))
        batch_size = first_obs.shape[0] if first_obs.ndim > 1 else 1

        num_images = sum(1 for k in obs_normalized if k.startswith("image_"))
        for i in range(num_images):
            mask_key = f"img_mask_{i}"
            if mask_key not in obs_normalized:
                obs_normalized[mask_key] = np.ones((batch_size,), dtype=np.float32)

        if self._state_dim is not None and "state" in obs_normalized:
            state = obs_normalized["state"]
            current_dim = state.shape[-1]
            if current_dim < self._state_dim:
                padding = np.zeros((*state.shape[:-1], self._state_dim - current_dim), dtype=state.dtype)
                obs_normalized["state"] = np.concatenate([state, padding], axis=-1)

        encoder_outputs = self._encoder_adapter.run(obs_normalized)

        prefix_pad_mask = encoder_outputs.get("prefix_pad_mask")
        if prefix_pad_mask is None:
            prefix_pad_mask = next(iter(encoder_outputs.values()))

        kv_cache = {}
        for key, value in encoder_outputs.items():
            if key.startswith("past_"):
                kv_cache[key] = value

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

            denoise_inputs = {
                "x_t": x_t,
                "timestep": timestep,
                "prefix_pad_mask": prefix_pad_mask,
                **kv_cache,
            }

            if "state" in obs_normalized:
                denoise_inputs["state"] = obs_normalized["state"]

            outputs = self._denoise_adapter.run(denoise_inputs)

            v_t = _get_output_by_names(
                outputs,
                primary_name="v_t",
                fallback_names=["velocity"],
                context="TwoPhaseRunner._run_denoise_loop",
            )

            x_t = x_t + dt * v_t

        action = self._normalizer.denormalize_outputs(x_t, key="action") if self._normalizer else x_t

        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]

        return action

    def reset(self) -> None:
        pass


class ActionChunkingWrapper:
    """Wraps InferenceRunner to provide single-action interface.

    This wrapper manages an action queue internally and provides actions
    one at a time, matching the semantics of Policy.select_action() from
    eager inference.
    """

    def __init__(self, runner: InferenceRunner):
        """Initialize the action chunking wrapper.

        Args:
            runner: Underlying InferenceRunner instance.
        """
        self.runner = runner
        self._queue: deque[NDArray[np.floating]] = deque()

    def reset(self) -> None:
        """Clear action queue and reset runner."""
        self._queue.clear()
        self.runner.reset()

    def select_action(self, observation: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Return single action, managing queue internally.

        Matches the semantics of Policy.select_action() from eager inference.

        Args:
            observation: Dict mapping input names to numpy arrays.

        Returns:
            Single action with shape (action_dim,).
        """
        if len(self._queue) == 0:
            chunk = self.runner.predict_action_chunk(observation)
            n_steps = self.runner.manifest.action.n_action_steps

            # Ensure chunk is 2D (chunk_size, action_dim)
            if chunk.ndim == 3:
                chunk = chunk[0]  # Remove batch dim

            # Add actions to queue (up to n_action_steps)
            for i in range(min(n_steps, len(chunk))):
                self._queue.append(chunk[i])

        return self._queue.popleft()


def create_runner(
    package_path: Path | str,
    backend: str | None = None,
    device: str = "cpu",
) -> InferenceRunner:
    """Create the appropriate runner for a PolicyPackage.

    Args:
        package_path: Path to the PolicyPackage directory.
        backend: Runtime adapter to use ("onnx" or "openvino"). Auto-detected if None.
        device: Device for inference.

    Returns:
        InferenceRunner instance (SinglePassRunner, IterativeRunner, or TwoPhaseRunner).
    """
    package_path = Path(package_path)
    manifest_path = package_path / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in {package_path}")

    manifest = Manifest.load(manifest_path)

    # Auto-detect backend if not specified
    if backend is None:
        # Prefer ONNX, then OpenVINO
        if "onnx" in manifest.artifacts:
            backend = "onnx"
        elif "openvino" in manifest.artifacts:
            backend = "openvino"
        else:
            backend = next(iter(manifest.artifacts.keys()))

    # Create appropriate runner based on inference config type
    if manifest.is_single_pass:
        return SinglePassRunner(package_path, manifest, backend, device)
    elif manifest.is_iterative:
        return IterativeRunner(package_path, manifest, backend, device)
    elif manifest.is_two_phase:
        return TwoPhaseRunner(package_path, manifest, backend, device)
    else:
        raise ValueError(f"Unknown inference config type: {type(manifest.inference)}")
