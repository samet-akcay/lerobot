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
"""Policy export: convert PyTorch policies to portable ONNX packages.

Produces a ``policy_package`` directory with ``manifest.json`` in the
converged format shared by LeRobot and PhysicalAI.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from .manifest import (
    CameraConfig,
    HardwareConfig,
    Manifest,
    Metadata,
    ModelConfig,
    PolicyInfo,
    PolicySource,
    ProcessorSpec,
    RobotConfig,
    TensorSpec,
)
from .normalize import save_stats_safetensors

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_policy(
    policy: PreTrainedPolicy,
    output_dir: str | Path,
    *,
    backend: str = "onnx",
    example_batch: dict[str, Tensor] | None = None,
    opset_version: int = 17,
    include_normalization: bool = True,
) -> Path:
    """Export a policy to a ``policy_package``.

    Args:
        policy: Trained policy instance.
        output_dir: Directory to write the package.
        backend: Export backend (``"onnx"``).
        example_batch: Example input for tracing (auto-generated if ``None``).
        opset_version: ONNX opset version.
        include_normalization: Include normalization stats in the package.

    Returns:
        Path to the created policy package directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = output_dir / "artifacts"
    assets_dir = output_dir / "assets"
    artifacts_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    policy_name = getattr(policy, "name", policy.__class__.__name__.lower())
    inference_type = _detect_inference_type(policy)

    if example_batch is None:
        example_batch = _generate_example_batch(policy)

    # ---- Export model artifacts -----------------------------------------

    kv_cache_extra: dict[str, Any] = {}
    if inference_type == "kv_cache":
        if backend != "onnx":
            raise ValueError(f"Unsupported backend for kv-cache export: {backend}")
        artifacts, kv_cache_extra = _export_kv_cache_onnx(policy, artifacts_dir, example_batch, opset_version)
    elif backend == "onnx":
        model_path = artifacts_dir / "model.onnx"
        _export_onnx(policy, model_path, example_batch, opset_version, inference_type)
        artifacts = {"model": "artifacts/model.onnx"}
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # ---- Normalization as preprocessors/postprocessors ------------------

    preprocessors: list[ProcessorSpec] = []
    postprocessors: list[ProcessorSpec] = []

    if include_normalization:
        stats = _get_policy_stats(policy)
        if stats:
            stats_path = artifacts_dir / "stats.safetensors"
            save_stats_safetensors(stats, stats_path)

            mode = _get_normalization_mode(policy)
            input_features = _get_normalized_input_features(policy)

            if input_features:
                preprocessors.append(
                    ProcessorSpec(
                        type="normalize",
                        mode=mode,
                        artifact="artifacts/stats.safetensors",
                        features=input_features,
                    )
                )
            postprocessors.append(
                ProcessorSpec(
                    type="denormalize",
                    mode=mode,
                    artifact="artifacts/stats.safetensors",
                    features=["action"],
                )
            )

    # ---- Save policy config as reference asset -------------------------

    _save_policy_config(policy, assets_dir / "config.json")

    # ---- Build manifest ------------------------------------------------

    manifest = _build_manifest(
        policy=policy,
        policy_name=policy_name,
        inference_type=inference_type,
        artifacts=artifacts,
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        kv_cache_extra=kv_cache_extra,
    )

    manifest.save(output_dir / "manifest.json")

    return output_dir


# ---------------------------------------------------------------------------
# Inference type detection
# ---------------------------------------------------------------------------


def _detect_inference_type(policy: PreTrainedPolicy) -> str:
    """Detect inference type: ``'action_chunking'``, ``'iterative'``, or ``'kv_cache'``."""
    valid_types = {"action_chunking", "iterative", "kv_cache"}

    if hasattr(policy, "get_inference_type"):
        declared = policy.get_inference_type()
        if declared is not None:
            if declared not in valid_types:
                raise ValueError(
                    f"Invalid inference type {declared!r} from {policy.__class__.__name__}. "
                    f"Must be one of: {valid_types}"
                )
            return declared

    # Fallback heuristics
    name = policy.__class__.__name__.lower()

    if "pi0" in name or "smolvla" in name:
        return "kv_cache"

    for pattern in ("diffusion", "flow"):
        if pattern in name:
            return "iterative"

    return "action_chunking"


# ---------------------------------------------------------------------------
# Example batch generation
# ---------------------------------------------------------------------------


def _generate_example_batch(policy: PreTrainedPolicy) -> dict[str, Tensor]:
    """Generate an example batch for ONNX export tracing."""
    config = policy.config
    batch_size = 1
    device = next(policy.parameters()).device

    batch: dict[str, Tensor] = {}

    if hasattr(config, "robot_state_feature") and config.robot_state_feature:
        state_dim = config.robot_state_feature.shape[0]
        batch["observation.state"] = torch.randn(batch_size, state_dim, device=device)

    if hasattr(config, "env_state_feature") and config.env_state_feature:
        env_dim = config.env_state_feature.shape[0]
        batch["observation.environment_state"] = torch.randn(batch_size, env_dim, device=device)

    if hasattr(config, "image_features") and config.image_features:
        for img_key in config.image_features:
            img_shape = config.image_features[img_key].shape
            batch[img_key] = torch.randn(batch_size, *img_shape, device=device)

    return batch


# ---------------------------------------------------------------------------
# ONNX export (single-model)
# ---------------------------------------------------------------------------


def _export_onnx(
    policy: PreTrainedPolicy,
    output_path: Path,
    example_batch: dict[str, Tensor],
    opset_version: int,
    inference_type: str,
) -> None:
    """Export a single-model policy to ONNX format."""
    from .protocols import is_iterative_exportable, is_single_phase_exportable

    policy.eval()

    if inference_type == "action_chunking":
        if is_single_phase_exportable(policy):
            _ = policy.get_single_phase_export_config()
            wrapper = policy.get_forward_module()
            example_inputs, input_names, output_names = policy.prepare_forward_inputs(example_batch)
        else:
            wrapper, input_names, output_names, export_batch = _create_single_pass_wrapper(
                policy, example_batch
            )
            example_inputs = tuple(export_batch[name] for name in input_names if name in export_batch)
    else:
        if is_iterative_exportable(policy):
            _ = policy.get_iterative_export_config()
            wrapper = policy.get_denoise_module()
            example_inputs, input_names, output_names = policy.prepare_denoise_inputs(example_batch)
        else:
            wrapper, input_names, output_names, export_batch = _create_iterative_wrapper(
                policy, example_batch
            )
            example_inputs = tuple(export_batch[name] for name in input_names if name in export_batch)

    dynamic_axes = {}
    for name in input_names:
        dynamic_axes[name] = {0: "batch_size"}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    torch.onnx.export(
        wrapper,
        example_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )


def _create_single_pass_wrapper(
    policy: PreTrainedPolicy,
    example_batch: dict[str, Tensor],
) -> tuple[nn.Module, list[str], list[str], dict[str, Tensor]]:
    class SinglePassWrapper(nn.Module):
        def __init__(self, policy: PreTrainedPolicy):
            super().__init__()
            self.policy = policy

        def forward(self, *args) -> Tensor:
            batch = dict(zip(input_names, args, strict=True))

            if hasattr(self.policy.config, "image_features") and self.policy.config.image_features:
                from lerobot.utils.constants import OBS_IMAGES

                batch[OBS_IMAGES] = [batch[key] for key in self.policy.config.image_features if key in batch]

            if hasattr(self.policy, "model"):
                actions, _ = self.policy.model(batch)
            else:
                actions = self.policy.predict_action_chunk(batch)

            return actions

    input_names = list(example_batch.keys())
    output_names = ["action"]

    return SinglePassWrapper(policy), input_names, output_names, example_batch


def _create_iterative_wrapper(
    policy: PreTrainedPolicy,
    example_batch: dict[str, Tensor],
) -> tuple[nn.Module, list[str], list[str], dict[str, Tensor]]:
    from lerobot.utils.constants import OBS_IMAGES

    config = policy.config
    batch_size = 1
    device = next(policy.parameters()).device

    horizon = getattr(config, "horizon", None) or getattr(config, "chunk_size", None)
    if horizon is None:
        raise ValueError("Policy config must have 'horizon' or 'chunk_size' for iterative export.")

    if hasattr(config, "action_feature") and config.action_feature is not None:
        action_dim = config.action_feature.shape[0]
    else:
        raise ValueError("Policy config must have 'action_feature' with shape for iterative export.")

    extended_batch = dict(example_batch)
    extended_batch["x_t"] = torch.randn(batch_size, horizon, action_dim, device=device)
    extended_batch["timestep"] = torch.tensor([1.0], dtype=torch.float32, device=device)

    is_diffusion = hasattr(policy, "diffusion") and hasattr(policy.diffusion, "unet")
    image_feature_keys = (
        list(config.image_features.keys())
        if hasattr(config, "image_features") and config.image_features
        else []
    )

    class DiffusionIterativeWrapper(nn.Module):
        def __init__(self, policy: PreTrainedPolicy, image_keys: list[str]):
            super().__init__()
            self.diffusion = policy.diffusion
            self.image_keys = image_keys
            if hasattr(self.diffusion, "rgb_encoder"):
                encoder = self.diffusion.rgb_encoder
                if isinstance(encoder, nn.ModuleList):
                    for enc in encoder:
                        enc.do_crop = False
                else:
                    encoder.do_crop = False

        def forward(self, *args) -> Tensor:
            batch = dict(zip(input_names, args, strict=True))
            x_t = batch.pop("x_t")
            timestep = batch.pop("timestep")

            if self.image_keys:
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.image_keys], dim=-4)

            global_cond = self.diffusion._prepare_global_conditioning(batch)
            timestep_long = timestep.long()
            return self.diffusion.unet(x_t, timestep_long, global_cond=global_cond)

    class GenericIterativeWrapper(nn.Module):
        def __init__(self, policy: PreTrainedPolicy):
            super().__init__()
            self.policy = policy

        def forward(self, *args) -> Tensor:
            batch = dict(zip(input_names, args, strict=True))
            x_t = batch.pop("x_t")
            timestep = batch.pop("timestep")

            if hasattr(self.policy, "denoise_step"):
                return self.policy.denoise_step(batch, x_t, timestep)
            elif hasattr(self.policy, "model") and hasattr(self.policy.model, "denoise_step"):
                return self.policy.model.denoise_step(batch, x_t, timestep)
            else:
                raise NotImplementedError(
                    f"Policy {type(self.policy).__name__} does not have a denoise_step method"
                )

    input_names = list(extended_batch.keys())
    output_names = ["v_t"]

    if is_diffusion:
        return (
            DiffusionIterativeWrapper(policy, image_feature_keys),
            input_names,
            output_names,
            extended_batch,
        )
    else:
        return GenericIterativeWrapper(policy), input_names, output_names, extended_batch


# ---------------------------------------------------------------------------
# ONNX export (kv-cache)
# ---------------------------------------------------------------------------


def _fix_onnx_gather_indices(onnx_path: Path) -> None:
    """Fix ONNX Gather nodes that have float indices from ScatterND outputs."""
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(onnx_path))
    nodes_to_insert: list[tuple[int, onnx.NodeProto]] = []

    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Gather" and "position_embedding" in node.name:
            indices_input = node.input[1]
            if "ScatterND" in indices_input:
                cast_output = indices_input + "_cast_to_int64"
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[indices_input],
                    outputs=[cast_output],
                    name=indices_input + "/Cast_to_int64",
                    to=TensorProto.INT64,
                )
                nodes_to_insert.append((idx, cast_node))
                node.input[1] = cast_output

    for idx, cast_node in reversed(nodes_to_insert):
        model.graph.node.insert(idx, cast_node)

    if nodes_to_insert:
        onnx.save(model, str(onnx_path))


def _fix_onnx_double_to_float(onnx_path: Path) -> None:
    """Fix ONNX nodes that use double precision where float32 is sufficient."""
    import onnx
    from onnx import TensorProto, numpy_helper

    model = onnx.load(str(onnx_path))
    modified = False

    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.DOUBLE:
                    data = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(data))
                    modified = True

        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.DOUBLE:
                    attr.i = TensorProto.FLOAT
                    modified = True

    if modified:
        onnx.save(model, str(onnx_path))


def _export_kv_cache_onnx(
    policy: PreTrainedPolicy,
    artifacts_dir: Path,
    example_batch: dict[str, Tensor],
    opset_version: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Export a KV-cache VLA policy to ONNX.

    Returns:
        Tuple of (artifacts dict, extra runner params dict).
    """
    from .protocols import is_kv_cache_exportable

    policy.eval()
    device = next(policy.parameters()).device

    is_pi0 = "pi0" in policy.__class__.__name__.lower()
    if not is_pi0:
        policy = policy.float()

    if not is_kv_cache_exportable(policy):
        raise ValueError(f"KV-cache policy {policy.__class__.__name__} must implement ExportableKVCache.")

    export_config = policy.get_kv_cache_export_config()
    num_layers = export_config.num_layers
    num_kv_heads = export_config.num_kv_heads
    head_dim = export_config.head_dim
    num_steps = export_config.num_steps
    chunk_size = export_config.chunk_size
    action_dim = export_config.action_dim

    encoder_inputs, encoder_input_names, num_images, input_mapping = policy.prepare_encoder_inputs(
        example_batch
    )
    encoder_wrapper = policy.get_encoder_module(num_images=num_images)

    with torch.no_grad():
        encoder_outputs = encoder_wrapper(*encoder_inputs)
        prefix_len = encoder_outputs[0].shape[1]

    denoise_inputs, denoise_input_names = policy.prepare_denoise_inputs(prefix_len, device)
    denoise_wrapper = policy.get_denoise_module()

    # Encoder output names
    encoder_output_names = ["prefix_pad_mask"]
    for layer_idx in range(num_layers):
        encoder_output_names.append(f"past_key_{layer_idx}")
        encoder_output_names.append(f"past_value_{layer_idx}")

    encoder_dynamic_axes = {name: {0: "batch_size"} for name in encoder_input_names + encoder_output_names}

    encoder_path = artifacts_dir / "encoder.onnx"
    torch.onnx.export(
        encoder_wrapper,
        encoder_inputs,
        str(encoder_path),
        input_names=encoder_input_names,
        output_names=encoder_output_names,
        dynamic_axes=encoder_dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    _fix_onnx_gather_indices(encoder_path)

    denoise_output_names = ["v_t"]
    denoise_dynamic_axes = {name: {0: "batch_size"} for name in denoise_input_names + denoise_output_names}

    denoise_path = artifacts_dir / "denoise.onnx"
    torch.onnx.export(
        denoise_wrapper,
        denoise_inputs,
        str(denoise_path),
        input_names=denoise_input_names,
        output_names=denoise_output_names,
        dynamic_axes=denoise_dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    _fix_onnx_double_to_float(denoise_path)

    artifacts = {
        "encoder": "artifacts/encoder.onnx",
        "denoise": "artifacts/denoise.onnx",
    }

    kv_cache_extra = {
        "num_inference_steps": num_steps,
        "scheduler": "euler",
        "action_dim": action_dim,
        "chunk_size": chunk_size,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "input_mapping": input_mapping,
        "state_dim": export_config.state_dim,
    }

    return artifacts, kv_cache_extra


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def _build_manifest(
    policy: PreTrainedPolicy,
    policy_name: str,
    inference_type: str,
    artifacts: dict[str, str],
    preprocessors: list[ProcessorSpec],
    postprocessors: list[ProcessorSpec],
    kv_cache_extra: dict[str, Any],
) -> Manifest:
    """Build the converged policy_package manifest."""
    config = policy.config

    # Build runner config dict
    runner: dict[str, Any] = {"type": inference_type}

    if inference_type == "action_chunking":
        chunk_size = getattr(config, "chunk_size", None) or getattr(config, "horizon", 100)
        n_action_steps = getattr(config, "n_action_steps", chunk_size)
        action_dim = _get_action_dim(config)
        runner["chunk_size"] = chunk_size
        runner["n_action_steps"] = n_action_steps
        runner["action_dim"] = action_dim

    elif inference_type == "iterative":
        runner.update(_build_iterative_runner_config(policy))

    elif inference_type == "kv_cache":
        runner.update(kv_cache_extra)
        runner["n_action_steps"] = kv_cache_extra.get("chunk_size", 50)

    # Build model config
    n_obs_steps = getattr(config, "n_obs_steps", 1)
    model_config = ModelConfig(
        n_obs_steps=n_obs_steps,
        runner=runner,
        artifacts=artifacts,
        preprocessors=preprocessors or None,
        postprocessors=postprocessors or None,
    )

    # Build policy info
    policy_info = PolicyInfo(
        name=policy_name,
        source=PolicySource(
            repo_id=getattr(config, "repo_id", None),
            revision=getattr(config, "revision", None),
        ),
    )

    # Build hardware config
    hardware = _build_hardware_config(policy)

    # Build metadata
    metadata = Metadata(
        created_at=datetime.now(UTC).isoformat(),
        created_by="lerobot.export",
    )

    return Manifest(
        policy=policy_info,
        model=model_config,
        hardware=hardware,
        metadata=metadata,
    )


def _build_iterative_runner_config(policy: PreTrainedPolicy) -> dict[str, Any]:
    """Build runner config params for iterative policies."""
    config = policy.config
    num_steps = getattr(config, "num_inference_steps", 10)
    horizon = getattr(config, "horizon", None) or getattr(config, "chunk_size", 16)
    n_action_steps = getattr(config, "n_action_steps", horizon)
    action_dim = _get_action_dim(config)

    is_diffusion = hasattr(policy, "diffusion") and hasattr(policy.diffusion, "noise_scheduler")

    runner: dict[str, Any] = {
        "horizon": horizon,
        "n_action_steps": n_action_steps,
        "action_dim": action_dim,
        "num_inference_steps": num_steps,
    }

    if is_diffusion:
        scheduler_type = config.noise_scheduler_type.lower()
        runner["scheduler"] = scheduler_type
        runner["timestep_spacing"] = "leading"
        runner["timestep_range"] = [config.num_train_timesteps - 1, 0]
        runner["num_train_timesteps"] = config.num_train_timesteps
        runner["beta_start"] = config.beta_start
        runner["beta_end"] = config.beta_end
        runner["beta_schedule"] = config.beta_schedule
        runner["prediction_type"] = config.prediction_type
        runner["clip_sample"] = config.clip_sample
        runner["clip_sample_range"] = config.clip_sample_range
    else:
        runner["scheduler"] = "euler"
        runner["timestep_range"] = [1.0, 0.0]

    return runner


def _build_hardware_config(policy: PreTrainedPolicy) -> HardwareConfig | None:
    """Extract hardware config from policy if available."""
    config = policy.config
    robots: list[RobotConfig] = []
    cameras: list[CameraConfig] = []

    # Robot state/action from config
    action_dim = _get_action_dim(config)
    state_dim = None
    if hasattr(config, "robot_state_feature") and config.robot_state_feature:
        state_dim = config.robot_state_feature.shape[0]

    if state_dim or action_dim:
        robot = RobotConfig(name="main")
        if state_dim:
            robot.state = TensorSpec(shape=[state_dim], dtype="float32")
        if action_dim:
            robot.action = TensorSpec(shape=[action_dim], dtype="float32")
        robots.append(robot)

    # Cameras from image features
    if hasattr(config, "image_features") and config.image_features:
        for img_key, img_feature in config.image_features.items():
            # Extract camera name from key like "observation.images.top"
            parts = img_key.split(".")
            cam_name = parts[-1] if len(parts) > 1 else img_key
            cameras.append(
                CameraConfig(
                    name=cam_name,
                    shape=list(img_feature.shape),
                    dtype="uint8",
                )
            )

    if not robots and not cameras:
        return None

    return HardwareConfig(
        robots=robots or None,
        cameras=cameras or None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_action_dim(config: Any) -> int:
    """Extract action dimension from policy config."""
    if hasattr(config, "max_action_dim") and config.max_action_dim is not None:
        return config.max_action_dim
    if hasattr(config, "action_feature") and config.action_feature is not None:
        return config.action_feature.shape[0]
    return 14  # fallback


def _get_policy_stats(policy: PreTrainedPolicy) -> dict[str, dict[str, Any]] | None:
    """Extract normalization statistics from policy."""
    if hasattr(policy, "policy_processor"):
        processor = policy.policy_processor
        for step in getattr(processor, "steps", []):
            if hasattr(step, "stats") and step.stats:
                return step.stats

    if hasattr(policy, "config") and hasattr(policy.config, "stats"):
        return policy.config.stats

    return None


def _get_normalization_mode(policy: PreTrainedPolicy) -> str:
    """Detect normalization mode from policy."""
    return "mean_std"


def _get_normalized_input_features(policy: PreTrainedPolicy) -> list[str]:
    """Get list of input features that should be normalized."""
    features = []
    if hasattr(policy.config, "robot_state_feature") and policy.config.robot_state_feature:
        features.append("observation.state")
    return features


def _save_policy_config(policy: PreTrainedPolicy, path: Path) -> None:
    """Save policy configuration as JSON reference."""
    try:
        config_dict = policy.config.__dict__.copy()
        config_dict = {k: v for k, v in config_dict.items() if _is_json_serializable(v)}
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
    except Exception as e:
        logger.debug("Could not save policy config to %s: %s", path, e)


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
