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

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from torch import Tensor

from . import (
    backends as _backends,  # noqa: F401
    runners as _runners,  # noqa: F401
)
from ._package_utils import (
    build_hardware_config,
    generate_example_batch,
    get_normalization_groups,
    get_normalized_input_features,
    get_policy_stats,
    save_policy_config,
)
from .backends import BACKENDS
from .manifest import Manifest, Metadata, ModelConfig, PolicyInfo, PolicySource, ProcessorSpec
from .normalize import save_stats_safetensors
from .processors import build_denormalize_processor_specs, build_normalize_processor_specs
from .runners.base import RUNNERS, Runner

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)

DEFAULT_ONNX_OPSET: int = 17
# ONNX opset 17 matches ORT 1.16+ and supports the operators used by ACT/PI05 exports.

__all__ = ["build_processor_specs", "export_policy"]


def build_processor_specs(
    policy: PreTrainedPolicy,
    *,
    include_normalization: bool,
    stats_artifact: str,
    tokenizer_artifact: str | None = None,
) -> tuple[list[ProcessorSpec], list[ProcessorSpec]]:
    if hasattr(policy, "export_preprocessors") and hasattr(policy, "export_postprocessors"):
        return (
            policy.export_preprocessors(
                include_normalization=include_normalization,
                stats_artifact=stats_artifact,
                tokenizer_artifact=tokenizer_artifact,
            ),
            policy.export_postprocessors(
                include_normalization=include_normalization,
                stats_artifact=stats_artifact,
                tokenizer_artifact=tokenizer_artifact,
            ),
        )

    preprocessors: list[ProcessorSpec] = []
    postprocessors: list[ProcessorSpec] = []
    if include_normalization:
        input_features = get_normalized_input_features(policy)
        preprocessors.extend(
            build_normalize_processor_specs(
                get_normalization_groups(policy, input_features),
                artifact=stats_artifact,
            )
        )
        postprocessors.extend(
            build_denormalize_processor_specs(
                get_normalization_groups(policy, ["action"]),
                artifact=stats_artifact,
            )
        )
    return preprocessors, postprocessors


def _export_assets(policy: PreTrainedPolicy, output_dir: Path) -> dict[str, str]:
    if hasattr(policy, "export_assets"):
        return policy.export_assets(output_dir)
    return {}


def _export_stats(policy: PreTrainedPolicy, output_dir: Path, *, include_normalization: bool) -> str | None:
    if hasattr(policy, "export_stats"):
        return policy.export_stats(output_dir, include_normalization=include_normalization)

    if not include_normalization:
        return None

    stats = get_policy_stats(policy)
    if not stats:
        raise ValueError(
            f"cannot export policy {type(policy).__name__}: normalization stats required but not available"
        )
    stats_path = output_dir / "stats.safetensors"
    save_stats_safetensors(stats, stats_path)
    return stats_path.name


def _policy_class_path(policy: PreTrainedPolicy) -> str:
    policy_cls = type(policy)
    return f"{policy_cls.__module__}.{policy_cls.__qualname__}"


def export_policy(
    policy: PreTrainedPolicy,
    output_dir: str | Path,
    *,
    backend: str = "onnx",
    example_batch: dict[str, Tensor] | None = None,
    opset_version: int = DEFAULT_ONNX_OPSET,
    include_normalization: bool = True,
) -> Path:
    """Export a trained policy to a self-contained ``policy_package`` directory.

    Traces the policy's inference graph, serialises model artifacts via the
    chosen backend, bundles normalisation statistics and (for PI05) the
    tokenizer, then writes a ``manifest.json`` that fully describes the
    package for runtime loading.

    Args:
        policy: Trained policy instance implementing the
            :class:`~lerobot.export.protocols.Exportable` protocol.
        output_dir: Destination directory.  Created (including parents) if it
            does not exist.
        backend: Serialisation backend.  ``"onnx"`` (default) or
            ``"openvino"`` (runtime-only; serialises as ONNX).
        example_batch: Optional representative input batch used for tracing.
            When ``None`` a synthetic batch is generated automatically.
        opset_version: ONNX opset version passed to ``torch.onnx.export``.
        include_normalization: When ``True``, save normalisation statistics and
            add normalise/denormalise processor specs to the manifest.

    Returns:
        The resolved ``output_dir`` path.

    Raises:
        ValueError: If the backend is unknown or runtime-only, or if no runner
            matches the policy.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    assets_dir = output_dir / "assets"
    artifacts_dir.mkdir(exist_ok=True)
    assets_dir.mkdir(exist_ok=True)

    if example_batch is None:
        example_batch = generate_example_batch(policy)

    runner_cls = _select_runner(policy)
    modules, runner_cfg = runner_cls.export(policy, example_batch)
    serialization_backend = "onnx" if backend == "openvino" else backend
    backend_impl = BACKENDS.get(serialization_backend)
    if backend_impl is None:
        raise ValueError(f"Unknown backend: {backend!r}. Known: {sorted(BACKENDS) + ['openvino']}")
    if backend_impl.runtime_only:
        raise ValueError(f"Backend {serialization_backend!r} is runtime-only and cannot serialize a model.")
    artifacts = backend_impl.serialize(modules, artifacts_dir, opset_version=opset_version)
    export_assets = _export_assets(policy, output_dir)
    stats_artifact = _export_stats(policy, output_dir, include_normalization=include_normalization)
    preprocessors, postprocessors = build_processor_specs(
        policy,
        include_normalization=include_normalization and bool(stats_artifact),
        stats_artifact=stats_artifact or "stats.safetensors",
        tokenizer_artifact=export_assets.get("tokenizer_artifact"),
    )

    save_policy_config(policy, assets_dir / "config.json")
    runner_block = {"type": runner_cls.type, **runner_cfg}
    manifest = Manifest(
        policy=PolicyInfo(
            name=getattr(policy, "name", policy.__class__.__name__.lower()),
            source=PolicySource(
                repo_id=getattr(policy.config, "repo_id", None),
                revision=getattr(policy.config, "revision", None),
                class_path=_policy_class_path(policy),
            ),
        ),
        model=ModelConfig(
            n_obs_steps=getattr(policy.config, "n_obs_steps", 1),
            runner=runner_block,
            artifacts=artifacts,
            preprocessors=preprocessors or None,
            postprocessors=postprocessors or None,
        ),
        hardware=build_hardware_config(policy),
        metadata=Metadata(created_at=datetime.now(UTC).isoformat(), created_by="lerobot.export"),
    )
    manifest.save(output_dir / "manifest.json")
    return output_dir


def _select_runner(policy: PreTrainedPolicy) -> type[Runner]:
    for runner_cls in RUNNERS:
        if runner_cls.matches(policy):
            return runner_cls
    known = ", ".join(r.type for r in RUNNERS)
    raise ValueError(
        f"No runner matches {type(policy).__name__}. Known runner types: {known}. "
        "Implement the Exportable protocol from lerobot.export.protocols "
        "(get_inference_type, get_export_config, get_export_modules, prepare_inputs)."
    )
