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
"""Manifest schema for PolicyPackage v1.0.

The manifest is the contract between export and runtime. It is pure JSON data—no code references.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

MANIFEST_FORMAT = "lerobot_exported_policy"
MANIFEST_VERSION = "1.0"


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return _dataclass_to_dict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        if not value:
            return None
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        if not value:
            return None
        return {key: _serialize_value(val) for key, val in value.items() if val is not None}
    return value


def _dataclass_to_dict(instance: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_info in fields(instance):
        value = getattr(instance, field_info.name)
        if value is None:
            continue
        serialized = _serialize_value(value)
        if serialized is None:
            continue
        result[field_info.name] = serialized
    return result


def _build_dataclass(
    cls: type[Any],
    data: dict[str, Any],
    converters: dict[str, Callable[[Any], Any]] | None = None,
) -> Any:
    values: dict[str, Any] = {}
    field_converters = converters or {}
    for field_info in fields(cls):
        if field_info.name in data:
            raw_value = data[field_info.name]
            if field_info.name in field_converters:
                values[field_info.name] = field_converters[field_info.name](raw_value)
            else:
                values[field_info.name] = raw_value
            continue
        if field_info.default is not MISSING or field_info.default_factory is not MISSING:
            continue
    return cls(**values)


class NormalizationType(str, Enum):
    """Normalization type for stats."""

    STANDARD = "standard"
    MIN_MAX = "min_max"
    QUANTILES = "quantiles"
    QUANTILE10 = "quantile10"
    IDENTITY = "identity"


@dataclass
class TensorSpec:
    """Specification for an input or output tensor."""

    name: str
    dtype: str
    shape: list[str | int]
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorSpec:
        return _build_dataclass(cls, data)


@dataclass
class PolicySource:
    """Source information for policy provenance."""

    repo_id: str | None = None
    revision: str | None = None
    commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicySource:
        return _build_dataclass(cls, data)


@dataclass
class PolicyInfo:
    """Policy metadata."""

    name: str
    kind: str | None = None
    source: PolicySource | None = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyInfo:
        return _build_dataclass(
            cls,
            data,
            converters={"source": PolicySource.from_dict},
        )


@dataclass
class IOSpec:
    """Input/output specifications."""

    inputs: list[TensorSpec]
    outputs: list[TensorSpec]

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IOSpec:
        return _build_dataclass(
            cls,
            data,
            converters={
                "inputs": lambda items: [TensorSpec.from_dict(item) for item in items],
                "outputs": lambda items: [TensorSpec.from_dict(item) for item in items],
            },
        )


@dataclass
class ActionSpec:
    """Action semantics specification."""

    dim: int
    chunk_size: int
    n_action_steps: int
    representation: str = "absolute"
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionSpec:
        return _build_dataclass(cls, data)


@dataclass
class IterativeConfig:
    """Configuration for iterative policies (flow matching, diffusion)."""

    num_steps: int = 10
    scheduler: str = "euler"
    timestep_spacing: str = "linear"
    timestep_range: list[float] = field(default_factory=lambda: [1.0, 0.0])
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        result = _dataclass_to_dict(self)
        if self.scheduler not in ("ddpm", "ddim"):
            for key in (
                "num_train_timesteps",
                "beta_start",
                "beta_end",
                "beta_schedule",
                "prediction_type",
                "clip_sample",
                "clip_sample_range",
            ):
                result.pop(key, None)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IterativeConfig:
        return _build_dataclass(cls, data)


@dataclass
class TwoPhaseConfig:
    """Configuration for two-phase VLA policies (PI0, SmolVLA)."""

    num_steps: int = 10
    encoder_artifact: str = "onnx_encoder"
    denoise_artifact: str = "onnx_denoise"
    num_layers: int = 18
    num_kv_heads: int = 8
    head_dim: int = 256
    input_mapping: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TwoPhaseConfig:
        return _build_dataclass(cls, data)


# Discriminated union type for inference configs
InferenceConfig = IterativeConfig | TwoPhaseConfig


def inference_config_from_dict(data: dict[str, Any]) -> InferenceConfig:
    """Parse inference config from dict, inferring type from structure.

    - Has `encoder_artifact` -> TwoPhaseConfig
    - Has `scheduler` -> IterativeConfig
    """
    if "encoder_artifact" in data:
        return TwoPhaseConfig.from_dict(data)
    elif "scheduler" in data:
        return IterativeConfig.from_dict(data)
    else:
        raise ValueError(
            "Cannot determine inference config type. "
            "Expected 'encoder_artifact' (TwoPhaseConfig) or 'scheduler' (IterativeConfig)."
        )


def inference_config_to_dict(config: InferenceConfig) -> dict[str, Any]:
    """Convert inference config to dict."""
    return config.to_dict()


def is_two_phase(config: InferenceConfig | None) -> bool:
    """Check if config is TwoPhaseConfig."""
    return isinstance(config, TwoPhaseConfig)


def is_iterative(config: InferenceConfig | None) -> bool:
    """Check if config is IterativeConfig."""
    return isinstance(config, IterativeConfig)


@dataclass
class NormalizationConfig:
    """Normalization configuration and stats location."""

    type: NormalizationType
    artifact: str
    input_features: list[str]
    output_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizationConfig:
        return _build_dataclass(
            cls,
            data,
            converters={"type": NormalizationType},
        )


@dataclass
class ExportMetadata:
    """Export metadata (timestamps, versions)."""

    created_at: str | None = None
    created_by: str = "lerobot.export"
    lerobot_version: str | None = None
    export_device: str = "cpu"
    export_dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        result = _dataclass_to_dict(self)
        result.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportMetadata:
        return _build_dataclass(cls, data)


@dataclass
class Manifest:
    """PolicyPackage manifest schema v1.0.

    The manifest is the contract between export and runtime. It is pure JSON data—no code references.

    The inference pattern is determined by the structure of the `inference` field:
    - None -> single-pass (ACT, Groot)
    - IterativeConfig -> iterative (Diffusion)
    - TwoPhaseConfig -> two-phase (PI0, SmolVLA)
    """

    policy: PolicyInfo
    artifacts: dict[str, str]
    io: IOSpec
    action: ActionSpec
    format: str = MANIFEST_FORMAT
    version: str = MANIFEST_VERSION
    inference: InferenceConfig | None = None
    normalization: NormalizationConfig | None = None
    metadata: ExportMetadata | None = None

    def __post_init__(self):
        """Validate manifest after construction."""
        self.validate()

    @property
    def is_single_pass(self) -> bool:
        """Check if this is a single-pass policy (no inference loop)."""
        return self.inference is None

    @property
    def is_iterative(self) -> bool:
        """Check if this is an iterative policy (Diffusion)."""
        return is_iterative(self.inference)

    @property
    def is_two_phase(self) -> bool:
        """Check if this is a two-phase policy (PI0, SmolVLA)."""
        return is_two_phase(self.inference)

    def validate(self) -> None:
        """Validate the manifest schema.

        Raises:
            ValueError: If validation fails.
        """
        if self.format != MANIFEST_FORMAT:
            raise ValueError(f"Invalid format: {self.format}, expected {MANIFEST_FORMAT}")

        if not self.version.startswith("1."):
            raise ValueError(f"Unsupported version: {self.version}, expected 1.x")

        if not self.artifacts:
            raise ValueError("At least one artifact is required")

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to a dictionary for JSON serialization."""
        result = _dataclass_to_dict(self)
        if self.inference:
            result["inference"] = inference_config_to_dict(self.inference)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        """Create a manifest from a dictionary (e.g., parsed JSON)."""
        inference: InferenceConfig | None = None
        if "inference" in data:
            inference = inference_config_from_dict(data["inference"])

        return _build_dataclass(
            cls,
            data,
            converters={
                "policy": PolicyInfo.from_dict,
                "io": IOSpec.from_dict,
                "action": ActionSpec.from_dict,
                "inference": lambda _: inference,
                "normalization": NormalizationConfig.from_dict,
                "metadata": ExportMetadata.from_dict,
            },
        )

    def save(self, path: Path | str) -> None:
        """Save manifest to a JSON file.

        Args:
            path: Path to save the manifest to.
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: Path | str) -> Manifest:
        """Load manifest from a JSON file.

        Args:
            path: Path to load the manifest from.

        Returns:
            Loaded manifest instance.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
