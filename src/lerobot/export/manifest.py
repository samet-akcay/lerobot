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
"""Manifest schema for the converged ``policy_package`` v1.0 format.

The manifest is the contract between export and runtime.  It is pure JSON
data — no code references, no framework-specific class paths.

Both LeRobot and PhysicalAI read and write this same schema.  LeRobot uses
the ``type`` + flat-params style for components (runners, preprocessors,
postprocessors); PhysicalAI can additionally use ``class_path`` +
``init_args`` for full-power component instantiation.

Schema overview::

    manifest.json
    ├── format + version          (envelope)
    ├── policy                    (identity — what policy is this?)
    │   ├── name
    │   └── source                (provenance: repo_id, class_path)
    ├── model                     (exported model — how to run it?)
    │   ├── n_obs_steps
    │   ├── runner                (execution pattern + parameters)
    │   ├── artifacts             (model files by named role)
    │   ├── preprocessors         (input transforms)
    │   └── postprocessors        (output transforms)
    ├── hardware                  (deployment — what hardware?)
    │   ├── robots
    │   └── cameras
    └── metadata                  (provenance — when/who created this?)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

MANIFEST_FORMAT = "policy_package"
MANIFEST_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Generic dataclass serialization helpers
# ---------------------------------------------------------------------------


def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value for JSON output."""
    if is_dataclass(value):
        return _to_dict(value)
    if isinstance(value, list):
        if not value:
            return None
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        if not value:
            return None
        return {key: _serialize_value(val) for key, val in value.items() if val is not None}
    return value


def _to_dict(instance: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a dict, omitting ``None`` values."""
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


def _from_dict(
    cls: type[Any],
    data: dict[str, Any],
    converters: dict[str, Callable[[Any], Any]] | None = None,
) -> Any:
    """Instantiate a dataclass from a dict, applying optional *converters*."""
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


# ---------------------------------------------------------------------------
# policy section
# ---------------------------------------------------------------------------


@dataclass
class PolicySource:
    """Provenance information for the exported policy."""

    repo_id: str | None = None
    revision: str | None = None
    class_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicySource:
        return _from_dict(cls, data)


@dataclass
class PolicyInfo:
    """Policy identity section."""

    name: str
    source: PolicySource | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyInfo:
        return _from_dict(cls, data, converters={"source": PolicySource.from_dict})


# ---------------------------------------------------------------------------
# model section
# ---------------------------------------------------------------------------


@dataclass
class ProcessorSpec:
    """Specification for a preprocessor or postprocessor entry.

    Uses the ``type`` + flat-params format for interoperability::

        {"type": "normalize", "mode": "mean_std", "artifact": "stats.safetensors", "features": [...]}
    """

    type: str
    mode: str | None = None
    artifact: str | None = None
    features: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessorSpec:
        return _from_dict(cls, data)


@dataclass
class ModelConfig:
    """Model configuration — how to run the exported policy.

    The ``runner`` field is an open-ended dict with a ``type`` key that
    determines the inference pattern.  Policy-specific parameters sit
    alongside ``type`` as flat keys.
    """

    n_obs_steps: int
    runner: dict[str, Any]
    artifacts: dict[str, str]
    preprocessors: list[ProcessorSpec] | None = None
    postprocessors: list[ProcessorSpec] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "n_obs_steps": self.n_obs_steps,
            "runner": self.runner,
            "artifacts": self.artifacts,
        }
        if self.preprocessors:
            result["preprocessors"] = [p.to_dict() for p in self.preprocessors]
        if self.postprocessors:
            result["postprocessors"] = [p.to_dict() for p in self.postprocessors]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        return _from_dict(
            cls,
            data,
            converters={
                "preprocessors": lambda items: [ProcessorSpec.from_dict(item) for item in items],
                "postprocessors": lambda items: [ProcessorSpec.from_dict(item) for item in items],
            },
        )


# ---------------------------------------------------------------------------
# hardware section
# ---------------------------------------------------------------------------


@dataclass
class TensorSpec:
    """Shape and dtype specification for a hardware tensor (state/action)."""

    shape: list[int]
    dtype: str = "float32"
    order: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorSpec:
        return _from_dict(cls, data)


@dataclass
class RobotConfig:
    """Robot hardware declaration."""

    name: str
    type: str | None = None
    state: TensorSpec | None = None
    action: TensorSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotConfig:
        return _from_dict(
            cls,
            data,
            converters={
                "state": TensorSpec.from_dict,
                "action": TensorSpec.from_dict,
            },
        )


@dataclass
class CameraConfig:
    """Camera hardware declaration."""

    name: str
    shape: list[int] | None = None
    dtype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraConfig:
        return _from_dict(cls, data)


@dataclass
class HardwareConfig:
    """Hardware section — what the policy expects at inference time."""

    robots: list[RobotConfig] | None = None
    cameras: list[CameraConfig] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareConfig:
        return _from_dict(
            cls,
            data,
            converters={
                "robots": lambda items: [RobotConfig.from_dict(item) for item in items],
                "cameras": lambda items: [CameraConfig.from_dict(item) for item in items],
            },
        )


# ---------------------------------------------------------------------------
# metadata section
# ---------------------------------------------------------------------------


@dataclass
class Metadata:
    """Export provenance metadata."""

    created_at: str | None = None
    created_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metadata:
        return _from_dict(cls, data)


# ---------------------------------------------------------------------------
# Top-level Manifest
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    """Policy-package manifest v1.0.

    This is the converged schema shared by LeRobot and PhysicalAI.  The
    runner ``type`` determines the inference pattern:

    - ``action_chunking`` — single forward pass with action chunk queue
    - ``iterative`` — multi-step denoising / flow-matching
    - ``two_phase`` — encode once, then iterative denoise
    """

    policy: PolicyInfo
    model: ModelConfig
    format: str = MANIFEST_FORMAT
    version: str = MANIFEST_VERSION
    hardware: HardwareConfig | None = None
    metadata: Metadata | None = None

    def __post_init__(self) -> None:
        self.validate()

    # -- convenience properties ------------------------------------------

    @property
    def runner_type(self) -> str:
        """Return the runner type string (e.g. ``"action_chunking"``)."""
        return self.model.runner.get("type", "action_chunking")

    @property
    def is_action_chunking(self) -> bool:
        return self.runner_type == "action_chunking"

    @property
    def is_iterative(self) -> bool:
        return self.runner_type == "iterative"

    @property
    def is_two_phase(self) -> bool:
        return self.runner_type == "two_phase"

    # -- validation ------------------------------------------------------

    def validate(self) -> None:
        """Validate required fields.

        Raises:
            ValueError: If validation fails.
        """
        if self.format != MANIFEST_FORMAT:
            raise ValueError(f"Invalid format: {self.format!r}, expected {MANIFEST_FORMAT!r}")
        if not self.version.startswith("1."):
            raise ValueError(f"Unsupported version: {self.version!r}, expected 1.x")
        if not self.model.artifacts:
            raise ValueError("At least one artifact is required in model.artifacts")
        if "type" not in self.model.runner:
            raise ValueError("model.runner must contain a 'type' key")

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        result: dict[str, Any] = {
            "format": self.format,
            "version": self.version,
            "policy": self.policy.to_dict(),
            "model": self.model.to_dict(),
        }
        if self.hardware:
            result["hardware"] = self.hardware.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        """Create a manifest from a dict (e.g. parsed JSON)."""
        return _from_dict(
            cls,
            data,
            converters={
                "policy": PolicyInfo.from_dict,
                "model": ModelConfig.from_dict,
                "hardware": HardwareConfig.from_dict,
                "metadata": Metadata.from_dict,
            },
        )

    # -- file I/O --------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save manifest to a JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> Manifest:
        """Load manifest from a JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
