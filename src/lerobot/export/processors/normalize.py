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

from dataclasses import dataclass


def _drop_none_values(spec: dict[str, object | None]) -> dict[str, object]:
    return {key: value for key, value in spec.items() if value is not None}


@dataclass(frozen=True)
class NormalizeProcessorEmitter:
    mode: str
    artifact: str | None
    features: list[str]

    def to_processor_spec(self) -> dict[str, object]:
        return _drop_none_values(
            {
                "type": "normalize",
                "mode": self.mode,
                "artifact": self.artifact,
                "features": self.features,
            }
        )


@dataclass(frozen=True)
class DenormalizeProcessorEmitter:
    mode: str
    artifact: str | None
    features: list[str]

    def to_processor_spec(self) -> dict[str, object]:
        return _drop_none_values(
            {
                "type": "denormalize",
                "mode": self.mode,
                "artifact": self.artifact,
                "features": self.features,
            }
        )


def emit_normalize_processor_specs(
    groups: list[tuple[str, list[str]]],
    *,
    artifact: str | None,
) -> list[dict[str, object]]:
    return [
        NormalizeProcessorEmitter(mode=mode, artifact=artifact, features=features).to_processor_spec()
        for mode, features in groups
    ]


def emit_denormalize_processor_specs(
    groups: list[tuple[str, list[str]]],
    *,
    artifact: str | None,
) -> list[dict[str, object]]:
    return [
        DenormalizeProcessorEmitter(mode=mode, artifact=artifact, features=features).to_processor_spec()
        for mode, features in groups
    ]


__all__ = [
    "DenormalizeProcessorEmitter",
    "NormalizeProcessorEmitter",
    "emit_denormalize_processor_specs",
    "emit_normalize_processor_specs",
]
