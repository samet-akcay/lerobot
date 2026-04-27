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

from ..manifest import ProcessorSpec


def build_normalize_processor_specs(
    groups: list[tuple[str, list[str]]],
    *,
    artifact: str | None,
) -> list[ProcessorSpec]:
    """Build a list of ``"normalize"`` processor specs from mode groups.

    Args:
        groups: List of ``(mode, features)`` tuples where ``mode`` is the
            normalisation mode and ``features`` is the list of feature keys.
        artifact: Relative path to the stats file shared by all specs.

    Returns:
        Ordered list of processor specs ready for the manifest.
    """
    return [
        ProcessorSpec(type="normalize", mode=mode, artifact=artifact, features=features)
        for mode, features in groups
    ]


def build_denormalize_processor_specs(
    groups: list[tuple[str, list[str]]],
    *,
    artifact: str | None,
) -> list[ProcessorSpec]:
    """Build a list of ``"denormalize"`` processor specs from mode groups.

    Args:
        groups: List of ``(mode, features)`` tuples where ``mode`` is the
            normalisation mode and ``features`` is the list of feature keys.
        artifact: Relative path to the stats file shared by all specs.

    Returns:
        Ordered list of processor specs ready for the manifest.
    """
    return [
        ProcessorSpec(type="denormalize", mode=mode, artifact=artifact, features=features)
        for mode, features in groups
    ]


__all__ = [
    "build_denormalize_processor_specs",
    "build_normalize_processor_specs",
]
