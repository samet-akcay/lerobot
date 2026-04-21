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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from torch import Tensor, nn

from ..interfaces import BackendSession
from ..manifest import ProcessorSpec
from ..normalize import Normalizer

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ExportModule:
    name: str
    wrapper: nn.Module
    example_inputs: tuple[Tensor, ...]
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]] | None = None
    hints: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Runner(Protocol):
    type: ClassVar[str]

    @classmethod
    def matches(cls, policy: object) -> bool: ...

    @classmethod
    def export(
        cls,
        policy: object,
        example_batch: dict[str, Tensor],
    ) -> tuple[list[ExportModule], dict[str, Any]]: ...

    @classmethod
    def load(
        cls,
        manifest: dict[str, Any],
        artifacts_dir: Path,
        backend_session: BackendSession,
    ) -> Runner: ...

    def predict_action_chunk(self, batch: dict[str, np.ndarray]) -> np.ndarray: ...

    def reset(self) -> None: ...


RUNNERS: list[type[Runner]] = []

# Maps deprecated runner type aliases to their canonical name. Kept so that
# packages exported by older versions still load. New manifests should use
# the canonical type name; aliases may be removed in a future release.
RUNNER_TYPE_ALIASES: dict[str, str] = {
    "action_chunking": "single_shot",
}


def resolve_runner_type(runner_type: str) -> str:
    """Return the canonical runner type, normalizing legacy aliases."""
    return RUNNER_TYPE_ALIASES.get(runner_type, runner_type)


def register_runner(cls: type[Runner]) -> type[Runner]:
    RUNNERS.append(cls)
    return cls


def build_dynamic_axes(input_names: list[str], output_names: list[str]) -> dict[str, dict[int, str]]:
    dynamic_axes: dict[str, dict[int, str]] = {}
    for name in input_names:
        dynamic_axes[name] = {0: "batch_size"}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}
    return dynamic_axes


def build_normalizer(manifest: dict[str, Any], package_path: Path) -> Normalizer | None:
    model = manifest["model"]
    preprocessors = model.get("preprocessors")
    postprocessors = model.get("postprocessors")
    return Normalizer.from_specs(
        [ProcessorSpec.from_dict(spec) for spec in preprocessors] if preprocessors else None,
        [ProcessorSpec.from_dict(spec) for spec in postprocessors] if postprocessors else None,
        package_path,
    )


def get_output_by_names(
    outputs: dict[str, NDArray[np.floating]],
    primary_name: str,
    fallback_names: list[str],
    context: str,
) -> NDArray[np.floating]:
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
