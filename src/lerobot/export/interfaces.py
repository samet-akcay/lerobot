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

"""Public runtime-neutral protocols for the export subsystem.

This module is the canonical home for the ``Backend`` and ``BackendSession``
protocols. Both ``backends`` and ``runners`` packages import from here so
neither needs to depend on the other; this is the boundary that keeps the
two halves of the export subsystem decoupled regardless of artifact format
or runtime engine.

External consumers should import from this module:

    from lerobot.export.interfaces import Backend, BackendSession
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from .runners.base import ExportModule


@runtime_checkable
class BackendSession(Protocol):
    def run(self, name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


@runtime_checkable
class Backend(Protocol):
    name: ClassVar[str]
    extension: ClassVar[str]
    runtime_only: ClassVar[bool] = False

    def serialize(
        self,
        modules: list[ExportModule],
        artifacts_dir: Path,
        **kwargs: Any,
    ) -> dict[str, str]: ...

    def open(
        self,
        artifacts_dir: Path,
        manifest: dict[str, Any],
        *,
        device: str = "cpu",
    ) -> BackendSession: ...
