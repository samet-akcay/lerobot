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

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from ..interfaces import BackendSession

if TYPE_CHECKING:
    from ..runners.base import ExportModule


__all__ = ["BACKENDS", "Backend", "BackendSession", "register_backend", "resolve_artifact_paths"]


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


BACKENDS: dict[str, Backend] = {}


def register_backend(cls: type[Backend]) -> type[Backend]:
    backend = cls()
    if not hasattr(backend, "runtime_only"):
        backend.runtime_only = False
    BACKENDS[cls.name] = backend
    return cls


def resolve_artifact_paths(artifacts_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    return {
        name: artifacts_dir / Path(relative_path).name
        for name, relative_path in manifest["model"]["artifacts"].items()
    }
