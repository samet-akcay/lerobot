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

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import backends as _backends  # noqa: F401
from .backends import BACKENDS
from .manifest import Manifest
from .runners.base import RUNNERS, Runner

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ExportedPolicy:
    def __init__(self, runner: Runner, manifest: Manifest):
        self._runner = runner
        self._manifest = manifest
        self._action_queue: deque[NDArray[np.floating]] = deque()

    @classmethod
    def load(
        cls,
        package_path: str | Path,
        backend: str | None = None,
        device: str = "cpu",
    ) -> ExportedPolicy:
        package_path = Path(package_path)
        manifest = Manifest.load(package_path / "manifest.json")
        manifest_dict = manifest.to_dict()
        runner_type = manifest.model.runner["type"]
        runner_cls = next((runner for runner in RUNNERS if runner.type == runner_type), None)
        if runner_cls is None:
            raise ValueError(f"Unknown runner type in manifest: {runner_type!r}")

        artifacts_dir = package_path / "artifacts"
        backend_name = backend or _detect_backend_name(manifest_dict, artifacts_dir)
        backend_impl = BACKENDS.get(backend_name)
        if backend_impl is None:
            raise ValueError(f"Unknown backend: {backend_name!r}. Known: {sorted(BACKENDS)}")
        sessions = backend_impl.open(artifacts_dir, manifest_dict, device=device)
        runner = runner_cls.load(manifest_dict, artifacts_dir, sessions)
        return cls(runner, manifest)

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    def reset(self) -> None:
        self._action_queue.clear()
        self._runner.reset()

    def predict_action_chunk(
        self,
        observation: dict[str, NDArray[np.floating]],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        return self._runner.predict_action_chunk(observation, **kwargs)

    def select_action(
        self,
        observation: dict[str, NDArray[np.floating]],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        if not self._action_queue:
            chunk = self.predict_action_chunk(observation, **kwargs)
            if chunk.ndim == 1:
                return chunk
            if chunk.ndim == 3:
                chunk = chunk[0]
            n_action_steps = self.manifest.model.runner.get("n_action_steps", len(chunk))
            for idx in range(min(n_action_steps, len(chunk))):
                self._action_queue.append(chunk[idx])
        return self._action_queue.popleft()


def _detect_backend_name(manifest: dict[str, Any], artifacts_dir: Path) -> str:
    declared = manifest["model"].get("backend")
    if declared:
        if declared not in BACKENDS:
            raise ValueError(
                f"Manifest declares backend={declared!r} but it is not registered. "
                f"Registered backends: {sorted(BACKENDS)}."
            )
        return declared

    artifact_names = {Path(path).name for path in manifest["model"]["artifacts"].values()}
    candidates = [
        backend_name
        for backend_name, backend_impl in BACKENDS.items()
        if not backend_impl.runtime_only
        and any((artifacts_dir / name).suffix == backend_impl.extension for name in artifact_names)
    ]
    if len(candidates) == 1:
        return candidates[0]
    suffixes = sorted({(artifacts_dir / name).suffix for name in artifact_names})
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple backends match artifacts {sorted(artifact_names)} "
            f"(suffixes: {suffixes}): {candidates}. "
            "Set model.backend in the manifest or pass backend=... explicitly to load_exported_policy()."
        )
    raise ValueError(
        f"Cannot detect backend for artifacts {sorted(artifact_names)} "
        f"(suffixes: {suffixes}). Registered backends: {sorted(BACKENDS)}. "
        "Pass backend=... explicitly to load_exported_policy()."
    )
