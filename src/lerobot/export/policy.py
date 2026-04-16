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
"""User-facing exported policy wrapper."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .runner import create_runner

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .manifest import Manifest
    from .runner import InferenceRunner


class ExportedPolicy:
    """User-facing wrapper around an exported policy package.

    This mirrors the eager policy interface by exposing ``select_action()`` and
    ``reset()`` as the primary API, while still allowing advanced callers to use
    ``predict_action_chunk()`` directly when they need raw chunk access.
    """

    def __init__(self, runner: InferenceRunner):
        self._runner = runner
        self._action_queue: deque[NDArray[np.floating]] = deque()

    @classmethod
    def from_package(
        cls,
        package_path: str | Path,
        backend: str | None = None,
        device: str = "cpu",
    ) -> ExportedPolicy:
        """Load an exported policy package."""
        runner = create_runner(package_path, backend=backend, device=device)
        return cls(runner)

    @property
    def manifest(self) -> Manifest:
        """Return the loaded export manifest."""
        return self._runner.manifest

    def reset(self) -> None:
        """Clear cached actions and reset the inner runner."""
        self._action_queue.clear()
        self._runner.reset()

    def predict_action_chunk(
        self,
        observation: dict[str, NDArray[np.floating]],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """Return the raw action chunk from the exported runtime."""
        return self._runner.predict_action_chunk(observation, **kwargs)

    def select_action(
        self,
        observation: dict[str, NDArray[np.floating]],
        **kwargs: Any,
    ) -> NDArray[np.floating]:
        """Return one action and manage chunk caching internally."""
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
