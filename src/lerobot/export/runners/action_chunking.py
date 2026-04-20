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

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ..interfaces import BackendSession
from ..protocols import Exportable, is_exportable
from .base import ExportModule, build_dynamic_axes, build_normalizer, get_output_by_names, register_runner

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


@register_runner
class ActionChunkingRunner:
    type: ClassVar[str] = "action_chunking"

    def __init__(self, manifest: dict[str, Any], artifacts_dir: Path, adapter: BackendSession):
        self._manifest = manifest
        self._adapter = adapter
        self._normalizer = build_normalizer(manifest, artifacts_dir.parent)

    @classmethod
    def matches(cls, policy: object) -> bool:
        return is_exportable(policy) and policy.get_inference_type() == cls.type

    @classmethod
    def export(
        cls,
        policy: object,
        example_batch: dict[str, Tensor],
    ) -> tuple[list[ExportModule], dict[str, Any]]:
        exportable = policy_as_exportable(policy)
        export_config = exportable.get_export_config()
        modules = exportable.get_export_modules()
        stage = exportable.prepare_inputs(example_batch)["model"]
        export_module = ExportModule(
            name="model",
            wrapper=modules["model"],
            example_inputs=stage.tensors,
            input_names=stage.input_names,
            output_names=["action"],
            dynamic_axes=build_dynamic_axes(stage.input_names, ["action"]),
        )
        runner_cfg = {
            "chunk_size": export_config.chunk_size,
            "n_action_steps": export_config.n_action_steps or export_config.chunk_size,
            "action_dim": export_config.action_dim,
        }
        return [export_module], runner_cfg

    @classmethod
    def load(
        cls,
        manifest: dict[str, Any],
        artifacts_dir: Path,
        backend_session: BackendSession,
    ) -> ActionChunkingRunner:
        return cls(manifest, artifacts_dir, backend_session)

    def predict_action_chunk(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        obs = self._normalizer.normalize_inputs(batch) if self._normalizer else batch
        obs = {k: v.astype(np.float32) for k, v in obs.items()}

        outputs = self._adapter.run("model", obs)

        action = get_output_by_names(
            outputs,
            primary_name="action",
            fallback_names=[],
            context="ActionChunkingRunner",
        )

        if self._normalizer:
            action = self._normalizer.denormalize_outputs(action, key="action")

        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]

        return action

    def reset(self) -> None:
        return None


def policy_as_exportable(policy: object) -> Exportable:
    if not is_exportable(policy):
        raise TypeError(f"{type(policy).__name__} does not implement Exportable Protocol.")
    return policy
