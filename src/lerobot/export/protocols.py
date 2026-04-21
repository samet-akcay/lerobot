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
"""Export protocols for policy classes.

A single :class:`Exportable` Protocol defines protocols that policies can
implement to provide clean, self-contained export logic. Policies declare their
inference pattern via :meth:`Exportable.get_inference_type` and provide one or
more named modules via :meth:`Exportable.get_export_modules`.

Multi-stage policies whose later-stage input shapes depend on a prior stage's
output (e.g. KV-cache VLAs needing ``prefix_len`` from the encoder) implement
the optional :meth:`Exportable.prepare_runtime_inputs` hook. Single-stage
policies leave it unimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor, nn


@dataclass
class SinglePhaseExportConfig:
    """Configuration for single-phase (single-pass) export.

    Used by policies like ACT and VQ-BeT that produce actions in one forward pass.
    """

    chunk_size: int
    action_dim: int
    n_action_steps: int | None = None


@dataclass
class IterativeExportConfig:
    """Configuration for iterative (denoising) export.

    Used by policies like Diffusion that iteratively refine actions.
    """

    horizon: int
    action_dim: int
    num_inference_steps: int
    scheduler_type: str = "ddpm"


@dataclass
class KVCacheExportConfig:
    """Configuration for KV-cache (VLA) export.

    Captures architecture-specific information needed to export a KV-cache
    policy and reconstruct the KV cache at runtime.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int

    chunk_size: int
    action_dim: int
    state_dim: int | None
    num_steps: int

    input_mapping: dict[str, str] = field(default_factory=dict)


ExportConfig = SinglePhaseExportConfig | IterativeExportConfig | KVCacheExportConfig


@dataclass
class ExportInputs:
    """Tensor inputs and ONNX naming for a single export stage (module).

    A stage is one entry returned by :meth:`Exportable.get_export_modules`.
    For KV-cache policies, the encoder stage may also populate ``metadata``
    with values needed by downstream stages (e.g. ``num_images``,
    ``input_mapping``).
    """

    tensors: tuple[Tensor, ...]
    input_names: list[str]
    output_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Exportable(Protocol):
    """Unified export protocol implemented by every exportable policy.

    A policy declares its inference pattern via :meth:`get_inference_type`,
    returns a config dataclass via :meth:`get_export_config`, and exposes one
    or more nn.Modules via :meth:`get_export_modules` keyed by stage name.

    Stage names by convention:
    - Single-stage policies (ACT, Diffusion): ``{"model": ...}``
    - KV-cache policies (PI0, PI05, SmolVLA): ``{"encoder": ..., "denoise": ...}``

    The runner consumes :meth:`prepare_inputs` for stages whose tracing tensors
    can be derived from ``example_batch`` alone, then invokes
    :meth:`prepare_runtime_inputs` for stages whose shapes depend on a prior
    stage's output (e.g. denoise needing ``prefix_len`` from encoder output).
    """

    def get_inference_type(self) -> str:
        """Return the inference pattern identifier.

        One of: ``"single_shot"``, ``"iterative"``, ``"kv_cache"``.
        The legacy value ``"action_chunking"`` is accepted as an alias
        for ``"single_shot"``.
        """
        ...

    def get_export_config(self) -> ExportConfig:
        """Return the export configuration dataclass for this policy."""
        ...

    def get_export_modules(self) -> dict[str, nn.Module]:
        """Return one or more nn.Modules to export, keyed by stage name.

        Single-stage policies return ``{"model": module}``. Multi-stage policies
        (KV-cache) return ``{"encoder": ..., "denoise": ...}``. Each returned
        module must be ``.eval()`` and ready for ONNX/ExecuTorch tracing.
        """
        ...

    def prepare_inputs(self, example_batch: dict[str, Tensor]) -> dict[str, ExportInputs]:
        """Return tracing inputs for stages that can be prepared from ``example_batch`` alone.

        Stages whose input shapes depend on a prior stage's runtime output
        (e.g. KV-cache ``denoise`` needing ``prefix_len``) MUST be omitted here
        and provided via :meth:`prepare_runtime_inputs`.
        """
        ...

    def prepare_runtime_inputs(
        self,
        stage_name: str,
        runtime_context: dict[str, Any],
    ) -> ExportInputs:
        """Return tracing inputs for a stage whose shapes depend on prior-stage output.

        Single-stage policies do not need to implement this method. Multi-stage
        policies implement it for stages with runtime data dependencies.

        Args:
            stage_name: Name of the stage being prepared (matches a key in
                ``get_export_modules``).
            runtime_context: Values produced by prior stages. For KV-cache
                ``denoise``, this contains ``{"prefix_len": int, "device": ...}``.
        """
        ...


def is_exportable(policy: Any) -> bool:
    """Check if a policy implements the unified Exportable Protocol."""
    return isinstance(policy, Exportable)
