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
"""ExecuTorch runtime adapter for model execution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    import torch
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ExecuTorchRuntimeAdapter:
    """ExecuTorch runtime adapter for model inference.

    This adapter loads and runs models exported to the ExecuTorch ``.pte``
    format. Input and output names are read from ``metadata.yaml``
    colocated with the model, since ``.pte`` files do not embed
    named I/O metadata like ONNX.
    """

    def __init__(self, model_path: Path | str, device: str = "cpu"):
        """Initialize the ExecuTorch runtime adapter.

        Args:
            model_path: Path to the ``.pte`` model file.
            device: Device hint (ExecuTorch Python runtime always runs on CPU).

        Raises:
            ImportError: If the ``executorch`` package is not installed.
            FileNotFoundError: If the model path does not exist.
        """
        try:
            from executorch.runtime import Runtime
        except ImportError as e:
            raise ImportError(
                "executorch is required for ExecuTorch backend. Install with: pip install executorch"
            ) from e

        self._model_path = Path(model_path)
        self._device = device

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")

        runtime = Runtime.get()
        self._program = runtime.load_program(self._model_path)
        self._method: Any = self._program.load_method("forward")

        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load input/output name metadata from ``metadata.yaml`` or ``{model_name}_metadata.yaml``."""
        model_stem = self._model_path.stem
        model_specific = self._model_path.parent / f"{model_stem}_metadata.yaml"
        generic = self._model_path.parent / "metadata.yaml"

        metadata_path = model_specific if model_specific.exists() else generic

        if not metadata_path.exists():
            logger.warning("No metadata.yaml found alongside %s; using positional I/O.", self._model_path)
            return

        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = yaml.safe_load(f) or {}
            self._input_names = [str(n) for n in metadata.get("input_names", [])]
            self._output_names = [str(n) for n in metadata.get("output_names", [])]
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Failed to read metadata from %s: %s", metadata_path, exc)

    @property
    def input_names(self) -> list[str]:
        """Return the list of input tensor names."""
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Return the list of output tensor names."""
        return self._output_names

    def run(self, inputs: dict[str, NDArray[np.floating]]) -> dict[str, NDArray[np.floating]]:
        """Execute one forward pass.

        Args:
            inputs: Dictionary mapping input names to numpy arrays.

        Returns:
            Dictionary mapping output names to numpy arrays.
        """
        import torch as _torch

        if self._input_names:
            missing = [n for n in self._input_names if n not in inputs]
            if missing:
                raise ValueError(f"Missing required inputs: {missing}. Expected: {self._input_names}")
            ordered: list[torch.Tensor] = [
                _torch.from_numpy(inputs[name])
                if not isinstance(inputs[name], _torch.Tensor)
                else inputs[name]
                for name in self._input_names
            ]
        else:
            ordered = [
                _torch.from_numpy(v) if not isinstance(v, _torch.Tensor) else v for v in inputs.values()
            ]

        outputs = self._method.execute(ordered)

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        if self._output_names and len(self._output_names) == len(outputs):
            names = self._output_names
        else:
            names = [f"output_{i}" for i in range(len(outputs))]

        result: dict[str, NDArray[np.floating]] = {}
        for name, out in zip(names, outputs, strict=True):
            result[name] = out.numpy() if isinstance(out, _torch.Tensor) else np.asarray(out)

        return result

    def __repr__(self) -> str:
        return f"ExecuTorchRuntimeAdapter(model={self._model_path.name}, device={self._device})"
