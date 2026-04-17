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
"""Runtime adapter protocol and factory for model execution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class RuntimeAdapter(Protocol):
    """Minimal interface for model execution.

    Runtime adapters are intentionally minimal—they execute a single forward pass.
    The runner handles the higher-level logic like normalization and iterative loops.
    """

    @property
    def input_names(self) -> list[str]:
        """Return the list of input tensor names."""
        ...

    @property
    def output_names(self) -> list[str]:
        """Return the list of output tensor names."""
        ...

    def run(self, inputs: dict[str, NDArray[np.floating]]) -> dict[str, NDArray[np.floating]]:
        """Execute one forward pass.

        Args:
            inputs: Dictionary mapping input names to numpy arrays.

        Returns:
            Dictionary mapping output names to numpy arrays.
        """
        ...


def get_runtime_adapter(adapter_name: str, model_path: Path, device: str = "cpu") -> RuntimeAdapter:
    """Factory function to get the appropriate runtime adapter.

    Args:
        adapter_name: Name of the runtime adapter ("onnx", "openvino", or "executorch").
        model_path: Path to the model file.
        device: Device for inference ("cpu", "cuda", "cuda:0").

    Returns:
        RuntimeAdapter instance ready for inference.

    Raises:
        ValueError: If the adapter is not supported.
    """
    if adapter_name == "onnx":
        from .onnx import ONNXRuntimeAdapter

        adapter = cast(object, ONNXRuntimeAdapter(model_path, device))
    elif adapter_name == "openvino":
        from .openvino import OpenVINORuntimeAdapter

        adapter = cast(object, OpenVINORuntimeAdapter(model_path, device))
    elif adapter_name == "executorch":
        from .executorch import ExecuTorchRuntimeAdapter

        adapter = cast(object, ExecuTorchRuntimeAdapter(model_path, device))
    else:
        raise ValueError(
            f"Unsupported runtime adapter: {adapter_name}. Supported: onnx, openvino, executorch"
        )

    if not isinstance(adapter, RuntimeAdapter):
        raise TypeError(f"Runtime adapter '{adapter_name}' must implement the RuntimeAdapter protocol.")

    return adapter
