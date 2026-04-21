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
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from .base import register_backend, resolve_artifact_paths

if TYPE_CHECKING:
    from ..interfaces import BackendSession
    from ..runners.base import ExportModule


class OpenVINOBackendSession:
    def __init__(
        self,
        infer_requests: dict[str, Any],
        input_names: dict[str, list[str]],
        output_names: dict[str, list[str]],
        input_name_mappings: dict[str, dict[str, str]],
        output_name_mappings: dict[str, dict[str, str]],
    ):
        self._infer_requests = infer_requests
        self._input_names = input_names
        self._output_names = output_names
        self._input_name_mappings = input_name_mappings
        self._output_name_mappings = output_name_mappings

    def run(self, name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        input_names = self._input_names[name]
        ov_inputs = {k: v for k, v in inputs.items() if k in input_names}
        missing = set(input_names) - set(ov_inputs)
        if missing:
            raise ValueError(f"Missing required inputs for {name!r}: {sorted(missing)}")

        mapped_inputs = {
            self._input_name_mappings[name].get(input_name, input_name): value
            for input_name, value in ov_inputs.items()
        }
        infer_request = self._infer_requests[name]
        infer_request.infer(mapped_inputs)

        outputs: dict[str, np.ndarray] = {}
        for output_name in self._output_names[name]:
            compiled_name = self._output_name_mappings[name].get(output_name, output_name)
            outputs[output_name] = infer_request.get_tensor(compiled_name).data.copy()
        return outputs


@register_backend
class OpenVINOBackend:
    name: ClassVar[str] = "openvino"
    extension: ClassVar[str] = ".onnx"
    runtime_only: ClassVar[bool] = True

    def serialize(
        self,
        modules: list[ExportModule],
        artifacts_dir: Path,
        **kwargs: Any,
    ) -> dict[str, str]:
        del modules, artifacts_dir, kwargs
        raise NotImplementedError("OpenVINO is runtime_only; export with backend='onnx' first")

    def open(
        self,
        artifacts_dir: Path,
        manifest: dict[str, Any],
        *,
        device: str = "cpu",
    ) -> BackendSession:
        try:
            import openvino as ov
        except ImportError as e:
            raise ImportError(
                "openvino is required for OpenVINO backend. Install with: pip install openvino"
            ) from e

        infer_requests: dict[str, Any] = {}
        input_names: dict[str, list[str]] = {}
        output_names: dict[str, list[str]] = {}
        input_name_mappings: dict[str, dict[str, str]] = {}
        output_name_mappings: dict[str, dict[str, str]] = {}

        core = ov.Core()
        for name, model_path in resolve_artifact_paths(artifacts_dir, manifest).items():
            if model_path.suffix != self.extension:
                continue
            model = core.read_model(str(model_path))
            original_input_names = [inp.get_any_name() for inp in model.inputs]
            original_output_names = [out.get_any_name() for out in model.outputs]
            compiled_model = core.compile_model(model, _normalize_device(device))
            infer_request = compiled_model.create_infer_request()
            compiled_input_names = [inp.get_any_name() for inp in compiled_model.inputs]
            compiled_output_names = [out.get_any_name() for out in compiled_model.outputs]

            infer_requests[name] = infer_request
            input_names[name] = original_input_names
            output_names[name] = original_output_names
            input_name_mappings[name] = {
                original: compiled
                for original, compiled in zip(original_input_names, compiled_input_names, strict=True)
                if original != compiled
            }
            output_name_mappings[name] = {
                original: compiled
                for original, compiled in zip(original_output_names, compiled_output_names, strict=True)
                if original != compiled
            }

        return OpenVINOBackendSession(
            infer_requests,
            input_names,
            output_names,
            input_name_mappings,
            output_name_mappings,
        )


def _normalize_device(device: str) -> str:
    device = device.lower()
    if device.startswith("cuda") or device.startswith("xpu"):
        return "GPU"
    mapping = {"cpu": "CPU", "gpu": "GPU", "npu": "NPU", "auto": "AUTO"}
    return mapping.get(device, device.upper())
