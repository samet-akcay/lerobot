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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from .base import register_backend, resolve_artifact_paths

if TYPE_CHECKING:
    from ..runners.base import ExportModule
    from .base import BackendSession

logger = logging.getLogger(__name__)


class ExecuTorchBackendSession:
    def __init__(self, methods: dict[str, Any], io_specs: dict[str, dict[str, list[str]]]):
        self._methods = methods
        self._io_specs = io_specs

    def run(self, name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        import torch

        io_spec = self._io_specs.get(name, {})
        input_names = io_spec.get("input_names", [])
        output_names = io_spec.get("output_names", [])

        if input_names:
            missing = [input_name for input_name in input_names if input_name not in inputs]
            if missing:
                raise ValueError(f"Missing required inputs for {name!r}: {missing}. Expected: {input_names}")
            ordered_inputs = [torch.from_numpy(inputs[input_name]) for input_name in input_names]
        else:
            ordered_inputs = [torch.from_numpy(value) for value in inputs.values()]

        outputs = self._methods[name].execute(ordered_inputs)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        names = (
            output_names
            if output_names and len(output_names) == len(outputs)
            else [f"output_{i}" for i in range(len(outputs))]
        )
        result: dict[str, np.ndarray] = {}
        for output_name, output in zip(names, outputs, strict=True):
            result[output_name] = output.numpy() if isinstance(output, torch.Tensor) else np.asarray(output)
        return result


class ExecuTorchRuntimeAdapter:
    def __init__(self, model_path: Path | str, device: str = "cpu"):
        self._session = ExecuTorchBackend().open(
            Path(model_path).parent,
            {"model": {"artifacts": {"model": f"artifacts/{Path(model_path).name}"}}},
            device=device,
        )

    @property
    def input_names(self) -> list[str]:
        return self._session._io_specs.get("model", {}).get("input_names", [])

    @property
    def output_names(self) -> list[str]:
        return self._session._io_specs.get("model", {}).get("output_names", [])

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return self._session.run("model", inputs)


@register_backend
class ExecuTorchBackend:
    name = "executorch"
    extension = ".pte"
    runtime_only = False

    def serialize(
        self,
        modules: list[ExportModule],
        artifacts_dir: Path,
        **kwargs: Any,
    ) -> dict[str, str]:
        del kwargs
        try:
            from executorch.exir import to_edge
            from torch.export import export as torch_export
        except ImportError as e:
            raise ImportError(
                "executorch is required for ExecuTorch backend. Install with: pip install executorch"
            ) from e

        artifacts: dict[str, str] = {}
        for module in modules:
            output_path = artifacts_dir / f"{module.name}{self.extension}"
            exported = torch_export(module.wrapper, module.example_inputs)
            edge = to_edge(exported)
            et_program = edge.to_executorch()
            with output_path.open("wb") as f:
                f.write(et_program.buffer)
            _write_io_spec_yaml(
                artifacts_dir / f"{module.name}_io_spec.yaml",
                module.input_names,
                module.output_names,
                extras=module.hints.get("executorch_io_spec_extras"),
            )
            if len(modules) == 1:
                _write_io_spec_yaml(
                    artifacts_dir / "io_spec.yaml",
                    module.input_names,
                    module.output_names,
                    extras=module.hints.get("executorch_io_spec_extras"),
                )
            artifacts[module.name] = f"artifacts/{output_path.name}"
        return artifacts

    def open(
        self,
        artifacts_dir: Path,
        manifest: dict[str, Any],
        *,
        device: str = "cpu",
    ) -> BackendSession:
        del device
        try:
            from executorch.runtime import Runtime
        except ImportError as e:
            raise ImportError(
                "executorch is required for ExecuTorch backend. Install with: pip install executorch"
            ) from e

        runtime = Runtime.get()
        methods: dict[str, Any] = {}
        io_specs: dict[str, dict[str, list[str]]] = {}
        artifact_paths = resolve_artifact_paths(artifacts_dir, manifest)
        for name, path in artifact_paths.items():
            if path.suffix != self.extension:
                continue
            program = runtime.load_program(path)
            methods[name] = program.load_method("forward")
            io_specs[name] = _read_io_spec(path)
        return ExecuTorchBackendSession(methods, io_specs)


def _write_io_spec_yaml(
    path: Path,
    input_names: list[str],
    output_names: list[str],
    *,
    extras: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {"input_names": input_names, "output_names": output_names}
    if extras:
        payload.update(extras)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False)


def _read_io_spec(model_path: Path) -> dict[str, list[str]]:
    model_specific = model_path.parent / f"{model_path.stem}_io_spec.yaml"
    generic = model_path.parent / "io_spec.yaml"
    metadata_path = model_specific if model_specific.exists() else generic
    if not metadata_path.exists():
        logger.warning("No io_spec.yaml found alongside %s; using positional I/O.", model_path)
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to read metadata from %s: %s", metadata_path, exc)
        return {}

    return {
        "input_names": [str(name) for name in metadata.get("input_names", [])],
        "output_names": [str(name) for name in metadata.get("output_names", [])],
    }
