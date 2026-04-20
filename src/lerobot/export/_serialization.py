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
from typing import Callable

import torch

from .runners.base import ExportModule


def serialize_modules(
    modules: list[ExportModule],
    artifacts_dir: Path,
    *,
    backend: str,
    opset_version: int,
    apply_onnx_fixups: Callable[[Path, ExportModule], None],
) -> dict[str, str]:
    if backend == "onnx":
        return _serialize_onnx(modules, artifacts_dir, opset_version, apply_onnx_fixups)
    if backend == "executorch":
        return _serialize_executorch(modules, artifacts_dir)
    raise ValueError(f"Unsupported backend: {backend}")


def _serialize_onnx(
    modules: list[ExportModule],
    artifacts_dir: Path,
    opset_version: int,
    apply_onnx_fixups: Callable[[Path, ExportModule], None],
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for module in modules:
        output_path = artifacts_dir / f"{module.name}.onnx"
        torch.onnx.export(
            module.wrapper,
            module.example_inputs,
            str(output_path),
            input_names=module.input_names,
            output_names=module.output_names,
            dynamic_axes=module.dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )
        apply_onnx_fixups(output_path, module)
        artifacts[module.name] = f"artifacts/{output_path.name}"
    return artifacts


def _serialize_executorch(modules: list[ExportModule], artifacts_dir: Path) -> dict[str, str]:
    from executorch.exir import to_edge
    from torch.export import export as torch_export

    artifacts: dict[str, str] = {}
    for module in modules:
        output_path = artifacts_dir / f"{module.name}.pte"
        exported = torch_export(module.wrapper, module.example_inputs)
        edge = to_edge(exported)
        et_program = edge.to_executorch()
        with output_path.open("wb") as f:
            f.write(et_program.buffer)
        if len(modules) == 1:
            _write_io_spec_yaml(artifacts_dir, module.input_names, module.output_names)
        else:
            _write_et_model_io_spec(artifacts_dir, module.name, module.input_names, module.output_names)
        artifacts[module.name] = f"artifacts/{output_path.name}"
    return artifacts


def _write_io_spec_yaml(artifacts_dir: Path, input_names: list[str], output_names: list[str]) -> None:
    import yaml

    with (artifacts_dir / "io_spec.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"input_names": input_names, "output_names": output_names}, f, default_flow_style=False
        )


def _write_et_model_io_spec(
    artifacts_dir: Path,
    model_name: str,
    input_names: list[str],
    output_names: list[str],
) -> None:
    import yaml

    with (artifacts_dir / f"{model_name}_io_spec.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"input_names": input_names, "output_names": output_names}, f, default_flow_style=False
        )
