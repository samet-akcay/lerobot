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

import builtins
from pathlib import Path

import pytest

from lerobot.export.backends.executorch import _read_io_spec, _write_io_spec_yaml


def _patch_missing_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_write_io_spec_yaml_raises_clean_importerror_when_yaml_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_missing_yaml(monkeypatch)

    with pytest.raises(ImportError, match="PyYAML is required for the ExecuTorch backend"):
        _write_io_spec_yaml(tmp_path / "io_spec.yaml", input_names=["x"], output_names=["y"])


def test_read_io_spec_raises_clean_importerror_when_yaml_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.pte"
    (tmp_path / "io_spec.yaml").write_text("input_names: [x]\noutput_names: [y]\n")

    _patch_missing_yaml(monkeypatch)

    with pytest.raises(ImportError, match="PyYAML is required for the ExecuTorch backend"):
        _read_io_spec(model_path)
