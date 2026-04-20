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

import json
from pathlib import Path

import pytest

from tests.export.conftest import create_smolvla_policy_and_batch, require_executorch


@require_executorch
@pytest.mark.slow
def test_kv_cache_executorch_export_creates_expected_artifacts(tmp_path: Path) -> None:
    from lerobot.export import export_policy

    policy, batch = create_smolvla_policy_and_batch(device="cuda")
    package_path = export_policy(policy, tmp_path / "smolvla_et", backend="executorch", example_batch=batch)

    assert (package_path / "artifacts" / "encoder.pte").exists()
    assert (package_path / "artifacts" / "denoise.pte").exists()
    assert (package_path / "artifacts" / "encoder_io_spec.yaml").exists()
    assert (package_path / "artifacts" / "denoise_io_spec.yaml").exists()

    manifest = json.loads((package_path / "manifest.json").read_text())
    runner = manifest["model"]["runner"]
    assert runner["type"] == "kv_cache"
    for key in [
        "num_layers",
        "num_kv_heads",
        "head_dim",
        "num_inference_steps",
        "chunk_size",
        "action_dim",
        "state_dim",
    ]:
        assert key in runner
