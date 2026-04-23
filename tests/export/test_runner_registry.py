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

from typing import Any

import pytest
from torch import nn

from lerobot.export.exporter import _select_runner
from lerobot.export.runners.base import RUNNERS, ExportModule
from lerobot.export.runners.iterative import IterativeRunner
from lerobot.export.runners.kv_cache import KVCacheRunner
from lerobot.export.runners.single_pass import SinglePassRunner
from tests.export.conftest import (
    create_act_policy_and_batch,
    create_diffusion_policy_and_batch,
    create_pi0_policy_and_batch,
    create_pi05_policy_and_batch,
    create_smolvla_policy_and_batch,
)


class UnknownPolicy(nn.Module):
    pass


def test_select_runner_returns_expected_class_for_act() -> None:
    policy, _ = create_act_policy_and_batch()
    assert _select_runner(policy) is SinglePassRunner


def test_select_runner_returns_expected_class_for_diffusion() -> None:
    policy, _ = create_diffusion_policy_and_batch()
    assert _select_runner(policy) is IterativeRunner


def test_select_runner_returns_expected_class_for_pi0() -> None:
    pytest.importorskip("transformers")
    policy, _ = create_pi0_policy_and_batch(device="cuda")
    assert _select_runner(policy) is KVCacheRunner


def test_select_runner_returns_expected_class_for_pi05() -> None:
    pytest.importorskip("transformers")
    policy, _ = create_pi05_policy_and_batch(device="cuda")
    assert _select_runner(policy) is KVCacheRunner


def test_select_runner_returns_expected_class_for_smolvla() -> None:
    policy, _ = create_smolvla_policy_and_batch(device="cuda")
    assert _select_runner(policy) is KVCacheRunner


def test_select_runner_raises_for_unrecognized_policy() -> None:
    with pytest.raises(ValueError, match="No runner matches UnknownPolicy"):
        _select_runner(UnknownPolicy())


def test_runtime_registry_lookup_returns_concrete_classes() -> None:
    assert next(r for r in RUNNERS if r.type == "single_pass") is SinglePassRunner
    assert next(r for r in RUNNERS if r.type == "iterative") is IterativeRunner
    assert next(r for r in RUNNERS if r.type == "kv_cache") is KVCacheRunner


@pytest.mark.parametrize(
    ("factory", "kwargs", "expected_type", "expected_names"),
    [
        (create_act_policy_and_batch, {"device": "cpu"}, SinglePassRunner, ["model"]),
        (create_diffusion_policy_and_batch, {"device": "cpu"}, IterativeRunner, ["model"]),
        (create_pi0_policy_and_batch, {"device": "cuda"}, KVCacheRunner, ["encoder", "denoise"]),
        (create_pi05_policy_and_batch, {"device": "cuda"}, KVCacheRunner, ["encoder", "denoise"]),
        (create_smolvla_policy_and_batch, {"device": "cuda"}, KVCacheRunner, ["encoder", "denoise"]),
    ],
)
def test_runner_exports_have_structural_invariants(
    factory: Any,
    kwargs: dict[str, Any],
    expected_type: type,
    expected_names: list[str],
) -> None:
    if factory in {create_pi0_policy_and_batch, create_pi05_policy_and_batch}:
        pytest.importorskip("transformers")

    policy, batch = factory(**kwargs)
    matching = [runner for runner in RUNNERS if runner.matches(policy)]
    assert matching == [expected_type]

    modules, runner_cfg = expected_type.export(policy, batch)

    assert modules
    assert runner_cfg
    assert [module.name for module in modules] == expected_names

    for module in modules:
        assert isinstance(module, ExportModule)
        assert isinstance(module.wrapper, nn.Module)
        assert not module.wrapper.training
        assert module.example_inputs
        assert module.input_names
        assert module.output_names


def test_single_pass_runner_export_shape() -> None:
    policy, batch = create_act_policy_and_batch()
    modules, _ = SinglePassRunner.export(policy, batch)

    assert len(modules) == 1
    assert modules[0].name == "model"
    assert modules[0].output_names == ["action"]


def test_iterative_runner_export_shape() -> None:
    policy, batch = create_diffusion_policy_and_batch()
    modules, _ = IterativeRunner.export(policy, batch)

    assert len(modules) == 1
    assert "x_t" in modules[0].input_names
    assert "timestep" in modules[0].input_names
    assert modules[0].output_names == ["v_t"]


def test_kv_cache_runner_export_shape() -> None:
    pytest.importorskip("transformers")
    policy, batch = create_pi0_policy_and_batch(device="cuda")
    modules, _ = KVCacheRunner.export(policy, batch)

    assert [module.name for module in modules] == ["encoder", "denoise"]
