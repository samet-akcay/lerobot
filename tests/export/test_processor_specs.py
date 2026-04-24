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
from types import SimpleNamespace

from lerobot.export.exporter import build_processor_specs
from lerobot.export.manifest import Manifest, ModelConfig, PolicyInfo, ProcessorSpec


class _Feature:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


class ACTPolicyStub:
    __module__ = "lerobot.policies.act.modeling_act"

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            robot_state_feature=_Feature((14,)),
            env_state_feature=None,
            image_features={},
            normalization_mapping={"STATE": "mean_std", "ACTION": "mean_std"},
        )


class PI05PolicyStub:
    __module__ = "lerobot.policies.pi05.modeling_pi05"

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            robot_state_feature=_Feature((14,)),
            env_state_feature=None,
            image_features={},
            normalization_mapping={"STATE": "quantiles", "ACTION": "quantiles"},
            use_relative_actions=True,
            relative_exclude_joints=["gripper"],
            action_feature_names=["joint_1", "gripper"],
            max_state_dim=32,
            tokenizer_max_length=200,
        )


def _assert_roundtrip(spec: ProcessorSpec) -> None:
    payload = spec.to_dict()
    validated = ProcessorSpec.from_dict(payload)
    assert validated.to_dict() == payload

    json_payload = json.loads(json.dumps(payload))
    assert ProcessorSpec.from_dict(json_payload).to_dict() == payload


def test_act_normalization_specs_validate_and_roundtrip() -> None:
    preprocessors, postprocessors = build_processor_specs(
        ACTPolicyStub(),
        include_normalization=True,
        stats_artifact="artifacts/stats.safetensors",
    )

    assert [spec.type for spec in preprocessors] == ["normalize"]
    assert [spec.type for spec in postprocessors] == ["denormalize"]

    for spec in [*preprocessors, *postprocessors]:
        _assert_roundtrip(spec)


def test_pi05_processor_specs_validate_and_roundtrip() -> None:
    preprocessors, postprocessors = build_processor_specs(
        PI05PolicyStub(),
        include_normalization=True,
        stats_artifact="artifacts/stats.safetensors",
    )

    assert [spec.type for spec in preprocessors] == [
        "relative_actions",
        "normalize",
        "pi05_prepare_state",
        "tokenize",
    ]
    assert [spec.type for spec in postprocessors] == ["denormalize", "absolute_actions"]

    relative_actions = preprocessors[0].to_dict()
    assert relative_actions["enabled"] is True
    assert relative_actions["exclude_joints"] == ["gripper"]
    assert relative_actions["action_names"] == ["joint_1", "gripper"]

    tokenize = preprocessors[-1].to_dict()
    assert tokenize["tokenizer_name"] == "google/paligemma-3b-pt-224"
    assert tokenize["max_length"] == 200

    for spec in [*preprocessors, *postprocessors]:
        _assert_roundtrip(spec)


def test_manifest_processor_specs_roundtrip_preserves_flat_custom_params(tmp_path) -> None:
    preprocessors, postprocessors = build_processor_specs(
        PI05PolicyStub(),
        include_normalization=True,
        stats_artifact="artifacts/stats.safetensors",
    )

    manifest = Manifest(
        policy=PolicyInfo(name="pi05"),
        model=ModelConfig(
            n_obs_steps=1,
            runner={"type": "kv_cache"},
            artifacts={"encoder": "artifacts/encoder.onnx", "denoise": "artifacts/denoise.onnx"},
            preprocessors=preprocessors,
            postprocessors=postprocessors,
        ),
    )

    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    loaded = Manifest.load(manifest_path)

    assert [spec.to_dict() for spec in loaded.model.preprocessors] == [spec.to_dict() for spec in preprocessors]
    assert [spec.to_dict() for spec in loaded.model.postprocessors] == [spec.to_dict() for spec in postprocessors]
