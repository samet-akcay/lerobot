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
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .manifest import CameraConfig, HardwareConfig, RobotConfig, TensorSpec

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


def generate_example_batch(policy: PreTrainedPolicy) -> dict[str, Tensor]:
    config = policy.config
    batch_size = 1
    device = next(policy.parameters()).device
    batch: dict[str, Tensor] = {}
    if hasattr(config, "robot_state_feature") and config.robot_state_feature:
        batch["observation.state"] = torch.randn(
            batch_size, config.robot_state_feature.shape[0], device=device
        )
    if hasattr(config, "env_state_feature") and config.env_state_feature:
        batch["observation.environment_state"] = torch.randn(
            batch_size, config.env_state_feature.shape[0], device=device
        )
    if hasattr(config, "image_features") and config.image_features:
        for img_key, img_feature in config.image_features.items():
            batch[img_key] = torch.randn(batch_size, *img_feature.shape, device=device)
    return batch


def build_hardware_config(policy: PreTrainedPolicy) -> HardwareConfig | None:
    config = policy.config
    robots: list[RobotConfig] = []
    cameras: list[CameraConfig] = []
    action_dim = get_action_dim(config)
    state_dim = (
        config.robot_state_feature.shape[0]
        if hasattr(config, "robot_state_feature") and config.robot_state_feature
        else None
    )
    if state_dim or action_dim:
        robot = RobotConfig(name="main")
        if state_dim:
            robot.state = TensorSpec(shape=[state_dim], dtype="float32")
        if action_dim:
            robot.action = TensorSpec(shape=[action_dim], dtype="float32")
        robots.append(robot)
    if hasattr(config, "image_features") and config.image_features:
        for img_key, img_feature in config.image_features.items():
            cameras.append(
                CameraConfig(name=img_key.split(".")[-1], shape=list(img_feature.shape), dtype="uint8")
            )
    if not robots and not cameras:
        return None
    return HardwareConfig(robots=robots or None, cameras=cameras or None)


def get_action_dim(config: Any) -> int:
    if hasattr(config, "max_action_dim") and config.max_action_dim is not None:
        return config.max_action_dim
    if hasattr(config, "action_feature") and config.action_feature is not None:
        return config.action_feature.shape[0]
    return 14


def get_policy_stats(policy: PreTrainedPolicy) -> dict[str, dict[str, Any]] | None:
    if hasattr(policy, "policy_processor"):
        for step in getattr(policy.policy_processor, "steps", []):
            if hasattr(step, "stats") and step.stats:
                return step.stats
    if hasattr(policy, "config") and hasattr(policy.config, "stats"):
        return policy.config.stats
    return None


def get_normalization_mode(policy: PreTrainedPolicy) -> str:
    return "mean_std"


def get_normalized_input_features(policy: PreTrainedPolicy) -> list[str]:
    return (
        ["observation.state"]
        if hasattr(policy.config, "robot_state_feature") and policy.config.robot_state_feature
        else []
    )


def save_policy_config(policy: PreTrainedPolicy, path: Path) -> None:
    try:
        config_dict = {k: v for k, v in policy.config.__dict__.copy().items() if is_json_serializable(v)}
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
    except Exception as e:
        logger.debug("Could not save policy config to %s: %s", path, e)


def is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
