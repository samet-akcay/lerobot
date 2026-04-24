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

from dataclasses import dataclass
from typing import Any

PI05_TOKENIZER_NAME = "google/paligemma-3b-pt-224"


def _drop_none_values(spec: dict[str, object | None]) -> dict[str, object]:
    return {key: value for key, value in spec.items() if value is not None}


@dataclass(frozen=True)
class TokenizeProcessorEmitter:
    tokenizer_name: str
    max_length: int
    padding_side: str
    padding: str
    truncation: bool = True

    def to_processor_spec(self) -> dict[str, object]:
        return {
            "type": "tokenize",
            "tokenizer_name": self.tokenizer_name,
            "max_length": self.max_length,
            "padding_side": self.padding_side,
            "padding": self.padding,
            "truncation": self.truncation,
        }


@dataclass(frozen=True)
class RelativeActionsProcessorEmitter:
    enabled: bool
    exclude_joints: list[str]
    action_names: list[str] | None = None

    def to_processor_spec(self) -> dict[str, object]:
        return _drop_none_values(
            {
                "type": "relative_actions",
                "enabled": self.enabled,
                "exclude_joints": self.exclude_joints,
                "action_names": self.action_names,
            }
        )


@dataclass(frozen=True)
class AbsoluteActionsProcessorEmitter:
    enabled: bool

    def to_processor_spec(self) -> dict[str, object]:
        return {
            "type": "absolute_actions",
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class Pi05PrepareStateProcessorEmitter:
    max_state_dim: int

    def to_processor_spec(self) -> dict[str, object]:
        return {
            "type": "pi05_prepare_state",
            "max_state_dim": self.max_state_dim,
        }


def emit_pi05_processor_specs(config: Any) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    preprocessors = [
        RelativeActionsProcessorEmitter(
            enabled=getattr(config, "use_relative_actions", False),
            exclude_joints=list(getattr(config, "relative_exclude_joints", [])),
            action_names=getattr(config, "action_feature_names", None),
        ).to_processor_spec(),
        Pi05PrepareStateProcessorEmitter(max_state_dim=config.max_state_dim).to_processor_spec(),
        TokenizeProcessorEmitter(
            tokenizer_name=PI05_TOKENIZER_NAME,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ).to_processor_spec(),
    ]
    postprocessors = [
        AbsoluteActionsProcessorEmitter(enabled=getattr(config, "use_relative_actions", False)).to_processor_spec()
    ]
    return preprocessors, postprocessors


__all__ = [
    "AbsoluteActionsProcessorEmitter",
    "PI05_TOKENIZER_NAME",
    "Pi05PrepareStateProcessorEmitter",
    "RelativeActionsProcessorEmitter",
    "TokenizeProcessorEmitter",
    "emit_pi05_processor_specs",
]
