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

"""Neutral interface module for the export subsystem.

Holds Protocols that are shared between ``backends`` and ``runners`` so that
neither package needs to import from the other to satisfy type references.
This breaks the latent circular dependency where ``runners.base`` referenced
``backends.base.BackendSession`` while ``backends.base`` referenced
``runners.base.ExportModule``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BackendSession(Protocol):
    def run(self, name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...
