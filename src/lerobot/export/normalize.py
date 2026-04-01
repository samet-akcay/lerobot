#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Normalizer for applying dataset statistics during inference.

Supports normalization modes:

- ``mean_std`` (a.k.a. "standard"): ``(x - mean) / std``
- ``min_max``: maps ``[min, max] → [-1, 1]``
- ``identity``: passthrough
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .manifest import ProcessorSpec

# Map various mode names to canonical mode identifiers
_MODE_ALIASES: dict[str, str] = {
    "mean_std": "mean_std",
    "standard": "mean_std",
    "min_max": "min_max",
    "identity": "identity",
}


class Normalizer:
    """Handles normalization and denormalization of inputs/outputs.

    Loads statistics from safetensors files and applies per-feature
    transforms compatible with numpy arrays for runtime inference.
    """

    def __init__(
        self,
        mode: str,
        stats: dict[str, dict[str, NDArray[np.floating]]],
        input_features: set[str],
        output_features: set[str],
        eps: float = 1e-8,
    ) -> None:
        self._mode = _MODE_ALIASES.get(mode, mode)
        self._stats = stats
        self._input_features = input_features
        self._output_features = output_features
        self._eps = eps

    # -- factory methods -------------------------------------------------

    @classmethod
    def from_specs(
        cls,
        preprocessors: list[ProcessorSpec] | None,
        postprocessors: list[ProcessorSpec] | None,
        package_path: Path | str,
        eps: float = 1e-8,
    ) -> Normalizer | None:
        """Build a Normalizer from preprocessor/postprocessor specs.

        Returns ``None`` if no normalize/denormalize specs are present.
        """
        package_path = Path(package_path)

        norm_spec = None
        input_features: set[str] = set()
        output_features: set[str] = set()

        for spec in preprocessors or []:
            if spec.type == "normalize":
                norm_spec = spec
                input_features.update(spec.features or [])

        for spec in postprocessors or []:
            if spec.type == "denormalize":
                if norm_spec is None:
                    norm_spec = spec
                output_features.update(spec.features or [])

        if norm_spec is None or not norm_spec.artifact:
            return None

        stats_path = package_path / norm_spec.artifact
        if not stats_path.exists():
            return None

        mode = norm_spec.mode or "mean_std"
        stats = _load_stats(stats_path)
        return cls(mode, stats, input_features, output_features, eps)

    @classmethod
    def from_safetensors(
        cls,
        path: Path | str,
        mode: str = "mean_std",
        input_features: list[str] | None = None,
        output_features: list[str] | None = None,
        eps: float = 1e-8,
    ) -> Normalizer:
        """Load normalizer directly from a safetensors file."""
        stats = _load_stats(path)
        return cls(
            mode=mode,
            stats=stats,
            input_features=set(input_features or []),
            output_features=set(output_features or []),
            eps=eps,
        )

    # -- public API ------------------------------------------------------

    def normalize_inputs(
        self,
        observation: dict[str, NDArray[np.floating]],
    ) -> dict[str, NDArray[np.floating]]:
        """Apply normalization to input features."""
        result = dict(observation)
        for key in self._input_features:
            if key in result and key in self._stats:
                result[key] = self._apply_transform(result[key], key, inverse=False)
        return result

    def denormalize_outputs(
        self,
        action: NDArray[np.floating],
        key: str = "action",
    ) -> NDArray[np.floating]:
        """Apply denormalization to an output array."""
        if key in self._output_features and key in self._stats:
            return self._apply_transform(action, key, inverse=True)
        return action

    # -- internals -------------------------------------------------------

    def _apply_transform(
        self,
        tensor: NDArray[np.floating],
        key: str,
        *,
        inverse: bool = False,
    ) -> NDArray[np.floating]:
        if self._mode == "identity" or key not in self._stats:
            return tensor

        stats = self._stats[key]

        if self._mode == "mean_std":
            return self._apply_mean_std(tensor, stats, inverse)
        elif self._mode == "min_max":
            return self._apply_min_max(tensor, stats, inverse)

        return tensor

    def _apply_mean_std(
        self,
        tensor: NDArray[np.floating],
        stats: dict[str, NDArray[np.floating]],
        inverse: bool,
    ) -> NDArray[np.floating]:
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None or std is None:
            return tensor

        if inverse:
            return tensor * std + mean
        return (tensor - mean) / (std + self._eps)

    def _apply_min_max(
        self,
        tensor: NDArray[np.floating],
        stats: dict[str, NDArray[np.floating]],
        inverse: bool,
    ) -> NDArray[np.floating]:
        min_val = stats.get("min")
        max_val = stats.get("max")
        if min_val is None or max_val is None:
            return tensor

        denom = max_val - min_val
        denom = np.where(denom == 0, self._eps, denom)

        if inverse:
            return (tensor + 1) / 2 * denom + min_val
        return 2 * (tensor - min_val) / denom - 1


# ---------------------------------------------------------------------------
# Stats I/O
# ---------------------------------------------------------------------------


def _load_stats(path: Path | str) -> dict[str, dict[str, NDArray[np.floating]]]:
    """Load stats from a safetensors file into a nested dict."""
    try:
        from safetensors.numpy import load_file
    except ImportError as e:
        raise ImportError("safetensors is required. Install with: pip install safetensors") from e

    flat = load_file(str(path))
    stats: dict[str, dict[str, NDArray[np.floating]]] = {}
    for flat_key, tensor in flat.items():
        feature_name, stat_name = flat_key.rsplit(".", 1)
        if feature_name not in stats:
            stats[feature_name] = {}
        stats[feature_name][stat_name] = tensor.astype(np.float32)
    return stats


def save_stats_safetensors(
    stats: dict[str, dict[str, Any]],
    path: Path | str,
) -> None:
    """Save normalization statistics to a safetensors file.

    Args:
        stats: Nested ``{feature: {stat_name: array}}`` dict.
        path: Output path for the safetensors file.
    """
    try:
        from safetensors.numpy import save_file
    except ImportError as e:
        raise ImportError("safetensors is required. Install with: pip install safetensors") from e

    flat: dict[str, NDArray[np.floating]] = {}
    for feature_name, feature_stats in stats.items():
        for stat_name, value in feature_stats.items():
            flat_key = f"{feature_name}.{stat_name}"
            if isinstance(value, np.ndarray):
                flat[flat_key] = value.astype(np.float32)
            else:
                flat[flat_key] = np.array(value, dtype=np.float32)

    save_file(flat, str(path))
