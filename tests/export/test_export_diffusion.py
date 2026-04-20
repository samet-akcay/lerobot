#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np
import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")
diffusers = pytest.importorskip("diffusers")

from tests.export.conftest import (  # noqa: E402
    assert_numerical_parity,
    create_diffusion_policy_and_batch,
    require_executorch,
    to_numpy,
)


class TestDiffusionExport:
    @pytest.mark.slow
    def test_export_creates_valid_package(self, tmp_path: Path):
        policy, batch = create_diffusion_policy_and_batch()

        package_path = policy.export(
            tmp_path / "diffusion_package",
            backend="onnx",
            example_batch=batch,
        )

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "model.onnx").exists()

    @pytest.mark.slow
    def test_detected_as_iterative(self, tmp_path: Path):
        from lerobot.export.manifest import Manifest

        policy, batch = create_diffusion_policy_and_batch()

        package_path = policy.to_onnx(
            tmp_path / "diffusion_package",
            example_batch=batch,
        )

        manifest = Manifest.load(package_path / "manifest.json")
        assert manifest.is_iterative
        assert manifest.runner_type == "iterative"
        assert manifest.model.runner["num_inference_steps"] > 0

    @pytest.mark.slow
    def test_scheduler_config_exported(self, tmp_path: Path):
        from lerobot.export.manifest import Manifest

        policy, batch = create_diffusion_policy_and_batch()

        package_path = policy.to_onnx(
            tmp_path / "diffusion_package",
            example_batch=batch,
        )

        manifest = Manifest.load(package_path / "manifest.json")
        runner = manifest.model.runner
        assert runner["type"] == "iterative"
        assert runner["scheduler"] == "ddim"
        assert runner["num_train_timesteps"] == policy.config.num_train_timesteps
        assert runner["beta_schedule"] == policy.config.beta_schedule
        assert runner["prediction_type"] == policy.config.prediction_type

    @pytest.mark.slow
    def test_onnx_numerical_parity(self, tmp_path: Path):
        from lerobot.export import load_exported_policy
        from lerobot.utils.constants import OBS_IMAGES

        policy, batch = create_diffusion_policy_and_batch()

        torch.manual_seed(42)
        np.random.seed(42)
        noise = torch.randn(1, policy.config.horizon, policy.config.action_feature.shape[0])

        stacked_batch = {
            "observation.state": batch["observation.state"],
            OBS_IMAGES: batch["observation.images.top"].unsqueeze(2),
        }

        with torch.no_grad():
            global_cond = policy.diffusion._prepare_global_conditioning(stacked_batch)
            pytorch_output = policy.diffusion.conditional_sample(1, global_cond=global_cond, noise=noise)

        package_path = policy.to_onnx(
            tmp_path / "diffusion_package",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load_exported_policy(package_path, backend="onnx", device="cpu")

        obs_numpy = to_numpy(batch)
        onnx_output = runtime.predict_action_chunk(obs_numpy, noise=noise.numpy())

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        assert_numerical_parity(
            onnx_output,
            pytorch_np,
            rtol=1e-4,
            atol=1e-4,
            msg="Diffusion ONNX output does not match PyTorch output",
        )

    @pytest.mark.slow
    def test_openvino_numerical_parity(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import load_exported_policy

        policy, batch = create_diffusion_policy_and_batch()

        package_path = policy.to_openvino(
            tmp_path / "diffusion_package",
            example_batch=batch,
            include_normalization=False,
        )

        np.random.seed(42)
        noise = np.random.randn(1, policy.config.horizon, policy.config.action_feature.shape[0]).astype(
            np.float32
        )

        obs_numpy = to_numpy(batch)

        runtime_onnx = load_exported_policy(package_path, backend="onnx", device="cpu")
        onnx_output = runtime_onnx.predict_action_chunk(obs_numpy, noise=noise)

        runtime_openvino = load_exported_policy(package_path, backend="openvino", device="cpu")
        openvino_output = runtime_openvino.predict_action_chunk(obs_numpy, noise=noise)

        assert_numerical_parity(
            openvino_output,
            onnx_output,
            rtol=1e-4,
            atol=1e-4,
            msg="Diffusion OpenVINO output does not match ONNX output",
        )

    @pytest.mark.slow
    def test_openvino_numerical_parity_with_pytorch(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import load_exported_policy
        from lerobot.utils.constants import OBS_IMAGES

        policy, batch = create_diffusion_policy_and_batch()

        torch.manual_seed(42)
        np.random.seed(42)
        noise = torch.randn(1, policy.config.horizon, policy.config.action_feature.shape[0])

        stacked_batch = {
            "observation.state": batch["observation.state"],
            OBS_IMAGES: batch["observation.images.top"].unsqueeze(2),
        }

        with torch.no_grad():
            global_cond = policy.diffusion._prepare_global_conditioning(stacked_batch)
            pytorch_output = policy.diffusion.conditional_sample(1, global_cond=global_cond, noise=noise)

        package_path = policy.to_openvino(
            tmp_path / "diffusion_package",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load_exported_policy(package_path, backend="openvino", device="cpu")
        obs_numpy = to_numpy(batch)
        ov_output = runtime.predict_action_chunk(obs_numpy, noise=noise.numpy())

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        assert_numerical_parity(
            ov_output,
            pytorch_np,
            rtol=1e-4,
            atol=1e-4,
            msg="Diffusion OpenVINO output does not match PyTorch output",
        )


class TestDiffusionNormalization:
    @pytest.mark.slow
    def test_per_feature_normalization_modes_in_manifest(self, tmp_path: Path):
        import json

        policy, batch = create_diffusion_policy_and_batch()
        policy.config.stats = {
            "observation.state": {
                "min": [0.0] * 6,
                "max": [1.0] * 6,
                "mean": [0.5] * 6,
                "std": [0.1] * 6,
            },
            "observation.images.top": {
                "mean": [[[0.5]], [[0.5]], [[0.5]]],
                "std": [[[0.1]], [[0.1]], [[0.1]]],
            },
            "action": {
                "min": [0.0] * 6,
                "max": [1.0] * 6,
                "mean": [0.5] * 6,
                "std": [0.1] * 6,
            },
        }

        package_path = policy.export(
            tmp_path / "diffusion_norm",
            backend="onnx",
            example_batch=batch,
            include_normalization=True,
        )
        manifest = json.loads((package_path / "manifest.json").read_text())

        preprocessors = manifest["model"]["preprocessors"]
        postprocessors = manifest["model"]["postprocessors"]

        pre_by_mode = {p["mode"]: set(p["features"]) for p in preprocessors}
        assert pre_by_mode.get("mean_std") == {"observation.images.top"}, (
            f"VISUAL must use mean_std per Diffusion's normalization_mapping; got {pre_by_mode}"
        )
        assert pre_by_mode.get("min_max") == {"observation.state"}, (
            f"STATE must use min_max per Diffusion's normalization_mapping; got {pre_by_mode}"
        )

        assert len(postprocessors) == 1
        assert postprocessors[0]["mode"] == "min_max"
        assert postprocessors[0]["features"] == ["action"]

    @pytest.mark.slow
    def test_normalizer_applies_per_feature_modes_at_runtime(self, tmp_path: Path):
        import numpy as np

        from lerobot.export.manifest import Manifest
        from lerobot.export.normalize import Normalizer

        policy, batch = create_diffusion_policy_and_batch()
        policy.config.stats = {
            "observation.state": {
                "min": np.full((6,), -2.0, dtype=np.float32),
                "max": np.full((6,), 2.0, dtype=np.float32),
                "mean": np.zeros(6, dtype=np.float32),
                "std": np.ones(6, dtype=np.float32),
            },
            "observation.images.top": {
                "mean": np.full((3, 1, 1), 0.25, dtype=np.float32),
                "std": np.full((3, 1, 1), 0.5, dtype=np.float32),
                "min": np.zeros((3, 1, 1), dtype=np.float32),
                "max": np.ones((3, 1, 1), dtype=np.float32),
            },
            "action": {
                "min": np.full((6,), -1.0, dtype=np.float32),
                "max": np.full((6,), 1.0, dtype=np.float32),
                "mean": np.zeros(6, dtype=np.float32),
                "std": np.ones(6, dtype=np.float32),
            },
        }

        package_path = policy.export(
            tmp_path / "diffusion_runtime_norm",
            backend="onnx",
            example_batch=batch,
            include_normalization=True,
        )

        manifest = Manifest.load(package_path / "manifest.json")
        normalizer = Normalizer.from_specs(
            manifest.model.preprocessors,
            manifest.model.postprocessors,
            package_path,
        )
        assert normalizer is not None

        state = np.full((1, 6), 1.0, dtype=np.float32)
        image = np.full((1, 3, 4, 4), 0.75, dtype=np.float32)
        normalized = normalizer.normalize_inputs(
            {"observation.state": state, "observation.images.top": image}
        )

        np.testing.assert_allclose(normalized["observation.state"], 0.5, atol=1e-6)
        np.testing.assert_allclose(normalized["observation.images.top"], 1.0, atol=1e-6)

        action = np.full((6,), 0.5, dtype=np.float32)
        recovered = normalizer.denormalize_outputs(action, key="action")
        np.testing.assert_allclose(recovered, 0.5, atol=1e-6)


class TestDiffusionRuntime:
    @pytest.mark.slow
    def test_from_exported_loads_user_facing_policy(self, tmp_path: Path):
        from lerobot.export import ExportedPolicy

        policy, batch = create_diffusion_policy_and_batch()

        package_path = policy.to_onnx(
            tmp_path / "diffusion_package",
            example_batch=batch,
        )

        runtime = policy.from_exported(package_path, backend="onnx", device="cpu")

        assert isinstance(runtime, ExportedPolicy)


class TestDiffusionExecuTorch:
    @require_executorch
    @pytest.mark.slow
    def test_executorch_export_creates_valid_package(self, tmp_path: Path):
        policy, batch = create_diffusion_policy_and_batch()

        package_path = policy.to_executorch(tmp_path / "diff_et", example_batch=batch)

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "model.pte").exists()
        assert (package_path / "artifacts" / "io_spec.yaml").exists()

    @require_executorch
    @pytest.mark.slow
    def test_executorch_numerical_parity_with_onnx(self, tmp_path: Path):
        from lerobot.export import load_exported_policy

        policy, batch = create_diffusion_policy_and_batch()

        onnx_pkg = policy.to_onnx(tmp_path / "diff_onnx", example_batch=batch, include_normalization=False)
        et_pkg = policy.to_executorch(tmp_path / "diff_et", example_batch=batch, include_normalization=False)

        noise = (
            np.random.default_rng(42)
            .standard_normal((1, policy.config.horizon, policy.config.action_feature.shape[0]))
            .astype(np.float32)
        )

        obs_numpy = to_numpy(batch)

        onnx_rt = load_exported_policy(onnx_pkg, backend="onnx", device="cpu")
        et_rt = load_exported_policy(et_pkg, backend="executorch", device="cpu")

        onnx_output = onnx_rt.predict_action_chunk(obs_numpy, noise=noise)
        et_output = et_rt.predict_action_chunk(obs_numpy, noise=noise)

        assert_numerical_parity(
            et_output,
            onnx_output,
            rtol=1e-4,
            atol=1e-4,
            msg="Diffusion ExecuTorch output does not match ONNX output",
        )

    @require_executorch
    @pytest.mark.slow
    def test_executorch_numerical_parity_with_openvino(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import load_exported_policy

        policy, batch = create_diffusion_policy_and_batch()

        onnx_pkg = policy.to_openvino(
            tmp_path / "diff_onnx", example_batch=batch, include_normalization=False
        )
        et_pkg = policy.to_executorch(tmp_path / "diff_et", example_batch=batch, include_normalization=False)

        noise = (
            np.random.default_rng(42)
            .standard_normal((1, policy.config.horizon, policy.config.action_feature.shape[0]))
            .astype(np.float32)
        )

        obs_numpy = to_numpy(batch)

        ov_rt = load_exported_policy(onnx_pkg, backend="openvino", device="cpu")
        et_rt = load_exported_policy(et_pkg, backend="executorch", device="cpu")

        ov_output = ov_rt.predict_action_chunk(obs_numpy, noise=noise)
        et_output = et_rt.predict_action_chunk(obs_numpy, noise=noise)

        assert_numerical_parity(
            et_output,
            ov_output,
            rtol=1e-4,
            atol=1e-4,
            msg="Diffusion ExecuTorch output does not match OpenVINO output",
        )
