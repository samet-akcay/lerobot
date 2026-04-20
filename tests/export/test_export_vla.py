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
transformers = pytest.importorskip("transformers")

from tests.export.conftest import (  # noqa: E402
    assert_numerical_parity,
    create_pi0_policy_and_batch,
    create_pi05_policy_and_batch,
    create_smolvla_policy_and_batch,
    require_executorch,
    to_numpy,
)


class TestPI0Export:
    @pytest.mark.slow
    def test_export_creates_valid_package(self, tmp_path: Path):
        pytest.importorskip("transformers")
        from lerobot.export import export_policy
        from lerobot.export.manifest import Manifest

        policy, batch = create_pi0_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "pi0_package",
            backend="onnx",
            example_batch=batch,
        )

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "encoder.onnx").exists()
        assert (package_path / "artifacts" / "denoise.onnx").exists()

        manifest = Manifest.load(package_path / "manifest.json")
        assert manifest.is_kv_cache
        assert manifest.model.runner["type"] == "kv_cache"
        assert manifest.model.runner["num_inference_steps"] == policy.config.num_inference_steps

    @pytest.mark.slow
    def test_onnx_numerical_parity(self, tmp_path: Path):
        pytest.importorskip("transformers")
        from lerobot.export import export_policy, load_exported_policy
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

        policy, batch = create_pi0_policy_and_batch(device="cuda")

        torch.manual_seed(42)
        np.random.seed(42)
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")

        with torch.no_grad():
            pytorch_output = policy.model.sample_actions(
                images=[batch["observation.images.top"]],
                img_masks=[torch.ones(1, dtype=torch.bool, device="cuda")],
                lang_tokens=batch[OBS_LANGUAGE_TOKENS],
                lang_masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
                state=torch.nn.functional.pad(batch[OBS_STATE], (0, 32 - 14)),
                noise=noise,
            )

        package_path = export_policy(
            policy,
            tmp_path / "pi0_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load_exported_policy(package_path, backend="onnx", device="cpu")
        obs_numpy = to_numpy(batch)
        obs_numpy[OBS_STATE] = np.pad(obs_numpy[OBS_STATE], ((0, 0), (0, 32 - 14)))
        onnx_output = runtime.predict_action_chunk(obs_numpy, noise=noise.cpu().numpy())

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        # Empirical CPU eager↔ONNX floor for PI0 kv_cache: max|abs|≈1.0e-6 over 3 denoise
        # steps. 1e-5 leaves margin for CPU↔GPU kernel differences in CI.
        assert_numerical_parity(
            onnx_output,
            pytorch_np,
            rtol=1e-5,
            atol=1e-5,
            msg="PI0 ONNX output does not match PyTorch output",
        )

    @pytest.mark.slow
    def test_openvino_numerical_parity(self, tmp_path: Path):
        pytest.importorskip("transformers")
        pytest.importorskip("openvino")
        from lerobot.export import export_policy, load_exported_policy
        from lerobot.utils.constants import OBS_STATE

        policy, batch = create_pi0_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "pi0_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        np.random.seed(42)
        noise = np.random.randn(1, policy.config.chunk_size, policy.config.max_action_dim).astype(np.float32)

        obs_numpy = to_numpy(batch)
        obs_numpy[OBS_STATE] = np.pad(obs_numpy[OBS_STATE], ((0, 0), (0, 32 - 14)))

        runtime_onnx = load_exported_policy(package_path, backend="onnx", device="cpu")
        onnx_output = runtime_onnx.predict_action_chunk(obs_numpy, noise=noise)

        runtime_openvino = load_exported_policy(package_path, backend="openvino", device="cpu")
        openvino_output = runtime_openvino.predict_action_chunk(obs_numpy, noise=noise)

        assert_numerical_parity(
            openvino_output,
            onnx_output,
            rtol=1e-4,
            atol=1e-4,
            msg="PI0 OpenVINO output does not match ONNX output",
        )


class TestPI05Export:
    @pytest.mark.slow
    def test_export_creates_valid_package(self, tmp_path: Path):
        pytest.importorskip("transformers")
        from lerobot.export import export_policy
        from lerobot.export.manifest import Manifest

        policy, batch = create_pi05_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "pi05_package",
            backend="onnx",
            example_batch=batch,
        )

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "encoder.onnx").exists()
        assert (package_path / "artifacts" / "denoise.onnx").exists()

        manifest = Manifest.load(package_path / "manifest.json")
        assert manifest.is_kv_cache
        assert manifest.model.runner["type"] == "kv_cache"
        assert manifest.model.runner["num_inference_steps"] == policy.config.num_inference_steps

    @pytest.mark.slow
    def test_onnx_numerical_parity(self, tmp_path: Path):
        pytest.importorskip("transformers")
        from lerobot.export import export_policy, load_exported_policy
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        policy, batch = create_pi05_policy_and_batch(device="cuda")

        torch.manual_seed(42)
        np.random.seed(42)
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")

        with torch.no_grad():
            pytorch_output = policy.model.sample_actions(
                images=[batch["observation.images.top"]],
                img_masks=[torch.ones(1, dtype=torch.bool, device="cuda")],
                tokens=batch[OBS_LANGUAGE_TOKENS],
                masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
                noise=noise,
            )

        package_path = export_policy(
            policy,
            tmp_path / "pi05_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load_exported_policy(package_path, backend="onnx", device="cpu")
        obs_numpy = to_numpy(batch)
        onnx_output = runtime.predict_action_chunk(obs_numpy, noise=noise.cpu().numpy())

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        # Empirical CPU eager↔ONNX floor for PI05 kv_cache: max|abs|≈7.2e-7 over 10 denoise
        # steps. 1e-5 leaves margin for CPU↔GPU kernel differences in CI.
        assert_numerical_parity(
            onnx_output,
            pytorch_np,
            rtol=1e-5,
            atol=1e-5,
            msg="PI05 ONNX output does not match PyTorch output",
        )

    @pytest.mark.slow
    def test_onnx_numerical_parity_with_normalization(self, tmp_path: Path):
        """End-to-end parity with PI05's real normalization defaults (QUANTILES).

        PI05 config defaults: STATE=QUANTILES, ACTION=QUANTILES, VISUAL=IDENTITY.
        Validates the new quantile path in the runtime Normalizer end-to-end.
        Note: PI05 predict_action_chunk does not consume state, so only ACTION
        quantile denormalization is exercised against raw model output here.
        """
        pytest.importorskip("transformers")
        from lerobot.configs.types import NormalizationMode
        from lerobot.export import export_policy, load_exported_policy
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        policy, batch = create_pi05_policy_and_batch(device="cuda")
        policy.config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
        action_q01 = np.full(14, -1.5, dtype=np.float32)
        action_q99 = np.full(14, 1.5, dtype=np.float32)
        state_q01 = np.full(14, -2.0, dtype=np.float32)
        state_q99 = np.full(14, 2.0, dtype=np.float32)
        policy.config.stats = {
            "observation.state": {"q01": state_q01.tolist(), "q99": state_q99.tolist()},
            "action": {"q01": action_q01.tolist(), "q99": action_q99.tolist()},
        }

        torch.manual_seed(42)
        np.random.seed(42)
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")

        with torch.no_grad():
            pytorch_normalized_output = policy.model.sample_actions(
                images=[batch["observation.images.top"]],
                img_masks=[torch.ones(1, dtype=torch.bool, device="cuda")],
                tokens=batch[OBS_LANGUAGE_TOKENS],
                masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
                noise=noise,
            )

        # PI05 internally pads action to max_action_dim; truncate to real action dim before denorm.
        original_action_dim = policy.config.output_features["action"].shape[0]
        pytorch_np = pytorch_normalized_output.cpu().numpy()[:, :, :original_action_dim]
        # Inverse quantiles: (x+1)/2 * (q99-q01) + q01
        pytorch_denormalized = (pytorch_np + 1.0) / 2.0 * (action_q99 - action_q01) + action_q01

        package_path = export_policy(
            policy,
            tmp_path / "pi05_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=True,
        )

        import json

        manifest = json.loads((package_path / "manifest.json").read_text())
        preprocessors = manifest["model"]["preprocessors"] or []
        postprocessors = manifest["model"]["postprocessors"] or []
        pre_modes = {p["mode"] for p in preprocessors}
        post_modes = {p["mode"] for p in postprocessors}
        assert "quantiles" in pre_modes or "quantiles" in post_modes, (
            f"Expected quantiles mode in manifest; pre={pre_modes}, post={post_modes}"
        )
        assert (package_path / "artifacts" / "stats.safetensors").exists()

        runtime = load_exported_policy(package_path, backend="onnx", device="cpu")
        obs_numpy = to_numpy(batch)
        onnx_output = runtime.predict_action_chunk(obs_numpy, noise=noise.cpu().numpy())

        if pytorch_denormalized.ndim == 3 and pytorch_denormalized.shape[0] == 1:
            pytorch_denormalized = pytorch_denormalized[0]

        assert_numerical_parity(
            onnx_output,
            pytorch_denormalized,
            rtol=1e-4,
            atol=1e-4,
            msg="PI05 ONNX (with QUANTILES normalization) does not match PyTorch raw→denormalize",
        )

    @pytest.mark.slow
    def test_openvino_numerical_parity(self, tmp_path: Path):
        pytest.importorskip("transformers")
        pytest.importorskip("openvino")
        from lerobot.export import export_policy, load_exported_policy

        policy, batch = create_pi05_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "pi05_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        np.random.seed(42)
        noise = np.random.randn(1, policy.config.chunk_size, policy.config.max_action_dim).astype(np.float32)

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
            msg="PI05 OpenVINO output does not match ONNX output",
        )

    @pytest.mark.slow
    def test_openvino_numerical_parity_with_pytorch(self, tmp_path: Path):
        pytest.importorskip("transformers")
        pytest.importorskip("openvino")
        from lerobot.export import export_policy, load_exported_policy
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        policy, batch = create_pi05_policy_and_batch(device="cuda")

        torch.manual_seed(42)
        np.random.seed(42)
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")

        with torch.no_grad():
            pytorch_output = policy.model.sample_actions(
                images=[batch["observation.images.top"]],
                img_masks=[torch.ones(1, dtype=torch.bool, device="cuda")],
                tokens=batch[OBS_LANGUAGE_TOKENS],
                masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
                noise=noise,
            )

        package_path = export_policy(
            policy,
            tmp_path / "pi05_package",
            backend="openvino",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load_exported_policy(package_path, backend="openvino", device="cpu")
        obs_numpy = to_numpy(batch)
        ov_output = runtime.predict_action_chunk(obs_numpy, noise=noise.cpu().numpy())

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        # OpenVINO loads the same ONNX serialization; floor matches PI05 ONNX (~7e-7).
        assert_numerical_parity(
            ov_output,
            pytorch_np,
            rtol=1e-5,
            atol=1e-5,
            msg="PI05 OpenVINO output does not match PyTorch output",
        )


class TestSmolVLAExport:
    @pytest.mark.slow
    def test_export_creates_valid_package(self, tmp_path: Path):
        from lerobot.export import export_policy
        from lerobot.export.manifest import Manifest

        policy, batch = create_smolvla_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "smolvla_package",
            backend="onnx",
            example_batch=batch,
        )

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "encoder.onnx").exists()
        assert (package_path / "artifacts" / "denoise.onnx").exists()

        manifest = Manifest.load(package_path / "manifest.json")
        assert manifest.is_kv_cache
        assert manifest.model.runner["type"] == "kv_cache"
        assert manifest.model.runner["num_inference_steps"] == policy.config.num_steps

    @pytest.mark.slow
    def test_runtime_forward_pass(self, tmp_path: Path):
        from lerobot.export import ExportedPolicy, export_policy, load_exported_policy

        policy, batch = create_smolvla_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "smolvla_package",
            backend="onnx",
            example_batch=batch,
        )

        runtime = load_exported_policy(package_path, backend="onnx", device="cpu")
        assert isinstance(runtime, ExportedPolicy)

        obs_numpy = to_numpy(batch)
        action_chunk = runtime.predict_action_chunk(obs_numpy)

        assert action_chunk.ndim == 2
        assert action_chunk.shape[0] == policy.config.chunk_size
        assert action_chunk.shape[1] == policy.config.max_action_dim

    @pytest.mark.slow
    def test_onnx_numerical_parity(self, tmp_path: Path):
        from lerobot.export import export_policy, load_exported_policy
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

        policy, batch = create_smolvla_policy_and_batch(device="cuda")
        policy_cpu = policy.cpu().float().eval()
        batch_cpu = {k: v.detach().cpu() for k, v in batch.items()}

        torch.manual_seed(42)
        np.random.seed(42)
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")
        noise_cpu = noise.detach().cpu()

        with torch.no_grad():
            pytorch_output = policy_cpu.model.sample_actions(
                images=[batch_cpu["observation.images.top"]],
                img_masks=[torch.ones(1, dtype=torch.bool)],
                lang_tokens=batch_cpu[OBS_LANGUAGE_TOKENS],
                lang_masks=batch_cpu[OBS_LANGUAGE_ATTENTION_MASK],
                state=torch.nn.functional.pad(batch_cpu[OBS_STATE], (0, 32 - 14)),
                noise=noise_cpu,
            )

        package_path = export_policy(
            policy,
            tmp_path / "smolvla_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load_exported_policy(package_path, backend="onnx", device="cpu")
        obs_numpy = to_numpy(batch)
        obs_numpy[OBS_STATE] = np.pad(obs_numpy[OBS_STATE], ((0, 0), (0, 32 - 14)))
        onnx_output = runtime.predict_action_chunk(obs_numpy, noise=noise_cpu.numpy())

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        # SmolVLA kv_cache parity: 1e-5 chosen conservatively (not yet empirically profiled
        # like PI0/PI05). Tighten further once an empirical floor is measured.
        assert_numerical_parity(
            onnx_output,
            pytorch_np,
            rtol=1e-5,
            atol=1e-5,
            msg="SmolVLA ONNX output does not match PyTorch output",
        )

    @pytest.mark.slow
    def test_openvino_numerical_parity(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import export_policy, load_exported_policy
        from lerobot.utils.constants import OBS_STATE

        policy, batch = create_smolvla_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "smolvla_package",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )

        np.random.seed(42)
        noise = np.random.randn(1, policy.config.chunk_size, policy.config.max_action_dim).astype(np.float32)

        obs_numpy = to_numpy(batch)
        obs_numpy[OBS_STATE] = np.pad(obs_numpy[OBS_STATE], ((0, 0), (0, 32 - 14)))

        runtime_onnx = load_exported_policy(package_path, backend="onnx", device="cpu")
        onnx_output = runtime_onnx.predict_action_chunk(obs_numpy, noise=noise)

        runtime_openvino = load_exported_policy(package_path, backend="openvino", device="cpu")
        openvino_output = runtime_openvino.predict_action_chunk(obs_numpy, noise=noise)

        assert_numerical_parity(
            openvino_output,
            onnx_output,
            rtol=1e-4,
            atol=1e-4,
            msg="SmolVLA OpenVINO output does not match ONNX output",
        )


class TestPI05ExecuTorch:
    """ExecuTorch parity for the kv_cache runner family (Pi05)."""

    @require_executorch
    @pytest.mark.slow
    def test_executorch_export_creates_valid_package(self, tmp_path: Path):
        pytest.importorskip("transformers")
        from lerobot.export import export_policy

        policy, batch = create_pi05_policy_and_batch(device="cuda")

        package_path = export_policy(
            policy,
            tmp_path / "pi05_et",
            backend="executorch",
            example_batch=batch,
        )

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "encoder.pte").exists()
        assert (package_path / "artifacts" / "denoise.pte").exists()
        assert (package_path / "artifacts" / "encoder_io_spec.yaml").exists()
        assert (package_path / "artifacts" / "denoise_io_spec.yaml").exists()

    @require_executorch
    @pytest.mark.slow
    def test_executorch_numerical_parity_with_pytorch(self, tmp_path: Path):
        """Exported ExecuTorch policy must match the torch policy on identical input.

        Tolerance is looser than ACT/Diffusion because kv_cache models accumulate
        more fp32 rounding error across the denoise loop and transformer layers.
        ExecuTorch runtime introduces additional kernel-level numerical drift vs ONNX;
        1e-5 is conservative until an empirical floor is profiled on hardware.
        """
        pytest.importorskip("transformers")
        from lerobot.export import export_policy, load_exported_policy
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        policy, batch = create_pi05_policy_and_batch(device="cuda")

        torch.manual_seed(42)
        np.random.seed(42)
        noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, device="cuda")

        with torch.no_grad():
            pytorch_output = policy.model.sample_actions(
                images=[batch["observation.images.top"]],
                img_masks=[torch.ones(1, dtype=torch.bool, device="cuda")],
                tokens=batch[OBS_LANGUAGE_TOKENS],
                masks=batch[OBS_LANGUAGE_ATTENTION_MASK],
                noise=noise,
            )

        package_path = export_policy(
            policy,
            tmp_path / "pi05_et",
            backend="executorch",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load_exported_policy(package_path, backend="executorch", device="cpu")
        obs_numpy = to_numpy(batch)
        et_output = runtime.predict_action_chunk(obs_numpy, noise=noise.cpu().numpy())

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        assert_numerical_parity(
            et_output,
            pytorch_np,
            rtol=1e-5,
            atol=1e-5,
            msg="PI05 ExecuTorch output does not match PyTorch output",
        )

    @require_executorch
    @pytest.mark.slow
    def test_executorch_numerical_parity_with_onnx(self, tmp_path: Path):
        pytest.importorskip("transformers")
        from lerobot.export import export_policy, load_exported_policy

        policy, batch = create_pi05_policy_and_batch(device="cuda")

        onnx_pkg = export_policy(
            policy,
            tmp_path / "pi05_onnx",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )
        et_pkg = export_policy(
            policy,
            tmp_path / "pi05_et",
            backend="executorch",
            example_batch=batch,
            include_normalization=False,
        )

        np.random.seed(42)
        noise = np.random.randn(1, policy.config.chunk_size, policy.config.max_action_dim).astype(np.float32)

        obs_numpy = to_numpy(batch)

        onnx_rt = load_exported_policy(onnx_pkg, backend="onnx", device="cpu")
        et_rt = load_exported_policy(et_pkg, backend="executorch", device="cpu")

        onnx_output = onnx_rt.predict_action_chunk(obs_numpy, noise=noise)
        et_output = et_rt.predict_action_chunk(obs_numpy, noise=noise)

        assert_numerical_parity(
            et_output,
            onnx_output,
            rtol=1e-3,
            atol=1e-3,
            msg="PI05 ExecuTorch output does not match ONNX output",
        )
