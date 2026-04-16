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

import pytest
import torch

onnxruntime = pytest.importorskip("onnxruntime")

from tests.export.conftest import (  # noqa: E402
    assert_numerical_parity,
    create_act_policy_and_batch,
    require_executorch,
    to_numpy,
)


class TestACTExport:
    @pytest.mark.slow
    def test_export_creates_valid_package(self, tmp_path: Path):
        policy, batch = create_act_policy_and_batch()

        package_path = policy.export(
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "model.onnx").exists()

    @pytest.mark.slow
    def test_onnx_forward_pass(self, tmp_path: Path):
        from lerobot.export import ExportedPolicy, load_exported_policy

        policy, batch = create_act_policy_and_batch()

        package_path = policy.to_onnx(
            tmp_path / "act_package",
            example_batch=batch,
        )

        runtime = load_exported_policy(package_path, backend="onnx", device="cpu")
        assert isinstance(runtime, ExportedPolicy)

        obs_numpy = to_numpy(batch)
        action_chunk = runtime.predict_action_chunk(obs_numpy)

        assert action_chunk.ndim == 2
        assert action_chunk.shape[0] == policy.config.chunk_size
        action_dim = policy.config.action_feature.shape[0] if policy.config.action_feature else 6
        assert action_chunk.shape[1] == action_dim

    @pytest.mark.slow
    def test_onnx_numerical_parity(self, tmp_path: Path):
        from lerobot.export import load_exported_policy

        policy, batch = create_act_policy_and_batch()

        with torch.no_grad():
            torch.manual_seed(42)
            pytorch_output = policy.predict_action_chunk(batch)

        package_path = policy.to_onnx(
            tmp_path / "act_package",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load_exported_policy(package_path, backend="onnx", device="cpu")
        obs_numpy = to_numpy(batch)
        onnx_output = runtime.predict_action_chunk(obs_numpy)

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        assert_numerical_parity(
            onnx_output,
            pytorch_np,
            rtol=1e-3,
            atol=1e-4,
            msg="ACT ONNX output does not match PyTorch output",
        )

    @pytest.mark.slow
    def test_select_action_is_default_api(self, tmp_path: Path):
        from lerobot.export import load_exported_policy

        policy, batch = create_act_policy_and_batch()

        package_path = policy.to_onnx(
            tmp_path / "act_package",
            example_batch=batch,
        )

        exported_policy = load_exported_policy(package_path, backend="onnx", device="cpu")

        obs_numpy = to_numpy(batch)
        exported_policy.reset()
        action = exported_policy.select_action(obs_numpy)

        assert action.ndim == 1
        action_dim = policy.config.action_feature.shape[0] if policy.config.action_feature else 6
        assert action.shape[0] == action_dim


class TestACTBackends:
    def test_onnx_backend_initialization(self, tmp_path: Path):
        from lerobot.export import export_policy

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        from lerobot.export.backends.onnx import ONNXRuntimeAdapter

        model_path = package_path / "artifacts" / "model.onnx"
        backend = ONNXRuntimeAdapter(model_path, device="cpu")

        assert backend.input_names is not None
        assert backend.output_names is not None
        assert len(backend.input_names) > 0
        assert len(backend.output_names) > 0

    def test_openvino_backend_initialization(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import export_policy

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        from lerobot.export.backends.openvino import OpenVINORuntimeAdapter

        model_path = package_path / "artifacts" / "model.onnx"
        backend = OpenVINORuntimeAdapter(model_path, device="cpu")

        assert backend.input_names is not None
        assert backend.output_names is not None
        assert len(backend.input_names) > 0
        assert len(backend.output_names) > 0

    @pytest.mark.slow
    def test_openvino_numerical_parity_with_onnx(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import export_policy

        policy, batch = create_act_policy_and_batch()

        package_path = export_policy(
            policy,
            tmp_path / "act_package",
            backend="onnx",
            example_batch=batch,
        )

        from lerobot.export.backends.onnx import ONNXRuntimeAdapter
        from lerobot.export.backends.openvino import OpenVINORuntimeAdapter

        model_path = package_path / "artifacts" / "model.onnx"

        onnx_backend = ONNXRuntimeAdapter(model_path, device="cpu")
        openvino_backend = OpenVINORuntimeAdapter(model_path, device="cpu")

        obs_numpy = to_numpy(batch)
        inputs = {k: v for k, v in obs_numpy.items() if k in onnx_backend.input_names}

        onnx_outputs = onnx_backend.run(inputs)
        openvino_outputs = openvino_backend.run(inputs)

        for name in onnx_outputs:
            assert_numerical_parity(
                openvino_outputs[name],
                onnx_outputs[name],
                rtol=1e-5,
                atol=1e-5,
                msg=f"OpenVINO output '{name}' does not match ONNX output",
            )


class TestACTRuntime:
    @pytest.mark.slow
    def test_from_exported_loads_user_facing_policy(self, tmp_path: Path):
        from lerobot.export import ExportedPolicy

        policy, batch = create_act_policy_and_batch()

        package_path = policy.to_onnx(
            tmp_path / "act_package",
            example_batch=batch,
        )

        runtime = policy.from_exported(package_path, backend="onnx", device="cpu")

        assert isinstance(runtime, ExportedPolicy)


class TestACTExecuTorch:
    @require_executorch
    @pytest.mark.slow
    def test_executorch_export_creates_valid_package(self, tmp_path: Path):
        policy, batch = create_act_policy_and_batch()

        package_path = policy.to_executorch(tmp_path / "act_et", example_batch=batch)

        assert (package_path / "manifest.json").exists()
        assert (package_path / "artifacts" / "model.pte").exists()
        assert (package_path / "artifacts" / "io_spec.yaml").exists()

    @require_executorch
    @pytest.mark.slow
    def test_executorch_forward_pass(self, tmp_path: Path):
        from lerobot.export import load_exported_policy

        policy, batch = create_act_policy_and_batch()

        package_path = policy.to_executorch(tmp_path / "act_et", example_batch=batch)

        runtime = load_exported_policy(package_path, backend="executorch", device="cpu")
        obs_numpy = to_numpy(batch)
        action_chunk = runtime.predict_action_chunk(obs_numpy)

        assert action_chunk.ndim == 2
        assert action_chunk.shape[0] == policy.config.chunk_size
        action_dim = policy.config.action_feature.shape[0] if policy.config.action_feature else 6
        assert action_chunk.shape[1] == action_dim

    @require_executorch
    @pytest.mark.slow
    def test_executorch_numerical_parity_with_pytorch(self, tmp_path: Path):
        from lerobot.export import load_exported_policy

        policy, batch = create_act_policy_and_batch()

        with torch.no_grad():
            torch.manual_seed(42)
            pytorch_output = policy.predict_action_chunk(batch)

        package_path = policy.to_executorch(
            tmp_path / "act_et",
            example_batch=batch,
            include_normalization=False,
        )

        runtime = load_exported_policy(package_path, backend="executorch", device="cpu")
        obs_numpy = to_numpy(batch)
        et_output = runtime.predict_action_chunk(obs_numpy)

        pytorch_np = pytorch_output.cpu().numpy()
        if pytorch_np.ndim == 3 and pytorch_np.shape[0] == 1:
            pytorch_np = pytorch_np[0]

        assert_numerical_parity(
            et_output,
            pytorch_np,
            rtol=1e-3,
            atol=1e-4,
            msg="ACT ExecuTorch output does not match PyTorch output",
        )

    @require_executorch
    @pytest.mark.slow
    def test_executorch_numerical_parity_with_onnx(self, tmp_path: Path):
        from lerobot.export import export_policy

        policy, batch = create_act_policy_and_batch()

        onnx_pkg = export_policy(
            policy,
            tmp_path / "act_onnx",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )
        et_pkg = export_policy(
            policy,
            tmp_path / "act_et",
            backend="executorch",
            example_batch=batch,
            include_normalization=False,
        )

        from lerobot.export.backends.executorch import ExecuTorchRuntimeAdapter
        from lerobot.export.backends.onnx import ONNXRuntimeAdapter

        onnx_be = ONNXRuntimeAdapter(onnx_pkg / "artifacts" / "model.onnx", device="cpu")
        et_be = ExecuTorchRuntimeAdapter(et_pkg / "artifacts" / "model.pte", device="cpu")

        obs_numpy = to_numpy(batch)
        inputs = {k: v for k, v in obs_numpy.items() if k in onnx_be.input_names}

        onnx_outputs = onnx_be.run(inputs)
        et_outputs = et_be.run(inputs)

        for name in onnx_outputs:
            et_name = name if name in et_outputs else next(iter(et_outputs))
            assert_numerical_parity(
                et_outputs[et_name],
                onnx_outputs[name],
                rtol=1e-5,
                atol=1e-5,
                msg=f"ExecuTorch output '{et_name}' does not match ONNX output '{name}'",
            )

    @require_executorch
    @pytest.mark.slow
    def test_executorch_numerical_parity_with_openvino(self, tmp_path: Path):
        pytest.importorskip("openvino")
        from lerobot.export import export_policy

        policy, batch = create_act_policy_and_batch()

        onnx_pkg = export_policy(
            policy,
            tmp_path / "act_onnx",
            backend="onnx",
            example_batch=batch,
            include_normalization=False,
        )
        et_pkg = export_policy(
            policy,
            tmp_path / "act_et",
            backend="executorch",
            example_batch=batch,
            include_normalization=False,
        )

        from lerobot.export.backends.executorch import ExecuTorchRuntimeAdapter
        from lerobot.export.backends.openvino import OpenVINORuntimeAdapter

        ov_be = OpenVINORuntimeAdapter(onnx_pkg / "artifacts" / "model.onnx", device="cpu")
        et_be = ExecuTorchRuntimeAdapter(et_pkg / "artifacts" / "model.pte", device="cpu")

        obs_numpy = to_numpy(batch)
        inputs = {k: v for k, v in obs_numpy.items() if k in ov_be.input_names}

        ov_outputs = ov_be.run(inputs)
        et_outputs = et_be.run(inputs)

        for name in ov_outputs:
            et_name = name if name in et_outputs else next(iter(et_outputs))
            assert_numerical_parity(
                et_outputs[et_name],
                ov_outputs[name],
                rtol=1e-5,
                atol=1e-5,
                msg=f"ExecuTorch output '{et_name}' does not match OpenVINO output '{name}'",
            )
