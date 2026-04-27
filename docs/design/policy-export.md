# Policy export framework: converged manifest carve-out

## Problem Statement

LeRobot policies are PyTorch-native, but deployment environments such as PhysicalAI's runtime need portable artifacts such as ONNX and OpenVINO packages to run policies on robots outside the training stack.

Without a converged manifest, every downstream consumer has to write and maintain its own adapter layer for policy metadata, processor wiring, normalization assets, tokenizer assets, and backend-specific artifact loading.

The single source of truth for the portable package shape is the converged schema defined in PhysicalAI's `docs/design/integrations/lerobot.md`. LeRobot exports must conform to that schema exactly so downstream consumers can call `InferenceModel.load()` directly without translation or policy-specific glue code.

## Architecture

```text
LeRobot Policy
    │
    ▼
Exporter
    ├── ONNX serializer
    ├── Manifest writer
    └── Package assets
        ├── manifest.json
        ├── artifacts/model.onnx          (ACT)
        ├── artifacts/encoder.onnx        (PI05)
        ├── artifacts/denoise.onnx        (PI05)
        ├── stats.safetensors             (optional)
        └── tokenizer/                    (PI05)
```

At a high level the exporter turns a PyTorch policy into:

- backend artifacts for runtime execution,
- ProcessorSpec arrays for normalization and policy-specific preprocessing,
- a manifest that describes how to run the package,
- supporting assets such as stats and tokenizer files.

Using `tests/export/fixtures/manifest_act_converged.json` as ground truth, the
manifest written by this carve-out contains:

- `policy.source.class_path` for the canonical LeRobot policy entry point
- `model.runner` as an object with `type` plus runner-specific flat params
- `model.preprocessors` as an array of `ProcessorSpec`
- `model.postprocessors` as an array of `ProcessorSpec`
- `model.artifacts` as a mapping from artifact role to relative path
- `model.n_obs_steps`, plus optional `hardware` and `metadata`

Processors are folded into ProcessorSpec arrays only. There is no sidecar JSON, no `RuntimeRegistry`, and no standalone `ProcessorPipeline` class in the export package.

This carve-out currently supports two runner families:

- `action_chunking` for single-pass chunk emitters such as ACT
- `kv_cache` for multi-stage encoder + iterative denoise policies such as PI05

The supported runtime backends are:

- ONNX Runtime for direct execution of exported ONNX artifacts
- OpenVINO as a runtime-only loader over the same ONNX artifacts

## Why ACT + PI05 first

- **ACT** is the smallest viable single-pass export target and validates the converged schema for non-iterative policies.
- **PI05** is the hardest currently-supported export case because it combines a multi-stage runner, KV cache handoff, tokenizer assets, and policy-specific processors.
- Together they exercise both runner types, both backends, and the full processor catalog implemented in this carve-out.

The following policy families are deferred to follow-up PRs:

- Diffusion
- SmolVLA
- PI0
- GROOT
- SAC
- TDMPC
- VQ-BeT
- X-VLA
- Wall-X

## Phased Rollout

- **Phase 1**: ProcessorSpec emitters (`aea1f435`)
- **Phase 2**: Schema cleanup + golden ACT manifest fixture (`7bb187e9`)
- **Phase 3**: Runner renames + `iterative.py` removal + orphan test cleanup (`c1a1dea1`)
- **Phase 4**: ACT end-to-end wiring + parity `1e-5` (`1cc865a5`)
- **Phase 5**: PI05 end-to-end wiring + parity restored (`e3d94662`)
- **Phase 6**: This RFC + PR description

Future work includes additional policies (e.g. SmolVLA, OpenVLA), an optional ExecuTorch backend for on-device deployment, and migration to the new `torch.export`-based ONNX exporter (the legacy TorchScript path is currently used).

## Known Limitations

- **PI05 parity** — ONNX runtime hits the strict design target: `rtol=0.02`, `atol=0.006` end-to-end across the 3-step Euler denoise loop. The previously-observed ~`0.052` drift was traced (Oracle-directed) to PI05's `create_sinusoidal_pos_embedding` helper, which built the time-embedding trig subgraph in `torch.float64`. Casting that subgraph to `torch.float32` at the source (no graph post-processing) restored parity. OpenVINO runtime parity is held to `rtol=0.05`, `atol=0.05` because OpenVINO's IR optimization passes (operator fusion, internal MatMul accumulation) introduce a measured ~`0.05 abs` drift on the chained Euler loop even with `INFERENCE_PRECISION_HINT=f32` pinned. Single-stage feedforward policies (ACT) are exempt: ACT OpenVINO matches ONNX at `1e-5`.
- PI05 manifest currently uses padded `action_dim=32` (model-internal) rather than the real `14`. This is orthogonal to schema conformance and will be addressed in follow-up work.
- Tokenizer bundling requires the PaliGemma tokenizer to be present in the local Hugging Face cache; tests skip on cache miss for CI portability.

## Companion PhysicalAI PR (non-blocking)

The converged schema in PhysicalAI's `docs/design/integrations/lerobot.md` will be extended in a non-blocking companion PR to enumerate the following as canonical:

- `kv_cache` runner type
- `tokenize` processor type
- `relative_actions` processor type
- `absolute_actions` processor type
- `pi05_prepare_state` processor type

This LeRobot PR can land independently because the companion change is documentation-only on the PhysicalAI side.

## Open Questions for Maintainers

1. Should the manifest stay under `lerobot.export`, or move to a top-level `lerobot.manifest` package once more policies opt in?
2. Should LeRobot publish the manifest as a machine-readable JSON Schema alongside the Python dataclasses?
3. Should this RFC be mirrored into the public documentation site, or remain repository-local design documentation?
