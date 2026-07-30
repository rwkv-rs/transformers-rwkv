# RWKV-7 kernel provenance

The Transformers Python implementation only adapts the published tensor and state contracts of these vendored
sources. It does not contain another WKV implementation.

## Inference

Source snapshot: `vllm-rwkv/csrc/libtorch_stable/rwkv7/`

- `rwkv7_wkv_fp32_v2.cpp`: `825E97ACAB883BE879864D0F027076D2C743EA8BE16F1EE2E03D44CBE77E9050`
- `rwkv7_wkv_fp32_v2.cu`: `DFA2F2C3B30F248B869CED64337F9D3E6578EBF927D389FB2626781107C02954`

The inference files retain their Apache-2.0 SPDX declarations.

## Training

Source snapshot: `RWKV-v7/train_temp/cuda/`

- `wkv7_op.cpp`: `F417F8EE8ADC57B45F64206CAF38FBABA5DFF575196BEA288E4E351310717C4B`
- `wkv7_cuda.cu`: `04C9EDEFF64824279CDC42FC29090592D33DF7D50D5FB212916F9C42805086FE`

The supplied RWKV-v7 snapshot has no repository license file or per-file license declaration. Confirm the upstream
redistribution terms before submitting these two training files to a public repository.
