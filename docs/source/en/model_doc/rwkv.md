<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was contributed to Hugging Face Transformers on 2023-05-09.*

# RWKV-7

## Overview

RWKV-7 is a recurrent language model whose TimeMix update uses a diagonal-plus-low-rank state transition. This
implementation replaces the former RWKV-4 model behind the existing `rwkv` model identity. RWKV-4 checkpoints are not
compatible and are rejected explicitly.

The Transformers implementation owns configuration, model composition, recurrent caching, generation and
serialization. All product computation is delegated to the public [FlashRWKV](https://github.com/rwkv-rs/FlashRWKV)
operator API. Training follows `RWKV-LM/RWKV-v7/train_temp`; inference follows Albatross. There is no CPU, PyTorch or
FLA product fallback.

The current canonical contract uses:

- `head_size=64`;
- BF16 CUDA tensors and sequence lengths divisible by 16 for pretraining;
- Albatross's mixed BF16 embedding / FP16 model layout for inference;
- FP32 recurrent WKV state;
- equal-length batches for the initial inference integration.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "path/to/converted-rwkv7"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).cuda().eval().prepare_for_inference()

inputs = tokenizer("RWKV-7", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32)
```

Canonical BlinkDL `.pth` checkpoints can be converted with:

```bash
./.venv/bin/python temp/convert_rwkv7_checkpoint.py \
    rwkv7-g1h-7.2b-20260710-ctx10240.pth \
    rwkv7-g1h-7.2b-hf \
    --context-length 10240
```

The converter preserves canonical tensor names under the single standard `model.` base-model prefix and writes
Safetensors without a per-tensor compatibility table. It also writes a standard fast `tokenizer.json`,
`tokenizer_config.json`, and `chat_template.jinja` from the pinned `RWKV/RWKV7-1.5B-20260805` tokenizer artifact.
Use `--tokenizer-source` to select an equivalent local standard tokenizer directory and `--tokenizer-revision` to
pin a different Hub revision.

For a new model, the four TimeMix low-rank dimensions are derived from `hidden_size` with the exact `train_temp`
formulas. Converted checkpoints instead record the dimensions found in `w1/a1/v1/g1` and preserve them verbatim;
loading a checkpoint never recomputes or replaces its serialized low-rank dimensions. Randomly initialized models
also use the final effective parameter initialization from `train_temp`, including SmallInitEmb, the
vocabulary-dependent orthogonal LM head, layer-scaled GroupNorm weights, and the depth-dependent TimeMix and
ChannelMix parameters.

## RwkvConfig

[[autodoc]] RwkvConfig

## RwkvTimeMix

[[autodoc]] RwkvTimeMix
    - forward

## RwkvCache

[[autodoc]] RwkvCache

## RwkvModel

[[autodoc]] RwkvModel
    - forward
    - prepare_for_inference

## RwkvForCausalLM

[[autodoc]] RwkvForCausalLM
    - forward
    - prepare_for_inference
