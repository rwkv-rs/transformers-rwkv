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
serialization. All product computation is delegated to the public [FlashRWKV2](https://github.com/rwkv-rs/FlashRWKV2)
operator API. Training follows `RWKV-LM/RWKV-v7/train_temp`; inference follows Albatross. There is no CPU, PyTorch or
FLA product fallback.

The current canonical contract uses:

- `head_size=64`;
- BF16 CUDA tensors and sequence lengths divisible by 16 for pretraining;
- Albatross's mixed BF16 embedding / FP16 model layout for inference;
- FP32 recurrent WKV state;
- equal-length batches for the initial inference integration.

## Usage

Install this checkout with its RWKV extra. This installs the published `FlashRWKV2==0.1.0a3` source distribution;
the pinned native `tokenizers-rwkv` dependency is installed automatically:

```bash
TORCH_CUDA_ARCH_LIST=12.0 uv pip install --pre -e ".[rwkv]"
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "path/to/converted-rwkv7"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).cuda().eval().prepare_for_inference()

inputs = tokenizer("RWKV-7", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32)
```

For chat generation, render the prompt and resolve its matching stop string from the same messages and tools. Pass the
tokenizer to `generate()` because Transformers uses it to match stop strings across token boundaries:

```python
messages = [{"role": "user", "content": "Explain recurrent state briefly."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
stop_strings = tokenizer.get_chat_stop_strings(messages)
outputs = model.generate(
    **inputs,
    tokenizer=tokenizer,
    stop_strings=stop_strings,
    max_new_tokens=256,
)
```

The native `bot` prompt stops at `✿`; the `assistant` prompt stops at `\nUser:`; and function-calling prompts stop at
`\n### User`. Passing tools, historical tool calls, or tool messages automatically selects the function-calling prompt
and stop string. One generation batch must use one effective prompt style; split mixed-style conversations before
generation. Transformers returns the generated stop delimiter in the token sequence, so remove that suffix after
decoding when it should not be shown to the user.

Canonical BlinkDL `.pth` checkpoints can be converted with:

```bash
./.venv/bin/python temp/convert_rwkv7_checkpoint.py \
    rwkv7-g1h-7.2b-20260710-ctx10240.pth \
    rwkv7-g1h-7.2b-hf \
    --context-length 10240
```

The converter preserves canonical tensor names under the single standard `model.` base-model prefix and writes
Safetensors without a per-tensor compatibility table. It builds a standard fast `tokenizer.json` from the pinned
`rwkv-rs/rwkv7-g1-st` `rwkv_vocab_v20230424.json` artifact and writes the native RWKV chat template. Pass
`--rwkv-vocab-json` to use a hash-verified local copy of that JSON artifact or `--chat-template` to select the template
file. The tokenizer uses the RWKV World byte-level greedy longest-match algorithm. BOS and EOS share token ID 0;
padding and unknown tokens are intentionally undefined.

For a new model, the four TimeMix low-rank dimensions are derived from `hidden_size` with the exact `train_temp`
formulas. Converted checkpoints instead record the dimensions found in `w1/a1/v1/g1` and preserve them verbatim;
loading a checkpoint never recomputes or replaces its serialized low-rank dimensions. Randomly initialized models
also use the final effective parameter initialization from `train_temp`, including SmallInitEmb, the
vocabulary-dependent orthogonal LM head, layer-scaled GroupNorm weights, and the depth-dependent TimeMix and
ChannelMix parameters.

## RwkvTokenizerFast

[[autodoc]] RwkvTokenizerFast

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
