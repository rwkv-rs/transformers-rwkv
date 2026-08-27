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

RWKV-7 is a recurrent language model whose linear-attention update uses a diagonal-plus-low-rank state transition. This
implementation replaces the former RWKV-4 model behind the existing `rwkv` model identity. RWKV-4 checkpoints are not
compatible and are rejected explicitly.

The Transformers implementation owns configuration, model composition, recurrent caching, generation and
serialization. All product computation is delegated to the public [FlashRWKV2](https://github.com/rwkv-rs/FlashRWKV2)
operator API. Training follows `RWKV-LM/RWKV-v7/train_temp`; inference follows Albatross. There is no CPU, PyTorch or
FLA product fallback.

Inference calls FlashRWKV2 at model-semantic fusion boundaries: each linear-attention layer uses PostNorm+TokenShift,
WKV Prepare, WKV7 and Readout; each MLP uses one complete fused operator. Transformers does not select
sparse/dense kernels or invoke standalone projection, activation, LN, Res, TokenShift, VRes or gate helpers.

The current canonical contract uses:

- `head_size=64`;
- BF16 CUDA tensors and sequence lengths divisible by 16 for pretraining;
- Albatross's mixed BF16 embedding / FP16 model layout for inference;
- FP32 recurrent WKV state;
- one active unmerged vanilla LoRA adapter on the linear-attention `r_proj`, `k_proj`, `v_proj`, and `o_proj` modules;
  multiple adapters, LoRA variants, LoRA bias, and per-sample mixed-adapter batches must be merged first;
- equal-length batches for training and inference. An attention mask may be omitted or may be a two-dimensional,
  batch-matched, all-ones tensor whose length covers the current input. Padding and ragged batches fail immediately;
  bucket inputs by length instead.

### Support boundaries

| Area | Supported | Not supported |
|---|---|---|
| Training | BF16 CUDA pretraining, stateful training, gradient checkpointing | CPU or FLA fallback |
| Inference | Equal-length batches, recurrent continuation, greedy and beam generation, CUDA Graph | Padding, ragged or continuous batching |
| Framework integration | Dynamic `RwkvCache`, standard generation entry points, supported vanilla LoRA | `StaticCache`, Transformers tensor-parallel plans, compiled generation, `torch.export` |

CUDA Graph support does not imply that `torch.compile`, compiled generation, or `torch.export` is supported. RWKV has
no Transformers tensor-parallel plan: FlashRWKV2's fused operators consume complete weight matrices, so Transformers
does not expose a framework-side pseudo-sharding plan. A missing or incompatible FlashRWKV2 installation, a missing
operator, or an input outside these boundaries raises an error instead of selecting another implementation.

## Usage

Every installation method installs the native tokenizer fork from the fixed Git commit
`rwkv-rs/tokenizers-rwkv@c5d8dde5ff49c70e4656199d5033a84e03c21b2b`. The standard `AutoTokenizer` path and
checkpoint converter require its `tokenizers.models.RwkvTrie`; there is no fallback tokenizer. Installing this checkout
with the RWKV extra additionally installs `FlashRWKV2==0.1.0a8`--the extra does not control the tokenizer dependency:

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

`prepare_for_inference()` is an in-place conversion, not a temporary execution mode. Call it after loading or changing
weights and after attaching adapters. It changes parameter dtypes and device placement, creates non-persistent
transposed runtime weights, and moves the serializable MLP `down_proj` weights to CPU so a second full GPU
copy is not retained. Repeating the call is safe. Saving still serializes the canonical weights, and loading the saved
checkpoint starts without runtime layouts.

Calling `train()` directly on a prepared model raises an error before executing a training operator. To resume training,
move the complete model back to CUDA BF16 first; `_apply()` then discards all inference-only runtime layouts and restores
the offloaded canonical weights to CUDA:

```python
model = model.to(device="cuda", dtype=torch.bfloat16).train()
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
./.venv/bin/python temp/rwkv_pth2st.py \
    rwkv7-g1h-7.2b-20260710-ctx10240.pth \
    rwkv7-g1h-7.2b-hf \
    --context-length 10240
```

The converter treats BlinkDL names as an input format and writes the native Transformers decoder layout. For example,
`blocks.0.att.receptance.weight` becomes `model.layers.0.linear_attn.r_proj.weight`,
`blocks.0.ffn.value.weight` becomes `model.layers.0.mlp.down_proj.weight`, and `head.weight` becomes
`lm_head.weight`. The converted Safetensors checkpoint therefore works with standard module discovery, adapter target
selection, and framework tooling without preserving a second set of runtime aliases. The converter also builds a
standard fast `tokenizer.json` from the pinned
`rwkv-rs/rwkv7-g1-st` `rwkv_vocab_v20230424.json` artifact and writes the native RWKV chat template. Pass
`--rwkv-vocab-json` to use a hash-verified local copy of that JSON artifact or `--chat-template` to select the template
file. The tokenizer uses the RWKV World byte-level greedy longest-match algorithm. BOS and EOS share token ID 0;
padding and unknown tokens are intentionally undefined.

For a new model, the four linear-attention low-rank dimensions are derived from `hidden_size` with the exact `train_temp`
formulas. Converted checkpoints instead record the dimensions found in `w1/a1/v1/g1` and preserve them verbatim;
loading a checkpoint never recomputes or replaces its serialized low-rank dimensions. Randomly initialized models
also use the final effective parameter initialization from `train_temp`, including SmallInitEmb, the
vocabulary-dependent orthogonal LM head, layer-scaled GroupNorm weights, and the depth-dependent linear-attention and
MLP parameters.

## RwkvTokenizerFast

[[autodoc]] RwkvTokenizerFast

## RwkvConfig

[[autodoc]] RwkvConfig

## RwkvLinearAttention

[[autodoc]] RwkvLinearAttention
    - forward

## RwkvCache

[[autodoc]] RwkvCache

## RwkvTrainingState

[[autodoc]] RwkvTrainingState

## RwkvModel

[[autodoc]] RwkvModel
    - forward
    - prepare_for_inference

## RwkvForCausalLM

[[autodoc]] RwkvForCausalLM
    - forward
    - prepare_for_inference
