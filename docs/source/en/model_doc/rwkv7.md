<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# RWKV-7

## Overview

RWKV-7 ("Goose") is an attention-free recurrent language model. Each layer replaces
self-attention with a *time-mix* block whose state is a fixed-size matrix per head,
updated by a generalised delta rule, and a *channel-mix* block that is a squared-ReLU
MLP over a one-token shift. Because the state does not grow with the sequence, there
is no KV cache: memory is constant in context length and each new token costs the
same as the first.

The model was released by the RWKV project; the reference implementation lives at
[BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM).

This implementation keeps the reference parameter names (`blocks.N.att.receptance`,
the LoRA factors as raw `w1`/`w2` tensors, `emb`, `head`, …) rather than renaming
them, so converting a native `.pth` checkpoint needs no per-tensor mapping table.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv7-0.1b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("RWKV/rwkv7-0.1b")

inputs = tokenizer("The Eiffel Tower is located in the city of", return_tensors="pt")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=16)[0]))
```

The recurrent state is returned as `state` and can be fed back to continue a
sequence; it replaces `past_key_values` and is O(1) in the sequence length.

```python
out = model(**inputs, use_cache=True)
next_out = model(input_ids=next_token, state=out.state, use_cache=True)
```

### DeepEmbed

`config.use_deep_embed` enables the RWKV-8 "DeepEmbed" hook: a per-layer, per-token
vector that channelwise-modulates the channel-mix. The table is deliberately not a
model weight — the design keeps it in RAM/SSD and prefetches per token, which is what
makes it cheap on VRAM — so it is passed to the forward as `deep_embeds` with shape
`[num_layers, batch, seq_len, deep_embed_size]`. A `deep_embed_size` of `hidden_size`
reproduces the reference "1x" variant and `intermediate_size` the "4x" variant. No
RWKV-7 checkpoint carries such a table; this is an extension point, off by default.

## Rwkv7Config

[[autodoc]] Rwkv7Config

## Rwkv7Model

[[autodoc]] Rwkv7Model
    - forward

## Rwkv7ForCausalLM

[[autodoc]] Rwkv7ForCausalLM
    - forward
