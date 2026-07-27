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

### Padded batches

Pass `attention_mask` whenever a batch is padded. There is no attention to mask here —
padding is neutralised inside the recurrence instead, by holding the state transition at
the identity for those positions, so a padded row decodes exactly as if it had been run
on its own. Without the mask the padding is fed through the recurrence like any other
token and moves the state before the real tokens arrive, which silently corrupts every
row shorter than the longest one. `generate` passes the mask through for you; an
all-ones mask costs nothing and changes nothing, and a single decoded token skips the
handling entirely.

### Packed (varlen) batches

When the lengths in a batch vary a lot, padding is expensive twice over for a
recurrent model — the pads cost time as well as memory, because every one of them
is a step of the recurrence. Pack the sequences into a single row instead and pass
`cu_seq_lens`, the cumulative boundaries starting at 0 and ending at `seq_len`:

```python
packed = torch.cat([a, b, c], dim=1)                     # [1, len(a)+len(b)+len(c)]
cu_seq_lens = torch.tensor([0, a.shape[1], a.shape[1] + b.shape[1], packed.shape[1]])
out = model(input_ids=packed, cu_seq_lens=cu_seq_lens)
```

Each segment then decodes exactly as if it had been run on its own: the recurrent
state restarts at every boundary, and so does the token shift, which would
otherwise hand a segment's first token the previous sequence's last hidden state.
A malformed boundary list raises rather than quietly restarting the recurrence in
the wrong places.

### Swapping the WKV kernel

The recurrence is the one part worth replacing — everything around it is ordinary
linear algebra — so it is looked up by name instead of hard-coded:

```python
from transformers.models.rwkv7.modeling_rwkv7 import RWKV7_WKV_FUNCTIONS

RWKV7_WKV_FUNCTIONS["my_kernel"] = my_wkv
model = Rwkv7ForCausalLM.from_pretrained(checkpoint, wkv_implementation="my_kernel")
```

An entry receives `[batch, seq_len, num_heads, head_dim]` tensors for `r, w_log, k,
v, kk, a`, the `[batch, num_heads, head_dim, head_dim]` state, and `cu_seq_lens` as
a keyword, and returns `(output, new_state)`. The contract is to reproduce
`rwkv7_recurrent`, which is also what the test suite checks against — so a fused or
varlen kernel drops in without forking the model.

### Performance notes

Prefill runs a chunk-parallel form of the recurrence rather than a per-token loop:
the running decay is factored out, which turns each chunk into one unit-lower-triangular
solve plus a few matmuls, leaving only the chunk-to-chunk carry serial. Decoding a
single token takes the plain sequential step, which is what it already is.

The model is `torch.compile`-friendly and benefits substantially from it, because the
time-mix is many small elementwise and low-rank operations whose per-op overhead
dominates at small batch.

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
