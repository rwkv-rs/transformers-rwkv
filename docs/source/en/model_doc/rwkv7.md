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

A worked example, against `vllm-rwkv`'s varlen WKV. Its calling convention is not
guessable from the signature and was read off that project's own call site, then
checked against `rwkv7_recurrent` before being relied on: `w` is the LoRA output
*before* the decay transform (the kernel applies that itself), the rank-one pair is
`(-kk, kk * a)`, activations are fp16 while the state is fp32, and `head_dim` must be
64. Single-token decode is a different entry point, `wkv_one`, whose state is fp16.

```python
import torch
from transformers.models.rwkv7.modeling_rwkv7 import RWKV7_WKV_FUNCTIONS

torch.ops.load_library("<vllm-rwkv>/vllm/rwkv7_ops.abi3.so")
_INV_SQRT_E = 0.6065306597126334


def vllm_varlen_wkv(r, w_log, k, v, kk, a, state, cu_seq_lens=None, **kwargs):
    batch, seq_len, heads, head_dim = r.shape
    channels = heads * head_dim
    # This model carries `w_log = -INV_SQRT_E * sigmoid(w_raw)`; the kernel wants
    # `w_raw` and applies the transform itself, so invert it here.
    sigmoid = (-w_log / _INV_SQRT_E).clamp(1e-6, 1 - 1e-6)
    w_raw = torch.log(sigmoid / (1 - sigmoid))
    flat = lambda t: t.reshape(batch * seq_len, channels).to(torch.float16).contiguous()

    # Without `cu_seq_lens` every row of the batch is its own segment.
    cu = cu_seq_lens if cu_seq_lens is not None else torch.arange(
        0, batch * seq_len + 1, seq_len, device=r.device)
    n_seq = cu.numel() - 1
    lengths = (cu[1:] - cu[:-1]).tolist()
    slot_state = torch.zeros(n_seq, heads, head_dim, head_dim, device=r.device, dtype=torch.float32)
    y = torch.empty(batch * seq_len, channels, device=r.device, dtype=torch.float16)
    torch.ops.rwkv7_wkv_fp32_v2.forward_varlen(
        n_seq, batch * seq_len, max(lengths), channels, heads, cu.to(torch.int32),
        torch.arange(n_seq, device=r.device, dtype=torch.int32), slot_state,
        flat(r), flat(w_raw), flat(k), flat(v), flat(-kk), flat(kk * a), y)
    # This model's contract returns the LAST segment's state.
    return y.view(batch, seq_len, heads, head_dim).to(r.dtype), slot_state[-1:].to(state.dtype)


RWKV7_WKV_FUNCTIONS["vllm_varlen"] = vllm_varlen_wkv
```

Checked against `rwkv7_recurrent` on three shapes — a packed row of uneven segments
(5 + 7), an unpacked `batch=1`, and an unpacked `batch=3` — at relative errors of
2.6e-04 to 5.2e-04, which is fp16 against an fp32 reference. On one RTX 5090 with the
7.2B checkpoint it takes `1x256` prefill from 94% to 105% of `albatross faster3a`;
single-token decode is unchanged, because that step is bound by streaming the weights
rather than by the recurrence.

### Reaching the quoted decode throughput

Single-stream decode on the 7.2B checkpoint, RTX 5090, fp16, measured five times
per row (spread ≤ 0.1%):

| how it is run | tok/s |
|---|---:|
| eager | ~62 |
| `torch.compile()` | 91.9 |
| `torch.compile(mode="reduce-overhead")` + sparse channel-mix | 110.4 |
| `torch.compile(mode="max-autotune")` + sparse channel-mix | **126.4** |

The top row is 4× the bottom one's cost, so it is worth being explicit that three
things have to line up — none of them is the default:

```python
model = Rwkv7ForCausalLM.from_pretrained(checkpoint, dtype=torch.float16, sparse_channel_mix=True)
model = model.eval().cuda()
state = model.rwkv7.allocate_state(batch_size=1)      # BEFORE compiling, not state=None
compiled = torch.compile(model, mode="max-autotune", dynamic=False)
```

1. **`sparse_channel_mix=True`.** No checkpoint config carries this flag, so it
   defaults to off and the dense path runs. It is worth +20%, and it is exact —
   the activation is a squared ReLU, so the rows it skips contribute nothing.
2. **`mode="max-autotune"`.** Plain `torch.compile()` gives 91.9 and
   `"reduce-overhead"` 110.4; the decode is a long chain of small kernels, which is
   what autotuning has the most to work with.
3. **Call `allocate_state` before compiling.** Anything first allocated *inside* the
   compiled region cannot have its address pinned — `mark_static_address` does not run
   during tracing — and inductor then declines CUDA graphs for a region that mutates
   its inputs. Both the recurrent state and the sparse path's transposed weight are in
   that category, which is why one call handles both; on some torch builds the
   cudagraph pass does not merely skip but segfaults. Passing `state=None` and letting
   the model allocate is correct, just several times slower.

For more than this, the kernels are a separate, optional package rather than part of
the model: `transformers` keeps the portable implementation that builds and runs
anywhere and that these are checked against.

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

## Rwkv7Cache

[[autodoc]] Rwkv7Cache

## Rwkv7Model

[[autodoc]] Rwkv7Model
    - forward

## Rwkv7ForCausalLM

[[autodoc]] Rwkv7ForCausalLM
    - forward
