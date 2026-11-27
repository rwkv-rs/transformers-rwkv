<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-26.*

# RWKV-7S

RWKV-7S lives in `transformers.models.rwkv7s`. The model source is
[`src/transformers/models/rwkv7s/modeling_rwkv7s.py`](https://github.com/rwkv-rs/transformers-rwkv/blob/main/src/transformers/models/rwkv7s/modeling_rwkv7s.py).
Import it with `from transformers import Rwkv7sForCausalLM, Rwkv7sConfig, RwkvTokenizer, Rwkv7sCache`.
`AutoModelForCausalLM` loads converted checkpoints without `trust_remote_code`. Loading an RWKV-4 checkpoint raises an error.

```bash
pip install "transformers[rwkv]"
```

| Transformers | Component | RWKV-LM | FLA / hf_rwkv |
| --- | --- | --- | --- |
| `Rwkv7sAttention` | TimeMix and WKV recurrence | `RWKV_Tmix_x070` | `RWKV7Attention` |
| `Rwkv7sFeedForward` | ChannelMix | `RWKV_CMix_x070` | `RWKV7FeedForward` |
| `Rwkv7sDecoderLayer` | TimeMix and ChannelMix block | `Block` | `RWKV7Block` |

## Cross-loop residual

The two-round RWKV-7S decoder runs its layers in the order `Prelude -> Core_1 -> Core_2 -> Coda`.
`loop_layer_index_start` and `loop_layer_index_end` define the half-open core range, split evenly between the two
rounds. A learned FP32 gate adds a per-channel fraction of the first round's change to the Prelude output at the
entry to `Core_2`. With the gate initialized to zero and `Core_2` initialized from the original core weights, the
two-round model starts with the same logits as the single-round model.

## Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "rwkv-rs/rwkv7-g1-st"
subfolder = "rwkv7-g1l-1.5b-20261031-ctx34304"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(checkpoint, subfolder=subfolder, dtype=torch.bfloat16).cuda()
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain recurrent language models briefly."}],
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
outputs = model.generate(**inputs, tokenizer=tokenizer, max_new_tokens=128)
print(tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Training

RWKV-7S uses CUDA and BF16 activations. A dense batch contains token IDs and labels of shape `[batch, length]`.
The model uses causal next-token loss.

```python
model.train()
loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
loss.backward()
```

### Packing

Use [`DataCollatorWithFlattening`] to pack independent training examples into a batch of shape `[1, total_tokens]`. Enable
`return_flash_attn_kwargs` so the collator supplies `cu_seq_lens_q`. TimeMix, ChannelMix, and the WKV state start
from zero at each sequence boundary. The collator masks the first label of each sequence to keep next-token loss
within that sequence.

```python
from transformers import DataCollatorWithFlattening, Trainer

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorWithFlattening(return_flash_attn_kwargs=True),
)
```

Packed batches run without a recurrent cache. Use dense batches to pass `cache` between chunks.

### Right padding

Dense training supports right padding with stateless batches. Set padded labels to `-100` so they are excluded from the
loss. For example, set `tokenizer.pad_token = tokenizer.eos_token` and use
[`DataCollatorForLanguageModeling`] with `mlm=False`. Left padding and padded stateful training are unsupported.
Use batches of equal-length prompts for inference and generation.

To score the last real token in each right-padded training batch, pass
`logits_to_keep=(lengths - 1)[:, None]` and read `outputs.logits[:, 0]`.
A one-dimensional index tensor selects the same positions for every sample.

### PEFT

A typical [PEFT](https://huggingface.co/docs/peft/) configuration targets these projections:

```python
from peft import LoraConfig, get_peft_model

model = get_peft_model(
    model,
    LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
    ),
)
```

| Adapter | Training and ordinary forward | CUDA Graph generation |
| --- | --- | --- |
| One active LoRA or NoRA adapter with no dropout or bias | Fused | Supported |
| One active LoRA or NoRA adapter with dropout | Unfused during training, fused in eval mode | Supported in eval mode |
| DoRA, LoRA with bias, multiple active adapters, IA3, VeRA, or LoKr | Unfused | Merge into fused-compatible BF16 weights first |
| QLoRA with bitsandbytes | Unfused | Use a BF16 model with compatible or merged adapters |

The output head is also a valid target. If any projection needs its own module forward, the entire forward uses the
unfused projection path. FlashRWKV2 still runs TimeMix token shift, WKV recurrence, and readout. NoRA requires the
Joluck/NoRA PEFT fork and `use_nora=True`.

## Chat and tool generation

The chat template selects `bot` by default or accepts `rwkv_prompt_template="assistant"`. Tool definitions and tool-call
history select the function-calling format. The default Open Think profile uses `generation_config.json`. Set
`fake_cot_mode=True` in `apply_chat_template()` for an empty thinking prefix and load the Fake Think generation profile.
When tools are present, use the tools profile.

```python
from transformers import GenerationConfig

config_file_name = (
    "tools_generation_config.json"
    if tools
    else "fake_think_generation_config.json"
    if fake_cot_mode
    else "generation_config.json"
)
generation_config = GenerationConfig.from_pretrained(checkpoint, subfolder=subfolder, config_file_name=config_file_name)
```

## Converting a checkpoint

Convert canonical BlinkDL release weights to sharded Safetensors. The converter also saves the tokenizer,
chat template, and generation profiles.

```bash
python temp/rwkv_pth2st.py \
  /path/to/rwkv7-g1i.pth \
  /path/to/converted-rwkv7 \
  --context-length 10240 \
  --max-shard-size 5GB
```

## Stateful execution

Pass `outputs.cache` back as `cache` to continue a stream. [`Rwkv7sCache`] stores two token-shift vectors and one
WKV matrix per layer. Use `cache.clone()` to branch a stream and `cache.detach()` to start a truncated
backpropagation segment.

```python
from transformers import Rwkv7sCache

cache = Rwkv7sCache(model.config)
model(input_ids=inputs.input_ids, cache=cache)
continued = model(input_ids=torch.tensor([[42]], device=model.device), cache=cache)
```

### WKV state precision

The default `wkv_mode="fp32io16"` keeps the WKV state in FP32 and uses BF16 activations.
Set `model.config.wkv_mode = "fp16"` to halve WKV state storage for fused inference.
FP16 inference rounds the state after each token. Differentiable and unfused forwards use FP32 state.
Create a fresh cache when changing the state precision.

### CUDA Graph generation

`generate()` supports greedy decoding and sampling. It captures one graph for each distinct prefill chunk length and
one graph for single-token decode. Reusing the same batch shape and generation settings reuses those graphs.
Set `prefill_chunk_size` to control the prefill capture size.

### AutoDeco dynamic decoding parameters

`Rwkv7sForCausalLM` attaches [`Rwkv7sAutoDeco`] on top of the normalized second-to-last decoder layer
(`model.norm(layers[-2])`). During `generate(..., do_sample=True)` with `use_autodeco=True` (default), per-token
`temperature` and `top_p` tensors are predicted inside the CUDA Graph decode step and passed directly to FlashRWKV2
sampling. Set `train_autodeco=True` to freeze the backbone forward and train the AutoDeco heads.

### DeepEmbed and token-level ROSA

RWKV-7S includes `rwkv7a`-style low-rank DeepEmbed (`deep_embed_dim=32`) on each MoE FFN and token-level ROSA injected
after `embedding_norm`. Both are identity-initialized (`s2=0, s0=1` and `rosa_gate=0`) for post-training. During
training, DataLoader CPU workers can precompute `rosa_ids = compute_rwkv7s_rosa_ids(input_ids, cu_seqlens)`
asynchronously and pass `rosa_ids` into `forward()`. During inference and CUDA Graph `generate()`, online suffix
automatons advance in O(1) amortized time per decode token on CPU using pinned staging buffers.

### LightMTP speculative decoding

`Rwkv7sForCausalLM` attaches a shared [`Rwkv7sMTP`] LightMTP block looped `mtp_depth` times with shared `embed_tokens`
and `lm_head`. Set `train_mtp=True` to train only the MTP head on a frozen backbone, or set
`mtp_loss_scaling_factor > 0` for joint LM + MTP training. During `generate(..., use_mtp=True)`, both greedy decoding
(`do_sample=False`, bit-exact) and zero-penalty AutoDeco sampling (`do_sample=True`, exact greedy-draft rejection
sampling) run with CUDA-graphed drafting and verification, using a single shared scratch WKV layer plus variable-length
WKV-only commit to keep high-concurrency VRAM low.

## Rwkv7sConfig

[[autodoc]] Rwkv7sConfig

## Rwkv7sCache

[[autodoc]] Rwkv7sCache

## Rwkv7sAutoDeco

[[autodoc]] Rwkv7sAutoDeco

## Rwkv7sMTP

[[autodoc]] Rwkv7sMTP

## Rwkv7sModelOutputWithPast

[[autodoc]] Rwkv7sModelOutputWithPast

## Rwkv7sCausalLMOutputWithPast

[[autodoc]] Rwkv7sCausalLMOutputWithPast

## Rwkv7sModel

[[autodoc]] Rwkv7sModel
    - forward

## Rwkv7sForCausalLM

[[autodoc]] Rwkv7sForCausalLM
    - forward

## Rwkv7sAttention

[[autodoc]] Rwkv7sAttention

## Rwkv7sFeedForward

[[autodoc]] Rwkv7sFeedForward

## Rwkv7sTopkRouter

[[autodoc]] Rwkv7sTopkRouter

## Rwkv7sExperts

[[autodoc]] Rwkv7sExperts

## Rwkv7sMLP

[[autodoc]] Rwkv7sMLP

## Rwkv7sDecoderLayer

[[autodoc]] Rwkv7sDecoderLayer

## Rwkv7sPreTrainedModel

[[autodoc]] Rwkv7sPreTrainedModel

## compute_rwkv7s_rosa_ids

[[autodoc]] compute_rwkv7s_rosa_ids
