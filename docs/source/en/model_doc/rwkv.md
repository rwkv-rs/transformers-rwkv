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
*This model was contributed to Hugging Face Transformers on 2023-05-09.*

# RWKV

## Overview

RWKV-7 is a recurrent language model whose TimeMix layers update a fixed-size matrix state instead of retaining a
key/value tensor for every earlier token. The implementation follows the canonical RWKV-LM training equations and uses
the FlashRWKV2 TileLang provider for both training and inference. The former RWKV-4 implementation and its state/output
types are not compatible with RWKV-7 checkpoints.

This integration requires CUDA and `FlashRWKV2==0.1.0a16`; it intentionally has no CPU or FLA execution fallback. Install
the RWKV dependencies with:

```bash
pip install "transformers[rwkv]"
```

Training, prefill, and decode use BF16 model tensors and activations with FP32 accumulation. WKV states and
head logits are FP32. The provider shares projection, normalization, and recurrence arithmetic across these paths.
Inputs in one batch must have equal length because RWKV-7 does not use a padding token.

## Inference

Converted checkpoints use standard `AutoModelForCausalLM`, `AutoTokenizer`, `generate()`, and [`RwkvCache`] APIs.

```python
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


checkpoint = "/path/to/converted-rwkv7"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    dtype=torch.bfloat16,
).cuda().eval()

messages = [{"role": "user", "content": "Explain recurrent language models briefly."}]
tools = None
generation_prompt = "open_think"
config_file_name = (
    "tools_generation_config.json"
    if tools
    else "fake_think_generation_config.json"
    if generation_prompt == "fake_think"
    else "generation_config.json"
)
generation_config = (
    model.generation_config
    if config_file_name == "generation_config.json"
    else GenerationConfig.from_pretrained(checkpoint, config_file_name=config_file_name)
)
inputs = tokenizer.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
    rwkv_generation_prompt=generation_prompt,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(
    **inputs,
    generation_config=generation_config,
    max_new_tokens=128,
    tokenizer=tokenizer,
)
print(tokenizer.decode(output_ids[0, inputs.input_ids.shape[1] :]))
```

The bundled Jinja template supports the RWKV bot and assistant prompt styles, tool-use conversations, and open/fake
thinking prompts through ordinary `apply_chat_template()` keyword arguments. The model repository contains three
matching generation configurations. `generation_config.json` is the default Open Think profile. Fake Think uses
`fake_think_generation_config.json`. Any request with tools uses `tools_generation_config.json`, which takes precedence
over the thinking prompt and disables penalties. All three profiles use FlashRWKV2 Rapid-Sampling and contain the stop
markers for every prompt style.

## Stateful execution

Pass the returned `past_key_values` back to the model to continue a stream. [`RwkvCache`] stores two token-shift vectors
and one FP32 WKV matrix per layer. It supports clone, detach, reset, batch repeat/select, and beam reorder operations.

```python
from transformers import RwkvCache


cache = RwkvCache(model.config)
first = model(input_ids=inputs.input_ids, past_key_values=cache, use_cache=True)
continued = model(input_ids=torch.tensor([[42]], device=model.device), past_key_values=cache, use_cache=True)
```

Stateful BF16 training keeps the recurrent state in the autograd graph. Call `cache.detach()` when starting a truncated
backpropagation segment.

## Training

Load the complete model in BF16 before training. Stateless training uses the canonical FlashRWKV2 pretraining
operators for sequence lengths divisible by 16; other non-empty chunks use the differentiable zero-initial-state
StateTune path.

```python
model.to(device="cuda", dtype=torch.bfloat16).train()
outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
outputs.loss.backward()
```

For recurrent training, create one [`RwkvCache`] and pass it through the chunks. The cache update occurs outside each
checkpointed decoder layer, so gradient checkpointing does not apply the state transition twice.

## LoRA and CUDA Graphs

BF16 inference supports one active PEFT LoRA or NoRA adapter on TimeMix R/K/V/O, ChannelMix key/value, and the
output head. Disabled and merged adapters are also supported. The provider receives raw adapter parameters and applies
NoRA normalization on each forward. Merge adapters with multiple active branches, DoRA, other variants, or bias.

Training uses differentiable operators whenever gradients are enabled. Use `model.eval()` with `torch.no_grad()` for
recurrent inference. CUDA Graph generation does this automatically. Weight transpose views refer to the canonical
parameters directly; no runtime weight copies or CPU offload are needed. Warm up fixed batch/sequence shapes and their
[`RwkvCache`] on the capture stream before capture so metadata and state addresses are ready.

## LoRA and NoRA training

BF16 training supports one active PEFT LoRA or NoRA adapter with zero dropout on
`r_proj`, `k_proj`, `v_proj`, `o_proj`, ChannelMix `key` / `value`, and `lm_head`.
FlashRWKV2 receives the original adapter parameters and PEFT scaling. For NoRA,
it computes column normalization and its gradient inside the training kernels;
PEFT continues to own adapter state, disabling, saving, loading, and merging.
NoRA requires the Joluck/NoRA PEFT fork (`use_nora=True`); `use_nora="init"`
retains ordinary LoRA behavior after initialization.

For a different output position in each sample, pass `logits_to_keep` as a
`LongTensor` of shape `[batch_size, positions]`. RWKV gathers hidden states
before the vocabulary projection. A 1D tensor continues to select the same
positions for every sample. For example, right-padded next-token scoring uses
`logits_to_keep=(lengths - 1)[:, None]` and reads `outputs.logits[:, 0]`.

## Converting a canonical checkpoint

The converter accepts canonical BlinkDL release keys only. It drops block 0's unused value-residual tensors, plans
Safetensors shards before cloning them, and writes one shard at a time to bound peak host memory.

```bash
python temp/rwkv_pth2st.py \
  /path/to/rwkv7-g1i.pth \
  /path/to/converted-rwkv7 \
  --context-length 10240 \
  --max-shard-size 5GB
```

The output includes the model config, the Open Think, Fake Think, and tools generation configs, sharded Safetensors,
the fixed RWKV World tokenizer, and the single standard chat template.

## RwkvConfig

[[autodoc]] RwkvConfig

## RwkvTokenizer

[[autodoc]] RwkvTokenizer

## RwkvCache

[[autodoc]] RwkvCache

## RwkvModel

[[autodoc]] RwkvModel
    - forward

## RwkvForCausalLM

[[autodoc]] RwkvForCausalLM
    - forward

## RwkvAttention

[[autodoc]] RwkvAttention

## RwkvFeedForward

[[autodoc]] RwkvFeedForward

## RwkvDecoderLayer

[[autodoc]] RwkvDecoderLayer

## RwkvPreTrainedModel

[[autodoc]] RwkvPreTrainedModel
