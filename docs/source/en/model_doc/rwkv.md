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
the FlashRWKV2 CUDA provider for both training and inference. The former RWKV-4 implementation and its state/output
types are not compatible with RWKV-7 checkpoints.

This integration requires CUDA and `FlashRWKV2==0.1.0a13`; it intentionally has no CPU or FLA execution fallback. Install
the RWKV dependencies with:

```bash
pip install "transformers[rwkv]"
```

Training uses BF16 model tensors. Inference uses FP16 model tensors, except for the BF16 embedding/LN0 fusion and FP32
WKV states. Inputs in one batch must have equal length because RWKV-7 does not use a padding token.

## Inference

Converted checkpoints use standard `AutoModelForCausalLM`, `AutoTokenizer`, `generate()`, and [`RwkvCache`] APIs.

```python
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


checkpoint = "/path/to/converted-rwkv7"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    dtype=torch.float16,
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

Move the complete model back to BF16 before training. Stateless training uses the canonical FlashRWKV2 pretraining
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

During FP16 inference, FlashRWKV2 accepts one active, unmerged vanilla PEFT LoRA adapter on each TimeMix R/K/V/O
projection. Disabled and merged adapters are also supported. Merge adapters that use multiple active branches, DoRA,
variants, bias, or ChannelMix targets before inference; unsupported layouts fail before a kernel launch.

Inference runtime layouts are created automatically by the first warmup forward. That warmup also moves the canonical
ChannelMix value weight to CPU after creating its transposed runtime layout. Calling `.to(...)` or loading a state dict
invalidates these non-persistent layouts and restores the canonical parameter. For CUDA Graph decode, warm up the fixed
batch/sequence shape and its [`RwkvCache`] on the capture stream before capture; cache state tensors retain fixed
addresses during eval updates.

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
