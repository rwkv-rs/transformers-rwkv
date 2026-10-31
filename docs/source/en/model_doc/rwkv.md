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

This integration requires CUDA and `FlashRWKV2==0.1.0a17`, which provides persistent verified tuning profiles and the
public preparation API. Install the RWKV dependencies with:

```bash
pip install "transformers[rwkv]"
```

Training, prefill, and decode use BF16 model tensors and activations with FP32 accumulation. By default, WKV states and
head logits are FP32. The provider shares projection, normalization, and recurrence arithmetic across these paths.
Inputs in one batch must have equal length because RWKV-7 does not use a padding token.

The default `wkv_mode="fp32io16"` matches training bitwise. Set `model.config.wkv_mode = "fp16"` before inference
to use optional FP16 WKV state in both ordinary forward and CUDA Graph generation. This keeps BF16 token IO and
FP32 arithmetic, but rounds the recurrent state to FP16 after every token; it does not promise bitwise equality with
the FP32 training trajectory. Training, including StateTune and recurrent training, always keeps FP32 state regardless
of this inference option. Create a fresh cache when changing state precision or returning from FP16-state inference
to training. The benchmark exposes the same option as `--wkv-mode fp16`.

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
fake_cot_mode = False
config_file_name = (
    "tools_generation_config.json"
    if tools
    else "fake_think_generation_config.json"
    if fake_cot_mode
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
    fake_cot_mode=fake_cot_mode,
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
thinking prompts through ordinary `apply_chat_template()` keyword arguments. `rwkv_prompt_template` selects `bot`
(the default), `assistant`, or `function_calling`. The boolean `fake_cot_mode` defaults to `False`, appending `<think`
to the generation prompt; `True` appends `<think></think`. Tools or tool-call history select the function-calling format.
The model repository contains three matching generation configurations. `generation_config.json` is the default
Open Think profile. Fake Think uses
`fake_think_generation_config.json`. Any request with tools uses `tools_generation_config.json`, which takes precedence
over the thinking prompt and disables penalties. All three profiles use FlashRWKV2 Rapid-Sampling and contain the stop
markers for every prompt style.

New conversions save `temp/rwkv_chat_template.jinja` as the checkpoint's `chat_template.jinja`. Existing converted
checkpoints need that file refreshed to use `fake_cot_mode`; updating this repository does not refresh saved templates.
The weight publisher reuses existing local artifacts and skips already published checkpoints.

## Stateful execution

Pass the returned `past_key_values` back to the model to continue a stream. [`RwkvCache`] stores two token-shift vectors
and one WKV matrix per layer: FP32 for training and default inference, or FP16 for `wkv_mode="fp16"` inference.
It supports clone, detach, reset, batch repeat/select, and beam reorder operations.

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

Generation checks FlashRWKV2 tuning coverage during each graph's real warmup. It covers the prompt prefix, each
distinct prefill chunk length, and the one-token decode path including the output head. The default
`tuning_mode="inspect"` reports missing configurations and warns when a graph uses untuned defaults. To prepare only
missing configurations and then require verified coverage, use the same model, inputs, and generation settings:

```python
model.generate(**inputs, max_new_tokens=128, tokenizer=tokenizer, tuning_mode="tune")
output_ids = model.generate(**inputs, max_new_tokens=128, tokenizer=tokenizer, tuning_mode="require")
```

`require` fails before capturing a graph whose warmup encounters a missing profile. Changing the preparation mode
creates new graphs, so a graph captured with defaults is not reused after tuning. Warmup restores recurrent state,
sampling state, and penalties before capture and generation.

FlashRWKV2 stores verified profiles in `~/.cache/flashrwkv2`, independently of the virtual environment,
`TORCH_HOME`, and `XDG_CACHE_HOME`. Set `FLASHRWKV2_CACHE_DIR` before importing FlashRWKV2 to relocate this cache.
To migrate an old namespace, pass `tuning_legacy_dir="/path/to/old/namespace"` with `tuning_mode="tune"`.
FlashRWKV2 revalidates matching profiles against the current kernels and tunes missing shapes.

For ordinary forward or training preparation, call the public `flashrwkv2.prepare_tuning` with zero-argument
callables that execute the actual workload. Training callables should include backward when preparing its kernels,
clear gradients between runs, and use disposable or reset recurrent state. Preserve the workload's gradient mode and
active LoRA/NoRA adapters. Preparation does not change the model's FP32 training state or head logits.

`temp/benchmark_rwkv_generate.py` defaults to `--tuning-mode require`; use `--tuning-mode tune` to fill missing
profiles before benchmarking. Its JSON records the loaded model configuration and every process's prefill/decode
coverage report, including selected kernel configurations and cache paths. Preparation and capture time are reported
separately from generation throughput.

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
