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

# RWKV-7

RWKV-7 lives in `transformers.models.rwkv`. The model source is
[`src/transformers/models/rwkv/modeling_rwkv.py`](https://github.com/rwkv-rs/transformers-rwkv/blob/main/src/transformers/models/rwkv/modeling_rwkv.py).
Import it with `from transformers import RwkvForCausalLM, RwkvConfig, RwkvTokenizer, RwkvCache`.
`AutoModelForCausalLM` loads converted checkpoints without `trust_remote_code`. Loading an RWKV-4 checkpoint raises an error.

```bash
pip install "transformers[rwkv]"
```

| Transformers | Component | RWKV-LM | FLA / hf_rwkv |
| --- | --- | --- | --- |
| `RwkvAttention` | TimeMix and WKV recurrence | `RWKV_Tmix_x070` | `RWKV7Attention` |
| `RwkvFeedForward` | ChannelMix | `RWKV_CMix_x070` | `RWKV7FeedForward` |
| `RwkvDecoderLayer` | TimeMix and ChannelMix block | `Block` | `RWKV7Block` |

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

RWKV-7 uses CUDA and BF16 activations. A dense batch contains token IDs and labels of shape `[batch, length]`.
The model uses causal next-token loss.

```python
model.train()
loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
loss.backward()
```

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

Pass the returned `past_key_values` back to continue a stream. [`RwkvCache`] stores two token-shift vectors and one
WKV matrix per layer. Use `cache.clone()` to branch a stream and `cache.detach()` to start a truncated
backpropagation segment.

```python
from transformers import RwkvCache

cache = RwkvCache(model.config)
model(input_ids=inputs.input_ids, past_key_values=cache)
continued = model(input_ids=torch.tensor([[42]], device=model.device), past_key_values=cache)
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
