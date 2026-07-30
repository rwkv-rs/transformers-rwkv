<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# RWKV-7

[RWKV-7](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7) is a recurrent language model with linear-time
sequence processing and constant-size recurrent state during decoding. Its time-mix recurrence can also process a
prompt in chunks, while producing the same continuation state as token-by-token execution.

The implementation supports RWKV-7 World and G1 checkpoints. Raw checkpoints must first be converted to the
Transformers format. The converter removes the unused block-0 `v0`, `v1`, and `v2` tensors found in some G1
checkpoints and can optionally fuse the block-0 layer normalization into the embedding table.

```bash
python -m transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf \
    --checkpoint_path /path/to/rwkv7.pth \
    --output_dir /path/to/rwkv7-hf \
    --tokenizer_name_or_path RWKV/rwkv-5-world-1b5
```

The recurrent `state` returned by the model can be passed to a later call to continue a sequence without processing
the prefix again.

```python
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = "/path/to/rwkv7-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="auto")

inputs = tokenizer("The Eiffel Tower is in", return_tensors="pt").to(model.device)
with torch.no_grad():
    prefix = model(**inputs, use_cache=True)
    next_token = prefix.logits[:, -1:].argmax(dim=-1)
    continuation = model(next_token, state=prefix.state, use_cache=True)
```

## WKV backends

RWKV-7 keeps the model projections and recurrent-state protocol separate from WKV execution. The default `"auto"`
backend loads the bundled vllm-rwkv packed-varlen inference operator for FP16 CUDA evaluation and the official
RWKV-v7 wind-backstepping operator for BF16 CUDA training. CUDA kernels are compiled lazily, so importing or
installing Transformers does not require CUDA. Runtime compilation requires a CUDA toolkit, a supported C++
compiler, and Ninja.

The inference adapter passes `raw_decay` before the sigmoid/decay transform, `negative_key` as the negated normalized
key, and `scaled_key` as the normalized key multiplied by the in-context learning rate. Inference activations use
FP16, recurrent WKV state uses FP32, and the bundled operators require a head size of 64. The training operator starts
from an empty state and does not support packed or padded batches.

An alternate existing kernel can be selected without forking the model by registering an adapter with
`register_rwkv7_wkv_backend` and setting `config.wkv_backend` to its registered name. Adapters receive the six named
WKV tensors, recurrent state, packed-sequence metadata, and `head_size`; they return `(output, updated_state)`.

## Rwkv7Config

[[autodoc]] Rwkv7Config

## Rwkv7Model

[[autodoc]] Rwkv7Model
    - forward

## Rwkv7ForCausalLM

[[autodoc]] Rwkv7ForCausalLM
    - forward
