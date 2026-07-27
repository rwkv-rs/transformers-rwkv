<!--Copyright 2026 The RWKV-7 and HuggingFace Inc. teams.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not
be rendered properly in your Markdown viewer.

-->

# RWKV-7

## Overview

RWKV-7 is a recurrent language model from the [RWKV-LM project](https://github.com/BlinkDL/RWKV-LM). TimeMix updates
a fixed-size per-layer matrix state, so cached autoregressive decoding does not retain a key/value tensor for every
previous token. ChannelMix supplies the feed-forward path.

This implementation follows the parameter names and matrix orientation used by the original RWKV-7 code. `TimeMix`,
`ChannelMix`, `DeepEmbedding`, and the model block are independent classes and can be replaced by optimized subclasses.

The recurrent precision is selected with `Rwkv7Config.wkv_mode`:

- `"fp32io16"` keeps the recurrent matrix and its update in float32 while model inputs and outputs use the model dtype.
- `"fp16"` keeps the recurrent matrix in the model dtype for lower memory use.

## Inference with recurrent state

```python
import torch

from transformers import AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained("path/to/rwkv7-checkpoint", dtype=torch.float16)
input_ids = torch.tensor([[1, 2, 3, 4]], device=model.device)

prefix = model(input_ids[:, :3], use_cache=True)
suffix = model(input_ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True)
```

Official BlinkDL checkpoints can be converted without semantic parameter renaming or matrix transposition:

```bash
python -m transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf \
    --checkpoint_file rwkv7-model.pth \
    --output_dir rwkv7-hf \
    --dtype float16 \
    --wkv_mode fp32io16
```

## Rwkv7Config

[[autodoc]] Rwkv7Config

## Rwkv7Model

[[autodoc]] Rwkv7Model
    - forward

## Rwkv7ForCausalLM

[[autodoc]] Rwkv7ForCausalLM
    - forward

## Rwkv7Cache

[[autodoc]] Rwkv7Cache

## Replaceable components

[[autodoc]] Rwkv7TimeMix

[[autodoc]] Rwkv7ChannelMix

[[autodoc]] Rwkv7DeepEmbedding
