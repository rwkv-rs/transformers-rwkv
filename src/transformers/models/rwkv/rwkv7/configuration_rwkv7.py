# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RWKV-7 configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="BlinkDL/rwkv-7-world")
class Rwkv7Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Rwkv7Model`]. It is used to instantiate
    an RWKV-7 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read
    the documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the RWKV-7 model. Defines the number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`Rwkv7Model`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        intermediate_size (`int`, *optional*):
            Dimensionality of the channel mix (FFN) inner states. Defaults to `int(hidden_size * 3.5 // 32 * 32)`.
        head_size (`int`, *optional*, defaults to 64):
            Size of each attention head.
        context_length (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model can be used with in a single forward pass.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by layer normalization layers.
        group_norm_epsilon (`float`, *optional*, defaults to 64e-5):
            The epsilon used by the GroupNorm layer inside TimeMix.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token.
        eos_token_id (`int` or `List[int]`, *optional*, defaults to 0):
            The id(s) of the end of sentence token(s).
        rescale_every (`int`, *optional*, defaults to 6):
            At inference, the hidden state (and weights of the corresponding output layers) are divided by 2
            every `rescale_every` blocks. If set to 0 or a negative number, no rescale is done.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the output projection weights (LM head) to the input embeddings.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use caching (state) for fast autoregressive generation.
        deep_embedding (`bool`, *optional*, defaults to `True`):
            Whether to use deep embedding (fuse the embedding weights with `ln0` of block 0 for
            better training stability and inference speed).
        activation_precision (`str`, *optional*, defaults to `"fp32io16"`):
            Precision mode for internal activations. Options: `"fp32io16"`, `"fp16"`, `"bf16"`.
            - `"fp32io16"`: Internal WKV computation uses fp32, input/output uses fp16/bf16.
            - `"fp16"`: All computation uses fp16.
            - `"bf16"`: All computation uses bf16.

    Example:

    ```python
    >>> from transformers import Rwkv7Config, Rwkv7Model

    >>> # Initializing a Rwkv7 configuration
    >>> configuration = Rwkv7Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = Rwkv7Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "rwkv7"
    attribute_map = {
        "max_position_embeddings": "context_length",
        "num_attention_heads": "num_hidden_layers",  # RWKV doesn't have traditional attention heads
    }

    vocab_size: int = 65536
    context_length: int = 4096
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int | None = None
    head_size: int = 64
    layer_norm_epsilon: float = 1e-5
    group_norm_epsilon: float = 64e-5
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 0
    rescale_every: int = 6
    tie_word_embeddings: bool = False
    use_cache: bool = True
    deep_embedding: bool = True
    activation_precision: str = "fp32io16"

    def __post_init__(self, **kwargs):
        if self.intermediate_size is None:
            # Default: ~3.5x hidden_size, rounded to multiple of 32
            self.intermediate_size = int((self.hidden_size * 3.5) // 32 * 32)

        super().__post_init__(**kwargs)

    @property
    def num_attention_heads(self) -> int:
        """Number of attention heads (derived from hidden_size / head_size)."""
        return self.hidden_size // self.head_size

    @property
    def head_size_a(self) -> int:
        """Head size, alias for compatibility with original RWKV-7 naming."""
        return self.head_size


__all__ = ["Rwkv7Config"]
