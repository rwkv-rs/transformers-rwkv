# Copyright 2026 The RWKV team and The HuggingFace Inc. team. All rights reserved.
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
"""RWKV-7 (Goose) model configuration."""

from ...configuration_utils import PreTrainedConfig


class Rwkv7Config(PreTrainedConfig):
    r"""
    Configuration for [`Rwkv7Model`], an all-recurrent (attention-free) RWKV-7 "Goose"
    model. Instantiating with the defaults yields the ~0.1B RWKV-7 configuration.

    Parameter names follow the upstream RWKV reference implementation
    (`BlinkDL/RWKV-LM`) rather than a renamed variant, so converting a native
    `.pth` checkpoint is close to a rename-free copy.

    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size (RWKV "world" tokenizer).
        hidden_size (`int`, *optional*, defaults to 768):
            Model width `C`.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of blocks.
        head_dim (`int`, *optional*, defaults to 64):
            Width of one WKV head. `hidden_size` must be divisible by it.
        num_heads (`int`, *optional*, defaults to 12):
            Number of WKV heads; must equal `hidden_size // head_dim`.
        decay_low_rank_dim (`int`, *optional*, defaults to 64):
            Rank of the decay (`w`) LoRA.
        a_low_rank_dim (`int`, *optional*, defaults to 64):
            Rank of the in-context-learning-rate (`a`) LoRA.
        v_low_rank_dim (`int`, *optional*, defaults to 32):
            Rank of the value-residual (`v`) LoRA. Unused on layer 0, which
            *produces* `v_first` instead of mixing towards it.
        gate_low_rank_dim (`int`, *optional*, defaults to 128):
            Rank of the output-gate (`g`) LoRA.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Channel-mix inner width (`hidden_ratio * hidden_size` by convention).
        hidden_ratio (`float`, *optional*, defaults to 4.0):
            Used to derive `intermediate_size` when it is not given explicitly.
        hidden_act (`str`, *optional*, defaults to `"sqrelu"`):
            Channel-mix activation. RWKV-7 uses squared ReLU.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            Epsilon of every LayerNorm/GroupNorm in the model.
        norm_bias (`bool`, *optional*, defaults to `True`):
            Whether the norms carry a bias.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            Training context length. RWKV is recurrent and not bounded by it at
            inference; it only sizes generation defaults.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the input embedding and the LM head.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the recurrent state.
        use_deep_embed (`bool`, *optional*, defaults to `False`):
            Enable the RWKV-8 "DeepEmbed" hook: a per-layer, per-token vector that
            channelwise-modulates the channel-mix. The table is deliberately NOT a
            model weight — it is meant to live in RAM/SSD and be prefetched per
            token, which is the whole point of the design (VRAM savings) — so it is
            passed to the forward as `deep_embeds` instead. No RWKV-7 checkpoint
            carries one; this is an extension point, off by default.
        deep_embed_size (`int`, *optional*):
            Width of one layer's DeepEmbed vector. `hidden_size` reproduces the
            reference "1x" variant (modulating the channel-mix output);
            `intermediate_size` reproduces "4x" (modulating its input). Defaults to
            `hidden_size` when `use_deep_embed` is set.
        wkv_state_dtype (`str`, *optional*, defaults to `"float32"`):
            Precision the recurrent WKV state is carried and accumulated in,
            independently of the activation dtype. The recurrence is unrolled over
            the whole sequence, so a narrow state drifts; `"float32"` with fp16
            activations is the combination the reference implementation uses.
            `"float16"`/`"bfloat16"` trade that for a smaller state.
        wkv_implementation (`str`, *optional*, defaults to `"eager"`):
            Which WKV recurrence to use, by name, from
            `models.rwkv7.modeling_rwkv7.RWKV7_WKV_FUNCTIONS`. `"eager"` is the
            portable PyTorch path — the sequential step when decoding, the
            chunk-parallel form otherwise, and per-segment when a packed batch is
            passed. Register an entry in that mapping to plug in a fused or varlen
            kernel without forking the model.
        sparse_channel_mix (`bool`, *optional*, defaults to `False`):
            Skip the channel-mix value-projection rows whose input channel is zero.
            The activation is a squared ReLU, so its zeros are exact and skipping
            them is exact too; on a 7.2B checkpoint only about a tenth of the
            channels are nonzero -- 10.07% measured over 16 decode steps, 6.85%
            low and 11.99% high -- and that projection is a third of the
            model's bytes. Costs a
            transposed copy of the value weight (about +30% weights), built lazily,
            and only pays once launches are captured — see the model doc.

            Exact but not bit-reproducible: the surviving channels are summed across
            several partitions that combine through an atomic add, so the order the
            partitions arrive in — and therefore the last unit in the last place of
            each output — varies between runs of identical input. Same rounding
            class as a split-K GEMM. Greedy decoding can turn that into a different
            token at a near-tie, so leave this off when you need a run to reproduce
            itself exactly.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning-of-sequence id. The RWKV world tokenizer has no dedicated BOS
            token and the reference implementation prepends nothing, so this exists to
            satisfy `GenerationMixin` rather than to be emitted.
        eos_token_id (`int`, *optional*, defaults to 0):
            End-of-sequence id, id 0 in the RWKV world vocabulary.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding id, the same id 0. Set deliberately rather than left `None`:
            `generate` needs one to pad a batch, and without it a batched call either
            raised or fell back to the eos id with a warning on every step.

    ```python
    >>> from transformers import Rwkv7Config, Rwkv7Model

    >>> configuration = Rwkv7Config()
    >>> model = Rwkv7Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "rwkv7"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=768,
        num_hidden_layers=12,
        head_dim=64,
        num_heads=12,
        decay_low_rank_dim=64,
        a_low_rank_dim=64,
        v_low_rank_dim=32,
        gate_low_rank_dim=128,
        intermediate_size=3072,
        hidden_ratio=4.0,
        hidden_act="sqrelu",
        norm_eps=1e-5,
        norm_bias=True,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
        use_cache=True,
        use_deep_embed=False,
        deep_embed_size=None,
        wkv_state_dtype="float32",
        wkv_implementation="eager",
        sparse_channel_mix=False,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.decay_low_rank_dim = decay_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else int(hidden_size * hidden_ratio)
        )
        self.hidden_act = hidden_act
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.use_deep_embed = use_deep_embed
        self.deep_embed_size = deep_embed_size if deep_embed_size is not None else hidden_size
        if wkv_state_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"wkv_state_dtype must be float32/float16/bfloat16, got {wkv_state_dtype}")
        self.wkv_state_dtype = wkv_state_dtype
        self.wkv_implementation = wkv_implementation
        self.sparse_channel_mix = sparse_channel_mix

        if hidden_size % head_dim != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by head_dim {head_dim}")
        if num_heads != hidden_size // head_dim:
            raise ValueError(f"num_heads must be hidden_size // head_dim = {hidden_size // head_dim}, got {num_heads}")

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Rwkv7Config"]
