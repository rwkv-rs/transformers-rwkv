# Copyright 2026 The RWKV-7 and HuggingFace Inc. teams.
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
"""PyTorch RWKV-7 model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, logging
from .configuration_rwkv7 import Rwkv7Config


logger = logging.get_logger(__name__)

RWKV7_EXP_NEG_HALF = 0.606531


class Rwkv7Cache(Cache):
    """Constant-size recurrent cache used by RWKV-7.

    RWKV-7 does not have key/value history. Each layer stores one recurrent
    matrix and the previous TimeMix and ChannelMix inputs.
    """

    is_compileable = False

    def __init__(
        self,
        recurrent_state: list[torch.Tensor] | None = None,
        time_mix_state: list[torch.Tensor] | None = None,
        channel_mix_state: list[torch.Tensor] | None = None,
        v_first: torch.Tensor | None = None,
        seen_tokens: int = 0,
    ):
        # The generic Cache constructor creates Transformer KV-cache layers,
        # which are not part of the RWKV state contract.
        self.layers = []
        self.recurrent_state = recurrent_state
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state
        self.v_first = v_first
        self.seen_tokens = int(seen_tokens)

    @property
    def is_initialized(self) -> bool:
        return all(
            value is not None
            for value in (self.recurrent_state, self.time_mix_state, self.channel_mix_state, self.v_first)
        )

    @property
    def batch_size(self) -> int:
        if self.recurrent_state:
            return int(self.recurrent_state[0].shape[0])
        if self.v_first is not None:
            return int(self.v_first.shape[0])
        return 0

    def __len__(self) -> int:
        return len(self.recurrent_state or ())

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < 0 or (self.recurrent_state is not None and layer_idx >= len(self.recurrent_state)):
            return 0
        return self.seen_tokens

    def get_max_length(self, layer_idx: int | None = None) -> int:
        return -1

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        return self.seen_tokens + int(query_length), 0

    def reset(self):
        self.recurrent_state = None
        self.time_mix_state = None
        self.channel_mix_state = None
        self.v_first = None
        self.seen_tokens = 0

    def reorder_cache(self, beam_idx: torch.LongTensor):
        def select(values):
            if values is None:
                return None
            return [value.index_select(0, beam_idx.to(value.device)) for value in values]

        self.recurrent_state = select(self.recurrent_state)
        self.time_mix_state = select(self.time_mix_state)
        self.channel_mix_state = select(self.channel_mix_state)
        if self.v_first is not None:
            self.v_first = self.v_first.index_select(0, beam_idx.to(self.v_first.device))
        return self

    def batch_repeat_interleave(self, repeats: int):
        def repeat(values):
            if values is None:
                return None
            return [value.repeat_interleave(repeats, dim=0) for value in values]

        self.recurrent_state = repeat(self.recurrent_state)
        self.time_mix_state = repeat(self.time_mix_state)
        self.channel_mix_state = repeat(self.channel_mix_state)
        if self.v_first is not None:
            self.v_first = self.v_first.repeat_interleave(repeats, dim=0)
        return self

    def batch_select_indices(self, indices: torch.Tensor):
        return self.reorder_cache(indices)

    def crop(self, max_length: int):
        target_length = self.seen_tokens + max_length if max_length < 0 else max_length
        if target_length >= self.seen_tokens:
            return self
        if target_length <= 0:
            self.reset()
            return self
        raise ValueError("RWKV-7 recurrent state cannot be cropped without a saved prefix state")


class Rwkv7DeepEmbedding(nn.Module):
    """Factorized DeepEmbedding lookup.

    Checkpoints store a direct per-token table and a residual projection from
    the ordinary token embedding. Keeping both factors avoids materializing a
    merged table for every layer.
    """

    def forward(
        self,
        input_ids: torch.LongTensor,
        token_embeddings: torch.Tensor,
        direct_embedding: nn.Embedding,
        residual_projection: nn.Linear,
    ) -> torch.Tensor:
        return direct_embedding(input_ids) + residual_projection(token_embeddings)


class Rwkv7TimeMix(nn.Module):
    """Bo-style RWKV-7 TimeMix with an eager recurrent reference backend."""

    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.attention_hidden_size = config.attention_hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads

        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, name, nn.Parameter(torch.empty(1, 1, config.hidden_size)))

        self.w0 = nn.Parameter(torch.empty(1, 1, config.attention_hidden_size))
        self.w1 = nn.Parameter(torch.empty(config.hidden_size, config.decay_low_rank_dim))
        self.w2 = nn.Parameter(torch.empty(config.decay_low_rank_dim, config.attention_hidden_size))
        self.a0 = nn.Parameter(torch.empty(1, 1, config.attention_hidden_size))
        self.a1 = nn.Parameter(torch.empty(config.hidden_size, config.a_low_rank_dim))
        self.a2 = nn.Parameter(torch.empty(config.a_low_rank_dim, config.attention_hidden_size))
        if layer_id > 0:
            self.v0 = nn.Parameter(torch.empty(1, 1, config.attention_hidden_size))
            self.v1 = nn.Parameter(torch.empty(config.hidden_size, config.value_low_rank_dim))
            self.v2 = nn.Parameter(torch.empty(config.value_low_rank_dim, config.attention_hidden_size))
        self.g1 = nn.Parameter(torch.empty(config.hidden_size, config.gate_low_rank_dim))
        self.g2 = nn.Parameter(torch.empty(config.gate_low_rank_dim, config.attention_hidden_size))

        self.k_k = nn.Parameter(torch.empty(1, 1, config.attention_hidden_size))
        self.k_a = nn.Parameter(torch.empty(1, 1, config.attention_hidden_size))
        self.r_k = nn.Parameter(torch.empty(config.num_attention_heads, config.head_dim))
        self.receptance = nn.Linear(config.hidden_size, config.attention_hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.attention_hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.attention_hidden_size, bias=False)
        self.output = nn.Linear(config.attention_hidden_size, config.hidden_size, bias=False)
        self.ln_x = nn.GroupNorm(
            config.num_attention_heads,
            config.attention_hidden_size,
            eps=config.head_dim * 1e-5,
        )

    def _step(
        self,
        hidden: torch.Tensor,
        previous: torch.Tensor,
        recurrent_state: torch.Tensor,
        v_first: torch.Tensor,
        active: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = hidden.shape[0]
        xx = previous - hidden
        xr = hidden + xx * self.x_r.reshape(1, self.hidden_size)
        xw = hidden + xx * self.x_w.reshape(1, self.hidden_size)
        xk = hidden + xx * self.x_k.reshape(1, self.hidden_size)
        xv = hidden + xx * self.x_v.reshape(1, self.hidden_size)
        xa = hidden + xx * self.x_a.reshape(1, self.hidden_size)
        xg = hidden + xx * self.x_g.reshape(1, self.hidden_size)

        r = self.receptance(xr)
        w = torch.tanh(xw @ self.w1) @ self.w2 + self.w0.reshape(1, -1)
        k = self.key(xk)
        v = self.value(xv)
        a = torch.sigmoid(xa @ self.a1 @ self.a2 + self.a0.reshape(1, -1))
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize(
            (k * self.k_k.reshape(1, -1)).view(batch_size, self.num_heads, self.head_dim),
            dim=-1,
            p=2,
        ).view(batch_size, self.attention_hidden_size)
        k = k * (1 + (a - 1) * self.k_a.reshape(1, -1))
        if self.layer_id == 0:
            next_v_first = v
        else:
            value_gate = torch.sigmoid(xv @ self.v1 @ self.v2 + self.v0.reshape(1, -1))
            v = v + (v_first - v) * value_gate
            next_v_first = v_first

        w = torch.exp(-RWKV7_EXP_NEG_HALF * torch.sigmoid(w.float()))
        state_dtype = torch.float32 if self.config.wkv_mode == "fp32io16" else hidden.dtype
        state = recurrent_state.to(state_dtype)
        k_state = k.to(state_dtype)
        v_state = v.to(state_dtype)
        kk_state = kk.to(state_dtype)
        a_state = a.to(state_dtype)
        w_state = w.to(state_dtype)
        vk = v_state.view(batch_size, self.num_heads, self.head_dim, 1) @ k_state.view(
            batch_size, self.num_heads, 1, self.head_dim
        )
        ab = (-kk_state).view(batch_size, self.num_heads, self.head_dim, 1) @ (kk_state * a_state).view(
            batch_size, self.num_heads, 1, self.head_dim
        )
        next_state = state * w_state.view(batch_size, self.num_heads, 1, self.head_dim) + state @ ab + vk

        out = next_state.to(hidden.dtype) @ r.view(batch_size, self.num_heads, self.head_dim, 1)
        out = out.view(batch_size, self.attention_hidden_size)
        out = F.group_norm(
            out,
            num_groups=self.num_heads,
            weight=self.ln_x.weight,
            bias=self.ln_x.bias,
            eps=self.head_dim * 1e-5,
        )
        residual = (
            r.view(batch_size, self.num_heads, self.head_dim)
            * k.view(batch_size, self.num_heads, self.head_dim)
            * self.r_k.reshape(1, self.num_heads, self.head_dim)
        ).sum(dim=-1, keepdim=True)
        out = out + (residual * v.view(batch_size, self.num_heads, self.head_dim)).view(
            batch_size, self.attention_hidden_size
        )
        out = self.output(out * g)

        if active is not None:
            row_mask = active[:, None]
            matrix_mask = active[:, None, None, None]
            out = torch.where(row_mask, out, torch.zeros_like(out))
            next_state = torch.where(matrix_mask, next_state, recurrent_state)
            next_previous = torch.where(row_mask, hidden, previous)
            next_v_first = torch.where(row_mask, next_v_first, v_first)
        else:
            next_previous = hidden
        return out, next_previous, next_state, next_v_first

    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        v_first: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = []
        next_v_first_tokens = []
        for token_idx in range(hidden_states.shape[1]):
            active = attention_mask[:, token_idx].bool() if attention_mask is not None else None
            output, previous_state, recurrent_state, token_v_first = self._step(
                hidden_states[:, token_idx], previous_state, recurrent_state, v_first[:, token_idx], active
            )
            outputs.append(output)
            next_v_first_tokens.append(token_v_first)
        return (
            torch.stack(outputs, dim=1),
            previous_state,
            recurrent_state,
            torch.stack(next_v_first_tokens, dim=1),
        )


class Rwkv7ChannelMix(nn.Module):
    """Bo-style RWKV-7 ChannelMix with optional DeepEmbedding."""

    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.x_k = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.deep_embedding = Rwkv7DeepEmbedding()
        if config.deep_embedding_size > 0:
            size = config.deep_embedding_size
            self.s_emb = nn.Embedding(config.vocab_size, size * size)
            self.s_emb_x = nn.Linear(config.hidden_size, size * size, bias=False)
            self.s1 = nn.Parameter(torch.empty(config.hidden_size, size))
            self.s2 = nn.Parameter(torch.empty(size, config.intermediate_size))
            self.s0 = nn.Parameter(torch.empty(1, 1, config.intermediate_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_state: torch.Tensor,
        input_ids: torch.LongTensor | None = None,
        token_embeddings: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        deep_embeddings = None
        if self.config.deep_embedding_size > 0:
            if input_ids is None or token_embeddings is None:
                raise ValueError("DeepEmbedding requires input_ids and their ordinary token embeddings")
            size = self.config.deep_embedding_size
            deep_embeddings = self.deep_embedding(input_ids, token_embeddings, self.s_emb, self.s_emb_x)
            deep_embeddings = deep_embeddings.view(*deep_embeddings.shape[:-1], size, size)

        outputs = []
        for token_idx in range(hidden_states.shape[1]):
            hidden = hidden_states[:, token_idx]
            mixed = hidden + (previous_state - hidden) * self.x_k.reshape(1, -1)
            activated = torch.relu(self.key(mixed)).square()
            if deep_embeddings is not None:
                selector = (hidden @ self.s1).unsqueeze(-2) @ deep_embeddings[:, token_idx]
                gate = selector.squeeze(-2) @ self.s2 + self.s0.reshape(1, -1)
                activated = activated * gate
            output = self.value(activated)
            if attention_mask is not None:
                active = attention_mask[:, token_idx].bool()[:, None]
                output = torch.where(active, output, torch.zeros_like(output))
                previous_state = torch.where(active, hidden, previous_state)
            else:
                previous_state = hidden
            outputs.append(output)
        return torch.stack(outputs, dim=1), previous_state


class Rwkv7Block(GradientCheckpointingLayer):
    time_mix_class = Rwkv7TimeMix
    channel_mix_class = Rwkv7ChannelMix

    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.att = self.time_mix_class(config, layer_id)
        self.ffn = self.channel_mix_class(config, layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor | None,
        token_embeddings: torch.Tensor,
        recurrent_state: torch.Tensor,
        time_mix_state: torch.Tensor,
        channel_mix_state: torch.Tensor,
        v_first: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: bool,
    ):
        if self.layer_id == 0:
            hidden_states = self.ln0(hidden_states)
        time_mix, time_mix_state, recurrent_state, v_first = self.att(
            self.ln1(hidden_states), time_mix_state, recurrent_state, v_first, attention_mask
        )
        hidden_states = hidden_states + time_mix
        channel_mix, channel_mix_state = self.ffn(
            self.ln2(hidden_states), channel_mix_state, input_ids, token_embeddings, attention_mask
        )
        hidden_states = hidden_states + channel_mix
        attention = time_mix if output_attentions else None
        return hidden_states, recurrent_state, time_mix_state, channel_mix_state, v_first, attention


@auto_docstring
class Rwkv7PreTrainedModel(PreTrainedModel):
    config: Rwkv7Config
    base_model_prefix = "model"
    _no_split_modules = ["Rwkv7Block"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (Rwkv7TimeMix, Rwkv7ChannelMix)):
            for parameter in module.parameters(recurse=False):
                nn.init.normal_(parameter, mean=0.0, std=self.config.initializer_range)


@auto_docstring(custom_intro="""Class for RWKV-7 base-model outputs.""")
@dataclass
class Rwkv7Output(ModelOutput):
    r"""
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Hidden states from the final layer normalization.
        past_key_values (`Rwkv7Cache`, *optional*):
            Constant-size recurrent state for continuing the sequence.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states returned when `output_hidden_states=True`.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            TimeMix outputs returned when `output_attentions=True`.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Rwkv7Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(custom_intro="""Class for RWKV-7 causal language-model outputs.""")
@dataclass
class Rwkv7CausalLMOutput(ModelOutput):
    r"""
    Args:
        loss (`torch.FloatTensor`, *optional*):
            Next-token prediction loss when `labels` are supplied.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Language-model prediction scores.
        past_key_values (`Rwkv7Cache`, *optional*):
            Constant-size recurrent state for continuing the sequence.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states returned when `output_hidden_states=True`.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            TimeMix outputs returned when `output_attentions=True`.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Rwkv7Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring
class Rwkv7Model(Rwkv7PreTrainedModel):
    block_class = Rwkv7Block

    def __init__(self, config: Rwkv7Config):
        super().__init__(config)
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [self.block_class(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        self.ln_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, value):
        self.emb = value

    def _new_cache(self, hidden_states: torch.Tensor) -> Rwkv7Cache:
        batch_size = hidden_states.shape[0]
        config = self.config
        state_dtype = torch.float32 if config.wkv_mode == "fp32io16" else hidden_states.dtype
        recurrent_state = [
            torch.zeros(
                batch_size,
                config.num_attention_heads,
                config.head_dim,
                config.head_dim,
                dtype=state_dtype,
                device=hidden_states.device,
            )
            for _ in self.blocks
        ]
        time_mix_state = [
            torch.zeros(batch_size, config.hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
            for _ in self.blocks
        ]
        channel_mix_state = [
            torch.zeros(batch_size, config.hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)
            for _ in self.blocks
        ]
        v_first = torch.zeros(
            batch_size,
            1,
            config.attention_hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return Rwkv7Cache(recurrent_state, time_mix_state, channel_mix_state, v_first)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Rwkv7Cache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | Rwkv7Output:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        token_embeddings = self.emb(input_ids) if inputs_embeds is None else inputs_embeds
        hidden_states = token_embeddings
        deep_token_embeddings = (
            self.blocks[0].ln0(token_embeddings) if self.config.deep_embedding_size > 0 else token_embeddings
        )
        if past_key_values is None:
            past_key_values = self._new_cache(hidden_states)
        elif not isinstance(past_key_values, Rwkv7Cache):
            raise TypeError("past_key_values must be a Rwkv7Cache")
        if past_key_values.batch_size != hidden_states.shape[0]:
            raise ValueError("RWKV-7 cache batch size must match the input batch size")

        sequence_length = hidden_states.shape[1]
        if past_key_values.v_first is None or past_key_values.v_first.shape[1] != sequence_length:
            past_key_values.v_first = torch.zeros(
                hidden_states.shape[0],
                sequence_length,
                self.config.attention_hidden_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for layer_id, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            (
                hidden_states,
                past_key_values.recurrent_state[layer_id],
                past_key_values.time_mix_state[layer_id],
                past_key_values.channel_mix_state[layer_id],
                past_key_values.v_first,
                attention,
            ) = block(
                hidden_states,
                input_ids,
                deep_token_embeddings,
                past_key_values.recurrent_state[layer_id],
                past_key_values.time_mix_state[layer_id],
                past_key_values.channel_mix_state[layer_id],
                past_key_values.v_first,
                attention_mask,
                output_attentions,
            )
            if output_attentions:
                all_attentions += (attention,)
        hidden_states = self.ln_out(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if use_cache:
            valid_tokens = sequence_length if attention_mask is None else int(attention_mask.sum(dim=-1).max().item())
            past_key_values.seen_tokens += valid_tokens
        else:
            past_key_values = None

        if not return_dict:
            return tuple(
                value
                for value in (hidden_states, past_key_values, all_hidden_states, all_attentions)
                if value is not None
            )
        return Rwkv7Output(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


@auto_docstring
class Rwkv7ForCausalLM(Rwkv7PreTrainedModel, GenerationMixin):
    model_class = Rwkv7Model

    def __init__(self, config: Rwkv7Config):
        super().__init__(config)
        self.model = self.model_class(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, value):
        self.head = value

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        model_embeddings = super()._resize_token_embeddings(
            new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            mean_resizing=mean_resizing,
        )
        if self.config.deep_embedding_size > 0:
            target_size = model_embeddings.num_embeddings
            for block in self.model.blocks:
                block.ffn.s_emb = self._get_resized_embeddings(
                    block.ffn.s_emb,
                    target_size,
                    mean_resizing=mean_resizing,
                )
        return model_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1:]
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        model_inputs = dict(kwargs)
        if inputs_embeds is not None and (past_key_values is None or past_key_values.get_seq_length() == 0):
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs["input_ids"] = None
        else:
            model_inputs["input_ids"] = input_ids
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "use_cache": kwargs.get("use_cache", True),
            }
        )
        return model_inputs

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Rwkv7Cache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | Rwkv7CausalLMOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else logits_to_keep
        )
        logits = self.head(hidden_states[:, indices, :]) if logits_to_keep else self.head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return Rwkv7CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Rwkv7Block",
    "Rwkv7Cache",
    "Rwkv7ChannelMix",
    "Rwkv7DeepEmbedding",
    "Rwkv7ForCausalLM",
    "Rwkv7Model",
    "Rwkv7PreTrainedModel",
    "Rwkv7TimeMix",
]
