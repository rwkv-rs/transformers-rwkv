# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2024-2026 Bo Peng, BlinkDL and contributors.
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

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple, logging
from .configuration_rwkv7 import Rwkv7Config
from .kernel_backends import run_rwkv7_wkv


logger = logging.get_logger(__name__)


def _token_shift(
    hidden_states: torch.Tensor,
    previous_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cu_seq_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the previous valid token at each position and the final valid token."""

    batch_size, sequence_length, _ = hidden_states.shape
    if cu_seq_lens is not None:
        shifted = torch.cat((previous_hidden_state.unsqueeze(1), hidden_states[:, :-1]), dim=1)
        positions = torch.arange(sequence_length, device=hidden_states.device)
        segment_starts = (positions[:, None] == cu_seq_lens[:-1]).any(dim=1)
        shifted = torch.where(segment_starts.view(1, sequence_length, 1), torch.zeros_like(shifted), shifted)
        return shifted - hidden_states, hidden_states[:, -1]
    if attention_mask is None:
        shifted = torch.cat((previous_hidden_state.unsqueeze(1), hidden_states[:, :-1]), dim=1)
        return shifted - hidden_states, hidden_states[:, -1]

    previous = previous_hidden_state
    shifted_tokens = []
    for token_index in range(sequence_length):
        shifted_tokens.append(previous)
        token_mask = attention_mask[:, token_index].to(torch.bool).view(batch_size, 1)
        previous = torch.where(token_mask, hidden_states[:, token_index], previous)
    return torch.stack(shifted_tokens, dim=1) - hidden_states, previous


class Rwkv7TimeMix(nn.Module):
    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.head_size = config.head_size
        self.num_heads = config.num_attention_heads

        hidden_size = config.hidden_size
        decay_rank = max(32, int(round(2.5 * math.sqrt(hidden_size) / 32) * 32))
        in_context_rank = max(32, int(round(2.5 * math.sqrt(hidden_size) / 32) * 32))
        value_rank = max(32, int(round(1.7 * math.sqrt(hidden_size) / 32) * 32))
        gate_rank = max(32, int(round(5.0 * math.sqrt(hidden_size) / 32) * 32))

        self.x_r = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.w0 = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.w1 = nn.Parameter(torch.empty(hidden_size, decay_rank))
        self.w2 = nn.Parameter(torch.empty(decay_rank, hidden_size))
        self.a0 = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.a1 = nn.Parameter(torch.empty(hidden_size, in_context_rank))
        self.a2 = nn.Parameter(torch.empty(in_context_rank, hidden_size))
        if layer_id > 0:
            self.v0 = nn.Parameter(torch.empty(1, 1, hidden_size))
            self.v1 = nn.Parameter(torch.empty(hidden_size, value_rank))
            self.v2 = nn.Parameter(torch.empty(value_rank, hidden_size))
        self.g1 = nn.Parameter(torch.empty(hidden_size, gate_rank))
        self.g2 = nn.Parameter(torch.empty(gate_rank, hidden_size))

        self.k_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.k_a = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.r_k = nn.Parameter(torch.empty(self.num_heads, self.head_size))

        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_x = nn.GroupNorm(self.num_heads, hidden_size, eps=config.group_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        wkv_state: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cu_seq_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        shifted, final_hidden_state = _token_shift(hidden_states, previous_hidden_state, attention_mask, cu_seq_lens)

        receptance_input = hidden_states + shifted * self.x_r
        decay_input = hidden_states + shifted * self.x_w
        key_input = hidden_states + shifted * self.x_k
        value_input = hidden_states + shifted * self.x_v
        in_context_input = hidden_states + shifted * self.x_a
        gate_input = hidden_states + shifted * self.x_g

        receptance = self.receptance(receptance_input)
        key = self.key(key_input)
        value = self.value(value_input)
        raw_decay = self.w0 + torch.tanh(decay_input @ self.w1) @ self.w2

        if self.layer_id == 0:
            v_first = value
        else:
            value_residual = torch.sigmoid(self.v0 + (value_input @ self.v1) @ self.v2)
            value = value + (v_first - value) * value_residual

        in_context_learning_rate = torch.sigmoid(self.a0 + (in_context_input @ self.a1) @ self.a2)
        gate = torch.sigmoid(gate_input @ self.g1) @ self.g2

        normalized_key = key * self.k_k
        normalized_key = F.normalize(
            normalized_key.view(batch_size, sequence_length, self.num_heads, self.head_size), dim=-1, p=2.0
        ).view(batch_size, sequence_length, hidden_size)
        key = key * (1 + (in_context_learning_rate - 1) * self.k_a)

        wkv_inputs = (
            receptance,
            raw_decay,
            key,
            value,
            -normalized_key,
            normalized_key * in_context_learning_rate,
        )
        wkv_output, wkv_state = run_rwkv7_wkv(
            self.config.wkv_backend,
            self.training,
            *wkv_inputs,
            state=wkv_state,
            attention_mask=attention_mask,
            cu_seq_lens=cu_seq_lens,
            head_size=self.head_size,
        )
        wkv_output = self.ln_x(wkv_output.reshape(batch_size * sequence_length, hidden_size)).view(
            batch_size, sequence_length, hidden_size
        )

        local_output = (
            (
                receptance.view(batch_size, sequence_length, self.num_heads, self.head_size)
                * key.view(batch_size, sequence_length, self.num_heads, self.head_size)
                * self.r_k
            ).sum(dim=-1, keepdim=True)
            * value.view(batch_size, sequence_length, self.num_heads, self.head_size)
        ).view(batch_size, sequence_length, hidden_size)
        output = self.output((wkv_output + local_output) * gate)
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)
        return output, v_first, final_hidden_state, wkv_state


class Rwkv7ChannelMix(nn.Module):
    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.x_k = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cu_seq_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shifted, final_hidden_state = _token_shift(hidden_states, previous_hidden_state, attention_mask, cu_seq_lens)
        key = hidden_states + shifted * self.x_k
        output = self.value(torch.relu(self.key(key)).square())
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)
        return output, final_hidden_state


class Rwkv7Block(GradientCheckpointingLayer):
    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        if layer_id == 0 and not config.embedding_layer_norm_fused:
            self.ln0 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.att = Rwkv7TimeMix(config, layer_id)
        self.ffn = Rwkv7ChannelMix(config, layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor,
        attention_previous_hidden_state: torch.Tensor,
        wkv_state: torch.Tensor,
        ffn_previous_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cu_seq_lens: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.layer_id == 0 and hasattr(self, "ln0"):
            hidden_states = self.ln0(hidden_states)

        attention_output, v_first, attention_previous_hidden_state, wkv_state = self.att(
            self.ln1(hidden_states),
            v_first,
            attention_previous_hidden_state,
            wkv_state,
            attention_mask,
            cu_seq_lens,
        )
        hidden_states = hidden_states + attention_output
        ffn_output, ffn_previous_hidden_state = self.ffn(
            self.ln2(hidden_states), ffn_previous_hidden_state, attention_mask, cu_seq_lens
        )
        hidden_states = hidden_states + ffn_output
        attention = attention_output if output_attentions else None
        return (
            hidden_states,
            v_first,
            attention_previous_hidden_state,
            wkv_state,
            ffn_previous_hidden_state,
            attention,
        )


@auto_docstring
class Rwkv7PreTrainedModel(PreTrainedModel):
    config_class = Rwkv7Config
    base_model_prefix = "rwkv7"
    _no_split_modules = ["Rwkv7Block"]
    _keep_in_fp32_modules = ["w0"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)

        if isinstance(module, Rwkv7TimeMix):
            layer_id = module.layer_id
            num_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            head_size = module.head_size
            layer_ratio = layer_id / (num_layers - 1) if num_layers > 1 else 0.0
            depth_ratio = 1.0 - layer_id / num_layers
            channel_ratio = torch.arange(hidden_size, dtype=module.x_r.dtype, device=module.x_r.device) / hidden_size
            channel_ratio = channel_ratio.view(1, 1, hidden_size)

            init.copy_(module.x_r, 1.0 - channel_ratio.pow(0.2 * depth_ratio))
            init.copy_(module.x_w, 1.0 - channel_ratio.pow(0.9 * depth_ratio))
            init.copy_(module.x_k, 1.0 - channel_ratio.pow(0.7 * depth_ratio))
            init.copy_(module.x_v, 1.0 - channel_ratio.pow(0.7 * depth_ratio))
            init.copy_(module.x_a, 1.0 - channel_ratio.pow(0.9 * depth_ratio))
            init.copy_(module.x_g, 1.0 - channel_ratio.pow(0.2 * depth_ratio))

            channels = torch.arange(hidden_size, dtype=module.w0.dtype, device=module.w0.device)
            head_positions = (channels.remainder(head_size) - (head_size - 1) / 2) / max((head_size - 1) / 2, 1)
            zigzag = head_positions * head_positions.abs()
            linear = channels / max(hidden_size - 1, 1) - 0.5
            decay = -6 + 6 * (channels / max(hidden_size - 1, 1)).pow(1 + layer_ratio**0.3)

            init.zeros_(module.w1)
            self._orthogonal_init(module.w2, 0.1)
            init.copy_(module.w0, (decay + 0.5 + zigzag * 2.5).view(1, 1, hidden_size))
            init.zeros_(module.a1)
            self._orthogonal_init(module.a2, 0.1)
            init.copy_(module.a0, (-0.19 + zigzag * 0.3 + linear * 0.4).view(1, 1, hidden_size))
            if layer_id > 0:
                init.zeros_(module.v1)
                self._orthogonal_init(module.v2, 0.1)
                init.copy_(module.v0, (0.73 - linear * 0.4).view(1, 1, hidden_size))
            init.zeros_(module.g1)
            self._orthogonal_init(module.g2, 0.1)
            init.copy_(module.k_k, (0.71 - linear * 0.1).view(1, 1, hidden_size))
            init.constant_(module.k_a, 1.02)
            init.constant_(module.r_k, -0.04)

            init.uniform_(module.receptance.weight, -0.5 / math.sqrt(hidden_size), 0.5 / math.sqrt(hidden_size))
            init.uniform_(module.key.weight, -0.05 / math.sqrt(hidden_size), 0.05 / math.sqrt(hidden_size))
            init.uniform_(module.value.weight, -0.5 / math.sqrt(hidden_size), 0.5 / math.sqrt(hidden_size))
            init.zeros_(module.output.weight)

        elif isinstance(module, Rwkv7ChannelMix):
            hidden_size = module.config.hidden_size
            depth_ratio = 1.0 - module.layer_id / module.config.num_hidden_layers
            channel_ratio = (
                torch.arange(hidden_size, dtype=module.x_k.dtype, device=module.x_k.device).view(1, 1, hidden_size)
                / hidden_size
            )
            init.copy_(module.x_k, 1.0 - channel_ratio.pow(depth_ratio**4))
            init.uniform_(module.key.weight, -0.5 / math.sqrt(hidden_size), 0.5 / math.sqrt(hidden_size))
            init.zeros_(module.value.weight)

        elif isinstance(module, nn.Linear):
            if module.bias is not None:
                init.zeros_(module.bias)
            gain = (
                math.sqrt(module.weight.shape[0] / module.weight.shape[1])
                if module.weight.shape[0] > module.weight.shape[1]
                else 1.0
            )
            if module.weight.shape == (self.config.vocab_size, self.config.hidden_size):
                gain *= 0.5
            init.orthogonal_(module.weight, gain=gain)
        elif isinstance(module, nn.Embedding):
            gain = 1e-4 * math.sqrt(max(module.weight.shape))
            init.orthogonal_(module.weight, gain=gain)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            init.ones_(module.weight)
            init.zeros_(module.bias)

    @staticmethod
    def _orthogonal_init(tensor: torch.Tensor, scale: float):
        gain = math.sqrt(tensor.shape[0] / tensor.shape[1]) if tensor.shape[0] > tensor.shape[1] else 1.0
        init.orthogonal_(tensor, gain=gain * scale)


@auto_docstring(custom_intro="""Class for RWKV-7 model outputs.""")
@dataclass
class Rwkv7Output(ModelOutput):
    r"""
    state (`list[torch.FloatTensor]`, *optional*):
        Recurrent state containing attention shifts of shape `(layers, batch, hidden)`, WKV matrices of shape
        `(layers, batch, heads, head_size, head_size)`, and FFN shifts of shape `(layers, batch, hidden)`.
    """

    last_hidden_state: torch.FloatTensor | None = None
    state: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(custom_intro="""Base class for RWKV-7 causal language model outputs.""")
@dataclass
class Rwkv7CausalLMOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss for next-token prediction.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head before softmax.
    state (`list[torch.FloatTensor]`, *optional*):
        Recurrent state containing attention shifts, WKV matrices, and FFN shifts. It can be passed to the next forward
        call to continue the sequence without recomputing earlier tokens.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    state: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring
class Rwkv7Model(Rwkv7PreTrainedModel):
    def __init__(self, config: Rwkv7Config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Rwkv7Block(config, layer_id) for layer_id in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.layers_are_rescaled = False
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def _init_state(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
        return [
            torch.zeros(
                self.config.num_hidden_layers, batch_size, self.config.hidden_size, dtype=dtype, device=device
            ),
            torch.zeros(
                self.config.num_hidden_layers,
                batch_size,
                self.config.num_attention_heads,
                self.config.head_size,
                self.config.head_size,
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                self.config.num_hidden_layers, batch_size, self.config.hidden_size, dtype=dtype, device=device
            ),
        ]

    def _validate_state(
        self,
        state: list[torch.Tensor] | tuple[torch.Tensor, ...],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        if not isinstance(state, (list, tuple)) or len(state) != 3:
            raise ValueError("RWKV-7 state must contain attention shifts, WKV matrices, and FFN shifts.")
        expected_shapes = (
            (self.config.num_hidden_layers, batch_size, self.config.hidden_size),
            (
                self.config.num_hidden_layers,
                batch_size,
                self.config.num_attention_heads,
                self.config.head_size,
                self.config.head_size,
            ),
            (self.config.num_hidden_layers, batch_size, self.config.hidden_size),
        )
        expected_dtypes = (dtype, torch.float32, dtype)
        for index, (tensor, expected_shape, expected_dtype) in enumerate(zip(state, expected_shapes, expected_dtypes)):
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                actual_shape = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
                raise ValueError(f"RWKV-7 state tensor {index} must have shape {expected_shape}, got {actual_shape}.")
            if tensor.device != device:
                raise ValueError(f"RWKV-7 state tensor {index} must be on device {device}, got {tensor.device}.")
            if tensor.dtype != expected_dtype:
                raise ValueError(f"RWKV-7 state tensor {index} must have dtype {expected_dtype}, got {tensor.dtype}.")
        return list(state)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        cu_seq_lens: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        state: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | Rwkv7Output:
        r"""
        state (`list[torch.FloatTensor]`, *optional*):
            State returned by an earlier forward pass. It is always used as initial context when provided; `use_cache`
            only controls whether the updated state is returned.
        cu_seq_lens (`torch.LongTensor`, *optional*):
            Cumulative boundaries for independent sequences packed into a single batch row. Packed sequences always
            start from empty recurrent and token-shift states.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both `input_ids` and `inputs_embeds`.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either `input_ids` or `inputs_embeds`.")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[1] == 0:
            raise ValueError("RWKV-7 inputs must have shape (batch, non-empty sequence, hidden).")
        if inputs_embeds.shape[1] > self.config.context_length:
            raise ValueError(
                f"Input sequence length {inputs_embeds.shape[1]} exceeds `context_length` {self.config.context_length}."
            )
        batch_size, sequence_length = inputs_embeds.shape[:2]
        if cu_seq_lens is not None:
            if attention_mask is not None:
                raise ValueError("`cu_seq_lens` and `attention_mask` are mutually exclusive.")
            if state is not None:
                raise ValueError("Packed sequences cannot continue from an existing recurrent state.")
            if batch_size != 1:
                raise ValueError("Packed RWKV-7 inputs must use a single batch row.")
            if cu_seq_lens.device != inputs_embeds.device:
                raise ValueError("`cu_seq_lens` must be on the same device as the model inputs.")
            if cu_seq_lens.dtype not in (torch.int32, torch.int64):
                raise ValueError("`cu_seq_lens` must use an integer dtype.")
            if cu_seq_lens.ndim != 1 or cu_seq_lens.numel() < 2:
                raise ValueError("`cu_seq_lens` must be one-dimensional and contain at least two boundaries.")
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    f"`attention_mask` must have shape {(batch_size, sequence_length)}, got {tuple(attention_mask.shape)}."
                )
            attention_mask = attention_mask.to(device=inputs_embeds.device)
            if torch.all(attention_mask != 0):
                attention_mask = None
            else:
                attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)

        if self.training == self.layers_are_rescaled:
            self._rescale_layers()
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting it to `False`.")
            use_cache = False

        if state is None:
            state = self._init_state(batch_size, inputs_embeds.dtype, inputs_embeds.device)
        else:
            state = self._validate_state(state, batch_size, inputs_embeds.dtype, inputs_embeds.device)

        hidden_states = inputs_embeds
        v_first = hidden_states
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        attention_shift_states = []
        wkv_states = []
        ffn_shift_states = []

        for layer_id, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            (
                hidden_states,
                v_first,
                attention_shift_state,
                wkv_state,
                ffn_shift_state,
                attention,
            ) = block(
                hidden_states,
                v_first,
                state[0][layer_id],
                state[1][layer_id],
                state[2][layer_id],
                attention_mask,
                cu_seq_lens,
                output_attentions,
            )
            attention_shift_states.append(attention_shift_state)
            wkv_states.append(wkv_state)
            ffn_shift_states.append(ffn_shift_state)
            if (
                self.layers_are_rescaled
                and self.config.rescale_every > 0
                and (layer_id + 1) % self.config.rescale_every == 0
            ):
                hidden_states = hidden_states / 2
            if output_attentions:
                all_attentions += (attention,)

        hidden_states = self.ln_out(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        new_state = None
        if use_cache:
            if any(wkv_state is None for wkv_state in wkv_states):
                raise ValueError("The selected RWKV-7 training backend does not return a recurrent state.")
            new_state = [
                torch.stack(attention_shift_states),
                torch.stack(wkv_states),
                torch.stack(ffn_shift_states),
            ]

        return Rwkv7Output(
            last_hidden_state=hidden_states,
            state=new_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def _rescale_layers(self):
        if self.layers_are_rescaled == (not self.training):
            return
        if self.config.rescale_every > 0:
            with torch.no_grad():
                for layer_id, block in enumerate(self.blocks):
                    factor = 2 ** int(layer_id // self.config.rescale_every)
                    if self.training:
                        block.att.output.weight.mul_(factor)
                        block.ffn.value.weight.mul_(factor)
                    else:
                        block.att.output.weight.div_(factor)
                        block.ffn.value.weight.div_(factor)
        self.layers_are_rescaled = not self.training


@auto_docstring
class Rwkv7ForCausalLM(Rwkv7PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"head.weight": "rwkv7.embeddings.weight"}

    def __init__(self, config: Rwkv7Config):
        super().__init__(config)
        self.rwkv7 = Rwkv7Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def get_input_embeddings(self):
        return self.rwkv7.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.rwkv7.embeddings = new_embeddings

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        cu_seq_lens: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        state: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | Rwkv7CausalLMOutput:
        r"""
        state (`list[torch.FloatTensor]`, *optional*):
            State returned by an earlier forward pass. It is used as the initial recurrent context for `input_ids`.
        cu_seq_lens (`torch.LongTensor`, *optional*):
            Cumulative boundaries for independent sequences packed into one batch row.
        """
        outputs = self.rwkv7(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cu_seq_lens=cu_seq_lens,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.head(outputs.last_hidden_state[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return Rwkv7CausalLMOutput(
            loss=loss,
            logits=logits,
            state=outputs.state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        state: list[torch.Tensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        **kwargs,
    ) -> dict:
        model_inputs = dict(kwargs)
        use_cache = model_inputs.get("use_cache", True)
        if state is not None:
            model_inputs["input_ids"] = input_ids[:, -1:]
            attention_mask = None
        elif inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "state": state,
                "use_cache": use_cache,
            }
        )
        return model_inputs

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict]:
        state = model_kwargs.pop("state", None)
        input_ids, model_kwargs = GenerationMixin._expand_inputs_for_generation(
            expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids,
            **model_kwargs,
        )
        if state is not None:
            model_kwargs["state"] = [tensor.repeat_interleave(expand_size, dim=1) for tensor in state]
        return input_ids, model_kwargs

    @staticmethod
    def _reorder_cache(state: list[torch.Tensor], beam_idx: torch.LongTensor) -> list[torch.Tensor]:
        return [tensor.index_select(1, beam_idx.to(tensor.device)) for tensor in state]


__all__ = [
    "Rwkv7ForCausalLM",
    "Rwkv7Model",
    "Rwkv7PreTrainedModel",
    "Rwkv7CausalLMOutput",
    "Rwkv7Output",
]
