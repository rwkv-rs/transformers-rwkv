# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ...generation import GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from .configuration_rwkv7 import Rwkv7Config


def _token_shift(
    hidden_states: torch.Tensor,
    previous_hidden_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shifted = torch.cat((previous_hidden_state[:, None], hidden_states[:, :-1]), dim=1)
    return shifted - hidden_states, hidden_states[:, -1]


def rwkv7_reference(
    receptance: torch.Tensor,
    raw_decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable token-by-token RWKV-7 recurrence."""
    batch_size, sequence_length, hidden_size = receptance.shape
    num_heads = hidden_size // head_size
    output_dtype = value.dtype
    tensors = [
        tensor.view(batch_size, sequence_length, num_heads, head_size).float()
        for tensor in (receptance, raw_decay, key, value, a, b)
    ]
    receptance, raw_decay, key, value, a, b = tensors
    log_decay = -F.softplus(-raw_decay) - 0.5
    outputs = []
    current_state = state.float()
    for token_index in range(sequence_length):
        decay = log_decay[:, token_index].exp().unsqueeze(-1)
        state_projection = torch.einsum("bhk,bhkv->bhv", a[:, token_index], current_state)
        current_state = (
            decay * current_state
            + b[:, token_index].unsqueeze(-1) * state_projection.unsqueeze(-2)
            + key[:, token_index].unsqueeze(-1) * value[:, token_index].unsqueeze(-2)
        )
        outputs.append(torch.einsum("bhk,bhkv->bhv", receptance[:, token_index], current_state))
    output = torch.stack(outputs, dim=1).reshape(batch_size, sequence_length, hidden_size).to(output_dtype)
    return output, current_state


class Rwkv7TimeMix(nn.Module):
    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        decay_rank = max(32, round(2.5 * math.sqrt(hidden_size) / 32) * 32)
        value_rank = max(32, round(1.7 * math.sqrt(hidden_size) / 32) * 32)
        gate_rank = max(32, round(5.0 * math.sqrt(hidden_size) / 32) * 32)

        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, name, nn.Parameter(torch.empty(1, 1, hidden_size)))
        self.w0 = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.w1 = nn.Parameter(torch.empty(hidden_size, decay_rank))
        self.w2 = nn.Parameter(torch.empty(decay_rank, hidden_size))
        self.a0 = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.a1 = nn.Parameter(torch.empty(hidden_size, decay_rank))
        self.a2 = nn.Parameter(torch.empty(decay_rank, hidden_size))
        if layer_id > 0:
            self.v0 = nn.Parameter(torch.empty(1, 1, hidden_size))
            self.v1 = nn.Parameter(torch.empty(hidden_size, value_rank))
            self.v2 = nn.Parameter(torch.empty(value_rank, hidden_size))
        self.g1 = nn.Parameter(torch.empty(hidden_size, gate_rank))
        self.g2 = nn.Parameter(torch.empty(gate_rank, hidden_size))
        self.k_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.k_a = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.r_k = nn.Parameter(torch.empty(config.num_attention_heads, config.head_size))
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln_x = nn.GroupNorm(
            config.num_attention_heads,
            hidden_size,
            eps=config.group_norm_epsilon,
        )

    def forward(self, hidden_states, v_first, previous_hidden_state, wkv_state):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        shifted, final_hidden_state = _token_shift(hidden_states, previous_hidden_state)
        inputs = {
            name: hidden_states + shifted * getattr(self, f"x_{name}") for name in ("r", "w", "k", "v", "a", "g")
        }
        receptance = self.receptance(inputs["r"])
        key = self.key(inputs["k"])
        value = self.value(inputs["v"])
        raw_decay = self.w0 + torch.tanh(inputs["w"] @ self.w1) @ self.w2
        if self.layer_id == 0:
            v_first = value
        else:
            value = value + (v_first - value) * torch.sigmoid(self.v0 + (inputs["v"] @ self.v1) @ self.v2)
        learning_rate = torch.sigmoid(self.a0 + (inputs["a"] @ self.a1) @ self.a2)
        gate = torch.sigmoid(inputs["g"] @ self.g1) @ self.g2
        normalized_key = F.normalize(
            (key * self.k_k).view(
                batch_size,
                sequence_length,
                self.config.num_attention_heads,
                self.config.head_size,
            ),
            dim=-1,
        ).view(batch_size, sequence_length, hidden_size)
        key = key * (1 + (learning_rate - 1) * self.k_a)
        output, wkv_state = rwkv7_reference(
            receptance,
            raw_decay,
            key,
            value,
            -normalized_key,
            normalized_key * learning_rate,
            wkv_state,
            self.config.head_size,
        )
        output = self.ln_x(output.flatten(0, 1)).view_as(output)
        local = (
            (
                receptance.view(
                    batch_size,
                    sequence_length,
                    self.config.num_attention_heads,
                    self.config.head_size,
                )
                * key.view(
                    batch_size,
                    sequence_length,
                    self.config.num_attention_heads,
                    self.config.head_size,
                )
                * self.r_k
            ).sum(-1, keepdim=True)
            * value.view(
                batch_size,
                sequence_length,
                self.config.num_attention_heads,
                self.config.head_size,
            )
        ).view_as(output)
        return self.output((output + local) * gate), v_first, final_hidden_state, wkv_state


class Rwkv7ChannelMix(nn.Module):
    def __init__(self, config: Rwkv7Config):
        super().__init__()
        self.x_k = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, previous_hidden_state):
        shifted, final_hidden_state = _token_shift(hidden_states, previous_hidden_state)
        output = self.value(F.relu(self.key(hidden_states + shifted * self.x_k)).square())
        return output, final_hidden_state


class Rwkv7Block(nn.Module):
    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.ln0 = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            if layer_id == 0 and not config.embedding_layer_norm_fused
            else nn.Identity()
        )
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.att = Rwkv7TimeMix(config, layer_id)
        self.ffn = Rwkv7ChannelMix(config)

    def forward(self, hidden_states, v_first, att_shift, wkv_state, ffn_shift):
        hidden_states = self.ln0(hidden_states)
        output, v_first, att_shift, wkv_state = self.att(self.ln1(hidden_states), v_first, att_shift, wkv_state)
        hidden_states = hidden_states + output
        output, ffn_shift = self.ffn(self.ln2(hidden_states), ffn_shift)
        return hidden_states + output, v_first, att_shift, wkv_state, ffn_shift


@dataclass
class Rwkv7Output(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    state: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Rwkv7CausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    state: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring
class Rwkv7PreTrainedModel(PreTrainedModel):
    config_class = Rwkv7Config
    base_model_prefix = "model"
    _no_split_modules = ["Rwkv7Block"]
    _is_stateful = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, Rwkv7TimeMix):
            for parameter in module.parameters(recurse=False):
                nn.init.zeros_(parameter)
        elif isinstance(module, Rwkv7ChannelMix):
            nn.init.zeros_(module.x_k)


@auto_docstring
class Rwkv7Model(Rwkv7PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Rwkv7Block(config, index) for index in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def _init_state(self, batch_size, dtype, device):
        layers = self.config.num_hidden_layers
        hidden = self.config.hidden_size
        return (
            torch.zeros(layers, batch_size, hidden, dtype=dtype, device=device),
            torch.zeros(
                layers,
                batch_size,
                self.config.num_attention_heads,
                self.config.head_size,
                self.config.head_size,
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(layers, batch_size, hidden, dtype=dtype, device=device),
        )

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        state=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise ValueError(
                "Rwkv7Model does not yet support padding in attention_mask; pass an all-ones mask or unpadded input."
            )
        hidden_states = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        if state is None:
            state = self._init_state(hidden_states.shape[0], hidden_states.dtype, hidden_states.device)
        elif len(state) != 3:
            raise ValueError("RWKV-7 state must contain attention shift, WKV, and FFN shift tensors.")
        next_att, next_wkv, next_ffn = [], [], []
        v_first = torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)
        for index, block in enumerate(self.blocks):
            hidden_states, v_first, att_shift, wkv_state, ffn_shift = block(
                hidden_states, v_first, state[0][index], state[1][index], state[2][index]
            )
            next_att.append(att_shift)
            next_wkv.append(wkv_state)
            next_ffn.append(ffn_shift)
        hidden_states = self.ln_out(hidden_states)
        use_cache = self.config.use_cache and not self.training if use_cache is None else use_cache
        next_state = None
        if use_cache:
            next_state = (
                torch.stack(next_att),
                torch.stack(next_wkv),
                torch.stack(next_ffn),
            )
        if return_dict is False:
            return hidden_states, next_state
        return Rwkv7Output(last_hidden_state=hidden_states, state=next_state)


@auto_docstring
class Rwkv7ForCausalLM(Rwkv7PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"head.weight": "model.embeddings.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Rwkv7Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, value):
        self.head = value

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, state=None, use_cache=None, **kwargs):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if state is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "state": state,
            "use_cache": use_cache,
        }

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder=False,
        num_new_tokens=1,
    ):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        model_kwargs["state"] = outputs.state
        return model_kwargs

    def forward(self, input_ids=None, labels=None, state=None, return_dict=None, **kwargs):
        outputs = self.model(input_ids=input_ids, state=state, return_dict=True, **kwargs)
        logits = self.head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        if return_dict is False:
            return tuple(value for value in (loss, logits, outputs.state) if value is not None)
        return Rwkv7CausalLMOutput(loss=loss, logits=logits, state=outputs.state)


__all__ = [
    "Rwkv7CausalLMOutput",
    "Rwkv7ForCausalLM",
    "Rwkv7Model",
    "Rwkv7Output",
    "Rwkv7PreTrainedModel",
    "rwkv7_reference",
]
