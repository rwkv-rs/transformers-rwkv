# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2024 BlinkDL and contributors.
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
"""PyTorch RWKV-7 model with unified training + inference support."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

try:
    from ....generation import GenerationMixin
    from ....modeling_layers import GradientCheckpointingLayer
    from ....modeling_utils import PreTrainedModel
    from ....utils import (
        ModelOutput,
        auto_docstring,
        logging,
    )
    logger = logging.get_logger(__name__)
except ImportError:
    # Standalone fallbacks (for testing without full transformers)
    from dataclasses import dataclass as _dc
    from torch import nn as _nn

    class ModelOutput:
        def __post_init__(self): pass
        def __iter__(self): return iter(self.__dict__.values())
        def __getitem__(self, i): return list(self.__dict__.values())[i]

    def auto_docstring(*args, **kwargs):
        if len(args) == 1 and callable(args[0]): return args[0]
        def d(c): return c
        return d

    class _FL:
        def info(self, *a, **kw): pass
        def warning(self, *a, **kw): pass
        def warning_once(self, *a, **kw): pass
        def error(self, *a, **kw): pass
    class _LM:
        get_logger = staticmethod(lambda n: _FL())
    logging = _LM()
    logger = _FL()

    class GradientCheckpointingLayer(_nn.Module):
        gradient_checkpointing: bool = False
        def __call__(self, *args, **kwargs):
            if getattr(self, "gradient_checkpointing", False) and self.training:
                from functools import partial
                from torch.utils.checkpoint import checkpoint
                f = getattr(self, "_gradient_checkpointing_func_impl", checkpoint)
                return f(partial(super().__call__, **kwargs), *args)
            return super().__call__(*args, **kwargs)

    class PreTrainedModel(_nn.Module):
        config_class = None
        _no_split_modules = []
        _keep_in_fp32_modules = []
        supports_gradient_checkpointing = False
        _is_stateful = False

        def __init__(self, config=None):
            super().__init__()
            self.config = config

        def _init_weights(self, module): pass

        def post_init(self):
            self.apply(self._init_weights)

        @staticmethod
        def loss_function(logits, labels, vocab_size, **kw):
            return _nn.functional.cross_entropy(
                logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)

        def _set_gradient_checkpointing(self, module=None, value=False):
            if module is None: module = self
            for m in module.modules():
                if hasattr(m, "gradient_checkpointing"):
                    m.gradient_checkpointing = value

    class GenerationMixin:
        def generate(self, input_ids, max_new_tokens=20, do_sample=False,
                     temperature=1.0, top_p=1.0, pad_token_id=None, eos_token_id=None, **kw):
            gen = input_ids.clone(); s = None
            for _ in range(max_new_tokens):
                o = self(gen[:, -1:] if s is not None else gen, state=s, use_cache=True)
                logits = getattr(o, 'logits', o[0])[:, -1, :]
                nt = torch.argmax(logits, dim=-1, keepdim=True)
                gen = torch.cat([gen, nt], dim=-1)
                s = getattr(o, 'state', o[1]) if len(o) > 1 else None
                if eos_token_id is not None and nt.item() == eos_token_id: break
            return gen

        def prepare_inputs_for_generation(self, input_ids, state=None, **kw):
            if state is not None: input_ids = input_ids[:, -1:]
            return {"input_ids": input_ids, "state": state, "use_cache": True}

from .configuration_rwkv7 import Rwkv7Config
from .wkv7_kernels import (
    compile_all as compile_all_kernels,
    wkv7_training,
    wkv7_prefill,
    wkv7_decode,
    wkv7_varlen as wkv7_varlen_batch,
    kernel_status as get_kernel_status,
)


# =============================================================================================
# RWKV-7 TimeMix (Attention block)
# =============================================================================================

class Rwkv7TimeMix(nn.Module):
    """
    RWKV-7 Time Mixing block.

    Computes:
        r, w, k, v, a, g = token_shift_and_project(x)
        wkv_out = WKV7(r, w, k, v, -kk, kk*a)  # dispatches to training/prefill/decode kernel
        output = output_norm(wkv_out + local_bonus) * g
    """

    def __init__(self, config: Rwkv7Config, layer_id: int = 0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        hidden_size = config.hidden_size
        head_size = config.head_size
        self.head_size = head_size
        self.n_head = hidden_size // head_size
        assert hidden_size % self.n_head == 0

        H, N, C = self.n_head, head_size, hidden_size

        # ── Time-mix parameters (token-shift coefficients) ──
        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        # ── Decay w: w0 + tanh(xw @ w1) @ w2 ──
        D_DECAY_LORA = max(32, int(round((2.5 * (C ** 0.5)) / 32) * 32))
        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        # ── ICL rate a: sigmoid(a0 + (xa @ a1) @ a2) ──
        D_AAA_LORA = max(32, int(round((2.5 * (C ** 0.5)) / 32) * 32))
        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        # ── Value residual v: v + sigmoid(v0 + (xv @ v1) @ v2) ──
        D_MV_LORA = max(32, int(round((1.7 * (C ** 0.5)) / 32) * 32))
        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))

        # ── Output gate g: sigmoid(xg @ g1) @ g2 ──
        D_GATE_LORA = max(32, int(round((5 * (C ** 0.5)) / 32) * 32))
        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

        # ── Key modulation ──
        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))

        # ── Local bonus ──
        self.r_k = nn.Parameter(torch.empty(H, N))

        # ── Token shift ──
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # ── Linear projections ──
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)

        # ── GroupNorm on WKV output ──
        self.ln_x = nn.GroupNorm(H, C, eps=config.group_norm_epsilon)

    def _token_shift(self, x, state_prev=None):
        B, T, C = x.shape
        if T == 1 and state_prev is not None:
            shifted = state_prev.unsqueeze(1)  # (B, C) -> (B, 1, C) to avoid broadcast issues
        else:
            shifted = self.time_shift(x)
            if state_prev is not None and T > 1:
                shifted[:, 0] = state_prev
        return shifted - x, x[:, -1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: Optional[torch.Tensor] = None,
        att_x_prev: Optional[torch.Tensor] = None,
        att_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (B, T, C)
            v_first: (B, T, C) or (B, C) — first-layer value for residual
            att_x_prev: (B, C) — previous x for token shift
            att_state: (B, H, N, N) — WKV state for RNN mode

        Returns:
            output: (B, T, C)
            v_first: updated
            x_last: (B, C) — for next state
            att_state_new: (B, H, N, N) — updated state
        """
        B, T, C = hidden_states.shape
        H, N = self.n_head, self.head_size

        # ── Token shift ──
        xx, x_last = self._token_shift(hidden_states, att_x_prev)

        # ── Apply time-mix coefficients ──
        xr = hidden_states + xx * self.x_r
        xw = hidden_states + xx * self.x_w
        xk = hidden_states + xx * self.x_k
        xv = hidden_states + xx * self.x_v
        xa = hidden_states + xx * self.x_a
        xg = hidden_states + xx * self.x_g

        # ── Projections ──
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)

        # ── Decay: soft-clamp to (-inf, -0.5) ──
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5

        # ── Value residual from first layer ──
        if self.layer_id == 0:
            v_first = v
        elif v_first is not None:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        # ── ICL learning rate ──
        a_icl = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)

        # ── Output gate ──
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        # ── Key normalization and modulation ──
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k_mod = k * (1 + (a_icl - 1) * self.k_a)

        # ── WKV7 operation ──
        # Dispatch based on mode:
        #   training + T>1 + bf16: WindBackstepping (forward+backward)
        #   inference, T>1: GPT prefill kernel
        #   inference, T==1: RNN decode kernel

        if self.training and T > 1:
            # ── Training path: WindBackstepping ──
            wkv_out = wkv7_training(
                q=r, w=w, k=k_mod, v=v,
                a=-kk, b=kk * a_icl,
                head_size=N,
            )
            att_state_new = None  # No state tracking needed during training
        elif T > 1:
            # ── Inference prefill (GPT-mode) ──
            wkv_out = wkv7_prefill(
                r=r, w=w, k=k_mod, v=v,
                a=-kk, b=kk * a_icl,
                head_size=N,
                use_cuda=(self.config.wkv_backend != "pytorch"),
            )
            att_state_new = None
        else:
            # ── Inference decode (RNN-mode) ──
            if att_state is None:
                att_state = torch.zeros(B, H, N, N, dtype=torch.float32, device=hidden_states.device)
            wkv_out, att_state_new = wkv7_decode(
                r=r.squeeze(1), w=w.squeeze(1),
                k=k_mod.squeeze(1), v=v.squeeze(1),
                a=-kk.squeeze(1), b=(kk * a_icl).squeeze(1),
                state=att_state,
                head_size=N,
                use_cuda=(self.config.wkv_backend != "pytorch"),
            )
            wkv_out = wkv_out.unsqueeze(1)  # (B, C) → (B, 1, C)

        # ── GroupNorm on output ──
        wkv_out = self.ln_x(wkv_out.view(B * T, C)).view(B, T, C)

        # ── Local attention bonus ──
        local_bonus = (
            (r.view(B, T, H, N) * k_mod.view(B, T, H, N) * self.r_k)
            .sum(dim=-1, keepdim=True) * v.view(B, T, H, N)
        ).view(B, T, C)
        wkv_out = wkv_out + local_bonus

        # ── Output projection with gate ──
        output = self.output(wkv_out * g)

        return output, v_first, x_last, att_state_new


# =============================================================================================
# RWKV-7 ChannelMix (FFN block)
# =============================================================================================

class Rwkv7ChannelMix(nn.Module):
    """RWKV-7 Channel Mixing block: token_shift → squared ReLU → value projection."""

    def __init__(self, config: Rwkv7Config, layer_id: int = 0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    def _token_shift(self, x, state_prev=None):
        B, T, C = x.shape
        if T == 1 and state_prev is not None:
            shifted = state_prev.unsqueeze(1)  # (B, C) -> (B, 1, C) to avoid broadcast issues
        else:
            shifted = self.time_shift(x)
            if state_prev is not None and T > 1:
                shifted[:, 0] = state_prev
        return shifted - x, x[:, -1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        ffn_x_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        xx, x_last = self._token_shift(hidden_states, ffn_x_prev)
        k = hidden_states + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        output = self.value(k)
        return output, x_last


# =============================================================================================
# RWKV-7 Block
# =============================================================================================

class Rwkv7Block(GradientCheckpointingLayer):
    """
    Single RWKV-7 block: ln → TimeMix → residual → ln → ChannelMix → residual.
    Block 0 additionally has a pre-ln (ln0) for DeepEmbedding support.
    """

    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.att = Rwkv7TimeMix(config, layer_id)
        self.ffn = Rwkv7ChannelMix(config, layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: Optional[torch.Tensor] = None,
        att_x_prev: Optional[torch.Tensor] = None,
        att_state: Optional[torch.Tensor] = None,
        ffn_x_prev: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # Deep embedding: if active, block 0's ln0 is fused into the embedding weights
        # at checkpoint load time, so we skip it here (LayerNorm with identity weights
        # is NOT identity — it still normalizes).
        if self.layer_id == 0 and not self.config.deep_embedding:
            hidden_states = self.ln0(hidden_states)

        # TimeMix
        residual = hidden_states
        attn_out, v_first, att_x_prev_new, att_state_new = self.att(
            self.ln1(hidden_states),
            v_first=v_first,
            att_x_prev=att_x_prev,
            att_state=att_state,
        )
        hidden_states = residual + attn_out

        # ChannelMix
        residual = hidden_states
        ffn_out, ffn_x_prev_new = self.ffn(
            self.ln2(hidden_states),
            ffn_x_prev=ffn_x_prev,
        )
        hidden_states = residual + ffn_out

        outputs = (hidden_states, v_first)
        if use_cache:
            outputs += ((att_x_prev_new, att_state_new, ffn_x_prev_new),)
        else:
            outputs += (None,)
        if output_attentions:
            outputs += (attn_out,)
        else:
            outputs += (None,)

        return outputs


# =============================================================================================
# RWKV-7 Pretrained Model
# =============================================================================================

@auto_docstring
class Rwkv7PreTrainedModel(PreTrainedModel):
    config_class = Rwkv7Config
    base_model_prefix = "rwkv7"
    _no_split_modules = ["Rwkv7Block"]
    _keep_in_fp32_modules = ["w0"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module: nn.Module):
        """Weight initialization matching RWKV-LM training code."""
        super()._init_weights(module)

        if isinstance(module, Rwkv7TimeMix):
            layer_id = module.layer_id
            n_layers = module.config.num_hidden_layers
            C = module.config.hidden_size
            H, N = module.n_head, module.head_size

            r1 = layer_id / (n_layers - 1) if n_layers > 1 else 0  # 0 to 1
            r2 = 1.0 - (layer_id / n_layers) if n_layers > 1 else 0  # 1 to ~0

            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            init_map = {
                "x_r": 1.0 - torch.pow(ddd, 0.2 * r2),
                "x_w": 1.0 - torch.pow(ddd, 0.9 * r2),
                "x_k": 1.0 - torch.pow(ddd, 0.7 * r2),
                "x_v": 1.0 - torch.pow(ddd, 0.7 * r2),
                "x_a": 1.0 - torch.pow(ddd, 0.9 * r2),
                "x_g": 1.0 - torch.pow(ddd, 0.2 * r2),
            }
            for name, val in init_map.items():
                with torch.no_grad():
                    getattr(module, name).copy_(val)

            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            www = torch.zeros(C)
            for n in range(C):
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                linear[n] = n / (C - 1) - 0.5
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + r1 ** 0.3)

            nn.init.zeros_(module.w1); self._ortho_init(module.w2, 0.1)
            with torch.no_grad():
                module.w0.copy_((www.reshape(1, 1, C) + 0.5 + zigzag * 2.5))

            nn.init.zeros_(module.a1); self._ortho_init(module.a2, 0.1)
            with torch.no_grad():
                module.a0.copy_((torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4))

            nn.init.zeros_(module.v1); self._ortho_init(module.v2, 0.1)
            with torch.no_grad():
                module.v0.copy_((torch.zeros(1, 1, C) + 0.73 - linear * 0.4))

            nn.init.zeros_(module.g1); self._ortho_init(module.g2, 0.1)

            with torch.no_grad():
                module.k_k.copy_(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
                module.k_a.copy_(torch.zeros(1, 1, C) + 1.02)
                module.r_k.copy_(torch.zeros(H, N) - 0.04)

            module.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            module.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            module.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            module.output.weight.data.zero_()

        elif isinstance(module, Rwkv7ChannelMix):
            layer_id = module.layer_id
            n_layers = module.config.num_hidden_layers
            C = module.config.hidden_size

            r2 = 1.0 - (layer_id / n_layers) if n_layers > 1 else 0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C
            with torch.no_grad():
                module.x_k.copy_(1.0 - torch.pow(ddd, r2 ** 4))

            module.key.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            module.value.weight.data.zero_()

        elif isinstance(module, nn.Linear):
            shape = module.weight.shape
            gain = 1.0; scale = 1.0
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if shape[0] > shape[1]:
                gain = math.sqrt(shape[0] / shape[1])
            if shape[0] == self.config.vocab_size and shape[1] == self.config.hidden_size:
                scale = 0.5
            nn.init.orthogonal_(module.weight, gain=gain * scale)

        elif isinstance(module, nn.Embedding):
            shape = module.weight.shape
            nn.init.orthogonal_(module.weight, gain=1e-4 * math.sqrt(max(shape[0], shape[1])))

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def _ortho_init(tensor, scale):
        shape = tensor.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
            nn.init.orthogonal_(tensor, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
            for i in range(shape[0]):
                nn.init.orthogonal_(tensor[i], gain=gain * scale)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Rwkv7Model):
            module.gradient_checkpointing = value


# =============================================================================================
# Output dataclasses
# =============================================================================================

@dataclass
class Rwkv7Output(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    state: list | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Rwkv7CausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    state: list | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


# =============================================================================================
# RWKV-7 Model
# =============================================================================================

@auto_docstring
class Rwkv7Model(Rwkv7PreTrainedModel):
    """Bare RWKV-7 model outputting raw hidden states."""

    def __init__(self, config: Rwkv7Config):
        super().__init__(config)

        # ── Compile CUDA kernels if enabled ──
        if config.auto_compile_kernels and torch.cuda.is_available():
            compile_all_kernels(config.head_size, chunk_len=config.wkv_chunk_len)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [Rwkv7Block(config, layer_id=i) for i in range(config.num_hidden_layers)]
        )
        self.ln_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.layers_are_rescaled = False
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def _init_state(self, batch_size, dtype, device):
        """Initialize empty state for all layers."""
        C = self.config.hidden_size
        H = C // self.config.head_size
        N = self.config.head_size
        state = []
        for _ in range(self.config.num_hidden_layers):
            state.append((
                torch.zeros(batch_size, C, dtype=dtype, device=device),
                torch.zeros(batch_size, H, N, N, dtype=torch.float32, device=device),
                torch.zeros(batch_size, C, dtype=dtype, device=device),
            ))
        return state

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Rwkv7Output]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.training == self.layers_are_rescaled:
            self._rescale_layers()

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        batch_size = inputs_embeds.size(0)
        if use_cache and state is None:
            state = self._init_state(batch_size, inputs_embeds.dtype, inputs_embeds.device)

        hidden_states = inputs_embeds
        v_first = None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        new_state = [] if use_cache else None

        for idx, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            att_x_prev = state[idx][0] if (use_cache and state is not None) else None
            att_kv_state = state[idx][1] if (use_cache and state is not None) else None
            ffn_x_prev = state[idx][2] if (use_cache and state is not None) else None

            hidden_states, v_first, layer_state, attentions = block(
                hidden_states,
                v_first=v_first,
                att_x_prev=att_x_prev,
                att_state=att_kv_state,
                ffn_x_prev=ffn_x_prev,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            if use_cache:
                new_state.append(layer_state)

            if self.layers_are_rescaled and self.config.rescale_every > 0 and (idx + 1) % self.config.rescale_every == 0:
                hidden_states = hidden_states / 2

            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)

        hidden_states = self.ln_out(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(x for x in [hidden_states, new_state, all_hidden_states, all_self_attentions] if x is not None)

        return Rwkv7Output(
            last_hidden_state=hidden_states,
            state=new_state,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _rescale_layers(self):
        if self.layers_are_rescaled == (not self.training):
            return
        if self.config.rescale_every > 0:
            with torch.no_grad():
                for block_id, block in enumerate(self.blocks):
                    if self.training:
                        block.att.output.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                        block.ffn.value.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                    else:
                        block.att.output.weight.div_(2 ** int(block_id // self.config.rescale_every))
                        block.ffn.value.weight.div_(2 ** int(block_id // self.config.rescale_every))
        self.layers_are_rescaled = not self.training


# =============================================================================================
# RWKV-7 For Causal LM
# =============================================================================================

@auto_docstring(custom_intro="RWKV-7 with language modeling head.")
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
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, Rwkv7CausalLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        rwkv7_outputs = self.rwkv7(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = rwkv7_outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + (rwkv7_outputs[1:])
            return ((loss,) + output) if loss is not None else output

        return Rwkv7CausalLMOutput(
            loss=loss, logits=logits,
            state=rwkv7_outputs.state,
            hidden_states=rwkv7_outputs.hidden_states,
            attentions=rwkv7_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, state=None, inputs_embeds=None, **kwargs):
        if state is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "state": state, "inputs_embeds": inputs_embeds, "use_cache": True}


__all__ = [
    "Rwkv7PreTrainedModel", "Rwkv7Model", "Rwkv7ForCausalLM",
    "Rwkv7TimeMix", "Rwkv7ChannelMix", "Rwkv7Block",
    "Rwkv7Output", "Rwkv7CausalLMOutput",
]
