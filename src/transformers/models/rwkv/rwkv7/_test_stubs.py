"""
Minimal stubs for running RWKV-7 tests without full transformers dependency chain.

This file provides drop-in replacements for the HF classes imported by modeling_rwkv7.py
so that tests can run in environments where transformers can't be fully imported.
"""

import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List, Callable


# =============================================================================
# PretrainedConfig stub
# =============================================================================

class PretrainedConfig:
    """Minimal PretrainedConfig stub."""
    model_type: str = ""
    is_encoder_decoder: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False
    return_dict: bool = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        import json, os
        config_file = os.path.join(path, "config.json")
        with open(config_file, 'r') as f:
            cfg_dict = json.load(f)
        return cls(**cfg_dict)

    def save_pretrained(self, path):
        import json, os
        os.makedirs(path, exist_ok=True)
        cfg_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(cfg_dict, f, indent=2)


# =============================================================================
# PreTrainedModel stub
# =============================================================================

class PreTrainedModel(nn.Module):
    """Minimal PreTrainedModel stub."""
    config_class = None
    base_model_prefix = "model"
    _no_split_modules = []
    _keep_in_fp32_modules = []
    supports_gradient_checkpointing = False
    _is_stateful = False
    _tied_weights_keys = {}

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = None

    def _init_weights(self, module):
        pass

    def post_init(self):
        self.apply(self._init_weights)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        raise NotImplementedError("Use convert script to create HF model, then load normally")

    def save_pretrained(self, path):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        self.config.save_pretrained(path)

    @staticmethod
    def loss_function(logits, labels, vocab_size, **kwargs):
        return nn.functional.cross_entropy(
            logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        from torch.utils.checkpoint import checkpoint
        self._set_gradient_checkpointing(value=True)
        func = gradient_checkpointing_kwargs.get("gradient_checkpointing_func", checkpoint) if gradient_checkpointing_kwargs else checkpoint
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = func
                module.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._set_gradient_checkpointing(value=False)

    def _set_gradient_checkpointing(self, module=None, value=False):
        if module is None:
            module = self
        for m in module.modules():
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = value


# =============================================================================
# GradientCheckpointingLayer stub
# =============================================================================

class GradientCheckpointingLayer(nn.Module):
    """Minimal GradientCheckpointingLayer stub."""
    gradient_checkpointing: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__()
        # _gradient_checkpointing_func is set by the parent model

    @property
    def _gradient_checkpointing_func(self) -> Callable:
        return getattr(self, "_gradient_checkpointing_func_impl", None)

    @_gradient_checkpointing_func.setter
    def _gradient_checkpointing_func(self, func):
        self._gradient_checkpointing_func_impl = func

    def __call__(self, *args, **kwargs):
        if getattr(self, "gradient_checkpointing", False) and self.training:
            from functools import partial
            from torch.utils.checkpoint import checkpoint
            func = getattr(self, "_gradient_checkpointing_func_impl", checkpoint)
            return func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)


# =============================================================================
# GenerationMixin stub
# =============================================================================

class GenerationMixin:
    """Minimal GenerationMixin stub (supports greedy + sampling)."""

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 20,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """Simple autoregressive generation."""
        generated = input_ids.clone()
        state = None

        for _ in range(max_new_tokens):
            outputs = self(
                generated[:, -1:] if state is not None else generated,
                state=state,
                use_cache=True,
            )
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            next_logits = logits[:, -1, :]

            if do_sample:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                if top_k > 0:
                    topk_vals, topk_idx = torch.topk(probs, min(top_k, probs.shape[-1]))
                    probs = torch.zeros_like(probs).scatter_(-1, topk_idx, topk_vals)
                    probs = probs / probs.sum(-1, keepdim=True)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum > top_p
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    sorted_probs[mask] = 0
                    probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
                    probs = probs / probs.sum(-1, keepdim=True)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            state = outputs.state if hasattr(outputs, 'state') else outputs[1]

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated

    def prepare_inputs_for_generation(self, input_ids, state=None, **kwargs):
        if state is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "state": state, "use_cache": True}


# =============================================================================
# Utilities
# =============================================================================

@dataclass
class ModelOutput:
    """Minimal ModelOutput stub."""
    def __post_init__(self):
        pass

    def __iter__(self):
        return iter(self.__dict__.values())

    def __getitem__(self, idx):
        return list(self.__dict__.values())[idx]

    def __len__(self):
        return len(self.__dict__)

    def to_tuple(self):
        return tuple(self.__dict__.values())


# auto_docstring decorator (no-op)
def auto_docstring(*args, **kwargs):
    if len(args) == 1 and callable(args[0]):
        return args[0]
    def decorator(cls_or_func):
        return cls_or_func
    return decorator


# logging
class _FakeLogger:
    def info(self, msg, *args, **kwargs): pass
    def warning(self, msg, *args, **kwargs): print(f"[WARN] {msg}")
    def warning_once(self, msg, *args, **kwargs): pass
    def error(self, msg, *args, **kwargs): print(f"[ERROR] {msg}")
    def debug(self, msg, *args, **kwargs): pass

def get_logger(name):
    return _FakeLogger()


# modeling_rwkv7.py imports 'logging' from utils which is a module with get_logger
class _LoggingModule:
    get_logger = staticmethod(get_logger)
logging = _LoggingModule()


__all__ = [
    "PretrainedConfig", "PreTrainedModel", "GradientCheckpointingLayer",
    "GenerationMixin", "ModelOutput", "auto_docstring", "get_logger", "logging",
]
