# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib
import inspect
import json
import subprocess
from dataclasses import dataclass
from functools import cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch
from torch import nn
from torch.nn import functional as F

from ...cache_utils import CacheLayerMixin, LinearAttentionLayer
from ...generation import GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from .configuration_rwkv7 import Rwkv7Config


_FLA_RWKV7_REQUIRED_PARAMETERS = frozenset(
    {"initial_state", "output_final_state", "cu_seqlens", "state_indices", "mode"}
)
_FLA_RWKV7_FUSED_INFERENCE_OPERATORS = frozenset(
    {
        "infer_cmix_mix_fp16",
        "infer_tmix_kk_a_gate_fp16",
        "infer_tmix_lnx_rkvres_xg_fp16",
        "infer_tmix_mix6_fp16",
        "infer_tmix_vres_gate_fp16",
    }
)
RWKV7_FLA_DISTRIBUTION = "flash-linear-attention"
RWKV7_FLA_EXTRA = "flash-rwkv"
RWKV7_FLA_REPOSITORY = "https://github.com/rwkv-rs/fla-rwkv.git"
RWKV7_FLA_REVISION = "8173df6ab27adb1c160a59d84b4ee02b6c6d8926"
RWKV7_FLA_REQUIREMENT = (
    f"{RWKV7_FLA_DISTRIBUTION}[{RWKV7_FLA_EXTRA}] @ git+{RWKV7_FLA_REPOSITORY}@{RWKV7_FLA_REVISION}"
)
RWKV7_FLASH_RWKV_DISTRIBUTION = "flash-rwkv"
RWKV7_FLASH_RWKV_REPOSITORY = "https://github.com/rwkv-rs/FlashRWKV.git"
RWKV7_FLASH_RWKV_REVISION = "5410491f0d6cff6058e5bd21cbab900b5b54f220"


@dataclass(frozen=True)
class _FlaRwkv7Contract:
    recurrent_rwkv7: object
    flash_rwkv: object
    can_use_flash_rwkv_inference: object
    get_last_provider: object
    get_last_kernel: object


class Rwkv7DynamicCacheLayer(LinearAttentionLayer, CacheLayerMixin):
    """Recurrent state cache that also tracks the token offset required by generation."""

    _layer_type = "rwkv7"
    is_sliding = False

    def __init__(self, number_of_states: int = 1, **kwargs):
        CacheLayerMixin.__init__(self)
        LinearAttentionLayer.__init__(self, number_of_states=number_of_states, **kwargs)
        self.cumulative_length = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        raise RuntimeError("RWKV-7 caches recurrent state through `update_conv_state` and `update_recurrent_state`.")

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        state_idx: int = 0,
        conv_kernel_size: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        updated_states = super().update_conv_state(
            conv_states,
            state_idx=state_idx,
            conv_kernel_size=conv_kernel_size,
            **kwargs,
        )
        if state_idx == 0:
            self.cumulative_length += conv_states.shape[-1]
        return updated_states

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return query_length, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def reset(self) -> None:
        super().reset()
        self.cumulative_length = 0


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
    """Numerical RWKV-7 oracle for tests; standard model execution never calls this helper."""
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


def _editable_git_value(source_dir: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_dir), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError("Editable RWKV7 runtime provenance requires a working Git executable.") from error
    if completed.returncode != 0:
        raise RuntimeError(f"Editable RWKV7 runtime provenance Git check failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _canonical_github_repository(url: str) -> str | None:
    """Return one strict ASCII identity for an HTTPS GitHub repository URL."""
    candidate = url.removeprefix("git+")
    if any(ord(character) <= 0x20 or ord(character) == 0x7F for character in candidate):
        return None
    try:
        candidate.encode("ascii")
    except UnicodeEncodeError:
        return None
    if "%" in candidate:
        return None

    parsed = urlparse(candidate)
    if parsed.scheme != "https" or parsed.netloc != "github.com" or parsed.params or parsed.query or parsed.fragment:
        return None

    repository_path = parsed.path.removesuffix("/")
    if repository_path.endswith(".git"):
        repository_path = repository_path[:-4]
        if repository_path.endswith(".git"):
            return None

    path_parts = repository_path.split("/")
    if len(path_parts) != 3 or path_parts[0]:
        return None
    owner, repository_name = path_parts[1:]
    allowed = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._")
    if (
        owner in {"", ".", ".."}
        or repository_name in {"", ".", ".."}
        or any(character not in allowed for character in owner)
        or any(character not in allowed for character in repository_name)
    ):
        return None
    return f"https://github.com/{owner.lower()}/{repository_name.lower()}.git"


def _validate_rwkv7_distribution_provenance(
    *,
    distribution_name: str,
    module_name: str,
    repository: str,
    revision: str,
) -> dict[str, str]:
    try:
        distribution = importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError as error:
        raise RuntimeError(f"RWKV-7 requires the pinned `{RWKV7_FLA_REQUIREMENT}` runtime.") from error
    if distribution.metadata.get("Name") != distribution_name:
        raise RuntimeError(f"RWKV-7 distribution identity does not match the pinned `{distribution_name}` package.")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(
            f"RWKV-7 `{distribution_name}` provenance requires PEP 610 direct_url.json; registry packages are rejected."
        )
    try:
        direct_url = json.loads(direct_url_text)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"RWKV-7 `{distribution_name}` direct_url.json is invalid.") from error

    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None or module_spec.origin is None:
        raise RuntimeError(f"RWKV-7 `{distribution_name}` does not provide an importable `{module_name}` package.")
    module_origin = Path(module_spec.origin).resolve()
    vcs_info = direct_url.get("vcs_info")
    source_kind = "vcs"
    if isinstance(vcs_info, dict):
        observed_repository = str(direct_url.get("url", ""))
        requested_revision = str(vcs_info.get("requested_revision", "")).lower()
        resolved_revision = str(vcs_info.get("commit_id", "")).lower()
        if vcs_info.get("vcs") != "git":
            raise RuntimeError(f"RWKV-7 `{distribution_name}` provenance must use Git.")
        expected_module = Path(distribution.locate_file(f"{module_name}/__init__.py")).resolve()
        if module_origin != expected_module:
            raise RuntimeError(f"Imported `{module_name}` package does not belong to `{distribution_name}`.")
    else:
        parsed_url = urlparse(str(direct_url.get("url", "")))
        if direct_url.get("dir_info", {}).get("editable") is not True or parsed_url.scheme != "file":
            raise RuntimeError(
                f"RWKV-7 `{distribution_name}` is neither a pinned Git install nor a verified editable checkout."
            )
        source_dir = Path(unquote(parsed_url.path)).resolve()
        try:
            module_origin.relative_to(source_dir)
        except ValueError as error:
            raise RuntimeError(f"Imported `{module_name}` is outside its editable provenance checkout.") from error
        observed_repository = _editable_git_value(source_dir, "remote", "get-url", "origin")
        requested_revision = revision
        resolved_revision = _editable_git_value(source_dir, "rev-parse", "HEAD").lower()
        if _editable_git_value(source_dir, "status", "--porcelain"):
            raise RuntimeError(f"Editable `{distribution_name}` provenance checkout must be clean.")
        source_kind = "editable"

    expected_repository = _canonical_github_repository(repository)
    if expected_repository is None:
        raise RuntimeError(f"RWKV-7 `{distribution_name}` pinned repository configuration is invalid.")
    if _canonical_github_repository(observed_repository) != expected_repository:
        raise RuntimeError(f"RWKV-7 `{distribution_name}` repository provenance mismatch.")
    if requested_revision != revision or resolved_revision != revision:
        raise RuntimeError(
            f"RWKV-7 `{distribution_name}` revision provenance mismatch: "
            f"requested={requested_revision!r}, resolved={resolved_revision!r}."
        )
    return {"source_kind": source_kind, "version": distribution.version}


def validate_rwkv7_runtime_provenance() -> dict[str, str]:
    """Fail closed unless FLA and FlashRWKV come from the pinned rwkv-rs revisions."""
    fla = _validate_rwkv7_distribution_provenance(
        distribution_name=RWKV7_FLA_DISTRIBUTION,
        module_name="fla",
        repository=RWKV7_FLA_REPOSITORY,
        revision=RWKV7_FLA_REVISION,
    )
    flash_rwkv = _validate_rwkv7_distribution_provenance(
        distribution_name=RWKV7_FLASH_RWKV_DISTRIBUTION,
        module_name="flash_rwkv",
        repository=RWKV7_FLASH_RWKV_REPOSITORY,
        revision=RWKV7_FLASH_RWKV_REVISION,
    )
    return {
        "distribution": RWKV7_FLA_DISTRIBUTION,
        "distribution_version": fla["version"],
        "extra": RWKV7_FLA_EXTRA,
        "flash_rwkv_distribution": RWKV7_FLASH_RWKV_DISTRIBUTION,
        "flash_rwkv_distribution_version": flash_rwkv["version"],
        "flash_rwkv_repository": RWKV7_FLASH_RWKV_REPOSITORY,
        "flash_rwkv_revision": RWKV7_FLASH_RWKV_REVISION,
        "flash_rwkv_source_kind": flash_rwkv["source_kind"],
        "repository": RWKV7_FLA_REPOSITORY,
        "requirement": RWKV7_FLA_REQUIREMENT,
        "revision": RWKV7_FLA_REVISION,
        "source_kind": fla["source_kind"],
    }


@cache
def _load_fla_rwkv7_contract():
    validate_rwkv7_runtime_provenance()
    rwkv7 = importlib.import_module("fla.ops.rwkv7")
    inference = importlib.import_module("fla.ops.rwkv7.inference")
    recurrent_rwkv7 = getattr(rwkv7, "recurrent_rwkv7", None)
    flash_rwkv = getattr(rwkv7, "flash_rwkv", None)
    can_use_flash_rwkv_inference = getattr(inference, "can_use_flash_rwkv_inference", None)
    get_last_provider = getattr(rwkv7, "get_last_rwkv7_provider", None)
    get_last_kernel = getattr(rwkv7, "get_last_rwkv7_kernel", None)
    if not all(
        callable(function)
        for function in (recurrent_rwkv7, can_use_flash_rwkv_inference, get_last_provider, get_last_kernel)
    ):
        raise RuntimeError(
            "The installed FLA RWKV-7 API must publicly expose recurrent_rwkv7, fused inference eligibility, "
            "and provider/kernel telemetry."
        )
    missing_operators = sorted(
        operator
        for operator in _FLA_RWKV7_FUSED_INFERENCE_OPERATORS
        if not callable(getattr(flash_rwkv, operator, None))
    )
    if missing_operators:
        raise RuntimeError(f"The installed FLA FlashRWKV API lacks public inference operators: {missing_operators}.")
    try:
        parameters = inspect.signature(recurrent_rwkv7).parameters
    except (TypeError, ValueError) as error:
        raise RuntimeError("The installed FLA recurrent_rwkv7 API is not inspectable.") from error
    missing = sorted(_FLA_RWKV7_REQUIRED_PARAMETERS - parameters.keys())
    if missing:
        raise RuntimeError(f"The installed FLA recurrent_rwkv7 API lacks required stateful parameters: {missing}.")
    return _FlaRwkv7Contract(
        recurrent_rwkv7=recurrent_rwkv7,
        flash_rwkv=flash_rwkv,
        can_use_flash_rwkv_inference=can_use_flash_rwkv_inference,
        get_last_provider=get_last_provider,
        get_last_kernel=get_last_kernel,
    )


def _require_flash_rwkv_telemetry(contract, expected_kernel):
    provider = contract.get_last_provider()
    kernel = contract.get_last_kernel()
    if provider != "flash_rwkv" or kernel != expected_kernel:
        raise RuntimeError(
            "FLA public RWKV-7 telemetry did not report the required FlashRWKV kernel: "
            f"provider={provider!r}, kernel={kernel!r}, expected_kernel={expected_kernel!r}."
        )


def _rwkv7_flash(
    receptance,
    raw_decay,
    key,
    value,
    a,
    b,
    state,
    head_size,
    *,
    cu_seqlens=None,
    state_indices=None,
    contract=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlashRWKV exclusively through FLA's public recurrent contract."""
    try:
        contract = _load_fla_rwkv7_contract() if contract is None else contract
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(f"RWKV-7 requires the pinned public FLA FlashRWKV contract: {error}") from error

    batch_size, sequence_length, hidden_size = receptance.shape
    num_heads = hidden_size // head_size
    tensors = [
        tensor.view(batch_size, sequence_length, num_heads, head_size).contiguous()
        for tensor in (receptance, raw_decay, key, value, a, b)
    ]
    tensors[1] = (-F.softplus(-tensors[1]) - 0.5).contiguous()

    try:
        result = contract.recurrent_rwkv7(
            *tensors,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp32io16",
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"FLA public recurrent_rwkv7 execution failed: {error}") from error
    requires_grad = any(tensor.requires_grad for tensor in (*tensors, state))
    expected_kernel = (
        "rwkv7_recurrent_stateful"
        if state_indices is not None
        else "pretrain_recurrent_fp32io16_forward"
        if requires_grad
        else "rwkv7"
    )
    _require_flash_rwkv_telemetry(contract, expected_kernel)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("FLA public recurrent_rwkv7 must return (output, final_state).")
    output, final_state = result
    if not isinstance(output, torch.Tensor) or output.shape != tensors[3].shape:
        raise RuntimeError("FLA public recurrent_rwkv7 returned an output with an incompatible shape.")
    if not isinstance(final_state, torch.Tensor) or final_state.shape != state.shape:
        raise RuntimeError("FLA public recurrent_rwkv7 returned an incompatible final state.")
    if state_indices is not None and final_state is not state:
        raise RuntimeError("FLA packed recurrent RWKV-7 must update the supplied state pool in place.")
    return output.reshape(batch_size, sequence_length, hidden_size), final_state


class Rwkv7TimeMix(nn.Module):
    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.last_wkv_backend = "uninitialized"
        hidden_size = config.hidden_size

        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, name, nn.Parameter(torch.empty(1, 1, hidden_size)))
        self.w0 = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.w1 = nn.Linear(hidden_size, config.decay_low_rank_dim, bias=False)
        self.w2 = nn.Linear(config.decay_low_rank_dim, hidden_size, bias=False)
        self.a0 = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.a1 = nn.Linear(hidden_size, config.a_low_rank_dim, bias=False)
        self.a2 = nn.Linear(config.a_low_rank_dim, hidden_size, bias=False)
        if layer_id > 0:
            self.v0 = nn.Parameter(torch.empty(1, 1, hidden_size))
            self.v1 = nn.Linear(hidden_size, config.v_low_rank_dim, bias=False)
            self.v2 = nn.Linear(config.v_low_rank_dim, hidden_size, bias=False)
        self.g1 = nn.Linear(hidden_size, config.gate_low_rank_dim, bias=False)
        self.g2 = nn.Linear(config.gate_low_rank_dim, hidden_size, bias=False)
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

    def _reset_low_rank_parameters(self):
        for name in ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2"):
            projection = getattr(self, name, None)
            if projection is not None:
                weight = getattr(projection, "weight", None)
                if weight is not None:
                    nn.init.zeros_(weight)

    def forward(self, hidden_states, v_first, previous_hidden_state, wkv_state):
        try:
            contract = _load_fla_rwkv7_contract()
        except (ImportError, RuntimeError) as error:
            raise RuntimeError(
                f"RWKV-7 FlashRWKV execution failed closed: pinned public FLA contract unavailable: {error}"
            ) from error
        batch_size, sequence_length, hidden_size = hidden_states.shape
        mix_names = ("r", "w", "k", "v", "a", "g")
        mixes = tuple(getattr(self, f"x_{name}").reshape(-1) for name in mix_names)
        if contract.can_use_flash_rwkv_inference(hidden_states, previous_hidden_state, *mixes):
            mixed = contract.flash_rwkv.infer_tmix_mix6_fp16(hidden_states, previous_hidden_state, mixes)
            _require_flash_rwkv_telemetry(contract, "infer_tmix_mix6_fp16")
            if not isinstance(mixed, tuple) or len(mixed) != len(mix_names):
                raise RuntimeError("FLA public infer_tmix_mix6_fp16 must return six mixed tensors.")
            inputs = dict(zip(mix_names, mixed, strict=True))
            final_hidden_state = previous_hidden_state
        else:
            shifted, final_hidden_state = _token_shift(hidden_states, previous_hidden_state)
            inputs = {name: hidden_states + shifted * getattr(self, f"x_{name}") for name in mix_names}
        receptance = self.receptance(inputs["r"])
        key = self.key(inputs["k"])
        value = self.value(inputs["v"])
        raw_decay = self.w0 + self.w2(torch.tanh(self.w1(inputs["w"])))
        if self.layer_id == 0:
            v_first = value
        else:
            value_delta = self.v2(self.v1(inputs["v"]))
            if contract.can_use_flash_rwkv_inference(value, v_first, self.v0, value_delta):
                value = contract.flash_rwkv.infer_tmix_vres_gate_fp16(value, v_first, self.v0, value_delta)
                _require_flash_rwkv_telemetry(contract, "infer_tmix_vres_gate_fp16")
            else:
                value = value + (v_first - value) * torch.sigmoid(self.v0 + value_delta)
        learning_rate_delta = self.a2(self.a1(inputs["a"]))
        gate = self.g2(torch.sigmoid(self.g1(inputs["g"])))
        if contract.can_use_flash_rwkv_inference(
            key,
            self.k_k,
            self.a0,
            learning_rate_delta,
            self.k_a,
            head_dim=self.config.head_size,
        ):
            key, recurrent_a, recurrent_b = contract.flash_rwkv.infer_tmix_kk_a_gate_fp16(
                key,
                self.k_k,
                self.a0,
                learning_rate_delta,
                self.k_a,
            )
            _require_flash_rwkv_telemetry(contract, "infer_tmix_kk_a_gate_fp16")
        else:
            learning_rate = torch.sigmoid(self.a0 + learning_rate_delta)
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
            recurrent_a = -normalized_key
            recurrent_b = normalized_key * learning_rate
        wkv_inputs = (
            receptance,
            raw_decay,
            key,
            value,
            recurrent_a,
            recurrent_b,
            wkv_state,
            self.config.head_size,
        )
        try:
            output, wkv_state = _rwkv7_flash(*wkv_inputs, contract=contract)
        except RuntimeError as error:
            raise RuntimeError(f"RWKV-7 FlashRWKV execution failed closed: {error}") from error
        self.last_wkv_backend = "flash_rwkv"
        if self.config.group_norm_epsilon == 64e-5 and contract.can_use_flash_rwkv_inference(
            output,
            receptance,
            key,
            value,
            self.r_k.reshape(-1),
            self.ln_x.weight,
            self.ln_x.bias,
            gate,
            head_dim=self.config.head_size,
        ):
            output = contract.flash_rwkv.infer_tmix_lnx_rkvres_xg_fp16(
                output,
                receptance,
                key,
                value,
                self.r_k.reshape(-1),
                self.ln_x.weight,
                self.ln_x.bias,
                gate,
            )
            _require_flash_rwkv_telemetry(contract, "infer_tmix_lnx_rkvres_xg_fp16")
        else:
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
            output = (output + local) * gate
        return self.output(output), v_first, final_hidden_state, wkv_state


class Rwkv7ChannelMix(nn.Module):
    def __init__(self, config: Rwkv7Config):
        super().__init__()
        self.x_k = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, previous_hidden_state):
        try:
            contract = _load_fla_rwkv7_contract()
        except (ImportError, RuntimeError) as error:
            raise RuntimeError(
                f"RWKV-7 FlashRWKV execution failed closed: pinned public FLA contract unavailable: {error}"
            ) from error
        mix = self.x_k.reshape(-1)
        if contract.can_use_flash_rwkv_inference(hidden_states, previous_hidden_state, mix):
            mixed = contract.flash_rwkv.infer_cmix_mix_fp16(hidden_states, previous_hidden_state, mix)
            _require_flash_rwkv_telemetry(contract, "infer_cmix_mix_fp16")
            final_hidden_state = previous_hidden_state
        else:
            shifted, final_hidden_state = _token_shift(hidden_states, previous_hidden_state)
            mixed = hidden_states + shifted * self.x_k
        output = self.value(F.relu(self.key(mixed)).square())
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
        super()._init_weights(module)
        if isinstance(module, Rwkv7TimeMix):
            for parameter in module.parameters(recurse=False):
                nn.init.zeros_(parameter)
            module._reset_low_rank_parameters()
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
    "RWKV7_FLA_DISTRIBUTION",
    "RWKV7_FLA_EXTRA",
    "RWKV7_FLA_REPOSITORY",
    "RWKV7_FLA_REQUIREMENT",
    "RWKV7_FLA_REVISION",
    "RWKV7_FLASH_RWKV_DISTRIBUTION",
    "RWKV7_FLASH_RWKV_REPOSITORY",
    "RWKV7_FLASH_RWKV_REVISION",
    "Rwkv7CausalLMOutput",
    "Rwkv7ForCausalLM",
    "Rwkv7Model",
    "Rwkv7Output",
    "Rwkv7PreTrainedModel",
    "validate_rwkv7_runtime_provenance",
]
