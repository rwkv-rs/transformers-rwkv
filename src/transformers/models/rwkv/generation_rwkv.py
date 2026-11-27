# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import copy
import queue
import threading
from typing import TYPE_CHECKING

import torch

from ...generation import (
    EosTokenCriteria,
    GenerationConfig,
    GenerationMixin,
    GenerationMode,
    LogitsProcessorList,
    MaxLengthCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from ...integrations.flash_rwkv2 import flash_rwkv2_linear_spec, load_flash_rwkv2
from .configuration_rwkv import RwkvConfig


if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer
    from .modeling_rwkv import RwkvForCausalLM, RwkvModel


def _recurrent_metadata(
    cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, int, object]],
    flash_rwkv2,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, object]:
    key = (device, batch_size, sequence_length, torch.cuda.current_stream(device).cuda_stream)
    if key not in cache:
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * sequence_length, sequence_length, dtype=torch.int32, device=device
        )
        state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        ticket = flash_rwkv2.prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=batch_size * sequence_length,
            state_pool_size=batch_size,
            max_seqlen=sequence_length,
        )
        cache[key] = (cu_seqlens, state_indices, sequence_length, ticket)
    return cache[key]


def _capture_graph(
    device: torch.device, capture_stream: torch.cuda.Stream, graph_pool, fn, restore
) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    generation_stream = torch.cuda.current_stream(device)
    capture_stream.wait_stream(generation_stream)
    with torch.cuda.stream(capture_stream):
        restore()
        fn()
        restore()
        with torch.cuda.graph(graph, pool=graph_pool, stream=capture_stream, capture_error_mode="thread_local"):
            fn()
        restore()
    generation_stream.wait_stream(capture_stream)
    return graph


def _recurrent_forward(
    flash_rwkv2,
    wkv_mode: str,
    recurrent_state,
    receptance: torch.Tensor | None,
    decay_logits: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    recurrent_a: torch.Tensor,
    recurrent_b: torch.Tensor,
    decay_bias: torch.Tensor,
    inference_metadata,
) -> torch.Tensor | None:
    cu_seqlens, state_indices, max_seqlen, ticket = inference_metadata
    return getattr(flash_rwkv2, f"infer_tmix_wkv7_recurrent_{wkv_mode}_forward_varlen")(
        receptance,
        decay_logits,
        key,
        value,
        recurrent_a,
        recurrent_b,
        state=recurrent_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        decay_bias=decay_bias,
        max_seqlen=max_seqlen,
        validated_metadata=ticket,
        save=False,
    )


def _materialize_recurrent_state(handle, pool: torch.Tensor) -> None:
    if pool.dtype == torch.float16 and handle.memory_layout["private_bytes_per_slot"]:
        slots = torch.arange(pool.shape[0], dtype=torch.int32, device=pool.device)
        handle.materialize_slots_(slots)


class _RwkvGraphState:
    def __init__(
        self,
        config: RwkvConfig,
        batch_size: int,
        device: torch.device,
        flash_rwkv2,
    ) -> None:
        shift_shape = (config.num_hidden_layers, batch_size, config.hidden_size)
        self.wkv_mode = config.wkv_mode
        self.attention_shift = torch.zeros(shift_shape, dtype=torch.bfloat16, device=device)
        self.feed_forward_shift = torch.zeros_like(self.attention_shift)
        prepare_state = getattr(flash_rwkv2, f"prepare_tmix_wkv7_recurrent_{self.wkv_mode}_state_from_tensor")
        state_shape = (batch_size, config.num_attention_heads, config.head_size, config.head_size)
        state_dtype = torch.float16 if self.wkv_mode == "fp16" else torch.float32
        self._state_pools = [
            torch.zeros(state_shape, dtype=state_dtype, device=device) for _ in range(config.num_hidden_layers)
        ]
        self.recurrent_states = [prepare_state(pool) for pool in self._state_pools]
        self._inference_metadata: dict[tuple, tuple[torch.Tensor, torch.Tensor, int, object]] = {}

    def get_recurrent_state(self, layer_idx: int) -> torch.Tensor:
        pool = self._state_pools[layer_idx]
        _materialize_recurrent_state(self.recurrent_states[layer_idx], pool)
        return pool

    def layer_states(self, layer_idx: int) -> tuple[torch.Tensor, object, torch.Tensor]:
        return self.attention_shift[layer_idx], self.recurrent_states[layer_idx], self.feed_forward_shift[layer_idx]

    def recurrent_metadata(
        self, flash_rwkv2, batch_size: int, sequence_length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, int, object]:
        return _recurrent_metadata(self._inference_metadata, flash_rwkv2, batch_size, sequence_length, device)

    def recurrent_forward(
        self,
        flash_rwkv2,
        receptance: torch.Tensor,
        decay_logits: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        recurrent_a: torch.Tensor,
        recurrent_b: torch.Tensor,
        recurrent_state,
        decay_bias: torch.Tensor,
        inference_metadata,
    ) -> torch.Tensor:
        return _recurrent_forward(
            flash_rwkv2,
            self.wkv_mode,
            recurrent_state,
            receptance,
            decay_logits,
            key,
            value,
            recurrent_a,
            recurrent_b,
            decay_bias,
            inference_metadata,
        )

    def zero_(self) -> None:
        self.attention_shift.zero_()
        self.feed_forward_shift.zero_()
        for state in self.recurrent_states:
            state.zero_()


class _RwkvVerifyState:
    """Verify on a single scratch layer and retain only BF16 inputs for commit."""

    def __init__(self, main: _RwkvGraphState, config: RwkvConfig, width: int, flash_rwkv2) -> None:
        self.main = main
        self.width = width
        self.scratch_attention_shift = torch.empty_like(main.attention_shift[0])
        self.scratch_feed_forward_shift = torch.empty_like(main.feed_forward_shift[0])
        self.scratch_S = torch.empty_like(main._state_pools[0])
        prepare_state = getattr(flash_rwkv2, f"prepare_tmix_wkv7_recurrent_{main.wkv_mode}_state_from_tensor")
        self.scratch_wkv = prepare_state(self.scratch_S)
        batch_size = self.scratch_S.shape[0]
        num_tokens = batch_size * width
        self.saved_wkv = torch.empty(
            (config.num_hidden_layers, 5, num_tokens, config.num_attention_heads, config.head_size),
            dtype=torch.bfloat16,
            device=self.scratch_S.device,
        )
        shift_shape = (config.num_hidden_layers, num_tokens, config.hidden_size)
        self.saved_tmix_norm = self.saved_wkv.new_empty(shift_shape)
        self.saved_cmix_norm = self.saved_wkv.new_empty(shift_shape)
        self.rows = torch.arange(batch_size, dtype=torch.int32, device=self.scratch_S.device)
        self.positions = torch.arange(num_tokens, dtype=torch.int32, device=self.scratch_S.device)

    def layer_states(self, layer_idx: int) -> tuple[torch.Tensor, object, torch.Tensor]:
        self.layer_idx = layer_idx
        self.scratch_attention_shift.copy_(self.main.attention_shift[layer_idx])
        self.scratch_feed_forward_shift.copy_(self.main.feed_forward_shift[layer_idx])
        self.scratch_wkv.copy_(self.main.recurrent_states[layer_idx])
        return self.scratch_attention_shift, self.scratch_wkv, self.scratch_feed_forward_shift

    def recurrent_metadata(self, flash_rwkv2, batch_size: int, sequence_length: int, device: torch.device):
        return self.main.recurrent_metadata(flash_rwkv2, batch_size, sequence_length, device)

    def recurrent_forward(
        self,
        flash_rwkv2,
        receptance: torch.Tensor,
        decay_logits: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        recurrent_a: torch.Tensor,
        recurrent_b: torch.Tensor,
        recurrent_state,
        decay_bias: torch.Tensor,
        inference_metadata,
    ) -> torch.Tensor:
        inputs = (decay_logits, key, value, recurrent_a, recurrent_b)
        for destination, tensor in zip(self.saved_wkv[self.layer_idx].unbind(), inputs, strict=True):
            destination.copy_(tensor)
        return _recurrent_forward(
            flash_rwkv2, self.main.wkv_mode, recurrent_state, receptance, *inputs, decay_bias, inference_metadata
        )

    def commit(self, flash_rwkv2, model: RwkvModel, prefixes: torch.Tensor) -> None:
        active = prefixes > 0
        batch_size = self.rows.numel()
        # Compact active slots and their accepted inputs into fixed-capacity GPU buffers for graph replay.
        order = torch.where(active, self.rows, batch_size).sort().values
        slots = order.clamp_max(batch_size - 1)
        lengths = torch.where(order < batch_size, prefixes[slots.long()], 0)
        offsets = torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        packed_rows = torch.searchsorted(offsets[1:], self.positions, right=True).clamp_max(batch_size - 1)
        gather = (slots[packed_rows] * self.width + self.positions - offsets[packed_rows]).clamp_max(
            self.positions.numel() - 1
        )
        ticket = flash_rwkv2.prepare_tmix_wkv7_recurrent_metadata(
            offsets,
            slots,
            state_pool_size=batch_size,
            token_capacity=self.positions.numel(),
            sequence_capacity=batch_size,
            max_seqlen_capacity=self.width,
            num_active_tokens=offsets[-1:],
            num_active_sequences=active.sum(dtype=torch.int32).view(1),
        )
        metadata = (offsets, slots, self.width, ticket)
        for layer_idx, layer in enumerate(model.layers):
            packed = self.saved_wkv[layer_idx].index_select(1, gather)
            _recurrent_forward(
                flash_rwkv2,
                self.main.wkv_mode,
                self.main.recurrent_states[layer_idx],
                None,
                *packed.unbind(),
                layer.linear_attn.w0,
                metadata,
            )
            del packed
        last = self.rows * self.width + (prefixes - 1).clamp_min(0)
        for shift, saved in (
            (self.main.attention_shift, self.saved_tmix_norm),
            (self.main.feed_forward_shift, self.saved_cmix_norm),
        ):
            shift.copy_(torch.where(active[None, :, None], saved.index_select(1, last), shift))


class _RwkvPrefillGraph:
    def __init__(
        self,
        model: RwkvModel,
        state: _RwkvGraphState,
        batch_size: int,
        sequence_length: int,
        capture_stream: torch.cuda.Stream,
        graph_pool,
    ) -> None:
        self.input_ids = torch.zeros(
            (batch_size, sequence_length), dtype=torch.long, device=model.embed_tokens.weight.device
        )
        self.graph = _capture_graph(
            self.input_ids.device,
            capture_stream,
            graph_pool,
            lambda: model._forward_state_only(self.input_ids, None, state, False),
            state.zero_,
        )

    def replay(self, input_ids: torch.LongTensor) -> None:
        self.input_ids.copy_(input_ids)
        self.graph.replay()


class _RwkvDecodeGraph:
    def __init__(
        self,
        model: RwkvForCausalLM,
        state: _RwkvGraphState,
        batch_size: int,
        max_new_tokens: int,
        generation_config: GenerationConfig,
        eos_token_ids: torch.Tensor | None,
        stop_string_criteria: tuple[StopStringCriteria, ...],
        capture_stream: torch.cuda.Stream,
        graph_pool,
        flash_rwkv2,
        mtp_depth: int = 0,
    ) -> None:
        device = model.lm_head.weight.device
        self.vocab_size = model.vocab_size
        self.state = state
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.do_sample = generation_config.do_sample
        self.presence_penalty = getattr(generation_config, "presence_penalty", 0.0)
        self.frequency_penalty = getattr(generation_config, "frequency_penalty", 0.0)
        self.penalty_decay = getattr(generation_config, "penalty_decay", 0.996)
        self.temperature = generation_config.temperature if generation_config.temperature is not None else 1.0
        self.top_k = generation_config.top_k if generation_config.top_k not in (None, 0) else -1
        self.top_p = generation_config.top_p if generation_config.top_p is not None else 1.0
        self.flash_rwkv2 = flash_rwkv2
        self.stop_string_criteria = tuple(copy.copy(criteria) for criteria in stop_string_criteria)
        self.input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.mtp_depth = mtp_depth
        self.completion = torch.empty(
            (batch_size, max_new_tokens + (mtp_depth + 1 if mtp_depth else 0)), dtype=torch.long, device=device
        )
        self.completion_index = torch.zeros(1, dtype=torch.long, device=device)
        self.completion_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.pad_token_id = torch.zeros((), dtype=torch.long, device=device)
        if generation_config._pad_token_tensor is not None:
            self.pad_token_id.copy_(generation_config._pad_token_tensor)
        self.eos_token_ids = None if eos_token_ids is None else eos_token_ids.to(device=device)
        suffix_length = max(
            (criteria.maximum_token_len for criteria in self.stop_string_criteria),
            default=1,
        )
        self.stop_suffix = torch.full((batch_size, suffix_length), self.vocab_size, dtype=torch.long, device=device)
        self.slot_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        self.sampling_states = flash_rwkv2.setup_sampling_states(0, batch_size) if self.do_sample else None
        self.penalties = (
            torch.zeros((batch_size, self.vocab_size), dtype=torch.float32, device=device) if self.do_sample else None
        )
        for criteria in self.stop_string_criteria:
            criteria(self.stop_suffix, None)

        sampling_snapshot = self.sampling_states.clone() if self.sampling_states is not None else None
        self.graph = _capture_graph(
            device,
            capture_stream,
            graph_pool,
            lambda: self._forward(model),
            lambda: self._restore_capture_inputs(sampling_snapshot),
        )

    def _reset_state(self) -> None:
        self.state.zero_()
        if self.penalties is not None:
            self.penalties.zero_()
        self.completion.fill_(self.pad_token_id)
        self.completion_index.zero_()
        self.completion_lengths.zero_()
        self.unfinished.fill_(True)
        self.stop_suffix.fill_(self.vocab_size)

    def _restore_capture_inputs(self, sampling_snapshot: torch.Tensor | None) -> None:
        self._reset_state()
        if sampling_snapshot is not None:
            self.sampling_states.copy_(sampling_snapshot)
        self.input_ids.zero_()

    def _forward(self, model: RwkvForCausalLM) -> None:
        hidden_states, residual, _ = model.model._forward_state_only(self.input_ids, None, self.state, False)
        hidden_states = self.flash_rwkv2.infer_post_norm_output_forward_varlen(
            hidden_states,
            residual,
            model.model.norm.weight,
            model.model.norm.bias,
            eps=model.config.layer_norm_epsilon,
        )
        if self.mtp_depth:
            self.hidden_states = hidden_states
        logits = model._project_lm_head(hidden_states, flash_rwkv2_linear_spec(model.lm_head))
        logits = logits.view(self.batch_size, model.vocab_size)
        self.logits = logits.float().contiguous()
        if self.do_sample:
            sampled_tokens = self.flash_rwkv2.infer_sampling_six_parameter_forward_varlen(
                self.logits,
                self.penalties,
                self.sampling_states,
                self.slot_indices,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                penalty_decay=self.penalty_decay,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            ).to(dtype=torch.long)
        else:
            sampled_tokens = torch.argmax(self.logits, dim=-1)

        active = self.unfinished
        self.sampled_tokens = torch.where(active, sampled_tokens, self.pad_token_id)
        completion_indices = self.completion_index.view(1, 1).expand(self.batch_size, 1)
        self.completion.scatter_(1, completion_indices, self.sampled_tokens[:, None])
        self.completion_lengths.add_(active)

        finished = torch.zeros_like(self.unfinished)
        if self.eos_token_ids is not None:
            finished.logical_or_(torch.isin(self.sampled_tokens, self.eos_token_ids))
        if self.stop_string_criteria:
            self.stop_suffix.copy_(torch.cat((self.stop_suffix[:, 1:], self.sampled_tokens[:, None]), dim=1))
            for criteria in self.stop_string_criteria:
                finished.logical_or_(criteria(self.stop_suffix, None))
        self.unfinished.logical_and_(~finished)
        self.input_ids[:, 0].copy_(self.sampled_tokens)
        self.completion_index.add_(1)

    def prepare(self, prompt: torch.LongTensor, seed: int) -> None:
        self._reset_state()
        self.input_ids.copy_(prompt[:, -1:])
        suffix = prompt[:, -self.stop_suffix.shape[1] :]
        self.stop_suffix[:, -suffix.shape[1] :].copy_(suffix)
        if self.sampling_states is not None:
            self.sampling_states.copy_(self.flash_rwkv2.setup_sampling_states(seed, self.batch_size))

    def replay(self) -> None:
        self.graph.replay()


class _RwkvMtpDecodeGraph:
    def __init__(
        self,
        model: RwkvForCausalLM,
        decode: _RwkvDecodeGraph,
        capture_stream: torch.cuda.Stream,
        graph_pool,
    ) -> None:
        from flashrwkv2.mtp import DraftLoop

        device = decode.input_ids.device
        flash_rwkv2 = decode.flash_rwkv2
        width = decode.mtp_depth + 1
        self.decode = decode
        self.state = _RwkvVerifyState(decode.state, model.config, width, flash_rwkv2)
        self.input_ids = torch.zeros((decode.batch_size, width), dtype=torch.long, device=device)
        self.steps = torch.arange(width, device=device)
        self.rows = decode.slot_indices.long()
        head_spec = flash_rwkv2_linear_spec(model.lm_head)
        weight, lora_a, lora_b, lora_scale, use_nora = head_spec
        self.draft = DraftLoop.from_state_dict(
            model.mtp.state_dict(prefix="mtp."),
            weight,
            depth=decode.mtp_depth,
            eps=model.mtp.norm_h.eps,
            lora_a=lora_a,
            lora_b=lora_b,
            lora_scale=lora_scale,
            use_nora=use_nora,
        )
        self.draft.capture(model.model.embed_tokens, decode.batch_size)
        self.graph = _capture_graph(
            device,
            capture_stream,
            graph_pool,
            lambda: self._forward(model, head_spec),
            lambda: decode._restore_capture_inputs(None),
        )

    def _forward(self, model: RwkvForCausalLM, head_spec) -> None:
        decode = self.decode
        flash_rwkv2 = decode.flash_rwkv2
        width = self.input_ids.shape[1]
        hidden_states, residual, _ = model.model._forward_state_only(self.input_ids, None, self.state, False)
        hidden_states = flash_rwkv2.infer_post_norm_output_forward_varlen(
            hidden_states,
            residual,
            model.model.norm.weight,
            model.model.norm.bias,
            eps=model.config.layer_norm_epsilon,
        ).view(decode.batch_size, width, -1)
        predictions = torch.empty_like(self.input_ids)
        for step in range(width):
            # Match ordinary decode's head shape and keep at most one logits plane live.
            logits = model._project_lm_head(hidden_states[:, step], head_spec)
            torch.argmax(logits, dim=-1, out=predictions[:, step])
            del logits

        matches = predictions[:, :-1] == self.input_ids[:, 1:]
        prefixes = torch.where(matches, width - 1, self.steps[:-1]).amin(dim=1) + 1
        finished = torch.zeros_like(predictions, dtype=torch.bool)
        if decode.eos_token_ids is not None:
            finished.logical_or_(torch.isin(predictions, decode.eos_token_ids))
        if decode.stop_string_criteria:
            suffix_length = decode.stop_suffix.shape[1]
            windows = torch.cat((decode.stop_suffix, predictions), dim=1).unfold(1, suffix_length, 1)
            suffixes = windows[:, 1:].reshape(-1, suffix_length)
            for criteria in decode.stop_string_criteria:
                finished.logical_or_(criteria(suffixes, None).view_as(finished))
        stop = torch.where(finished, self.steps, width).amin(dim=1) + 1
        remaining = decode.max_new_tokens - decode.completion_lengths
        prefixes = torch.minimum(prefixes, torch.minimum(remaining, stop)).masked_fill(~decode.unfinished, 0)
        self.state.commit(flash_rwkv2, model.model, prefixes)

        valid = self.steps < prefixes[:, None]
        destination = torch.where(
            valid, decode.completion_lengths[:, None] + self.steps, decode.max_new_tokens + self.steps
        )
        decode.completion.scatter_(1, destination, predictions)
        decode.completion_lengths.add_(prefixes)
        decode.unfinished.logical_and_((prefixes < remaining) & (prefixes < stop))
        if decode.stop_string_criteria:
            decode.stop_suffix.copy_(windows[self.rows, prefixes])
        last = (prefixes - 1).clamp_min(0)
        self.hidden_states = hidden_states[self.rows, last]
        decode.input_ids[:, 0].copy_(predictions[self.rows, last])

    def replay(self, hidden_states: torch.Tensor) -> torch.Tensor:
        tokens = self.decode.input_ids[:, 0]
        self.input_ids[:, 0].copy_(tokens)
        self.input_ids[:, 1:].copy_(self.draft.replay(hidden_states, tokens))
        self.graph.replay()
        return self.hidden_states


class _RwkvAsyncTokenStreamer:
    def __init__(self, streamer: BaseStreamer, batch_size: int, max_new_tokens: int, device: torch.device) -> None:
        self.streamer = streamer
        self.ring = torch.empty(
            (max_new_tokens, batch_size),
            dtype=torch.long,
            device="cpu",
            pin_memory=True,
        )
        self.copy_stream = torch.cuda.Stream(device=device)
        self.copied_events = [torch.cuda.Event() for _ in range(max_new_tokens)]
        self.pending: queue.SimpleQueue[int | None] = queue.SimpleQueue()
        self.error: BaseException | None = None
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self) -> None:
        try:
            while (item := self.pending.get()) is not None:
                self.copied_events[item].synchronize()
                self.streamer.put(self.ring[item])
            self.streamer.end()
        except Exception as error:
            self.error = error

    def submit(self, completion: torch.LongTensor, position: int, generation_stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_stream(generation_stream)
            self.ring[position].copy_(completion[:, position], non_blocking=True)
            self.copied_events[position].record(self.copy_stream)
        self.pending.put(position)

    def finish(self) -> None:
        self.pending.put(None)
        self.worker.join()
        if self.error is not None:
            raise self.error


class _RwkvGenerationModelRunner:
    def __init__(
        self,
        model: RwkvForCausalLM,
        input_ids: torch.LongTensor,
        generation_config: GenerationConfig,
        max_length: int,
        eos_token_ids: torch.Tensor | None,
        stop_string_criteria: tuple[StopStringCriteria, ...],
        mtp_depth: int,
    ) -> None:
        batch_size, prompt_length = input_ids.shape
        prefill_length = prompt_length - 1
        chunk_size = generation_config.prefill_chunk_size
        if prefill_length <= 0:
            self.prefill_lengths: tuple[int, ...] = ()
        elif chunk_size is None:
            self.prefill_lengths = (prefill_length,)
        else:
            full_chunks, remainder = divmod(prefill_length, chunk_size)
            self.prefill_lengths = (chunk_size,) * full_chunks + ((remainder,) if remainder else ())
        max_new_tokens = max_length - prompt_length
        flash_rwkv2 = load_flash_rwkv2()
        self.state = _RwkvGraphState(
            model.config,
            batch_size,
            input_ids.device,
            flash_rwkv2,
        )
        self.capture_stream = torch.cuda.Stream(device=input_ids.device)
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.prefill_graphs = {
            sequence_length: _RwkvPrefillGraph(
                model.model,
                self.state,
                batch_size,
                sequence_length,
                self.capture_stream,
                self.graph_pool,
            )
            for sequence_length in dict.fromkeys(self.prefill_lengths)
        }
        self.decode_graph = _RwkvDecodeGraph(
            model,
            self.state,
            batch_size,
            max_new_tokens,
            generation_config,
            eos_token_ids,
            stop_string_criteria,
            self.capture_stream,
            self.graph_pool,
            flash_rwkv2,
            mtp_depth,
        )
        self.mtp_graph = None

    def generate(
        self, input_ids: torch.LongTensor, seed: int, streamer: BaseStreamer | None = None, *, use_mtp: bool = False
    ) -> torch.LongTensor:
        self.decode_graph.prepare(input_ids, seed)
        generation_stream = torch.cuda.current_stream(input_ids.device)
        async_streamer = (
            None
            if streamer is None
            else _RwkvAsyncTokenStreamer(
                streamer,
                input_ids.shape[0],
                self.decode_graph.max_new_tokens,
                input_ids.device,
            )
        )
        start_event = torch.cuda.Event(enable_timing=True)
        first_token_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(generation_stream)
        offset = 0
        for sequence_length in self.prefill_lengths:
            self.prefill_graphs[sequence_length].replay(input_ids[:, offset : offset + sequence_length])
            offset += sequence_length

        produced_tokens = 0
        synchronize_stopping = self.decode_graph.eos_token_ids is not None or bool(
            self.decode_graph.stop_string_criteria
        )
        if use_mtp:
            decode = self.decode_graph
            decode.replay()
            first_token_event.record(generation_stream)
            produced_tokens = 1
            if async_streamer is not None:
                async_streamer.submit(decode.completion, 0, generation_stream)
            if not synchronize_stopping or bool(decode.unfinished.any().item()):
                streamed = 1
                hidden_states = decode.hidden_states
                while True:
                    hidden_states = self.mtp_graph.replay(hidden_states)
                    ready = torch.where(decode.unfinished, decode.completion_lengths, decode.max_new_tokens).min()
                    produced_tokens, ready = torch.stack((decode.completion_lengths.max(), ready)).tolist()
                    done = ready == decode.max_new_tokens
                    if async_streamer is not None:
                        end = produced_tokens if done else ready
                        for position in range(streamed, end):
                            async_streamer.submit(decode.completion, position, generation_stream)
                        streamed = end
                    if done:
                        break
        else:
            for _ in range(self.decode_graph.max_new_tokens):
                self.decode_graph.replay()
                if produced_tokens == 0:
                    first_token_event.record(generation_stream)
                if async_streamer is not None:
                    async_streamer.submit(self.decode_graph.completion, produced_tokens, generation_stream)
                produced_tokens += 1
                if synchronize_stopping and not bool(self.decode_graph.unfinished.any().item()):
                    break
        end_event.record(generation_stream)
        self.last_generation_events = start_event, first_token_event, end_event
        output = torch.cat((input_ids, self.decode_graph.completion[:, :produced_tokens]), dim=-1)
        if async_streamer is not None:
            async_streamer.finish()
        return output

    def reset(self) -> None:
        if self.mtp_graph is not None:
            self.mtp_graph.graph.reset()
            self.mtp_graph = None
        for prefill_graph in self.prefill_graphs.values():
            prefill_graph.graph.reset()
        self.decode_graph.graph.reset()


class _RwkvGenerationCache(dict[tuple, _RwkvGenerationModelRunner]):
    """Owns RWKV generation runners and releases their CUDA graphs before the tensors they reference."""

    def clear(self) -> None:
        for runner in self.values():
            runner.reset()
        super().clear()


class RwkvGenerationMixin(GenerationMixin):
    _supported_generation_modes = [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]

    def adjust_generation_fn(self, *args, **kwargs):
        super().adjust_generation_fn(*args, **kwargs)
        self.generation_config.presence_penalty = getattr(self.generation_config, "presence_penalty", 0.0)
        self.generation_config.frequency_penalty = getattr(self.generation_config, "frequency_penalty", 0.0)
        self.generation_config.penalty_decay = getattr(self.generation_config, "penalty_decay", 0.996)

    def _validate_generation_mode(self, generation_mode, generation_config, generation_mode_kwargs):
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if (
                not generation_config.use_mtp
                or generation_mode_kwargs.get("assistant_model") is not None
                or generation_config.prompt_lookup_num_tokens is not None
                or generation_config.assistant_early_exit is not None
            ):
                raise ValueError("RWKV-7 assisted generation supports only built-in LightMTP (`use_mtp=True`).")
            if self.mtp is None or self.config.mtp_depth < 1 or generation_config.do_sample:
                raise ValueError("LightMTP requires a loaded MTP head and greedy generation (`do_sample=False`).")
            generation_mode = GenerationMode.GREEDY_SEARCH
        super()._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)
        if generation_config.token_healing:
            raise ValueError("RWKV-7 CUDA graph generation does not support token healing.")
        if generation_config.cache_implementation is not None:
            raise ValueError("RWKV-7 CUDA graph generation does not support cache_implementation.")
        if generation_config.compile_config is not None:
            raise ValueError("RWKV-7 CUDA graph generation does not support compile_config.")
        if generation_config.continuous_batching_config is not None:
            raise ValueError("RWKV-7 CUDA graph generation does not support continuous batching.")

    def _get_logits_processor(self, generation_config, *args, **kwargs):
        if generation_config.top_k == -1:
            generation_config = copy.copy(generation_config)
            generation_config.top_k = 0
        return super()._get_logits_processor(generation_config, *args, **kwargs)

    def _validate_graph_generation(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        model_kwargs: dict,
    ) -> tuple[int, torch.Tensor | None, tuple[StopStringCriteria, ...]]:
        if flash_rwkv2_linear_spec(self.lm_head) is None or not self.model._supports_fused_projections():
            raise ValueError("RWKV-7 generation requires fused projections. Merge unsupported PEFT adapters first.")
        if input_ids.device.type != "cuda" or self.lm_head.weight.device != input_ids.device:
            raise ValueError("RWKV-7 generation requires the model and a static input batch on one CUDA device.")
        if self.training or self.lm_head.weight.dtype != torch.bfloat16:
            raise ValueError("RWKV-7 generation requires an eval-mode bfloat16 model.")
        attention_mask = model_kwargs.get("attention_mask")
        if attention_mask is not None and (
            attention_mask.shape != input_ids.shape or not bool(torch.all(attention_mask != 0))
        ):
            raise ValueError("RWKV-7 does not support padding or ragged batches; bucket inputs by sequence length.")
        if synced_gpus:
            raise ValueError("RWKV-7 CUDA graph generation does not support synced_gpus.")
        if not generation_config.use_cache:
            raise ValueError("RWKV-7 CUDA graph generation requires use_cache=True.")
        if generation_config.num_return_sequences != 1:
            raise ValueError("RWKV-7 CUDA graph generation requires num_return_sequences=1.")
        if (
            generation_config.return_dict_in_generate
            or generation_config.output_attentions
            or generation_config.output_hidden_states
            or generation_config.output_scores
            or generation_config.output_logits
        ):
            raise ValueError("RWKV-7 CUDA graph generation does not support detailed generation outputs.")
        if model_kwargs.get("cache") is not None or model_kwargs.get("inputs_embeds") is not None:
            raise ValueError("RWKV-7 CUDA graph generation does not accept an external cache or inputs_embeds.")
        unsupported_kwargs = {
            name
            for name, value in model_kwargs.items()
            if value is not None
            and name not in {"attention_mask", "logits_to_keep", "position_ids", "tokenizer", "use_cache"}
        }
        if unsupported_kwargs:
            raise ValueError(
                f"RWKV-7 CUDA graph generation does not support model kwargs {sorted(unsupported_kwargs)}."
            )
        if generation_config.prefill_chunk_size is not None and generation_config.prefill_chunk_size <= 0:
            raise ValueError("`prefill_chunk_size` must be a positive integer.")
        supported_criteria = (MaxLengthCriteria, EosTokenCriteria, StopStringCriteria)
        if any(type(criteria) not in supported_criteria for criteria in stopping_criteria):
            raise ValueError(
                "RWKV-7 CUDA graph generation supports only max length, EOS, and StopStringCriteria stopping."
            )
        max_length = next(
            criteria.max_length for criteria in stopping_criteria if isinstance(criteria, MaxLengthCriteria)
        )
        eos_token_ids = next(
            (criteria.eos_token_id for criteria in stopping_criteria if isinstance(criteria, EosTokenCriteria)), None
        )
        stop_string_criteria = tuple(
            criteria for criteria in stopping_criteria if isinstance(criteria, StopStringCriteria)
        )
        expected_processors = []
        if generation_config.do_sample:
            if generation_config.temperature is not None and generation_config.temperature != 1.0:
                expected_processors.append(TemperatureLogitsWarper)
            if generation_config.top_k not in (None, 0, -1):
                expected_processors.append(TopKLogitsWarper)
            if generation_config.top_p is not None and generation_config.top_p < 1.0:
                expected_processors.append(TopPLogitsWarper)
        if [type(processor) for processor in logits_processor] != expected_processors:
            raise ValueError(
                "RWKV-7 CUDA graph generation supports only Rapid-Sampling and its six parameters; "
                f"custom logits processors are not supported, got {[type(p).__name__ for p in logits_processor]}."
            )
        return max_length, eos_token_ids, stop_string_criteria

    def _sample(
        self: RwkvForCausalLM,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: BaseStreamer | None = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        max_length, eos_token_ids, stop_string_criteria = self._validate_graph_generation(
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            synced_gpus,
            model_kwargs,
        )
        mtp_depth = 0
        if self.mtp is not None and not generation_config.do_sample:
            mtp_depth = max(0, min(self.config.mtp_depth, max_length - input_ids.shape[1] - 1))
        use_mtp = generation_config.use_mtp is not False and mtp_depth > 0
        tokenizer = model_kwargs.get("tokenizer")
        pad_token_id = generation_config._pad_token_tensor
        key = (
            input_ids.device,
            *input_ids.shape,
            max_length,
            generation_config.prefill_chunk_size,
            generation_config.do_sample,
            getattr(generation_config, "presence_penalty", 0.0),
            getattr(generation_config, "frequency_penalty", 0.0),
            getattr(generation_config, "penalty_decay", 0.996),
            generation_config.temperature,
            generation_config.top_k,
            generation_config.top_p,
            None if pad_token_id is None else int(pad_token_id.item()),
            None if eos_token_ids is None else tuple(eos_token_ids.tolist()),
            tuple(
                (criteria.stop_strings, tokenizer if tokenizer is not None else criteria)
                for criteria in stop_string_criteria
            ),
            self.config.wkv_mode,
            mtp_depth,
            self.mtp.norm_h.eps if mtp_depth else None,
        ) + tuple(
            (
                name,
                module.get_base_layer().weight.data_ptr(),
                module.disable_adapters,
                tuple(module.active_adapters),
                tuple(module.merged_adapters),
                tuple(
                    (
                        adapter,
                        branch.weight.data_ptr(),
                        module.lora_B[adapter].weight.data_ptr(),
                        module.scaling[adapter],
                        getattr(module, "use_nora", {}).get(adapter, False),
                    )
                    for adapter, branch in module.lora_A.items()
                ),
            )
            for name, module in self.named_modules()
            if hasattr(module, "lora_A")
        )
        with torch.cuda.device(input_ids.device):
            # Captured kernels use the storage addresses recorded during capture.
            weight_layout = tuple((p.data_ptr(), p.shape, p.stride(), p.dtype, p.device) for p in self.parameters())
            cache = getattr(self, "_rwkv_generation_graphs", None)
            if cache is None:
                cache = self._rwkv_generation_graphs = _RwkvGenerationCache()
            elif self._rwkv_graph_weight_layout != weight_layout:
                cache.clear()
            self._rwkv_graph_weight_layout = weight_layout
            runner = cache.get(key)
            if runner is None:
                runner = _RwkvGenerationModelRunner(
                    self,
                    input_ids,
                    generation_config,
                    max_length,
                    eos_token_ids,
                    stop_string_criteria,
                    mtp_depth,
                )
                cache[key] = runner
            if use_mtp and runner.mtp_graph is None:
                runner.mtp_graph = _RwkvMtpDecodeGraph(
                    self, runner.decode_graph, runner.capture_stream, runner.graph_pool
                )
            seed = torch.empty((), dtype=torch.int64).random_().item()
            return runner.generate(input_ids, seed, streamer, use_mtp=use_mtp)

    _assisted_decoding = _sample
