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
from .configuration_rwkv7s import Rwkv7sConfig


if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer
    from .modeling_rwkv7s import Rwkv7sForCausalLM, Rwkv7sModel


def _rwkv7s_prefill_lengths(prompt_length: int, chunk_size: int | None) -> tuple[int, ...]:
    prefill_length = prompt_length - 1
    if prefill_length <= 0:
        return ()
    if chunk_size is None:
        return (prefill_length,)
    full_chunks, remainder = divmod(prefill_length, chunk_size)
    return (chunk_size,) * full_chunks + ((remainder,) if remainder else ())


class _Rwkv7sRosa:
    """Online ROSA (Rapid Online Suffix Automaton): longest-suffix most-recent match successor."""

    __slots__ = ("endpos", "last", "length", "link", "tokens", "transitions")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tokens: list[int] = []
        self.transitions: list[dict[int, int]] = [{}]
        self.link: list[int] = [-1]
        self.length: list[int] = [0]
        self.endpos: list[int] = [-1]
        self.last: int = 0

    def clone(self) -> _Rwkv7sRosa:
        copied = _Rwkv7sRosa.__new__(_Rwkv7sRosa)
        copied.tokens, copied.link, copied.length, copied.endpos = (
            self.tokens.copy(),
            self.link.copy(),
            self.length.copy(),
            self.endpos.copy(),
        )
        copied.transitions = [t.copy() for t in self.transitions]
        copied.last = self.last
        return copied

    def step(self, token: int, undo: list[tuple] | None = None) -> int:
        pos = len(self.tokens)
        self.tokens.append(token)
        cur = len(self.transitions)
        self.transitions.append({})
        self.length.append(self.length[self.last] + 1)
        self.link.append(-1)
        self.endpos.append(-1)

        p = self.last
        while p != -1 and token not in self.transitions[p]:
            if undo is not None:
                undo.append(("t", p, token, None))
            self.transitions[p][token] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.transitions[p][token]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = len(self.transitions)
                self.transitions.append(self.transitions[q].copy())
                self.length.append(self.length[p] + 1)
                self.link.append(self.link[q])
                self.endpos.append(self.endpos[q])
                while p != -1 and self.transitions[p].get(token) == q:
                    if undo is not None:
                        undo.append(("t", p, token, q))
                    self.transitions[p][token] = clone
                    p = self.link[p]
                if undo is not None:
                    undo.append(("l", q, self.link[q]))
                self.link[q] = self.link[cur] = clone

        self.last = cur
        link = self.link[cur]
        matched = self.tokens[self.endpos[link] + 1] if link > 0 else -1
        v = cur
        while v != -1 and self.endpos[v] < pos:
            if undo is not None:
                undo.append(("e", v, self.endpos[v]))
            self.endpos[v] = pos
            v = self.link[v]
        return matched

    def extend(self, tokens: list[int]) -> list[int]:
        return [self.step(token) for token in tokens]

    def probe(self, tokens: list[int]) -> list[int]:
        saved_tokens = len(self.tokens)
        saved_states = len(self.transitions)
        saved_last = self.last
        undo: list[tuple] = []
        matched = [self.step(token, undo) for token in tokens]
        for op in reversed(undo):
            if op[0] == "t":
                _, state, tok, prev = op
                if prev is None:
                    del self.transitions[state][tok]
                else:
                    self.transitions[state][tok] = prev
            elif op[0] == "l":
                self.link[op[1]] = op[2]
            else:
                self.endpos[op[1]] = op[2]
        del self.tokens[saved_tokens:]
        del self.transitions[saved_states:]
        del self.length[saved_states:]
        del self.link[saved_states:]
        del self.endpos[saved_states:]
        self.last = saved_last
        return matched


def compute_rwkv7s_rosa_ids(input_ids: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
    """Compute token-level ROSA successor IDs on CPU (`-1` when unmatched). Safe to run in DataLoader workers."""
    rows = input_ids.detach().cpu().tolist()
    if cu_seqlens is not None:
        bounds = cu_seqlens.detach().cpu().tolist()
        flat = rows[0] if input_ids.ndim == 2 else rows
        matched = [
            tok
            for start, end in zip(bounds[:-1], bounds[1:], strict=True)
            for tok in _Rwkv7sRosa().extend(flat[start:end])
        ]
        return torch.tensor(matched, dtype=torch.long, device=input_ids.device).view_as(input_ids)
    matched = _Rwkv7sRosa().extend(rows) if input_ids.ndim == 1 else [_Rwkv7sRosa().extend(row) for row in rows]
    return torch.tensor(matched, dtype=torch.long, device=input_ids.device)


def _recurrent_metadata(flash_rwkv2, batch_size: int, sequence_length: int, device: torch.device):
    cu_seqlens = torch.arange(0, (batch_size + 1) * sequence_length, sequence_length, dtype=torch.int32, device=device)
    state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    ticket = flash_rwkv2.prepare_tmix_wkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=batch_size * sequence_length,
        state_pool_size=batch_size,
        max_seqlen=sequence_length,
    )
    return cu_seqlens, state_indices, sequence_length, ticket


def _recurrent_forward(
    flash_rwkv2,
    wkv_mode: str,
    recurrent_state,
    receptance: torch.Tensor,
    decay_logits: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    recurrent_a: torch.Tensor,
    recurrent_b: torch.Tensor,
    decay_bias: torch.Tensor,
    inference_metadata,
) -> torch.Tensor:
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
    )


def _materialize_recurrent_state(handle, pool: torch.Tensor) -> None:
    if pool.dtype == torch.float16 and handle.memory_layout["private_bytes_per_slot"]:
        slots = torch.arange(pool.shape[0], dtype=torch.int32, device=pool.device)
        handle.materialize_slots_(slots)


def _capture_graph(device: torch.device, capture_stream: torch.cuda.Stream, graph_pool, fn, restore=None):
    graph = torch.cuda.CUDAGraph()
    generation_stream = torch.cuda.current_stream(device)
    capture_stream.wait_stream(generation_stream)
    with torch.cuda.stream(capture_stream):
        fn()
        if restore is not None:
            restore()
        with torch.cuda.graph(graph, pool=graph_pool, stream=capture_stream, capture_error_mode="thread_local"):
            fn()
        if restore is not None:
            restore()
    generation_stream.wait_stream(capture_stream)
    return graph


def _fused_lm_head(flash_rwkv2, lm_head, hidden_states: torch.Tensor) -> torch.Tensor:
    weight, lora_a, lora_b, scale, use_nora = flash_rwkv2_linear_spec(lm_head)
    return flash_rwkv2.pretrain_head_linear_bf16(
        hidden_states.contiguous(), weight, lora_a=lora_a, lora_b=lora_b, lora_scale=scale, use_nora=use_nora
    )


class _Rwkv7sGraphState:
    def __init__(
        self,
        config: Rwkv7sConfig,
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

    def layer_states(self, layer_idx: int, _hidden_states: torch.Tensor) -> tuple[torch.Tensor, object, torch.Tensor]:
        return self.attention_shift[layer_idx], self.recurrent_states[layer_idx], self.feed_forward_shift[layer_idx]

    def recurrent_metadata(
        self, flash_rwkv2, batch_size: int, sequence_length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, int, object]:
        key = (device, batch_size, sequence_length, torch.cuda.current_stream(device).cuda_stream)
        if key not in self._inference_metadata:
            self._inference_metadata[key] = _recurrent_metadata(flash_rwkv2, batch_size, sequence_length, device)
        return self._inference_metadata[key]

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

    def clone(self) -> tuple:
        return (
            self.attention_shift.clone(),
            self.feed_forward_shift.clone(),
            tuple(state.clone() for state in self.recurrent_states),
        )

    def copy_(self, snapshot: tuple) -> None:
        attention_shift, feed_forward_shift, saved_recurrent_states = snapshot
        self.attention_shift.copy_(attention_shift)
        self.feed_forward_shift.copy_(feed_forward_shift)
        for state, saved_state in zip(self.recurrent_states, saved_recurrent_states, strict=True):
            state.copy_(saved_state)

    def zero_(self) -> None:
        self.attention_shift.zero_()
        self.feed_forward_shift.zero_()
        for state in self.recurrent_states:
            state.zero_()


class _Rwkv7sPrefillGraph:
    def __init__(
        self,
        model: Rwkv7sModel,
        state: _Rwkv7sGraphState,
        batch_size: int,
        sequence_length: int,
        capture_stream: torch.cuda.Stream,
        graph_pool,
    ) -> None:
        device = model.embed_tokens.weight.device
        self.input_ids = torch.zeros((batch_size, sequence_length), dtype=torch.long, device=device)
        self.rosa_ids = torch.full((batch_size, sequence_length), -1, dtype=torch.long, device=device)
        snapshot = state.clone()
        self.graph = _capture_graph(
            device,
            capture_stream,
            graph_pool,
            lambda: model._forward_state_only(self.input_ids, None, state, False, rosa_ids=self.rosa_ids),
            lambda: state.copy_(snapshot),
        )

    def replay(self, input_ids: torch.LongTensor, rosa_ids: torch.LongTensor) -> None:
        self.input_ids.copy_(input_ids)
        self.rosa_ids.copy_(rosa_ids, non_blocking=True)
        self.graph.replay()


class _Rwkv7sDecodeGraph:
    def __init__(
        self,
        model: Rwkv7sForCausalLM,
        state: _Rwkv7sGraphState,
        batch_size: int,
        max_new_tokens: int,
        generation_config: GenerationConfig,
        use_autodeco: bool,
        eos_token_ids: torch.Tensor | None,
        stop_string_criteria: tuple[StopStringCriteria, ...],
        capture_stream: torch.cuda.Stream,
        graph_pool,
        flash_rwkv2,
    ) -> None:
        device = model.lm_head.weight.device
        self.vocab_size = model.vocab_size
        self.state = state
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.do_sample = generation_config.do_sample
        self.use_autodeco = use_autodeco
        self.autodeco_temperature = self.autodeco_top_p = None
        self.presence_penalty = getattr(generation_config, "presence_penalty", 0.0)
        self.frequency_penalty = getattr(generation_config, "frequency_penalty", 0.0)
        self.penalty_decay = getattr(generation_config, "penalty_decay", 0.996)
        self.temperature = generation_config.temperature if generation_config.temperature is not None else 1.0
        self.top_k = generation_config.top_k if generation_config.top_k not in (None, 0) else -1
        self.top_p = generation_config.top_p if generation_config.top_p is not None else 1.0
        self.flash_rwkv2 = flash_rwkv2
        self.stop_string_criteria = tuple(copy.copy(criteria) for criteria in stop_string_criteria)
        self.input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.rosa_ids = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
        self._rosa_token_cpu = torch.empty(batch_size, dtype=torch.long, pin_memory=True)
        self._rosa_next_cpu = torch.empty((batch_size, 1), dtype=torch.long, pin_memory=True)
        self.completion = torch.empty((batch_size, max_new_tokens), dtype=torch.long, device=device)
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

        state_snapshot = state.clone()
        sampling_snapshot = self.sampling_states.clone() if self.sampling_states is not None else None
        self.graph = _capture_graph(
            device,
            capture_stream,
            graph_pool,
            lambda: self._forward(model),
            lambda: self._restore_capture_inputs(state_snapshot, sampling_snapshot),
        )

    def _restore_capture_inputs(self, state_snapshot: tuple, sampling_snapshot: torch.Tensor | None) -> None:
        self.state.copy_(state_snapshot)
        if sampling_snapshot is not None:
            self.sampling_states.copy_(sampling_snapshot)
            self.penalties.zero_()
        self.input_ids.zero_()
        self.rosa_ids.fill_(-1)
        self.completion.fill_(self.pad_token_id)
        self.completion_index.zero_()
        self.completion_lengths.zero_()
        self.unfinished.fill_(True)
        self.stop_suffix.fill_(self.vocab_size)

    def _forward(self, model: Rwkv7sForCausalLM) -> None:
        hidden_states, residual, _, autodeco_hidden = model.model._forward_state_only(
            self.input_ids,
            None,
            self.state,
            False,
            return_autodeco_hidden=self.use_autodeco,
            rosa_ids=self.rosa_ids,
        )
        hidden_states = self.flash_rwkv2.infer_post_norm_output_forward_varlen(
            hidden_states,
            residual,
            model.model.norm.weight,
            model.model.norm.bias,
            eps=model.config.layer_norm_epsilon,
        )
        logits = _fused_lm_head(self.flash_rwkv2, model.lm_head, hidden_states).view(self.batch_size, model.vocab_size)
        self.logits = logits.float().contiguous()
        if self.do_sample:
            if self.use_autodeco:
                temp, top_p = model.autodeco(
                    autodeco_hidden if autodeco_hidden is not None else hidden_states, self.logits
                )
                self.autodeco_temperature = temp.float().clamp_min(1e-2).view(self.batch_size).contiguous()
                self.autodeco_top_p = top_p.float().view(self.batch_size).contiguous()
                temperature, top_p = self.autodeco_temperature, self.autodeco_top_p
            else:
                temperature, top_p = self.temperature, self.top_p
            sampled_tokens = self.flash_rwkv2.infer_sampling_six_parameter_forward_varlen(
                self.logits,
                self.penalties,
                self.sampling_states,
                self.slot_indices,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                penalty_decay=self.penalty_decay,
                temperature=temperature,
                top_k=self.top_k,
                top_p=top_p,
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
        self.state.zero_()
        self.input_ids.copy_(prompt[:, -1:])
        self.completion.fill_(self.pad_token_id)
        self.completion_index.zero_()
        self.completion_lengths.zero_()
        self.unfinished.fill_(True)
        self.stop_suffix.fill_(self.vocab_size)
        suffix = prompt[:, -self.stop_suffix.shape[1] :]
        self.stop_suffix[:, -suffix.shape[1] :].copy_(suffix)
        if self.sampling_states is not None:
            self.sampling_states.copy_(self.flash_rwkv2.setup_sampling_states(seed, self.batch_size))
            self.penalties.zero_()

    def step_rosa(self, rosa_states: list[_Rwkv7sRosa]) -> None:
        self._rosa_token_cpu.copy_(self.sampled_tokens)
        for i, (state, tok) in enumerate(zip(rosa_states, self._rosa_token_cpu.tolist(), strict=True)):
            self._rosa_next_cpu[i, 0] = state.step(tok)
        self.rosa_ids.copy_(self._rosa_next_cpu, non_blocking=True)

    def replay(self) -> None:
        self.graph.replay()


class _Rwkv7sAsyncTokenStreamer:
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


def _replay_prefill(
    prefill_graphs: dict[int, _Rwkv7sPrefillGraph],
    prefill_lengths: tuple[int, ...],
    input_ids: torch.LongTensor,
    prompt_rosa_ids: torch.LongTensor,
) -> None:
    offset = 0
    for sequence_length in prefill_lengths:
        prefill_graphs[sequence_length].replay(
            input_ids[:, offset : offset + sequence_length],
            prompt_rosa_ids[:, offset : offset + sequence_length],
        )
        offset += sequence_length


class _Rwkv7sGenerationModelRunner:
    def __init__(
        self,
        model: Rwkv7sForCausalLM,
        input_ids: torch.LongTensor,
        generation_config: GenerationConfig,
        max_length: int,
        use_autodeco: bool,
        eos_token_ids: torch.Tensor | None,
        stop_string_criteria: tuple[StopStringCriteria, ...],
    ) -> None:
        batch_size, prompt_length = input_ids.shape
        self.prefill_lengths = _rwkv7s_prefill_lengths(prompt_length, generation_config.prefill_chunk_size)
        max_new_tokens = max_length - prompt_length
        flash_rwkv2 = load_flash_rwkv2()
        self.state = _Rwkv7sGraphState(model.config, batch_size, input_ids.device, flash_rwkv2)
        capture_stream = torch.cuda.Stream(device=input_ids.device)
        graph_pool = torch.cuda.graph_pool_handle()
        original_experts_impl = model.config._experts_implementation
        if original_experts_impl in ("eager", "grouped_mm"):
            model.config._experts_implementation = "batched_mm"
        try:
            self.prefill_graphs = {
                length: _Rwkv7sPrefillGraph(model.model, self.state, batch_size, length, capture_stream, graph_pool)
                for length in dict.fromkeys(self.prefill_lengths)
            }
            self.decode_graph = _Rwkv7sDecodeGraph(
                model,
                self.state,
                batch_size,
                max_new_tokens,
                generation_config,
                use_autodeco,
                eos_token_ids,
                stop_string_criteria,
                capture_stream,
                graph_pool,
                flash_rwkv2,
            )
        finally:
            model.config._experts_implementation = original_experts_impl

    def generate(
        self, input_ids: torch.LongTensor, seed: int, streamer: BaseStreamer | None = None
    ) -> torch.LongTensor:
        self.decode_graph.prepare(input_ids, seed)
        rosa_states = [_Rwkv7sRosa() for _ in range(input_ids.shape[0])]
        prompt_rosa_ids = torch.tensor(
            [state.extend(row) for state, row in zip(rosa_states, input_ids.detach().cpu().tolist(), strict=True)],
            dtype=torch.long,
            pin_memory=True,
        )
        self.decode_graph.rosa_ids.copy_(prompt_rosa_ids[:, -1:], non_blocking=True)
        generation_stream = torch.cuda.current_stream(input_ids.device)
        async_streamer = (
            None
            if streamer is None
            else _Rwkv7sAsyncTokenStreamer(
                streamer, input_ids.shape[0], self.decode_graph.max_new_tokens, input_ids.device
            )
        )
        start_event = torch.cuda.Event(enable_timing=True)
        first_token_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(generation_stream)
        _replay_prefill(self.prefill_graphs, self.prefill_lengths, input_ids, prompt_rosa_ids)

        produced_tokens = 0
        synchronize_stopping = self.decode_graph.eos_token_ids is not None or bool(
            self.decode_graph.stop_string_criteria
        )
        for step in range(self.decode_graph.max_new_tokens):
            self.decode_graph.replay()
            if produced_tokens == 0:
                first_token_event.record(generation_stream)
            if async_streamer is not None:
                async_streamer.submit(self.decode_graph.completion, produced_tokens, generation_stream)
            produced_tokens += 1
            if synchronize_stopping and not bool(self.decode_graph.unfinished.any().item()):
                break
            if step + 1 < self.decode_graph.max_new_tokens:
                self.decode_graph.step_rosa(rosa_states)
        end_event.record(generation_stream)
        self.last_generation_events = start_event, first_token_event, end_event
        output = torch.cat((input_ids, self.decode_graph.completion[:, :produced_tokens]), dim=-1)
        if async_streamer is not None:
            async_streamer.finish()
        return output

    def reset(self) -> None:
        for prefill_graph in self.prefill_graphs.values():
            prefill_graph.graph.reset()
        self.decode_graph.graph.reset()


def _top_p_probs(
    logits: torch.Tensor,
    temperature: torch.Tensor | float,
    top_p: torch.Tensor | float,
    top_k: int,
) -> torch.Tensor:
    denom = (
        temperature.float().clamp_min(1e-2).unsqueeze(-1)
        if isinstance(temperature, torch.Tensor)
        else max(float(temperature), 1e-2)
    )
    scaled = logits.float() / denom
    if 0 < top_k < scaled.shape[-1]:
        scaled = scaled.masked_fill(scaled < torch.topk(scaled, top_k, dim=-1).values[..., -1:], float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    if isinstance(top_p, torch.Tensor) or float(top_p) < 1.0:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        p_limit = top_p.float().unsqueeze(-1) if isinstance(top_p, torch.Tensor) else float(top_p)
        keep = (sorted_probs.cumsum(dim=-1) - sorted_probs) < p_limit
        keep[..., 0] = True
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, torch.where(keep, sorted_probs, 0.0))
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs


class _Rwkv7sVerifyState:
    """Read-only view over `_Rwkv7sGraphState` using 1 shared scratch layer + BF16 WKV inputs for varlen commit."""

    def __init__(
        self,
        main_state: _Rwkv7sGraphState,
        config: Rwkv7sConfig,
        batch_size: int,
        width: int,
        device: torch.device,
        flash_rwkv2,
    ) -> None:
        self.main_state = main_state
        self.wkv_mode = config.wkv_mode
        self.batch_size = batch_size
        self.width = width
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.head_size = config.head_size
        self._active_layer = 0

        self.scratch_attn_shift = torch.empty((batch_size, config.hidden_size), dtype=torch.bfloat16, device=device)
        self.scratch_ffn_shift = torch.empty_like(self.scratch_attn_shift)
        state_dtype = torch.float16 if self.wkv_mode == "fp16" else torch.float32
        self._scratch_pool = torch.empty(
            (batch_size, self.num_heads, self.head_size, self.head_size), dtype=state_dtype, device=device
        )
        prepare_state = getattr(flash_rwkv2, f"prepare_tmix_wkv7_recurrent_{self.wkv_mode}_state_from_tensor")
        self.scratch_wkv = prepare_state(self._scratch_pool)

        self.saved_wkv = torch.empty(
            (self.num_layers, 5, batch_size, width, self.num_heads, self.head_size),
            dtype=torch.bfloat16,
            device=device,
        )
        self.saved_tmix_norm = torch.empty(
            (self.num_layers, batch_size * width, config.hidden_size), dtype=torch.bfloat16, device=device
        )
        self.saved_cmix_norm = torch.empty(
            (self.num_layers, batch_size, width, config.hidden_size), dtype=torch.bfloat16, device=device
        )
        self._dummy_r = torch.zeros(
            (batch_size * width, self.num_heads, self.head_size), dtype=torch.bfloat16, device=device
        )
        self._state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        self._verify_metadata = None

    def layer_states(self, layer_idx: int, _hidden_states: torch.Tensor) -> tuple[torch.Tensor, object, torch.Tensor]:
        self._active_layer = layer_idx
        self.scratch_attn_shift.copy_(self.main_state.attention_shift[layer_idx])
        self.scratch_ffn_shift.copy_(self.main_state.feed_forward_shift[layer_idx])
        self.scratch_wkv.copy_(self.main_state.recurrent_states[layer_idx])
        return self.scratch_attn_shift, self.scratch_wkv, self.scratch_ffn_shift

    def recurrent_metadata(self, flash_rwkv2, batch_size: int, sequence_length: int, device: torch.device):
        if self._verify_metadata is None:
            self._verify_metadata = _recurrent_metadata(flash_rwkv2, batch_size, sequence_length, device)
        return self._verify_metadata

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
        slot = self.saved_wkv[self._active_layer]
        shape = (self.batch_size, self.width, self.num_heads, self.head_size)
        for idx, tensor in enumerate((decay_logits, key, value, recurrent_a, recurrent_b)):
            slot[idx].copy_(tensor.view(shape))
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

    def commit(self, flash_rwkv2, model: Rwkv7sModel, prefixes: list[int]) -> None:
        device = self.saved_wkv.device
        batch_size, width = self.batch_size, self.width
        last_pos = torch.tensor([p - 1 for p in prefixes], dtype=torch.long, device=device)
        rows = torch.arange(batch_size, device=device)
        self.main_state.attention_shift.copy_(
            self.saved_tmix_norm.view(self.num_layers, batch_size, width, -1)[:, rows, last_pos]
        )
        self.main_state.feed_forward_shift.copy_(self.saved_cmix_norm[:, rows, last_pos])

        offsets = [0]
        gather_indices: list[int] = []
        for row_idx, length in enumerate(prefixes):
            offsets.append(offsets[-1] + length)
            gather_indices.extend(range(row_idx * width, row_idx * width + length))
        total, max_len = offsets[-1], max(prefixes)
        cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
        ticket = flash_rwkv2.prepare_tmix_wkv7_recurrent_metadata(
            cu_seqlens, self._state_indices, total_tokens=total, state_pool_size=batch_size, max_seqlen=max_len
        )
        metadata = (cu_seqlens, self._state_indices, max_len, ticket)
        flat_idx = torch.tensor(gather_indices, dtype=torch.long, device=device)
        dummy_r, all_full = self._dummy_r[:total], total == batch_size * width
        for layer_idx, layer in enumerate(model.layers):
            slot = self.saved_wkv[layer_idx].view(5, batch_size * width, self.num_heads, self.head_size)
            packed = slot if all_full else slot[:, flat_idx]
            _recurrent_forward(
                flash_rwkv2,
                self.wkv_mode,
                self.main_state.recurrent_states[layer_idx],
                dummy_r,
                *(packed[i].contiguous() for i in range(5)),
                layer.linear_attn.w0,
                metadata,
            )


class _Rwkv7sMtpDraftGraph:
    """CUDA-graphed greedy LightMTP chain with O(B * V) transient logits (no [B, depth, V] buffer)."""

    def __init__(self, model: Rwkv7sForCausalLM, batch_size: int, depth: int, capture_stream, graph_pool, flash_rwkv2):
        device = model.lm_head.weight.device
        self.hidden = torch.zeros((batch_size, model.config.hidden_size), dtype=torch.bfloat16, device=device)
        self.tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.drafts = torch.zeros((batch_size, depth), dtype=torch.long, device=device)

        def forward() -> None:
            h, tok = self.hidden, self.tokens
            for step in range(depth):
                h = model.mtp(h, model.model.embed_tokens(tok))
                tok = _fused_lm_head(flash_rwkv2, model.lm_head, model.mtp.norm_out(h)).view(batch_size, -1).argmax(-1)
                self.drafts[:, step].copy_(tok)

        self.graph = _capture_graph(device, capture_stream, graph_pool, forward)

    def replay(self, hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        self.hidden.copy_(hidden)
        self.tokens.copy_(tokens)
        self.graph.replay()
        return self.drafts


class _Rwkv7sMtpVerifyGraph:
    """CUDA-graphed verify pass writing only lightweight BF16 WKV inputs without mutating `main_state`."""

    def __init__(
        self,
        model: Rwkv7sForCausalLM,
        verify_state: _Rwkv7sVerifyState,
        batch_size: int,
        width: int,
        use_autodeco: bool,
        capture_stream,
        graph_pool,
        flash_rwkv2,
    ) -> None:
        device = model.lm_head.weight.device
        self.input_ids = torch.zeros((batch_size, width), dtype=torch.long, device=device)
        self.rosa_ids = torch.full((batch_size, width), -1, dtype=torch.long, device=device)
        self.hidden_states = self.logits = self.autodeco_temperature = self.autodeco_top_p = None

        def forward() -> None:
            hidden_states, residual, _, autodeco_hidden = model.model._forward_state_only(
                self.input_ids, None, verify_state, False, return_autodeco_hidden=use_autodeco, rosa_ids=self.rosa_ids
            )
            hidden_states = flash_rwkv2.infer_post_norm_output_forward_varlen(
                hidden_states,
                residual,
                model.model.norm.weight,
                model.model.norm.bias,
                eps=model.config.layer_norm_epsilon,
            )
            self.hidden_states = hidden_states.view(batch_size, width, -1)
            self.logits = (
                _fused_lm_head(flash_rwkv2, model.lm_head, hidden_states)
                .view(batch_size, width, -1)
                .float()
                .contiguous()
            )
            if use_autodeco:
                prev = autodeco_hidden if autodeco_hidden is not None else hidden_states
                temp, top_p = model.autodeco(prev.view(batch_size, width, -1), self.logits)
                self.autodeco_temperature = temp.float().clamp_min(1e-2).view(batch_size, width).contiguous()
                self.autodeco_top_p = top_p.float().view(batch_size, width).contiguous()

        self.graph = _capture_graph(device, capture_stream, graph_pool, forward)

    def replay(self, input_ids: torch.Tensor, rosa_ids: torch.Tensor) -> None:
        self.input_ids.copy_(input_ids)
        self.rosa_ids.copy_(rosa_ids, non_blocking=True)
        self.graph.replay()


class _Rwkv7sMtpGenerationRunner:
    """Batch MTP speculative decoding with 1-layer scratch verify + varlen WKV commit."""

    def __init__(
        self,
        model: Rwkv7sForCausalLM,
        input_ids: torch.LongTensor,
        generation_config: GenerationConfig,
        max_length: int,
        use_autodeco: bool,
        eos_token_ids: torch.Tensor | None,
        stop_string_criteria: tuple[StopStringCriteria, ...],
    ) -> None:
        batch_size, prompt_length = input_ids.shape
        self.model = model
        self.batch_size = batch_size
        self.depth = model.config.mtp_depth
        self.width = self.depth + 1
        self.max_new_tokens = max_length - prompt_length
        self.do_sample = generation_config.do_sample
        self.use_autodeco = use_autodeco
        self.temperature = generation_config.temperature if generation_config.temperature is not None else 1.0
        self.top_k = generation_config.top_k if generation_config.top_k not in (None, 0) else -1
        self.top_p = generation_config.top_p if generation_config.top_p is not None else 1.0
        self.pad_token_id = (
            int(generation_config._pad_token_tensor.item()) if generation_config._pad_token_tensor is not None else 0
        )
        self.eos_set = set() if eos_token_ids is None else set(eos_token_ids.tolist())
        self.stop_string_criteria = tuple(copy.copy(criteria) for criteria in stop_string_criteria)
        self.prefill_lengths = _rwkv7s_prefill_lengths(prompt_length, generation_config.prefill_chunk_size)
        self.flash_rwkv2 = load_flash_rwkv2()
        device = input_ids.device
        self.state = _Rwkv7sGraphState(model.config, batch_size, device, self.flash_rwkv2)
        self.verify_state = _Rwkv7sVerifyState(
            self.state, model.config, batch_size, self.width, device, self.flash_rwkv2
        )
        capture_stream = torch.cuda.Stream(device=device)
        graph_pool = torch.cuda.graph_pool_handle()
        original_experts_impl = model.config._experts_implementation
        if original_experts_impl in ("eager", "grouped_mm"):
            model.config._experts_implementation = "batched_mm"
        try:
            self.prefill_graphs = {
                length: _Rwkv7sPrefillGraph(model.model, self.state, batch_size, length, capture_stream, graph_pool)
                for length in dict.fromkeys(self.prefill_lengths)
            }
            self.draft_graph = _Rwkv7sMtpDraftGraph(
                model, batch_size, self.depth, capture_stream, graph_pool, self.flash_rwkv2
            )
            self.verify_graph = _Rwkv7sMtpVerifyGraph(
                model,
                self.verify_state,
                batch_size,
                self.width,
                use_autodeco,
                capture_stream,
                graph_pool,
                self.flash_rwkv2,
            )
        finally:
            model.config._experts_implementation = original_experts_impl
        self.total_rounds = self.total_accepted = 0

    def _emit_token(
        self, row: int, tok: int, completion, cursors: list[int], unfinished: list[bool], histories, prompt_tails
    ) -> None:
        if not unfinished[row] or cursors[row] >= self.max_new_tokens:
            return
        completion[row, cursors[row]] = tok
        cursors[row] += 1
        histories[row].append(tok)
        stop_hit = bool(self.stop_string_criteria) and any(
            bool(c(torch.tensor([prompt_tails[row] + histories[row]], dtype=torch.long), None).item())
            for c in self.stop_string_criteria
        )
        if cursors[row] >= self.max_new_tokens or tok in self.eos_set or stop_hit:
            unfinished[row] = False

    def _select_tokens(self, verify_logits, verify_rows, drafts, generator) -> tuple[list[int], list[int]]:
        if not self.do_sample:
            preds = verify_logits.argmax(dim=-1).tolist()
            hits = [
                next((s for s in range(self.depth) if preds[r][s] != verify_rows[r][s + 1]), self.depth)
                for r in range(self.batch_size)
            ]
            return hits, [preds[r][hits[r]] for r in range(self.batch_size)]

        temp = self.verify_graph.autodeco_temperature if self.use_autodeco else self.temperature
        top_p = self.verify_graph.autodeco_top_p if self.use_autodeco else self.top_p
        probs = _top_p_probs(verify_logits, temp, top_p, self.top_k)
        p_draft = probs[:, : self.depth, :].gather(-1, drafts.unsqueeze(-1)).squeeze(-1)
        accept = (
            torch.rand((self.batch_size, self.depth), device=probs.device, generator=generator) < p_draft
        ).tolist()
        hits: list[int] = []
        bonuses: list[int] = []
        for row in range(self.batch_size):
            hit = next((s for s, ok in enumerate(accept[row]) if not ok), self.depth)
            hits.append(hit)
            dist = probs[row, hit]
            if hit < self.depth:
                dist = dist.clone()
                dist[verify_rows[row][hit + 1]] = 0.0
                mass = dist.sum()
                dist = dist / mass if mass > 0 else probs[row, hit]
            bonuses.append(int(torch.multinomial(dist, 1, generator=generator).item()))
        return hits, bonuses

    def generate(
        self, input_ids: torch.LongTensor, seed: int, streamer: BaseStreamer | None = None
    ) -> torch.LongTensor:
        device, batch_size = input_ids.device, self.batch_size
        self.state.zero_()
        rosa_states = [_Rwkv7sRosa() for _ in range(batch_size)]
        prompt_rows = input_ids.detach().cpu().tolist()
        prompt_rosa_ids = torch.tensor(
            [state.extend(row) for state, row in zip(rosa_states, prompt_rows, strict=True)],
            dtype=torch.long,
            pin_memory=True,
        )
        suffix_len = max((c.maximum_token_len for c in self.stop_string_criteria), default=0)
        prompt_tails = [row[-suffix_len:] if suffix_len else [] for row in prompt_rows]

        generation_stream = torch.cuda.current_stream(device)
        async_streamer = (
            None if streamer is None else _Rwkv7sAsyncTokenStreamer(streamer, batch_size, self.max_new_tokens, device)
        )
        start_event, first_token_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        start_event.record(generation_stream)
        _replay_prefill(self.prefill_graphs, self.prefill_lengths, input_ids, prompt_rosa_ids)

        hidden, residual, _, autodeco_hidden = self.model.model._forward_state_only(
            input_ids[:, -1:],
            None,
            self.state,
            False,
            return_autodeco_hidden=self.use_autodeco,
            rosa_ids=prompt_rosa_ids[:, -1:].to(device=device, non_blocking=True),
        )
        hidden = self.flash_rwkv2.infer_post_norm_output_forward_varlen(
            hidden,
            residual,
            self.model.model.norm.weight,
            self.model.model.norm.bias,
            eps=self.model.config.layer_norm_epsilon,
        ).view(batch_size, -1)
        first_logits = _fused_lm_head(self.flash_rwkv2, self.model.lm_head, hidden).view(batch_size, -1).float()
        generator = torch.Generator(device=device).manual_seed(seed) if self.do_sample else None
        if self.do_sample:
            if self.use_autodeco:
                prev = autodeco_hidden if autodeco_hidden is not None else hidden
                temp, top_p = self.model.autodeco(prev.view(batch_size, -1), first_logits)
                first_probs = _top_p_probs(first_logits, temp.view(batch_size), top_p.view(batch_size), self.top_k)
            else:
                first_probs = _top_p_probs(first_logits, self.temperature, self.top_p, self.top_k)
            cur_tokens = torch.multinomial(first_probs, 1, generator=generator).squeeze(-1)
        else:
            cur_tokens = first_logits.argmax(dim=-1)
        first_token_event.record(generation_stream)

        completion = torch.full((batch_size, self.max_new_tokens), self.pad_token_id, dtype=torch.long, device=device)
        cursors, unfinished = [0] * batch_size, [self.max_new_tokens > 0] * batch_size
        histories: list[list[int]] = [[] for _ in range(batch_size)]
        for row, tok in enumerate(cur_tokens.tolist()):
            self._emit_token(row, tok, completion, cursors, unfinished, histories, prompt_tails)

        streamed = 0
        if async_streamer is not None and any(cursors):
            async_streamer.submit(completion, 0, generation_stream)
            streamed = 1

        cur_hidden = hidden
        verify_ids = torch.empty((batch_size, self.width), dtype=torch.long, device=device)
        rows_idx = torch.arange(batch_size, device=device)
        while any(unfinished):
            drafts = self.draft_graph.replay(cur_hidden, cur_tokens)
            verify_ids[:, 0].copy_(cur_tokens)
            verify_ids[:, 1:].copy_(drafts)
            verify_rows = verify_ids.detach().cpu().tolist()
            probe_rosa = torch.tensor(
                [state.probe(row) for state, row in zip(rosa_states, verify_rows, strict=True)],
                dtype=torch.long,
                pin_memory=True,
            )
            self.verify_graph.replay(verify_ids, probe_rosa)
            hits, bonuses = self._select_tokens(self.verify_graph.logits, verify_rows, drafts, generator)

            prefixes = [hit + 1 if unfinished[row] else 1 for row, hit in enumerate(hits)]
            self.verify_state.commit(self.flash_rwkv2, self.model.model, prefixes)
            for row, prefix in enumerate(prefixes):
                rosa_states[row].extend(verify_rows[row][:prefix])

            self.total_rounds += sum(unfinished)
            self.total_accepted += sum(hit for row, hit in enumerate(hits) if unfinished[row])
            for row, hit in enumerate(hits):
                if unfinished[row]:
                    for tok in (*verify_rows[row][1 : hit + 1], bonuses[row]):
                        self._emit_token(row, tok, completion, cursors, unfinished, histories, prompt_tails)

            if async_streamer is not None:
                ready = min((cursors[r] for r in range(batch_size) if unfinished[r]), default=max(cursors, default=0))
                while streamed < ready:
                    async_streamer.submit(completion, streamed, generation_stream)
                    streamed += 1

            hit_tensor = torch.tensor(
                [h if unfinished[r] else 0 for r, h in enumerate(hits)], dtype=torch.long, device=device
            )
            cur_hidden = self.verify_graph.hidden_states[rows_idx, hit_tensor]
            cur_tokens = torch.tensor(bonuses, dtype=torch.long, device=device)

        produced_tokens = max(cursors, default=0)
        if async_streamer is not None:
            while streamed < produced_tokens:
                async_streamer.submit(completion, streamed, generation_stream)
                streamed += 1
        end_event.record(generation_stream)
        self.last_generation_events = start_event, first_token_event, end_event
        output = torch.cat((input_ids, completion[:, :produced_tokens]), dim=-1)
        if async_streamer is not None:
            async_streamer.finish()
        return output

    def reset(self) -> None:
        for prefill_graph in self.prefill_graphs.values():
            prefill_graph.graph.reset()
        self.draft_graph.graph.reset()
        self.verify_graph.graph.reset()


class _Rwkv7sGenerationCache(dict[tuple, _Rwkv7sGenerationModelRunner | _Rwkv7sMtpGenerationRunner]):
    """Owns RWKV generation runners and releases their CUDA graphs before the tensors they reference."""

    def clear(self) -> None:
        for runner in self.values():
            runner.reset()
        super().clear()


class Rwkv7sGenerationMixin(GenerationMixin):
    _supported_generation_modes = [
        GenerationMode.GREEDY_SEARCH,
        GenerationMode.SAMPLE,
        GenerationMode.ASSISTED_GENERATION,
    ]

    def adjust_generation_fn(
        self,
        generation_config,
        from_auto_class,
        from_pipeline,
        pretrained_model_name_or_path,
        cache_dir,
        force_download,
        proxies,
        local_files_only,
        token,
        revision,
        subfolder,
        trust_remote_code,
        **kwargs,
    ):
        super().adjust_generation_fn(
            generation_config,
            from_auto_class,
            from_pipeline,
            pretrained_model_name_or_path,
            cache_dir,
            force_download,
            proxies,
            local_files_only,
            token,
            revision,
            subfolder,
            trust_remote_code,
            **kwargs,
        )
        self.generation_config.presence_penalty = getattr(self.generation_config, "presence_penalty", 0.0)
        self.generation_config.frequency_penalty = getattr(self.generation_config, "frequency_penalty", 0.0)
        self.generation_config.penalty_decay = getattr(self.generation_config, "penalty_decay", 0.996)
        self.generation_config.use_autodeco = getattr(self.generation_config, "use_autodeco", True)

    def _validate_generation_mode(self, generation_mode, generation_config, generation_mode_kwargs):
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if (
                not generation_config.use_mtp
                or generation_mode_kwargs.get("assistant_model") is not None
                or generation_config.prompt_lookup_num_tokens is not None
                or generation_config.assistant_early_exit is not None
            ):
                raise ValueError("RWKV-7S assisted generation supports only built-in LightMTP (`use_mtp=True`).")
            base_mode = GenerationMode.SAMPLE if generation_config.do_sample else GenerationMode.GREEDY_SEARCH
            super()._validate_generation_mode(base_mode, generation_config, generation_mode_kwargs)
        else:
            super()._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)
        if generation_config.token_healing:
            raise ValueError("RWKV-7S CUDA graph generation does not support token healing.")
        if generation_config.cache_implementation is not None:
            raise ValueError("RWKV-7S CUDA graph generation does not support cache_implementation.")
        if generation_config.compile_config is not None:
            raise ValueError("RWKV-7S CUDA graph generation does not support compile_config.")
        if generation_config.continuous_batching_config is not None:
            raise ValueError("RWKV-7S CUDA graph generation does not support continuous batching.")

    def _get_logits_processor(self, generation_config, *args, **kwargs):
        if generation_config.top_k == -1:
            generation_config = copy.copy(generation_config)
            generation_config.top_k = 0
        return super()._get_logits_processor(generation_config, *args, **kwargs)

    @staticmethod
    def _validate_graph_logits_processors(
        logits_processor: LogitsProcessorList, generation_config: GenerationConfig
    ) -> None:
        expected = []
        if generation_config.do_sample:
            if generation_config.temperature is not None and generation_config.temperature != 1.0:
                expected.append(TemperatureLogitsWarper)
            if generation_config.top_k not in (None, 0, -1):
                expected.append(TopKLogitsWarper)
            if generation_config.top_p is not None and generation_config.top_p < 1.0:
                expected.append(TopPLogitsWarper)
        if [type(processor) for processor in logits_processor] != expected:
            raise ValueError(
                "RWKV-7S CUDA graph generation supports only Rapid-Sampling and its six parameters; "
                f"custom logits processors are not supported, got {[type(p).__name__ for p in logits_processor]}."
            )

    def _validate_graph_generation(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        model_kwargs: dict,
    ) -> tuple[int, torch.Tensor | None, tuple[StopStringCriteria, ...], bool, bool]:
        if flash_rwkv2_linear_spec(self.lm_head) is None or not self.model._supports_fused_projections():
            raise ValueError("RWKV-7S generation requires fused projections. Merge unsupported PEFT adapters first.")
        if input_ids.device.type != "cuda" or self.lm_head.weight.device != input_ids.device:
            raise ValueError("RWKV-7S generation requires the model and a static input batch on one CUDA device.")
        if self.training or self.lm_head.weight.dtype != torch.bfloat16:
            raise ValueError("RWKV-7S generation requires an eval-mode bfloat16 model.")
        attention_mask = model_kwargs.get("attention_mask")
        if attention_mask is not None and (
            attention_mask.shape != input_ids.shape or not bool(torch.all(attention_mask != 0))
        ):
            raise ValueError("RWKV-7S does not support padding or ragged batches; bucket inputs by sequence length.")
        if synced_gpus:
            raise ValueError("RWKV-7S CUDA graph generation does not support synced_gpus.")
        if not generation_config.use_cache:
            raise ValueError("RWKV-7S CUDA graph generation requires use_cache=True.")
        if generation_config.num_return_sequences != 1:
            raise ValueError("RWKV-7S CUDA graph generation requires num_return_sequences=1.")
        if (
            generation_config.return_dict_in_generate
            or generation_config.output_attentions
            or generation_config.output_hidden_states
            or generation_config.output_scores
            or generation_config.output_logits
        ):
            raise ValueError("RWKV-7S CUDA graph generation does not support detailed generation outputs.")
        if model_kwargs.get("cache") is not None or model_kwargs.get("inputs_embeds") is not None:
            raise ValueError("RWKV-7S CUDA graph generation does not accept an external cache or inputs_embeds.")
        unsupported_kwargs = {
            name
            for name, value in model_kwargs.items()
            if value is not None
            and name not in {"attention_mask", "logits_to_keep", "position_ids", "tokenizer", "use_cache"}
        }
        if unsupported_kwargs:
            raise ValueError(
                f"RWKV-7S CUDA graph generation does not support model kwargs {sorted(unsupported_kwargs)}."
            )
        use_autodeco = generation_config.do_sample and bool(getattr(generation_config, "use_autodeco", True))
        has_penalty = (
            getattr(generation_config, "presence_penalty", 0.0) != 0.0
            or getattr(generation_config, "frequency_penalty", 0.0) != 0.0
        )
        use_mtp = bool(generation_config.use_mtp) and not (generation_config.do_sample and has_penalty)
        if generation_config.prefill_chunk_size is not None and generation_config.prefill_chunk_size <= 0:
            raise ValueError("`prefill_chunk_size` must be a positive integer.")
        supported_criteria = (MaxLengthCriteria, EosTokenCriteria, StopStringCriteria)
        if any(type(criteria) not in supported_criteria for criteria in stopping_criteria):
            raise ValueError(
                "RWKV-7S CUDA graph generation supports only max length, EOS, and StopStringCriteria stopping."
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
        self._validate_graph_logits_processors(logits_processor, generation_config)
        return max_length, eos_token_ids, stop_string_criteria, use_autodeco, use_mtp

    @staticmethod
    def _graph_generation_key(
        input_ids: torch.LongTensor,
        generation_config: GenerationConfig,
        max_length: int,
        eos_token_ids: torch.Tensor | None,
        stop_string_criteria: tuple[StopStringCriteria, ...],
        tokenizer,
        wkv_mode: str,
        use_autodeco: bool,
        use_mtp: bool,
    ) -> tuple:
        eos_key = None if eos_token_ids is None else tuple(eos_token_ids.tolist())
        pad_token_id = generation_config._pad_token_tensor
        pad_key = None if pad_token_id is None else int(pad_token_id.item())
        stop_key = tuple(
            (criteria.stop_strings, tokenizer if tokenizer is not None else criteria)
            for criteria in stop_string_criteria
        )
        return (
            input_ids.device,
            *input_ids.shape,
            max_length,
            generation_config.prefill_chunk_size,
            generation_config.do_sample,
            use_autodeco,
            use_mtp,
            getattr(generation_config, "presence_penalty", 0.0),
            getattr(generation_config, "frequency_penalty", 0.0),
            getattr(generation_config, "penalty_decay", 0.996),
            generation_config.temperature,
            generation_config.top_k,
            generation_config.top_p,
            pad_key,
            eos_key,
            stop_key,
            wkv_mode,
        )

    def _assisted_decoding(
        self: Rwkv7sForCausalLM,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: BaseStreamer | None = None,
        inputs_tensor: torch.Tensor | None = None,
        assistant_model=None,
        assistant_tokenizer=None,
        tokenizer=None,
        **model_kwargs,
    ) -> torch.LongTensor:
        del inputs_tensor, assistant_model, assistant_tokenizer
        if tokenizer is not None:
            model_kwargs["tokenizer"] = tokenizer
        return self._sample(
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    def _sample(
        self: Rwkv7sForCausalLM,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: BaseStreamer | None = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        max_length, eos_token_ids, stop_string_criteria, use_autodeco, use_mtp = self._validate_graph_generation(
            input_ids,
            logits_processor,
            stopping_criteria,
            generation_config,
            synced_gpus,
            model_kwargs,
        )
        key = self._graph_generation_key(
            input_ids,
            generation_config,
            max_length,
            eos_token_ids,
            stop_string_criteria,
            model_kwargs.get("tokenizer"),
            self.config.wkv_mode,
            use_autodeco,
            use_mtp,
        )
        key += tuple(
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
            cache = getattr(self, "_rwkv7s_generation_graphs", None)
            if cache is None:
                cache = self._rwkv7s_generation_graphs = _Rwkv7sGenerationCache()
            elif self._rwkv7s_graph_weight_layout != weight_layout:
                cache.clear()
            self._rwkv7s_graph_weight_layout = weight_layout
            runner = cache.get(key)
            if runner is None:
                runner_cls = _Rwkv7sMtpGenerationRunner if use_mtp else _Rwkv7sGenerationModelRunner
                runner = runner_cls(
                    self,
                    input_ids,
                    generation_config,
                    max_length,
                    use_autodeco,
                    eos_token_ids,
                    stop_string_criteria,
                )
                cache[key] = runner
            seed = torch.empty((), dtype=torch.int64).random_().item()
            return runner.generate(input_ids, seed, streamer)
