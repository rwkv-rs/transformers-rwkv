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
from ...integrations.flash_rwkv2 import load_flash_rwkv2
from .configuration_rwkv import RwkvConfig


if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer
    from .modeling_rwkv import RwkvForCausalLM, RwkvModel


def _rwkv_prefill_lengths(prompt_length: int, chunk_size: int | None) -> tuple[int, ...]:
    prefill_length = prompt_length - 1
    if prefill_length <= 0:
        return ()
    if chunk_size is None:
        return (prefill_length,)
    full_chunks, remainder = divmod(prefill_length, chunk_size)
    return (chunk_size,) * full_chunks + ((remainder,) if remainder else ())


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
        self.attention_shift = torch.zeros(shift_shape, dtype=torch.float16, device=device)
        self.feed_forward_shift = torch.zeros_like(self.attention_shift)
        prepare_state = getattr(flash_rwkv2, f"prepare_tmix_wkv7_recurrent_{self.wkv_mode}_state")
        self.recurrent_states = [
            prepare_state(
                batch_size,
                config.hidden_size,
                sequence_capacity=batch_size,
                head_size=config.head_size,
                device=device,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self._inference_metadata: dict[tuple, tuple[torch.Tensor, torch.Tensor, int, object]] = {}

    def layer_states(self, layer_idx: int, _hidden_states: torch.Tensor) -> tuple[torch.Tensor, object, torch.Tensor]:
        return self.attention_shift[layer_idx], self.recurrent_states[layer_idx], self.feed_forward_shift[layer_idx]

    def recurrent_metadata(
        self, flash_rwkv2, batch_size: int, sequence_length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, int, object]:
        stream = torch.cuda.current_stream(device)
        key = (device, batch_size, sequence_length, stream.cuda_stream)
        if key not in self._inference_metadata:
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * sequence_length,
                sequence_length,
                dtype=torch.int32,
                device=device,
            )
            state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
            ticket = flash_rwkv2.prepare_tmix_wkv7_recurrent_metadata(
                cu_seqlens,
                state_indices,
                total_tokens=batch_size * sequence_length,
                state_pool_size=batch_size,
                max_seqlen=sequence_length,
            )
            self._inference_metadata[key] = (cu_seqlens, state_indices, sequence_length, ticket)
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
        inference_metadata: tuple[torch.Tensor, torch.Tensor, int, object],
    ) -> torch.Tensor:
        cu_seqlens, state_indices, max_seqlen, ticket = inference_metadata
        recurrent_forward = getattr(flash_rwkv2, f"infer_tmix_wkv7_recurrent_{self.wkv_mode}_forward_varlen")
        return recurrent_forward(
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
        self.graph = torch.cuda.CUDAGraph()
        snapshot = state.clone()
        generation_stream = torch.cuda.current_stream(self.input_ids.device)
        capture_stream.wait_stream(generation_stream)
        with torch.cuda.stream(capture_stream):
            model._forward_state_only(self.input_ids, None, state, False)
            state.copy_(snapshot)
            with torch.cuda.graph(
                self.graph,
                pool=graph_pool,
                stream=capture_stream,
                capture_error_mode="thread_local",
            ):
                model._forward_state_only(self.input_ids, None, state, False)
            state.copy_(snapshot)
        generation_stream.wait_stream(capture_stream)

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
    ) -> None:
        device = model.lm_head.weight.device
        self.vocab_size = model.vocab_size
        self.state = state
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.do_sample = generation_config.do_sample
        self.temperature = generation_config.temperature if generation_config.temperature is not None else 1.0
        self.top_k = generation_config.top_k if generation_config.top_k not in (None, 0) else -1
        self.top_p = generation_config.top_p if generation_config.top_p is not None else 1.0
        self.flash_rwkv2 = flash_rwkv2
        self.stop_string_criteria = tuple(copy.copy(criteria) for criteria in stop_string_criteria)
        self.input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
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
        for criteria in self.stop_string_criteria:
            criteria(self.stop_suffix, None)

        self.graph = torch.cuda.CUDAGraph()
        state_snapshot = state.clone()
        sampling_snapshot = self.sampling_states.clone() if self.sampling_states is not None else None
        generation_stream = torch.cuda.current_stream(device)
        capture_stream.wait_stream(generation_stream)
        with torch.cuda.stream(capture_stream):
            self._forward(model)
            self._restore_capture_inputs(state_snapshot, sampling_snapshot)
            with torch.cuda.graph(
                self.graph,
                pool=graph_pool,
                stream=capture_stream,
                capture_error_mode="thread_local",
            ):
                self._forward(model)
            self._restore_capture_inputs(state_snapshot, sampling_snapshot)
        generation_stream.wait_stream(capture_stream)

    def _restore_capture_inputs(self, state_snapshot: tuple, sampling_snapshot: torch.Tensor | None) -> None:
        self.state.copy_(state_snapshot)
        if sampling_snapshot is not None:
            self.sampling_states.copy_(sampling_snapshot)
        self.input_ids.zero_()
        self.completion.fill_(self.pad_token_id)
        self.completion_index.zero_()
        self.completion_lengths.zero_()
        self.unfinished.fill_(True)
        self.stop_suffix.fill_(self.vocab_size)

    def _forward(self, model: RwkvForCausalLM) -> None:
        hidden_states, residual, _ = model.model._forward_state_only(self.input_ids, None, self.state, False)
        hidden_states = self.flash_rwkv2.infer_post_norm_output_forward_varlen(
            hidden_states,
            residual,
            model.model.norm.weight,
            model.model.norm.bias,
            eps=model.config.layer_norm_epsilon,
        )
        logits = self.flash_rwkv2.infer_head_linear_last_forward_varlen(
            hidden_states,
            model.lm_head.weight,
            tokens_count=self.batch_size,
        ).view(self.batch_size, model.vocab_size)
        self.logits = logits.float().contiguous()
        if self.do_sample:
            sampled_tokens = self.flash_rwkv2.infer_sampling_temperature_topk_topp_forward_varlen(
                self.logits,
                self.sampling_states,
                self.slot_indices,
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

    def replay(self) -> None:
        self.graph.replay()


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
    ) -> None:
        batch_size, prompt_length = input_ids.shape
        self.prefill_lengths = _rwkv_prefill_lengths(prompt_length, generation_config.prefill_chunk_size)
        max_new_tokens = max_length - prompt_length
        operators = [
            "infer_head_linear_last_forward_varlen",
            "infer_post_norm_output_forward_varlen",
            "prepare_tmix_wkv7_recurrent_metadata",
        ]
        wkv_mode = model._rwkv_wkv_mode
        operators.extend(
            (
                f"infer_tmix_wkv7_recurrent_{wkv_mode}_forward_varlen",
                f"prepare_tmix_wkv7_recurrent_{wkv_mode}_state",
            )
        )
        if generation_config.do_sample:
            operators.extend(("infer_sampling_temperature_topk_topp_forward_varlen", "setup_sampling_states"))
        flash_rwkv2 = load_flash_rwkv2(tuple(operators), model.lm_head.weight, "CUDA graph generation")
        self.state = _RwkvGraphState(
            model.config,
            batch_size,
            input_ids.device,
            flash_rwkv2,
        )
        capture_stream = torch.cuda.Stream(device=input_ids.device)
        graph_pool = torch.cuda.graph_pool_handle()
        self.prefill_graphs = {
            sequence_length: _RwkvPrefillGraph(
                model.model,
                self.state,
                batch_size,
                sequence_length,
                capture_stream,
                graph_pool,
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
            capture_stream,
            graph_pool,
            flash_rwkv2,
        )

    def generate(
        self, input_ids: torch.LongTensor, seed: int, streamer: BaseStreamer | None = None
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

    def _get_rwkv_generation_cache(self) -> _RwkvGenerationCache:
        cache = getattr(self, "_rwkv_generation_graphs", None)
        if cache is None:
            cache = _RwkvGenerationCache()
            self._rwkv_generation_graphs = cache
        return cache

    def _clear_rwkv_generation_cache(self) -> None:
        cache = getattr(self, "_rwkv_generation_graphs", None)
        if cache is not None:
            cache.clear()

    def _validate_generation_mode(self, generation_mode, generation_config, generation_mode_kwargs):
        super()._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)
        if generation_config.token_healing:
            raise ValueError("RWKV-7 CUDA graph generation does not support token healing.")
        if generation_config.cache_implementation is not None:
            raise ValueError("RWKV-7 CUDA graph generation does not support cache_implementation.")
        if generation_config.compile_config is not None:
            raise ValueError("RWKV-7 CUDA graph generation does not support compile_config.")
        if generation_config.continuous_batching_config is not None:
            raise ValueError("RWKV-7 CUDA graph generation does not support continuous batching.")

    @staticmethod
    def _validate_graph_logits_processors(
        logits_processor: LogitsProcessorList, generation_config: GenerationConfig
    ) -> None:
        expected = []
        if generation_config.do_sample:
            if generation_config.temperature is not None and generation_config.temperature != 1.0:
                expected.append(TemperatureLogitsWarper)
            if generation_config.top_k is not None and generation_config.top_k != 0:
                expected.append(TopKLogitsWarper)
            if generation_config.top_p is not None and generation_config.top_p < 1.0:
                expected.append(TopPLogitsWarper)
        if [type(processor) for processor in logits_processor] != expected:
            raise ValueError(
                "RWKV-7 CUDA graph generation supports only temperature, top-k, and top-p sampling; "
                "custom logits processors are not supported."
            )

    def _validate_graph_generation(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        model_kwargs: dict,
    ) -> tuple[int, torch.Tensor | None, tuple[StopStringCriteria, ...]]:
        if input_ids.device.type != "cuda" or self.lm_head.weight.device != input_ids.device:
            raise ValueError("RWKV-7 generation requires the model and a static input batch on one CUDA device.")
        if self.training or self.lm_head.weight.dtype != torch.float16:
            raise ValueError("RWKV-7 generation requires an eval-mode float16 model.")
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
        if model_kwargs.get("past_key_values") is not None or model_kwargs.get("inputs_embeds") is not None:
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
        self._validate_graph_logits_processors(logits_processor, generation_config)
        return max_length, eos_token_ids, stop_string_criteria

    @staticmethod
    def _graph_generation_key(
        input_ids: torch.LongTensor,
        generation_config: GenerationConfig,
        max_length: int,
        eos_token_ids: torch.Tensor | None,
        stop_string_criteria: tuple[StopStringCriteria, ...],
        tokenizer,
        wkv_mode: str,
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
            generation_config.temperature,
            generation_config.top_k,
            generation_config.top_p,
            pad_key,
            eos_key,
            stop_key,
            wkv_mode,
        )

    def _sample(
        self,
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
        key = self._graph_generation_key(
            input_ids,
            generation_config,
            max_length,
            eos_token_ids,
            stop_string_criteria,
            model_kwargs.get("tokenizer"),
            self._rwkv_wkv_mode,
        )
        with torch.cuda.device(input_ids.device):
            cache = self._get_rwkv_generation_cache()
            runner = cache.get(key)
            if runner is None:
                runner = _RwkvGenerationModelRunner(
                    self,
                    input_ids,
                    generation_config,
                    max_length,
                    eos_token_ids,
                    stop_string_criteria,
                )
                cache[key] = runner
            seed = torch.empty((), dtype=torch.int64).random_().item()
            return runner.generate(input_ids, seed, streamer)
