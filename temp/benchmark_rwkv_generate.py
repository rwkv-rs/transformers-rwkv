#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Benchmark RWKV-7 CUDA-graph generation and asynchronous detokenization."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from transformers import AutoTokenizer, RwkvForCausalLM


BATCH_SIZES = (1, 4, 64, 320, 512)
NUM_PROCESSES = 3


class DetokenizeStreamer:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.tokens = 0

    def put(self, token_ids: torch.LongTensor) -> None:
        if token_ids.ndim != 1:
            return
        self.tokenizer.batch_decode(token_ids[:, None], skip_special_tokens=False)
        self.tokens += token_ids.numel()

    def end(self) -> None:
        pass


def event_metrics(runtime, prompt_length: int, completion_length: int) -> dict[str, float]:
    start, first_token, end = runtime.last_generation_events
    end.synchronize()
    first_token_latency = start.elapsed_time(first_token) / 1000
    total_generation_time = start.elapsed_time(end) / 1000
    decode_time = total_generation_time - first_token_latency
    return {
        "first_token_latency": first_token_latency,
        "total_generation_time": total_generation_time,
        "prefill_speed": prompt_length / first_token_latency,
        "decode_speed": completion_length / decode_time,
    }


def run_generate(model, input_ids, args, streamer=None):
    torch.manual_seed(args.seed)
    output = model.generate(
        input_ids,
        max_new_tokens=args.completion_length,
        prefill_chunk_size=args.prefill_chunk_size,
        do_sample=args.do_sample,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        penalty_decay=args.penalty_decay,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=None,
        stop_strings=None,
        streamer=streamer,
    )
    runtime = next(iter(model._rwkv_generation_graphs.values()))
    return output, event_metrics(runtime, args.prompt_length, args.completion_length)


def benchmark_worker(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = RwkvForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map={"": args.device},
    ).eval()
    model.config.wkv_mode = args.wkv_mode
    base_tokens = tokenizer(args.prompt, add_special_tokens=False).input_ids
    if not base_tokens:
        raise ValueError("The benchmark prompt must produce at least one token.")
    repetitions = (args.prompt_length + len(base_tokens) - 1) // len(base_tokens)
    prompt_tokens = (base_tokens * repetitions)[: args.prompt_length]
    prompt_hash = hashlib.sha256(torch.tensor(prompt_tokens, dtype=torch.int32).numpy().tobytes()).hexdigest()
    results = []

    for batch_size in BATCH_SIZES:
        input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=args.device).repeat(batch_size, 1)
        torch.cuda.synchronize()
        capture_started = time.perf_counter()
        _, capture_metrics = run_generate(model, input_ids, args)
        torch.cuda.synchronize()
        capture_warmup_time = time.perf_counter() - capture_started - capture_metrics["total_generation_time"]

        disabled_output, disabled = run_generate(model, input_ids, args)
        streamer = DetokenizeStreamer(tokenizer)
        async_output, asynchronous = run_generate(model, input_ids, args, streamer)
        torch.testing.assert_close(async_output, disabled_output)
        output_hash = hashlib.sha256(disabled_output.cpu().numpy().tobytes()).hexdigest()
        results.append(
            {
                "batch_size": batch_size,
                "capture_warmup_time": capture_warmup_time,
                "detokenize_disabled": disabled,
                "detokenize_async": asynchronous,
                "detokenized_tokens": streamer.tokens,
                "output_hash": output_hash,
            }
        )
        model._rwkv_generation_graphs.clear()
        del input_ids, disabled_output, async_output
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "model": str(args.model),
        "device": args.device,
        "wkv_mode": args.wkv_mode,
        "prompt_length": args.prompt_length,
        "completion_length": args.completion_length,
        "prefill_chunk_size": args.prefill_chunk_size,
        "do_sample": args.do_sample,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "penalty_decay": args.penalty_decay,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "prompt_hash": prompt_hash,
        "batches": results,
    }


def median_results(runs: list[dict]) -> dict:
    summary = {key: runs[0][key] for key in runs[0] if key != "batches"}
    batches = []
    for batch_index, batch_size in enumerate(BATCH_SIZES):
        run_batches = [run["batches"][batch_index] for run in runs]
        if any(batch["batch_size"] != batch_size for batch in run_batches):
            raise ValueError("Worker batch order does not match the fixed benchmark contract.")
        if len({batch["output_hash"] for batch in run_batches}) != 1:
            raise ValueError(f"Batch {batch_size} generated different tokens across worker processes.")
        expected_detokenized_tokens = batch_size * summary["completion_length"]
        if any(batch["detokenized_tokens"] != expected_detokenized_tokens for batch in run_batches):
            raise RuntimeError(f"Batch {batch_size} asynchronous detokenization dropped completion tokens.")
        batch = {
            "batch_size": batch_size,
            "capture_warmup_time": statistics.median(item["capture_warmup_time"] for item in run_batches),
            "output_hash": run_batches[0]["output_hash"],
        }
        for mode in ("detokenize_disabled", "detokenize_async"):
            batch[mode] = {
                metric: statistics.median(item[mode][metric] for item in run_batches)
                for metric in run_batches[0][mode]
            }
        batch["prefill_regression"] = max(
            0.0,
            1 - batch["detokenize_async"]["prefill_speed"] / batch["detokenize_disabled"]["prefill_speed"],
        )
        batch["decode_regression"] = max(
            0.0,
            1 - batch["detokenize_async"]["decode_speed"] / batch["detokenize_disabled"]["decode_speed"],
        )
        batches.append(batch)
    summary["batches"] = batches
    return summary


def validate_results(summary: dict, args) -> None:
    for batch in summary["batches"]:
        if max(batch["prefill_regression"], batch["decode_regression"]) > args.max_detokenize_regression:
            raise RuntimeError(
                f"Batch {batch['batch_size']} asynchronous detokenization regression exceeds "
                f"{args.max_detokenize_regression:.1%}."
            )
    if args.albatross_results is None:
        return
    reference = json.loads(args.albatross_results.read_text())
    for field in (
        "wkv_mode",
        "prompt_length",
        "completion_length",
        "prefill_chunk_size",
        "do_sample",
        "presence_penalty",
        "frequency_penalty",
        "penalty_decay",
        "temperature",
        "top_k",
        "top_p",
        "seed",
        "prompt_hash",
    ):
        if summary[field] != reference[field]:
            raise RuntimeError(f"Transformers and Albatross use different {field} values.")
    reference_batches = {batch["batch_size"]: batch for batch in reference["batches"]}
    for batch in summary["batches"]:
        expected = reference_batches[batch["batch_size"]]
        if batch["output_hash"] != expected["output_hash"]:
            raise RuntimeError(f"Batch {batch['batch_size']} tokens differ from Albatross.")
        for metric in ("prefill_speed", "decode_speed"):
            ratio = batch["detokenize_disabled"][metric] / expected[metric]
            batch[f"albatross_{metric}_ratio"] = ratio
            if ratio < args.min_albatross_ratio:
                raise RuntimeError(f"Batch {batch['batch_size']} {metric} is {ratio:.2%} of the Albatross reference.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wkv-mode", choices=("fp32io16", "fp16"), default="fp16")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--completion-length", type=int, default=128)
    parser.add_argument("--prefill-chunk-size", type=int)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--penalty-decay", type=float, default=0.996)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-detokenize-regression", type=float, default=0.01)
    parser.add_argument("--albatross-results", type=Path)
    parser.add_argument("--min-albatross-ratio", type=float, default=0.99)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prompt_length < 1 or args.completion_length < 2:
        raise ValueError("Benchmark prompt_length must be positive and completion_length must be at least 2.")
    if args.worker_output is not None:
        args.worker_output.write_text(json.dumps(benchmark_worker(args)))
        return
    with tempfile.TemporaryDirectory(prefix="rwkv-generate-benchmark-") as directory:
        outputs = []
        for run_index in range(NUM_PROCESSES):
            output = Path(directory) / f"run-{run_index}.json"
            subprocess.run(
                [sys.executable, __file__, *sys.argv[1:], "--worker-output", str(output)],
                check=True,
            )
            outputs.append(json.loads(output.read_text()))
    summary = median_results(outputs)
    validate_results(summary, args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
