"""RWKV-7 forward throughput across a batch x length grid.

The numbers quoted for this model are single-process, single-GPU throughput at a
handful of `batch x seq_len` shapes, run in one session so that every row shares a
driver, a clock state and a set of weights. This is the script that produced them.

Two things are deliberate. Inputs are seeded, so a rerun measures the same work
rather than the same shape. And the reference column is an argument, not a
constant: quoting "N% of X" is only meaningful if X is named, and a table whose
denominator lives in prose is one edit away from mixing two of them. Pass
`--baseline` a JSON of `{"<batch>x<len>": tok_per_s}` and the label rides along
into the output.

    python benchmark_v2/benchmark_scripts/rwkv7_throughput.py \
        --checkpoint ./rwkv7-7.2b-hf --dtype float16 \
        --sparse-channel-mix --compile-mode max-autotune \
        --baseline baselines/albatross_faster3a_2607.json \
        --baseline-label "albatross faster3a_2607"

Nothing here is RWKV-specific except the state allocation, which has to happen
before `torch.compile` -- a state first allocated inside the compiled region cannot
have its address pinned, and inductor then declines CUDA graphs for a region that
mutates its inputs. Passing `state=None` is correct and several times slower, which
is the kind of difference that makes an unlabelled benchmark misleading.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM


DEFAULT_GRID = ["1x1", "1x16", "1x32", "1x128", "1x256", "32x1", "128x1", "256x1", "16x16"]


def parse_shape(text: str) -> tuple[int, int]:
    batch, _, seq_len = text.partition("x")
    return int(batch), int(seq_len)


def measure(model, batch: int, seq_len: int, compile_mode: str | None, warmup: int, seed: int) -> float:
    """Median-free mean wall-clock throughput for one shape, in tokens/second.

    The iteration count is higher for single-token steps because they are short
    enough that timer overhead is visible at eight repeats.
    """
    torch.manual_seed(seed)
    state = model.rwkv7.allocate_state(batch)
    runner = torch.compile(model, mode=compile_mode) if compile_mode else model
    input_ids = torch.randint(1, model.config.vocab_size, (batch, seq_len), device=model.device)

    with torch.no_grad():
        for _ in range(warmup):
            runner(input_ids=input_ids, state=state, use_cache=True)
        torch.cuda.synchronize()

        repeats = 32 if seq_len == 1 else 8
        started = time.perf_counter()
        for _ in range(repeats):
            runner(input_ids=input_ids, state=state, use_cache=True)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) / repeats

    return batch * seq_len / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="converted RWKV-7 checkpoint directory")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--wkv-implementation", default="eager", help="RWKV7_WKV_FUNCTIONS key")
    parser.add_argument("--wkv-state-dtype", default="float32")
    parser.add_argument("--sparse-channel-mix", action="store_true")
    parser.add_argument("--compile-mode", default=None, help='e.g. "max-autotune"; omit to run eager')
    parser.add_argument("--shapes", nargs="*", default=DEFAULT_GRID, help='"<batch>x<len>" entries')
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", type=Path, default=None, help="JSON of shape -> tok/s to divide by")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--output", type=Path, default=None, help="write the run, environment included, as JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark measures GPU throughput and there is no CUDA device")

    baseline = json.loads(args.baseline.read_text()) if args.baseline else {}
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype=getattr(torch, args.dtype),
        sparse_channel_mix=args.sparse_channel_mix,
        wkv_implementation=args.wkv_implementation,
        wkv_state_dtype=args.wkv_state_dtype,
    )
    model = model.eval().cuda()

    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "dtype": args.dtype,
        "wkv_implementation": args.wkv_implementation,
        "wkv_state_dtype": args.wkv_state_dtype,
        "sparse_channel_mix": args.sparse_channel_mix,
        "compile_mode": args.compile_mode,
        "seed": args.seed,
    }
    for key, value in environment.items():
        print(f"{key:20s} {value}")
    print()

    rows = {}
    for shape in args.shapes:
        batch, seq_len = parse_shape(shape)
        try:
            throughput = measure(model, batch, seq_len, args.compile_mode, args.warmup, args.seed)
        except (torch.OutOfMemoryError, RuntimeError) as error:
            # A shape that will not fit is a result; anything else is a bug in the run
            # and must not be recorded as "this shape did not work". The default used
            # to be a `wkv_implementation` that is not a registry key, so every row
            # came back None and the table read as nine failed shapes rather than one
            # wrong flag.
            print(f"  {shape:8s} FAILED {type(error).__name__}: {error}")
            rows[shape] = None
        else:
            rows[shape] = throughput
            reference = baseline.get(shape)
            share = f"{throughput / reference * 100:6.1f}% of {args.baseline_label}" if reference else ""
            print(f"  {shape:8s} {throughput:10.1f} tok/s {share}")
        torch._dynamo.reset()

    if args.output:
        args.output.write_text(
            json.dumps(
                {"environment": environment, "baseline_label": args.baseline_label, "tok_per_s": rows},
                indent=2,
            )
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
