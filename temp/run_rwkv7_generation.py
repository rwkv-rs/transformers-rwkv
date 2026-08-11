#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Generate text with native Transformers RWKV-7 and FlashRWKV2 sampling."""

from __future__ import annotations

import argparse

import torch
from flashrwkv2 import infer_sampling_six_parameter_forward_varlen, setup_sampling_states

from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "rwkv-rs/rwkv7-g1-st"
DEFAULT_REVISION = "9140a362f42aa023a4d2a72d94217fad2580f685"
SAMPLING_PARAMETERS = {
    "temperature": 0.96,
    "top_p": 0.76,
    "top_k": 32,
    "presence_penalty": 1.0,
    "frequency_penalty": 0.1,
    "penalty_decay": 0.988,
}


def flashrwkv2_generate(
    model,
    tokenizer,
    model_inputs,
    stop_strings: list[str],
    *,
    max_new_tokens: int,
    seed: int,
) -> list[int]:
    """Decode one request with the persistent FlashRWKV2 sampler state."""
    if model_inputs["input_ids"].shape[0] != 1:
        raise ValueError("This minimal generation example accepts exactly one request.")
    if model.device.type != "cuda":
        raise RuntimeError(f"FlashRWKV2 generation requires a CUDA model, got {model.device}.")

    model_vocab_size = model.config.vocab_size
    tokenizer_vocab_size = tokenizer.vocab_size
    if not 0 < tokenizer_vocab_size <= model_vocab_size:
        raise ValueError(
            f"Expected 0 < tokenizer vocab <= model vocab, got {tokenizer_vocab_size} and {model_vocab_size}."
        )

    with torch.cuda.device(model.device):
        sampling_states = setup_sampling_states(seed, num_slots=1)
    penalties = torch.zeros(1, model_vocab_size, dtype=torch.float32, device=model.device)
    slot_indices = torch.tensor([0], dtype=torch.int32, device=model.device)
    generated_ids: list[int] = []

    with torch.inference_mode():
        outputs = model(**model_inputs, use_cache=True, logits_to_keep=1)
        for _ in range(max_new_tokens):
            logits = outputs.logits[:, -1, :].float().contiguous()
            # RWKV World defines tokenizer IDs 0..65529 while the model keeps six
            # padded rows. Keep those reserved model-only IDs unreachable.
            logits[:, tokenizer_vocab_size:] = -torch.inf
            sampled = infer_sampling_six_parameter_forward_varlen(
                logits,
                penalties,
                sampling_states,
                slot_indices,
                **SAMPLING_PARAMETERS,
            )
            token_id = int(sampled.item())
            generated_ids.append(token_id)

            if token_id == tokenizer.eos_token_id:
                break
            decoded = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if any(decoded.endswith(stop_string) for stop_string in stop_strings):
                break

            outputs = model(
                input_ids=sampled.to(dtype=torch.long).view(1, 1),
                past_key_values=outputs.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )

    return generated_ids


def clean_response(text: str, stop_strings: list[str]) -> str:
    """Remove the prompt-completing angle bracket and a generated turn delimiter."""
    text = text.removeprefix(">")
    for stop_string in stop_strings:
        if text.endswith(stop_string):
            text = text[: -len(stop_string)]
            break
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="?",
        default="你好！请用中文简单介绍一下你自己，并告诉我 2+2 等于多少。",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prompt-style", choices=("bot", "assistant", "function_calling"), default="bot")
    parser.add_argument("--thinking", choices=("open_think", "fake_think"), default="fake_think")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=False,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            dtype=torch.float16,
            trust_remote_code=False,
        )
        .cuda()
        .eval()
        .prepare_for_inference()
    )

    messages = [{"role": "user", "content": args.prompt}]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        rwkv_prompt_template=args.prompt_style,
        rwkv_generation_prompt=args.thinking,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    stop_strings = tokenizer.get_chat_stop_strings(
        messages,
        rwkv_prompt_template=args.prompt_style,
    )
    generated_ids = flashrwkv2_generate(
        model,
        tokenizer,
        model_inputs,
        stop_strings,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(clean_response(response, stop_strings))


if __name__ == "__main__":
    main()
