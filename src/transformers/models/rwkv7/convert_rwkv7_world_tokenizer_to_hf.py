# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Convert the official RWKV World vocabulary to a standard fast tokenizer artifact."""

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ..auto.tokenization_auto import AutoTokenizer


RWKV_WORLD_VOCAB_REPOSITORY = "https://github.com/BlinkDL/ChatRWKV"
RWKV_WORLD_VOCAB_REVISION = "02058ba0624a77c20f0913f83550835eb03a8db4"
RWKV_WORLD_VOCAB_PATH = "tokenizer/rwkv_vocab_v20230424.txt"
RWKV_WORLD_VOCAB_SHA256 = "e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89"
RWKV_WORLD_VOCAB_SIZE = 65536
RWKV_WORLD_LAST_TOKEN_ID = 65529
RWKV_WORLD_RESERVED_TOKEN_IDS = tuple(range(RWKV_WORLD_LAST_TOKEN_ID + 1, RWKV_WORLD_VOCAB_SIZE))
RWKV_WORLD_TOKEN_ZERO = "<|endoftext|>"

_METADATA_FILENAME = "rwkv7_tokenizer_conversion.json"
_VALIDATION_MARKER = "RWKV7_TOKENIZER_VALIDATION="
_RESERVED_TOKEN_PREFIX = "\ue000rwkv_reserved_"
_VALIDATION_PROBES = (
    "RWKV-7 tokenizer BOS probe",
    "hello world\n",
    "你好，世界！",
    "é e\u0301 😀🧑\u200d🚀",
    RWKV_WORLD_TOKEN_ZERO,
    *(_RESERVED_TOKEN_PREFIX + str(token_id) for token_id in RWKV_WORLD_RESERVED_TOKEN_IDS),
)
_VALIDATION_SCRIPT = r"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


artifact_dir, encoded_expected = sys.argv[1:]
artifact_path = Path(artifact_dir)
expected = json.loads(encoded_expected)
tokenizer_config = json.loads((artifact_path / "tokenizer_config.json").read_text(encoding="utf-8"))
if tokenizer_config.get("auto_map"):
    raise RuntimeError("RWKV World tokenizer artifact must not require auto_map or remote code.")
tokenizer = AutoTokenizer.from_pretrained(
    artifact_dir,
    local_files_only=True,
    use_fast=True,
)
if not tokenizer.is_fast or len(tokenizer) != 65536:
    raise RuntimeError("RWKV World artifact did not load as the 65536-entry standard fast tokenizer.")
special_ids = {
    name: getattr(tokenizer, f"{name}_token_id")
    for name in ("bos", "eos", "pad", "unk")
}
if set(special_ids.values()) != {0}:
    raise RuntimeError(f"RWKV World BOS/EOS/PAD/UNK must all use token 0, got {special_ids}.")
if tokenizer.split_special_tokens is not True:
    raise RuntimeError("RWKV World special-token text must remain in the ordinary greedy byte path.")
reserved_ids = set(range(65530, 65536))
for item in expected:
    token_ids = tokenizer.encode(item["text"], add_special_tokens=False)
    if token_ids != item["token_ids"]:
        raise RuntimeError(f"RWKV World fresh-process token IDs drifted for {item['text']!r}.")
    if reserved_ids.intersection(token_ids):
        raise RuntimeError(f"RWKV World reserved token IDs became reachable for {item['text']!r}.")
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != item["text"]:
        raise RuntimeError(f"RWKV World fresh-process decode drifted for {item['text']!r}.")
probe = "RWKV-7 tokenizer BOS probe"
if tokenizer.encode(probe, add_special_tokens=True) != tokenizer.encode(
    probe,
    add_special_tokens=False,
):
    raise RuntimeError("RWKV World tokenizer must not insert a BOS token.")
print(
    "RWKV7_TOKENIZER_VALIDATION="
    + json.dumps(
        {
            "auto_map": False,
            "is_fast": True,
            "special_token_ids": special_ids,
            "split_special_tokens": True,
            "tokenizer_class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
        },
        sort_keys=True,
    )
)
"""


def convert_rwkv7_world_vocab_to_fast_tokenizer(
    vocab_file: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    model_max_length: int = 10240,
) -> dict:
    """Build and fresh-process validate a standard tokenizer.json artifact."""

    if not isinstance(model_max_length, int) or isinstance(model_max_length, bool) or model_max_length < 1:
        raise ValueError("RWKV World tokenizer model_max_length must be a positive integer.")
    source_path = Path(vocab_file).resolve()
    token_bytes = _load_official_world_vocab(source_path)
    tokenizer = _build_standard_fast_tokenizer(token_bytes, model_max_length=model_max_length)

    destination = Path(output_dir).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise RuntimeError(f"RWKV World tokenizer destination must be absent or empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        tokenizer.save_pretrained(staging)
        expected = _validate_reloaded_tokenizer(staging, tokenizer)
        validation = _validate_tokenizer_in_subprocess(staging, expected)
        tokenizer_files = _file_digests(staging)
        metadata = {
            "schema_version": 1,
            "source": {
                "repository": RWKV_WORLD_VOCAB_REPOSITORY,
                "revision": RWKV_WORLD_VOCAB_REVISION,
                "path": RWKV_WORLD_VOCAB_PATH,
                "sha256": RWKV_WORLD_VOCAB_SHA256,
                "entry_count": RWKV_WORLD_LAST_TOKEN_ID,
                "first_token_id": 1,
                "last_token_id": RWKV_WORLD_LAST_TOKEN_ID,
            },
            "tokenizer": {
                "artifact_format": "standard_fast_tokenizer_json",
                "decoder": "ByteLevel",
                "model": "WordLevel",
                "model_max_length": model_max_length,
                "pre_tokenizers": [
                    "ByteLevel(add_prefix_space=False,use_regex=False)",
                    "Split(longest-first-rwkv-world-regex)",
                ],
                "reserved_token_ids": list(RWKV_WORLD_RESERVED_TOKEN_IDS),
                "special_token": RWKV_WORLD_TOKEN_ZERO,
                "special_token_ids": {"bos": 0, "eos": 0, "pad": 0, "unk": 0},
                "split_special_tokens": True,
                "vocab_size": RWKV_WORLD_VOCAB_SIZE,
            },
            "tokenizer_files": tokenizer_files,
            "validation": validation,
        }
        (staging / _METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            destination.rmdir()
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_official_world_vocab(path: Path) -> dict[int, bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"RWKV World vocabulary does not exist: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != RWKV_WORLD_VOCAB_SHA256:
        raise ValueError(
            f"RWKV World vocabulary SHA-256 mismatch: expected={RWKV_WORLD_VOCAB_SHA256} actual={actual_sha256}"
        )
    entries = {}
    seen_values = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        first_space = line.find(" ")
        last_space = line.rfind(" ")
        if first_space < 1 or last_space <= first_space:
            raise ValueError(f"RWKV World vocabulary line {line_number} is malformed.")
        token_id = int(line[:first_space])
        value = ast.literal_eval(line[first_space:last_space])
        value = value.encode("utf-8") if isinstance(value, str) else value
        if not isinstance(value, bytes) or len(value) != int(line[last_space:]):
            raise ValueError(f"RWKV World vocabulary line {line_number} has an invalid byte length.")
        if token_id in entries or value in seen_values:
            raise ValueError(f"RWKV World vocabulary line {line_number} is duplicated.")
        entries[token_id] = value
        seen_values.add(value)
    expected_ids = set(range(1, RWKV_WORLD_LAST_TOKEN_ID + 1))
    if set(entries) != expected_ids:
        raise ValueError("RWKV World vocabulary must define every token ID from 1 through 65529 exactly once.")
    if {entries[token_id] for token_id in range(1, 257)} != {bytes([value]) for value in range(256)}:
        raise ValueError("RWKV World vocabulary must contain a complete one-byte fallback alphabet.")
    return entries


def _byte_level_alphabet() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = byte_values.copy()
    extra_codepoint = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra_codepoint)
            extra_codepoint += 1
    return dict(zip(byte_values, map(chr, codepoints), strict=True))


def _byte_level_text(value: bytes, alphabet: dict[int, str]) -> str:
    return "".join(alphabet[byte] for byte in value)


def _regex_literal(value: str) -> str:
    return re.sub(r"([\\.^$|?*+()\[\]{}])", r"\\\1", value)


def _reserved_token(token_id: int) -> str:
    return _RESERVED_TOKEN_PREFIX + str(token_id)


def _build_standard_fast_tokenizer(
    token_bytes: dict[int, bytes],
    *,
    model_max_length: int,
) -> PreTrainedTokenizerFast:
    alphabet = _byte_level_alphabet()
    vocabulary = {_byte_level_text(value, alphabet): token_id for token_id, value in token_bytes.items()}
    vocabulary[RWKV_WORLD_TOKEN_ZERO] = 0
    vocabulary.update({_reserved_token(token_id): token_id for token_id in RWKV_WORLD_RESERVED_TOKEN_IDS})
    alternatives = [
        _regex_literal(_byte_level_text(value, alphabet))
        for _, value in sorted(token_bytes.items(), key=lambda item: (-len(item[1]), item[0]))
    ]
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token=RWKV_WORLD_TOKEN_ZERO))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            pre_tokenizers.Split(Regex("(?:" + "|".join(alternatives) + ")"), behavior="isolated"),
        ]
    )
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=RWKV_WORLD_TOKEN_ZERO,
        eos_token=RWKV_WORLD_TOKEN_ZERO,
        pad_token=RWKV_WORLD_TOKEN_ZERO,
        unk_token=RWKV_WORLD_TOKEN_ZERO,
        model_max_length=model_max_length,
        clean_up_tokenization_spaces=False,
        split_special_tokens=True,
    )


def _expected_probe_results(tokenizer: PreTrainedTokenizerFast) -> list[dict]:
    reserved_ids = set(RWKV_WORLD_RESERVED_TOKEN_IDS)
    expected = []
    for text in _VALIDATION_PROBES:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if reserved_ids.intersection(token_ids):
            raise RuntimeError(f"RWKV World reserved token IDs became reachable for {text!r}.")
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != text:
            raise RuntimeError(f"RWKV World tokenizer failed to round-trip {text!r}.")
        expected.append({"text": text, "token_ids": token_ids})
    return expected


def _validate_reloaded_tokenizer(
    artifact_dir: Path,
    original: PreTrainedTokenizerFast,
) -> list[dict]:
    tokenizer_config = json.loads((artifact_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    if tokenizer_config.get("auto_map"):
        raise RuntimeError("RWKV World tokenizer artifact must not require auto_map or remote code.")
    reloaded = AutoTokenizer.from_pretrained(
        artifact_dir,
        local_files_only=True,
        use_fast=True,
    )
    expected = _expected_probe_results(original)
    if (
        not reloaded.is_fast
        or len(reloaded) != RWKV_WORLD_VOCAB_SIZE
        or reloaded.split_special_tokens is not True
        or any(getattr(reloaded, f"{name}_token_id") != 0 for name in ("bos", "eos", "pad", "unk"))
    ):
        raise RuntimeError("RWKV World tokenizer did not preserve its standard fast-tokenizer contract.")
    for item in expected:
        if reloaded.encode(item["text"], add_special_tokens=False) != item["token_ids"]:
            raise RuntimeError(f"RWKV World tokenizer reload drifted for {item['text']!r}.")
    probe = "RWKV-7 tokenizer BOS probe"
    if reloaded.encode(probe, add_special_tokens=True) != reloaded.encode(probe, add_special_tokens=False):
        raise RuntimeError("RWKV World tokenizer must not insert a BOS token.")
    return expected


def _validate_tokenizer_in_subprocess(artifact_dir: Path, expected: list[dict]) -> dict:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _VALIDATION_SCRIPT, str(artifact_dir), json.dumps(expected)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(f"RWKV World tokenizer fresh-process validation failed: {details}") from error
    marker_lines = [line for line in completed.stdout.splitlines() if line.startswith(_VALIDATION_MARKER)]
    if len(marker_lines) != 1:
        raise RuntimeError("RWKV World tokenizer validation did not emit exactly one result payload.")
    return json.loads(marker_lines[0].removeprefix(_VALIDATION_MARKER))


def _file_digests(directory: Path) -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != _METADATA_FILENAME
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab_file", required=True, help="Official local rwkv_vocab_v20230424.txt.")
    parser.add_argument("--output_dir", required=True, help="Destination for the standard fast tokenizer artifact.")
    parser.add_argument("--model_max_length", type=int, default=10240)
    args = parser.parse_args(argv)
    result = convert_rwkv7_world_vocab_to_fast_tokenizer(
        args.vocab_file,
        args.output_dir,
        model_max_length=args.model_max_length,
    )
    print(f"RWKV7_TOKENIZER_CONVERSION={json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
