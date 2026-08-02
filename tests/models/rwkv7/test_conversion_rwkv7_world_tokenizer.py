# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import ast
import json
import os
import random
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from transformers import AutoTokenizer
from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import _load_standard_fast_tokenizer
from transformers.models.rwkv7.convert_rwkv7_world_tokenizer_to_hf import (
    RWKV_WORLD_LAST_TOKEN_ID,
    RWKV_WORLD_RESERVED_TOKEN_IDS,
    RWKV_WORLD_TOKEN_ZERO,
    RWKV_WORLD_VOCAB_REVISION,
    RWKV_WORLD_VOCAB_SHA256,
    RWKV_WORLD_VOCAB_SIZE,
    convert_rwkv7_world_vocab_to_fast_tokenizer,
)


_OFFICIAL_VOCAB_ENV = "RWKV7_WORLD_VOCAB_FILE"
_RESERVED_TOKEN_PREFIX = "\ue000rwkv_reserved_"


def _official_vocab_file() -> Path:
    configured = os.environ.get(_OFFICIAL_VOCAB_ENV)
    if not configured:
        pytest.skip(f"set {_OFFICIAL_VOCAB_ENV} to the official local rwkv_vocab_v20230424.txt")
    path = Path(configured).resolve()
    if not path.is_file():
        pytest.fail(f"{_OFFICIAL_VOCAB_ENV} does not name a file: {path}")
    return path


def _parse_reference_vocab(path: Path) -> dict[int, bytes]:
    entries = {0: RWKV_WORLD_TOKEN_ZERO.encode("utf-8")}
    for line in path.read_text(encoding="utf-8").splitlines():
        first_space = line.index(" ")
        last_space = line.rindex(" ")
        token_id = int(line[:first_space])
        value = ast.literal_eval(line[first_space:last_space])
        value = value.encode("utf-8") if isinstance(value, str) else value
        assert len(value) == int(line[last_space:])
        entries[token_id] = value
    return entries


def _reference_trie(entries: dict[int, bytes]) -> dict:
    root = {}
    for token_id, value in entries.items():
        if token_id == 0:
            continue
        node = root
        for byte in value:
            node = node.setdefault(byte, {})
        node[None] = token_id
    return root


def _reference_encode(text: str, trie: dict) -> list[int]:
    source = text.encode("utf-8")
    token_ids = []
    position = 0
    while position < len(source):
        node = trie
        best = None
        cursor = position
        while cursor < len(source) and source[cursor] in node:
            node = node[source[cursor]]
            cursor += 1
            if None in node:
                best = (cursor, node[None])
        assert best is not None
        position, token_id = best
        token_ids.append(token_id)
    return token_ids


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


def _assert_reference_equivalence(tokenizer, trie: dict, texts: list[str]) -> None:
    reserved_ids = set(RWKV_WORLD_RESERVED_TOKEN_IDS)
    for text in texts:
        expected = _reference_encode(text, trie)
        actual = tokenizer.encode(text, add_special_tokens=False)
        assert actual == expected
        assert not reserved_ids.intersection(actual)
        assert (
            tokenizer.decode(
                actual,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            == text
        )


def test_rejects_non_official_rwkv_world_vocabulary(tmp_path) -> None:
    vocab_file = tmp_path / "rwkv_vocab_v20230424.txt"
    vocab_file.write_text("1 'a' 1\n", encoding="utf-8")
    output_dir = tmp_path / "tokenizer"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        convert_rwkv7_world_vocab_to_fast_tokenizer(vocab_file, output_dir)

    assert not output_dir.exists()


def test_official_rwkv_world_vocab_is_a_strict_standard_fast_tokenizer(tmp_path) -> None:
    vocab_file = _official_vocab_file()
    output_dir = tmp_path / "tokenizer"
    metadata = convert_rwkv7_world_vocab_to_fast_tokenizer(vocab_file, output_dir)
    reference_entries = _parse_reference_vocab(vocab_file)
    reference_trie = _reference_trie(reference_entries)

    tokenizer = AutoTokenizer.from_pretrained(
        output_dir,
        local_files_only=True,
        use_fast=True,
    )
    tokenizer_config = json.loads((output_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer_payload = json.loads((output_dir / "tokenizer.json").read_text(encoding="utf-8"))
    backend = Tokenizer.from_file(str(output_dir / "tokenizer.json"))
    assert tokenizer.is_fast
    assert len(tokenizer) == RWKV_WORLD_VOCAB_SIZE
    assert tokenizer.all_special_ids == [0]
    assert tokenizer.bos_token_id == tokenizer.eos_token_id == 0
    assert tokenizer.pad_token_id == tokenizer.unk_token_id == 0
    assert tokenizer.split_special_tokens is True
    assert tokenizer_config.get("auto_map") is None
    assert tokenizer_config["tokenizer_class"] == "TokenizersBackend"
    assert tokenizer_payload["model"]["type"] == "WordLevel"
    assert tokenizer_payload["decoder"]["type"] == "ByteLevel"
    assert tokenizer_payload["normalizer"] is None
    assert tokenizer_payload["pre_tokenizer"]["type"] == "Sequence"
    assert [item["type"] for item in tokenizer_payload["pre_tokenizer"]["pretokenizers"]] == [
        "ByteLevel",
        "Split",
    ]
    assert tokenizer_payload["post_processor"]["special_tokens"] == {}
    assert tokenizer.get_added_vocab() == {RWKV_WORLD_TOKEN_ZERO: 0}
    assert metadata["source"]["sha256"] == RWKV_WORLD_VOCAB_SHA256
    assert metadata["source"]["revision"] == RWKV_WORLD_VOCAB_REVISION
    assert metadata["source"]["entry_count"] == RWKV_WORLD_LAST_TOKEN_ID
    assert metadata["tokenizer"]["reserved_token_ids"] == list(RWKV_WORLD_RESERVED_TOKEN_IDS)
    assert metadata["validation"]["tokenizer_class"] == "TokenizersBackend"
    assert set(metadata["tokenizer_files"]) == {"tokenizer.json", "tokenizer_config.json"}
    assert {path.name for path in output_dir.iterdir()} == {
        "rwkv7_tokenizer_conversion.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    converter_tokenizer = _load_standard_fast_tokenizer(
        str(output_dir),
        vocab_size=RWKV_WORLD_VOCAB_SIZE,
        context_length=10240,
    )
    assert converter_tokenizer.is_fast
    assert converter_tokenizer.model_max_length == 10240

    alphabet = _byte_level_alphabet()
    assert set(reference_entries) == set(range(RWKV_WORLD_LAST_TOKEN_ID + 1))
    for token_id in range(1, RWKV_WORLD_LAST_TOKEN_ID + 1):
        assert backend.id_to_token(token_id) == _byte_level_text(reference_entries[token_id], alphabet)
    for token_id in RWKV_WORLD_RESERVED_TOKEN_IDS:
        assert backend.id_to_token(token_id) == _RESERVED_TOKEN_PREFIX + str(token_id)
        assert token_id not in tokenizer.all_special_ids

    valid_token_texts = []
    valid_token_ids = []
    for token_id in range(1, RWKV_WORLD_LAST_TOKEN_ID + 1):
        try:
            text = reference_entries[token_id].decode("utf-8")
        except UnicodeDecodeError:
            continue
        valid_token_texts.append(text)
        valid_token_ids.append(token_id)
    for offset in range(0, len(valid_token_texts), 2048):
        encodings = backend.encode_batch(valid_token_texts[offset : offset + 2048])
        expected_ids = valid_token_ids[offset : offset + 2048]
        assert [encoding.ids for encoding in encodings] == [[token_id] for token_id in expected_ids]

    overlap_texts = []
    for character in (" ", "#", "*", "-", "/", "="):
        for length in (1, 2, 3, 7, 8, 15, 16, 31, 32, 57, 58, 59, 62, 64, 65, 66, 72, 74, 76, 80, 127, 128, 129):
            overlap_texts.append(character * length)
    overlap_texts.extend(
        [
            "",
            " hello",
            "hello world\n",
            "\n\nUser: hi\n\nAssistant:",
            "你好，世界！",
            "é e\u0301",
            "😀🧑\u200d🚀",
            "a\x00b\t\r\n",
            RWKV_WORLD_TOKEN_ZERO,
            *(_RESERVED_TOKEN_PREFIX + str(token_id) for token_id in RWKV_WORLD_RESERVED_TOKEN_IDS),
        ]
    )
    _assert_reference_equivalence(tokenizer, reference_trie, overlap_texts)

    random_generator = random.Random(20260802)
    random_texts = []
    unicode_ranges = ((0, 0xD7FF), (0xE000, 0x10FFFF))
    for _ in range(512):
        text = []
        for _ in range(random_generator.randrange(0, 65)):
            lower, upper = random_generator.choice(unicode_ranges)
            text.append(chr(random_generator.randint(lower, upper)))
        random_texts.append("".join(text))
    _assert_reference_equivalence(tokenizer, reference_trie, random_texts)

    probe = "RWKV-7 tokenizer BOS probe"
    assert tokenizer.encode(probe, add_special_tokens=True) == tokenizer.encode(
        probe,
        add_special_tokens=False,
    )
