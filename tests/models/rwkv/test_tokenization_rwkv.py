# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import json
import tempfile
import unittest
from importlib.metadata import distribution
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, processors

from transformers import AutoTokenizer, RwkvTokenizer


SPECIAL_TOKEN = "<|endoftext|>"
CHAT_TEMPLATE = Path(__file__).parents[3] / "temp" / "rwkv_chat_template.jinja"


def byte_encoder() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = byte_values.copy()
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, map(chr, codepoints), strict=True))


def make_tokenizer() -> RwkvTokenizer:
    encoder = byte_encoder()
    vocabulary = {SPECIAL_TOKEN: 0, **{encoder[byte]: byte + 1 for byte in range(256)}}
    backend = Tokenizer(models.RwkvTrie(vocabulary))
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.TemplateProcessing(
        single=f"{SPECIAL_TOKEN} $A",
        pair=f"{SPECIAL_TOKEN} $A $B",
        special_tokens=[(SPECIAL_TOKEN, 0)],
    )
    tokenizer = RwkvTokenizer(
        tokenizer_object=backend,
        bos_token=SPECIAL_TOKEN,
        eos_token=SPECIAL_TOKEN,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = CHAT_TEMPLATE.read_text(encoding="utf-8")
    return tokenizer


class RwkvTokenizerTest(unittest.TestCase):
    def test_installed_fork_and_public_class_contract(self):
        installed = distribution("tokenizers")
        direct_url = json.loads(installed.read_text("direct_url.json"))
        self.assertEqual(direct_url["url"], "https://github.com/rwkv-rs/tokenizers-rwkv.git")
        self.assertEqual(
            direct_url["vcs_info"]["commit_id"],
            "c5d8dde5ff49c70e4656199d5033a84e03c21b2b",
        )
        self.assertEqual(direct_url["subdirectory"], "bindings/python")
        self.assertIs(RwkvTokenizer.model, models.RwkvTrie)
        self.assertTrue(callable(models.RwkvTrie.from_file))
        self.assertFalse(hasattr(RwkvTokenizer, "get_chat_stop_strings"))
        self.assertNotIn("RwkvTokenizerFast", __import__("transformers").__dict__)

    def test_longest_match_byte_roundtrip_and_special_tokens(self):
        encoder = byte_encoder()
        backend = Tokenizer(models.RwkvTrie({encoder[ord("a")]: 1, "ab": 2, encoder[ord("b")]: 3}))
        self.assertEqual(backend.encode("ab").ids, [2])

        tokenizer = make_tokenizer()
        for text in ("hello\n", "你好，世界！", "é e\u0301 \U0001f600", "a\x00b\t\r\n"):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            self.assertNotIn(0, token_ids)
            self.assertEqual(
                tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                text,
            )
        payload = tokenizer.encode("hello", add_special_tokens=False)
        self.assertEqual(tokenizer.encode("hello", add_special_tokens=True), [0, *payload])
        self.assertEqual((tokenizer.bos_token_id, tokenizer.eos_token_id), (0, 0))
        self.assertIsNone(tokenizer.pad_token_id)
        self.assertIsNone(tokenizer.unk_token_id)

    def test_bot_assistant_and_think_golden(self):
        tokenizer = make_tokenizer()
        messages = [{"role": "user", "content": "A  \r\n\r\nB\t  \n\n"}]
        self.assertEqual(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "User✿A\nB✿\nBot✿<think",
        )
        assistant_messages = [
            {"role": "system", "content": "System  "},
            {"role": "user", "content": "  indented\n\nquestion  "},
            {"role": "assistant", "content": "first  \n\nsecond  "},
        ]
        self.assertEqual(
            tokenizer.apply_chat_template(
                assistant_messages,
                tokenize=False,
                rwkv_prompt_template="assistant",
            ),
            "System: System\n\nUser:   indented\nquestion\n\nAssistant: first\n\nsecond",
        )
        self.assertEqual(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hi"}],
                tokenize=False,
                add_generation_prompt=True,
                rwkv_generation_prompt="fake_think",
            ),
            "User✿Hi✿\nBot✿<think></think",
        )

    def test_tool_chat_golden(self):
        tokenizer = make_tokenizer()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        messages = [
            {"role": "system", "content": "Use tools.  "},
            {"role": "user", "content": "Paris?\n\nUse Celsius.  "},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "weather", "arguments": {"city": "Paris"}}}],
            },
            {"role": "tool", "content": {"temperature": 20}},
        ]
        rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        self.assertIn("### System\nUse tools.\n### `weather`", rendered)
        self.assertIn("### User\nParis?\nUse Celsius.", rendered)
        self.assertIn('"name": "weather"', rendered)
        self.assertIn('"temperature": 20', rendered)
        self.assertTrue(rendered.endswith("### Assistant\n<think"))

    def test_save_reload_uses_auto_tokenizer_without_remote_code(self):
        tokenizer = make_tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save_pretrained(directory)
            reloaded = AutoTokenizer.from_pretrained(
                directory,
                local_files_only=True,
            )
            self.assertIs(type(reloaded), RwkvTokenizer)
            self.assertEqual(
                reloaded.apply_chat_template(
                    [{"role": "user", "content": "hello"}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "User✿hello✿\nBot✿<think",
            )


if __name__ == "__main__":
    unittest.main()
