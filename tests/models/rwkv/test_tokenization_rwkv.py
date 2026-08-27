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

import tokenizers
import torch
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from tokenizers import Tokenizer, decoders, models, processors

from transformers import AutoTokenizer, RwkvTokenizerFast, StopStringCriteria
from transformers.dependency_versions_table import deps


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


def make_tokenizer() -> RwkvTokenizerFast:
    encoder = byte_encoder()
    vocabulary = {SPECIAL_TOKEN: 0, **{encoder[byte]: byte + 1 for byte in range(256)}}
    backend = Tokenizer(models.RwkvTrie(vocabulary))
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.TemplateProcessing(
        single=f"{SPECIAL_TOKEN} $A",
        pair=f"{SPECIAL_TOKEN} $A $B",
        special_tokens=[(SPECIAL_TOKEN, 0)],
    )
    tokenizer = RwkvTokenizerFast(
        tokenizer_object=backend,
        bos_token=SPECIAL_TOKEN,
        eos_token=SPECIAL_TOKEN,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = CHAT_TEMPLATE.read_text(encoding="utf-8")
    return tokenizer


class RwkvTokenizerTest(unittest.TestCase):
    def test_installed_git_fork_contract(self):
        installed = distribution("tokenizers")
        direct_url = json.loads(installed.read_text("direct_url.json"))
        self.assertEqual(direct_url["url"], "https://github.com/rwkv-rs/tokenizers-rwkv.git")
        self.assertEqual(
            direct_url["vcs_info"]["commit_id"],
            "c5d8dde5ff49c70e4656199d5033a84e03c21b2b",
        )
        self.assertEqual(direct_url["subdirectory"], "bindings/python")
        self.assertTrue(Path(tokenizers.__file__).is_relative_to(installed.locate_file("")))
        self.assertIn(Version(tokenizers.__version__), SpecifierSet(deps["tokenizers"].removeprefix("tokenizers")))
        self.assertTrue(callable(models.RwkvTrie.from_file))

        with tempfile.TemporaryDirectory() as directory:
            vocab = Path(directory) / "vocab.json"
            vocab.write_text(json.dumps({SPECIAL_TOKEN: 0, "a": 1}), encoding="utf-8")
            backend = Tokenizer(models.RwkvTrie.from_file(str(vocab)))
        self.assertEqual(backend.encode("a").ids, [1])

    def test_special_token_contract(self):
        tokenizer = make_tokenizer()
        payload = tokenizer.encode("hello", add_special_tokens=False)
        self.assertEqual(tokenizer.encode("hello", add_special_tokens=True), [0, *payload])
        self.assertEqual(tokenizer("hello", add_special_tokens=False).input_ids, payload)
        self.assertNotIn(0, payload)
        self.assertEqual(tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False), [0])
        self.assertEqual((tokenizer.bos_token_id, tokenizer.eos_token_id), (0, 0))
        self.assertIsNone(tokenizer.pad_token_id)
        self.assertIsNone(tokenizer.unk_token_id)
        with self.assertRaisesRegex(ValueError, "padding token"):
            tokenizer(["a", "longer"], padding=True)
        tokenizer.truncation_side = "left"
        self.assertEqual(tokenizer("hello", truncation=True, max_length=1).input_ids, [0])

    def test_native_chat_normalization_and_generation_prompt(self):
        tokenizer = make_tokenizer()
        messages = [{"role": "user", "content": "A  \r\n\r\nB\t  \n\n\n"}]
        original = json.loads(json.dumps(messages))
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(rendered, "User✿A\nB✿\nBot✿<think")
        self.assertEqual(messages, original)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True)
        self.assertEqual(ids[0], 0)
        self.assertNotEqual(ids[-1], 0)

    def test_custom_chat_template_uses_standard_content_and_special_token_behavior(self):
        tokenizer = make_tokenizer()
        messages = [{"role": "user", "content": "A  \r\n\r\nB  "}]
        rendered = tokenizer.apply_chat_template(
            messages,
            chat_template="{{ messages[0]['content'] }}",
            tokenize=False,
        )
        self.assertEqual(rendered, "A  \r\n\r\nB  ")
        ids = tokenizer.apply_chat_template(
            messages,
            chat_template="{{ messages[0]['content'] }}",
            tokenize=True,
            return_dict=False,
        )
        self.assertNotEqual(ids[0], 0)

    def test_assistant_whitespace_is_preserved_inside_content(self):
        tokenizer = make_tokenizer()
        messages = [
            {"role": "system", "content": "System  "},
            {"role": "user", "content": "  indented\n\nquestion  "},
            {"role": "assistant", "content": "first  \n\nsecond  "},
        ]
        self.assertEqual(
            tokenizer.apply_chat_template(messages, tokenize=False, rwkv_prompt_template="assistant"),
            "System: System\n\nUser:   indented\nquestion\n\nAssistant: first\n\nsecond",
        )

    def test_fake_think_prompt(self):
        tokenizer = make_tokenizer()
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            tokenize=False,
            add_generation_prompt=True,
            rwkv_generation_prompt="fake_think",
        )
        self.assertEqual(rendered, "User✿Hi✿\nBot✿<think></think")

    def test_tool_chat(self):
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
                "tool_calls": [{"function": {"name": "weather", "arguments": '{"city": "Paris"}'}}],
            },
            {"role": "tool", "content": '{"temperature": 20}'},
        ]
        rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        self.assertIn("### System\nUse tools.\n### `weather`", rendered)
        self.assertIn("### User\nParis?\nUse Celsius.", rendered)
        self.assertIn('"name": "weather"', rendered)
        self.assertIn('"temperature": 20', rendered)
        self.assertTrue(rendered.endswith("### Assistant\n<think"))

    def test_chat_stop_strings_follow_the_effective_prompt_style(self):
        tokenizer = make_tokenizer()
        messages = [{"role": "user", "content": "Hi"}]
        self.assertEqual(tokenizer.get_chat_stop_strings(messages), ["✿"])
        self.assertEqual(
            tokenizer.get_chat_stop_strings(messages, rwkv_prompt_template="assistant"),
            ["\nUser:"],
        )
        self.assertEqual(
            tokenizer.get_chat_stop_strings(messages, rwkv_prompt_template="function_calling"),
            ["\n### User"],
        )
        self.assertEqual(tokenizer.get_chat_stop_strings(messages, tools=[{"name": "search"}]), ["\n### User"])
        self.assertEqual(
            tokenizer.get_chat_stop_strings(
                [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search"}}]}]
            ),
            ["\n### User"],
        )
        self.assertEqual(
            tokenizer.get_chat_stop_strings([{"role": "tool", "content": "{}"}]),
            ["\n### User"],
        )
        empty_tool_calls = [{"role": "assistant", "content": "Done", "tool_calls": []}]
        self.assertEqual(tokenizer.get_chat_stop_strings(empty_tool_calls), ["✿"])
        self.assertEqual(tokenizer.apply_chat_template(empty_tool_calls, tokenize=False), "Bot✿Done✿")

    def test_chat_stop_strings_work_with_transformers_generation(self):
        tokenizer = make_tokenizer()
        for stop_string in ("✿", "\nUser:", "\n### User"):
            criterion = StopStringCriteria(tokenizer=tokenizer, stop_strings=stop_string)
            token_ids = tokenizer.encode(f"generated answer{stop_string}", add_special_tokens=False)
            self.assertTrue(criterion(torch.tensor([token_ids]), scores=None).item())

    def test_mixed_prompt_style_batch_requires_separate_generation(self):
        tokenizer = make_tokenizer()
        conversations = [
            [{"role": "user", "content": "Hello"}],
            [
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search"}}]},
                {"role": "tool", "content": "{}"},
            ],
        ]
        with self.assertRaisesRegex(ValueError, "different prompt styles"):
            tokenizer.get_chat_stop_strings(conversations)

    def test_save_reload_and_auto_tokenizer(self):
        tokenizer = make_tokenizer()
        messages = [{"role": "user", "content": "hello"}]
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save_pretrained(directory)
            reloaded = AutoTokenizer.from_pretrained(directory, use_fast=True, local_files_only=True)
            self.assertIsInstance(reloaded, RwkvTokenizerFast)
            self.assertEqual(
                reloaded.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                "User✿hello✿\nBot✿<think",
            )
            self.assertEqual(reloaded.get_chat_stop_strings(messages), ["✿"])


if __name__ == "__main__":
    unittest.main()
