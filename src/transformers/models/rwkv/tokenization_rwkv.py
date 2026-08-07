# Copyright 2026 The HuggingFace Inc. team.
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
"""Fast tokenizer for the RWKV World vocabulary."""

import copy
import json
import re
from typing import Any

from ...tokenization_utils_base import BatchEncoding, TruncationStrategy
from ...tokenization_utils_tokenizers import PreTrainedTokenizerFast
from ...utils import PaddingStrategy


RWKV_BOS_EOS_TOKEN = "<|endoftext|>"
RWKV_BOS_EOS_TOKEN_ID = 0
RWKV_GENERATION_PROMPT_MODES = ("open_think", "fake_think")
RWKV_PROMPT_STYLES = ("bot", "assistant", "function_calling")
RWKV_CHAT_STOP_STRINGS = {
    "bot": "✿",
    "assistant": "\nUser:",
    "function_calling": "\n### User",
}


def _message_field(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and (item.get("type") == "text" or "text" in item):
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _normalize_content(content: Any, role: str) -> str:
    text = _stringify_content(content).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip(" \t") for line in text.split("\n")).rstrip(" \t\n")
    if role == "user":
        text = re.sub(r"\n{2,}", "\n", text)
    return text


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _normalize_message(message: Any) -> dict[str, Any]:
    role = str(_message_field(message, "role", ""))
    normalized = {
        "role": role,
        "content": _normalize_content(_message_field(message, "content", ""), role),
    }
    tool_calls = copy.deepcopy(_message_field(message, "tool_calls", None))
    if tool_calls is not None:
        for tool_call in tool_calls:
            function = _message_field(tool_call, "function", tool_call)
            if isinstance(function, dict) and "arguments" in function:
                function["arguments"] = _json_value(function["arguments"])
        normalized["tool_calls"] = tool_calls
    if role == "tool":
        normalized["rwkv_json_content"] = _json_value(normalized["content"])
    return normalized


def _normalize_conversation(conversation: Any) -> list[dict[str, Any]]:
    messages = getattr(conversation, "messages", conversation)
    return [_normalize_message(message) for message in messages]


def _resolve_prompt_style(conversation: Any, tools: Any, prompt_style: str) -> str:
    messages = getattr(conversation, "messages", conversation)
    has_tool_history = any(
        _message_field(message, "role", "") == "tool" or bool(_message_field(message, "tool_calls", None))
        for message in messages
    )
    return "function_calling" if tools or has_tool_history or prompt_style == "function_calling" else prompt_style


class RwkvTokenizerFast(PreTrainedTokenizerFast):
    """Rust-backed greedy byte tokenizer for RWKV World models."""

    model_input_names = ["input_ids", "attention_mask"]

    def get_chat_stop_strings(
        self,
        conversation,
        tools=None,
        rwkv_prompt_template="bot",
    ) -> list[str]:
        """Return the stop string required by the effective RWKV chat prompt style.

        Tool definitions, historical tool calls, and tool messages select the
        ``function_calling`` style just like the native RWKV chat template. A
        batched call must resolve to one style because ``generate()`` accepts
        one shared set of stop strings for the whole batch.
        """
        if rwkv_prompt_template not in RWKV_PROMPT_STYLES:
            raise ValueError(
                f"Unsupported RWKV prompt style {rwkv_prompt_template!r}; expected one of {RWKV_PROMPT_STYLES}."
            )

        values = getattr(conversation, "messages", conversation)
        is_batched = bool(values) and (isinstance(values[0], (list, tuple)) or hasattr(values[0], "messages"))
        conversations = values if is_batched else [values]
        prompt_styles = {_resolve_prompt_style(item, tools, rwkv_prompt_template) for item in conversations}
        if len(prompt_styles) > 1:
            raise ValueError(
                "RWKV chat batches resolved to different prompt styles. Split the conversations by prompt style "
                "before calling generate(), which accepts one shared set of stop strings per batch."
            )
        effective_style = prompt_styles.pop() if prompt_styles else rwkv_prompt_template
        return [RWKV_CHAT_STOP_STRINGS[effective_style]]

    def encode(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        padding=False,
        truncation=None,
        max_length=None,
        stride=0,
        padding_side=None,
        return_tensors=None,
        **kwargs,
    ):
        model = self.backend_tokenizer.model
        if (
            isinstance(text, str)
            and text_pair is None
            and not padding
            and not truncation
            and max_length is None
            and not stride
            and return_tensors is None
            and not kwargs
            and hasattr(model, "encode")
        ):
            token_ids = model.encode(text)
            if not add_special_tokens:
                return token_ids
            first_non_bos = 0
            while first_non_bos < len(token_ids) and token_ids[first_non_bos] == RWKV_BOS_EOS_TOKEN_ID:
                first_non_bos += 1
            return [RWKV_BOS_EOS_TOKEN_ID, *token_ids[first_non_bos:]]
        return super().encode(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            padding_side=padding_side,
            return_tensors=return_tensors,
            **kwargs,
        )

    def _encode_plus(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
        max_length=None,
        stride=0,
        is_split_into_words=False,
        pad_to_multiple_of=None,
        padding_side=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        split_special_tokens=None,
        **kwargs,
    ):
        if (
            add_special_tokens
            and truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE
            and max_length is not None
            and max_length < 1
        ):
            raise ValueError("RWKV sequences require max_length >= 1 to preserve BOS/EOS token ID 0.")
        use_rwkv_fast_path = (
            text_pair is None
            and not is_split_into_words
            and padding_strategy == PaddingStrategy.DO_NOT_PAD
            and not stride
            and not return_overflowing_tokens
            and not return_special_tokens_mask
            and not return_offsets_mapping
            and not split_special_tokens
            and (
                isinstance(text, str)
                or isinstance(text, (list, tuple))
                and all(isinstance(item, str) for item in text)
            )
        )
        model = self.backend_tokenizer.model
        if not use_rwkv_fast_path or not hasattr(model, "encode_batch"):
            return super()._encode_plus(
                text=text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding_strategy=padding_strategy,
                truncation_strategy=truncation_strategy,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                split_special_tokens=split_special_tokens,
                **kwargs,
            )

        is_batched = not isinstance(text, str)
        sequences = list(text) if is_batched else [text]
        input_ids = model.encode_batch(sequences)
        for index, token_ids in enumerate(input_ids):
            if add_special_tokens:
                first_non_bos = 0
                while first_non_bos < len(token_ids) and token_ids[first_non_bos] == RWKV_BOS_EOS_TOKEN_ID:
                    first_non_bos += 1
                token_ids = [RWKV_BOS_EOS_TOKEN_ID, *token_ids[first_non_bos:]]
            if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length is not None:
                if self.truncation_side == "left" and add_special_tokens:
                    tail_length = max_length - 1
                    token_ids = (
                        [RWKV_BOS_EOS_TOKEN_ID, *token_ids[-tail_length:]]
                        if tail_length > 0
                        else [RWKV_BOS_EOS_TOKEN_ID]
                    )
                elif self.truncation_side == "left":
                    token_ids = token_ids[-max_length:]
                else:
                    token_ids = token_ids[:max_length]
            input_ids[index] = token_ids

        data = {"input_ids": input_ids if is_batched else input_ids[0]}
        if return_attention_mask is not False:
            attention_mask = [[1] * len(token_ids) for token_ids in input_ids]
            data["attention_mask"] = attention_mask if is_batched else attention_mask[0]
        if return_token_type_ids:
            token_type_ids = [[0] * len(token_ids) for token_ids in input_ids]
            data["token_type_ids"] = token_type_ids if is_batched else token_type_ids[0]
        if return_length:
            lengths = [len(token_ids) for token_ids in input_ids]
            data["length"] = lengths if is_batched else lengths[0]
        return BatchEncoding(data, tensor_type=return_tensors)

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        documents=None,
        chat_template=None,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize=True,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_dict=True,
        return_assistant_tokens_mask=False,
        tokenizer_kwargs=None,
        **kwargs,
    ):
        if return_assistant_tokens_mask:
            raise ValueError("RWKV chat templates do not define an assistant token mask.")
        generation_prompt = kwargs.setdefault("rwkv_generation_prompt", "open_think")
        if generation_prompt not in RWKV_GENERATION_PROMPT_MODES:
            raise ValueError(
                f"Unsupported RWKV generation prompt mode {generation_prompt!r}; "
                f"expected one of {RWKV_GENERATION_PROMPT_MODES}."
            )
        prompt_style = kwargs.setdefault("rwkv_prompt_template", "bot")
        if prompt_style not in RWKV_PROMPT_STYLES:
            raise ValueError(f"Unsupported RWKV prompt style {prompt_style!r}; expected one of {RWKV_PROMPT_STYLES}.")

        values = getattr(conversation, "messages", conversation)
        is_batched = bool(values) and (isinstance(values[0], (list, tuple)) or hasattr(values[0], "messages"))
        if is_batched:
            normalized = [_normalize_conversation(item) for item in values]
        else:
            normalized = _normalize_conversation(values)
        rendered = super().apply_chat_template(
            normalized,
            tools=tools,
            documents=documents,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
            **kwargs,
        )
        if not tokenize:
            return rendered
        encoded = self(
            rendered,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors=return_tensors,
            **(tokenizer_kwargs or {}),
        )
        return encoded if return_dict else encoded["input_ids"]


__all__ = ["RwkvTokenizerFast"]
