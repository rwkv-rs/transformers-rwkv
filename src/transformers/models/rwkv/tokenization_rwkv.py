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

from ...tokenization_utils_tokenizers import TokenizersBackend


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


class RwkvTokenizerFast(TokenizersBackend):
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
        native_template = chat_template is None or chat_template == self.chat_template
        if not native_template:
            return super().apply_chat_template(
                conversation,
                tools=tools,
                documents=documents,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=tokenize,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_dict=return_dict,
                return_assistant_tokens_mask=return_assistant_tokens_mask,
                tokenizer_kwargs=tokenizer_kwargs,
                **kwargs,
            )
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
        kwargs["rwkv_add_bos"] = tokenize
        return super().apply_chat_template(
            normalized,
            tools=tools,
            documents=documents,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )


__all__ = ["RwkvTokenizerFast"]
