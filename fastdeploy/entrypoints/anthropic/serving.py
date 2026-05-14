"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

"""Anthropic Messages API serving handler for FastDeploy.

Converts Anthropic Messages API requests to OpenAI ChatCompletion format,
delegates to the existing inference pipeline, and converts responses back.
"""

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Optional, Union

from fastdeploy.entrypoints.anthropic.protocol import (
    AnthropicContentBlock,
    AnthropicDelta,
    AnthropicError,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)
from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionToolsParam,
    ErrorResponse,
    FunctionDefinition,
    StreamOptions,
)
from fastdeploy.logger.request_logger import (
    RequestLogLevel,
    log_request,
    log_request_error,
)


def wrap_data_with_event(data: str, event: str) -> str:
    """Format a serialized JSON string as an Anthropic SSE event."""
    return f"event: {event}\ndata: {data}\n\n"


# OpenAI finish_reason -> Anthropic stop_reason
_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "recover_stop": "end_turn",
    "abort": "end_turn",
}


class AnthropicServingMessages:
    """Anthropic Messages API handler for FastDeploy.

    Handles the full request/response lifecycle:
    - Converts Anthropic request to OpenAI ChatCompletion format
    - Delegates inference to the existing chat_handler (Legacy or V1)
    - Converts OpenAI response back to Anthropic format (streaming or non-streaming)
    """

    def __init__(self, chat_handler):
        self.chat_handler = chat_handler

    # ========================================================================
    # Public API
    # ========================================================================

    async def create_messages(
        self, request: AnthropicMessagesRequest
    ) -> Union[AsyncGenerator[str, None], AnthropicMessagesResponse, ErrorResponse]:
        """Handle an Anthropic Messages API request."""
        log_request(RequestLogLevel.FULL, message="Anthropic request: {request}", request=request.model_dump_json())

        # 1. Convert request
        chat_req = self._convert_request(request)

        log_request(RequestLogLevel.FULL, message="Adapted to OpenAI: {request}", request=chat_req.model_dump_json())

        # 2. Delegate to inference engine
        result = await self.chat_handler.create_chat_completion(chat_req)

        # 3. Convert response
        if isinstance(result, ErrorResponse):
            return result
        elif isinstance(result, ChatCompletionResponse):
            return self._convert_full_response(result)
        else:
            return self._convert_stream_response(result)

    # ========================================================================
    # Request Conversion (Anthropic -> OpenAI)
    # ========================================================================

    def _convert_request(self, request: AnthropicMessagesRequest) -> ChatCompletionRequest:
        """Convert an Anthropic Messages request into a ChatCompletion request."""
        messages = self._build_messages(request)

        chat_req = ChatCompletionRequest(
            model=request.model,
            messages=messages,
            max_tokens=request.max_tokens,
            max_completion_tokens=request.max_tokens,
            stop=request.stop_sequences,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

        if request.stream:
            chat_req.stream = True
            chat_req.stream_options = StreamOptions(include_usage=True, continuous_usage_stats=True)

        if request.tools:
            chat_req.tools = [
                ChatCompletionToolsParam(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.input_schema,
                    ),
                )
                for tool in request.tools
            ]

        return chat_req

    def _build_messages(self, request: AnthropicMessagesRequest) -> list[dict[str, Any]]:
        """Build the full OpenAI messages list from Anthropic request."""
        messages: list[dict[str, Any]] = []

        # System message
        if request.system:
            if isinstance(request.system, str):
                messages.append({"role": "system", "content": request.system})
            else:
                system_prompt = ""
                for block in request.system:
                    if block.type == "text" and block.text:
                        # Strip Claude Code's attribution header which contains
                        # a per-request hash that defeats prefix caching.
                        if block.text.startswith("x-anthropic-billing-header"):
                            continue
                        system_prompt += block.text
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

        # Conversation messages
        for msg in request.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
            else:
                self._convert_complex_message(msg, messages)

        return messages

    def _convert_complex_message(self, msg, messages: list[dict[str, Any]]) -> None:
        """Convert a message with structured content blocks to OpenAI format."""
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        extra_messages: list[dict[str, Any]] = []

        for block in msg.content:
            if block.type == "text":
                if block.text:
                    content_parts.append({"type": "text", "text": block.text})

            elif block.type == "image":
                if block.source:
                    url = self._convert_image_source(block.source)
                    content_parts.append({"type": "image_url", "image_url": {"url": url}})

            elif block.type == "thinking":
                if block.thinking is not None:
                    reasoning_parts.append(block.thinking)

            elif block.type == "redacted_thinking":
                pass  # Intentionally skipped

            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id or f"call_{int(time.time())}",
                        "type": "function",
                        "function": {
                            "name": block.name or "",
                            "arguments": json.dumps(block.input or {}),
                        },
                    }
                )

            elif block.type == "tool_result":
                if msg.role == "user":
                    self._extract_tool_result(block, extra_messages)
                else:
                    text = str(block.content) if block.content else ""
                    content_parts.append({"type": "text", "text": f"Tool result: {text}"})

        # Build the OpenAI message
        openai_msg: dict[str, Any] = {"role": msg.role}
        if reasoning_parts:
            openai_msg["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            openai_msg["tool_calls"] = tool_calls
        if content_parts:
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                openai_msg["content"] = content_parts[0]["text"]
            else:
                openai_msg["content"] = content_parts

        has_content = "content" in openai_msg or tool_calls or reasoning_parts
        if has_content or msg.role != "user":
            messages.append(openai_msg)

        messages.extend(extra_messages)

    def _extract_tool_result(self, block, extra_messages: list[dict[str, Any]]) -> None:
        """Extract tool_result block into separate tool messages for OpenAI format."""
        tool_text = ""
        image_urls: list[str] = []

        if isinstance(block.content, str):
            tool_text = block.content
        elif isinstance(block.content, list):
            text_parts: list[str] = []
            for item in block.content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        url = self._convert_image_source(item.get("source", {}))
                        if url:
                            image_urls.append(url)
                elif hasattr(item, "type"):
                    if item.type == "text" and hasattr(item, "text"):
                        text_parts.append(item.text or "")
                    elif item.type == "image" and hasattr(item, "source"):
                        url = self._convert_image_source(item.source or {})
                        if url:
                            image_urls.append(url)
            tool_text = "\n".join(text_parts)

        extra_messages.append(
            {
                "role": "tool",
                "tool_call_id": block.tool_use_id or "",
                "content": tool_text or "",
            }
        )

        if image_urls:
            extra_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": u}} for u in image_urls],
                }
            )

    @staticmethod
    def _convert_image_source(source: dict[str, Any]) -> str:
        """Convert Anthropic image source to data URI or URL."""
        if source.get("type") == "url":
            return source.get("url", "")
        media_type = source.get("media_type", "image/jpeg")
        data = source.get("data", "")
        return f"data:{media_type};base64,{data}"

    # ========================================================================
    # Non-streaming Response Conversion (OpenAI -> Anthropic)
    # ========================================================================

    def _convert_full_response(self, response: ChatCompletionResponse) -> AnthropicMessagesResponse:
        """Convert a non-streaming OpenAI response to Anthropic format."""
        choice = response.choices[0]
        content: list[AnthropicContentBlock] = []

        # Order: thinking -> text -> tool_use
        if choice.message.reasoning_content:
            content.append(
                AnthropicContentBlock(
                    type="thinking",
                    thinking=choice.message.reasoning_content,
                    signature=uuid.uuid4().hex,
                )
            )

        if choice.message.content:
            content.append(AnthropicContentBlock(type="text", text=choice.message.content))

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    tool_input = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    tool_input = {}
                content.append(
                    AnthropicContentBlock(type="tool_use", id=tc.id, name=tc.function.name, input=tool_input)
                )

        return AnthropicMessagesResponse(
            id=response.id,
            content=content,
            model=response.model,
            stop_reason=_STOP_REASON_MAP.get(choice.finish_reason or "stop", "end_turn"),
            usage=AnthropicUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens or 0,
            ),
        )

    # ========================================================================
    # Streaming Response Conversion (OpenAI SSE -> Anthropic SSE)
    # ========================================================================

    async def _convert_stream_response(self, generator: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """Convert an OpenAI SSE stream to Anthropic SSE stream.

        Manages a state machine that tracks the current content block type
        (thinking/text/tool_use) and handles block open/close lifecycle.
        Uses protocol models (AnthropicStreamEvent, AnthropicDelta, etc.)
        for type-safe event construction.
        """
        # Stream state
        block_index = 0
        block_type: Optional[str] = None
        block_signature: Optional[str] = None
        signature_emitted = False
        tool_use_id: Optional[str] = None
        tool_index_map: dict[int, str] = {}
        finish_reason: Optional[str] = None
        first_chunk = True

        def _open_block(btype: str, **kwargs) -> str:
            nonlocal block_type, block_signature, signature_emitted, tool_use_id

            if btype == "text":
                content_block = AnthropicContentBlock(type="text", text="")
            elif btype == "thinking":
                content_block = AnthropicContentBlock(type="thinking", thinking="", signature="")
                block_signature = uuid.uuid4().hex
                signature_emitted = False
            elif btype == "tool_use":
                content_block = AnthropicContentBlock(
                    type="tool_use",
                    id=kwargs.get("id", ""),
                    name=kwargs.get("name", ""),
                    input={},
                )
                tool_use_id = kwargs.get("id")
            else:
                content_block = AnthropicContentBlock(type=btype)

            block_type = btype
            chunk = AnthropicStreamEvent(
                type="content_block_start",
                index=block_index,
                content_block=content_block,
            )
            data = chunk.model_dump_json(exclude_unset=True)
            return wrap_data_with_event(data, "content_block_start")

        def _close_block() -> list[str]:
            nonlocal block_index, block_type, block_signature, signature_emitted, tool_use_id
            if block_type is None:
                return []

            events: list[str] = []

            # Thinking blocks need a signature_delta before closing
            if block_type == "thinking" and block_signature and not signature_emitted:
                chunk = AnthropicStreamEvent(
                    type="content_block_delta",
                    index=block_index,
                    delta=AnthropicDelta(type="signature_delta", signature=block_signature),
                )
                data = chunk.model_dump_json(exclude_unset=True)
                events.append(wrap_data_with_event(data, "content_block_delta"))
                signature_emitted = True

            stop_chunk = AnthropicStreamEvent(
                type="content_block_stop",
                index=block_index,
            )
            data = stop_chunk.model_dump_json(exclude_unset=True)
            events.append(wrap_data_with_event(data, "content_block_stop"))

            block_index += 1
            block_type = None
            block_signature = None
            signature_emitted = False
            tool_use_id = None
            return events

        def _transition_to(target: str, **kwargs) -> list[str]:
            nonlocal block_type, tool_use_id
            events: list[str] = []
            if block_type != target or (target == "tool_use" and tool_use_id != kwargs.get("id")):
                events.extend(_close_block())
                events.append(_open_block(target, **kwargs))
            return events

        try:
            async for raw_line in generator:
                if not raw_line.startswith("data:"):
                    continue

                data_str = raw_line[5:].strip().rstrip("\n")

                # Terminal sentinel
                if data_str == "[DONE]":
                    chunk = AnthropicStreamEvent(type="message_stop")
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield wrap_data_with_event(data, "message_stop")
                    continue

                openai_chunk = ChatCompletionStreamResponse.model_validate_json(data_str)

                # First chunk -> message_start
                if first_chunk:
                    first_chunk = False
                    input_tokens = openai_chunk.usage.prompt_tokens if openai_chunk.usage else 0
                    chunk = AnthropicStreamEvent(
                        type="message_start",
                        message=AnthropicMessagesResponse(
                            id=openai_chunk.id,
                            type="message",
                            role="assistant",
                            content=[],
                            model=openai_chunk.model,
                            stop_reason=None,
                            stop_sequence=None,
                            usage=AnthropicUsage(input_tokens=input_tokens, output_tokens=0),
                        ),
                    )
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield wrap_data_with_event(data, "message_start")
                    continue

                # Usage-only chunk (empty choices) -> finalize
                if not openai_chunk.choices:
                    input_tokens = openai_chunk.usage.prompt_tokens if openai_chunk.usage else 0
                    output_tokens = openai_chunk.usage.completion_tokens if openai_chunk.usage else 0

                    for ev in _close_block():
                        yield ev

                    stop = _STOP_REASON_MAP.get(finish_reason or "stop", "end_turn")
                    chunk = AnthropicStreamEvent(
                        type="message_delta",
                        delta=AnthropicDelta(stop_reason=stop, stop_sequence=None),
                        usage=AnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens or 0),
                    )
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield wrap_data_with_event(data, "message_delta")
                    continue

                choice = openai_chunk.choices[0]

                # Track finish_reason
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason

                delta = choice.delta

                # Reasoning content (thinking)
                if delta.reasoning_content:
                    for ev in _transition_to("thinking"):
                        yield ev
                    chunk = AnthropicStreamEvent(
                        type="content_block_delta",
                        index=block_index,
                        delta=AnthropicDelta(type="thinking_delta", thinking=delta.reasoning_content),
                    )
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield wrap_data_with_event(data, "content_block_delta")

                # Text content
                if delta.content:
                    for ev in _transition_to("text"):
                        yield ev
                    chunk = AnthropicStreamEvent(
                        type="content_block_delta",
                        index=block_index,
                        delta=AnthropicDelta(type="text_delta", text=delta.content),
                    )
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield wrap_data_with_event(data, "content_block_delta")

                # Tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.id is not None:
                            # New tool call
                            tool_index_map[tc.index] = tc.id
                            tool_name = tc.function.name if tc.function else None
                            if tool_use_id != tc.id and tool_name:
                                for ev in _transition_to("tool_use", id=tc.id, name=tool_name):
                                    yield ev
                            if tc.function and tc.function.arguments and tool_use_id == tc.id:
                                chunk = AnthropicStreamEvent(
                                    type="content_block_delta",
                                    index=block_index,
                                    delta=AnthropicDelta(type="input_json_delta", partial_json=tc.function.arguments),
                                )
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_delta")
                        else:
                            # Incremental update
                            tid = tool_index_map.get(tc.index)
                            if tid and tc.function and tc.function.arguments and tool_use_id == tid:
                                chunk = AnthropicStreamEvent(
                                    type="content_block_delta",
                                    index=block_index,
                                    delta=AnthropicDelta(type="input_json_delta", partial_json=tc.function.arguments),
                                )
                                data = chunk.model_dump_json(exclude_unset=True)
                                yield wrap_data_with_event(data, "content_block_delta")

        except Exception as e:
            log_request_error(message=f"Error in Anthropic stream conversion: {e}")
            error_event = AnthropicStreamEvent(
                type="error",
                error=AnthropicError(type="internal_error", message=str(e)),
            )
            data = error_event.model_dump_json(exclude_unset=True)
            yield wrap_data_with_event(data, "error")
