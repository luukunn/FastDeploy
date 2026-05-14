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

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class AnthropicError(BaseModel):
    """Anthropic API error detail."""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Anthropic API error response."""

    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    """Token usage for Anthropic Messages API."""

    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicContentBlock(BaseModel):
    """
    Anthropic content block supporting multiple types:
    text, image, tool_use, tool_result, thinking, redacted_thinking.
    """

    type: Literal[
        "text",
        "image",
        "tool_use",
        "tool_result",
        "tool_reference",
        "thinking",
        "redacted_thinking",
    ]
    # text type
    text: Optional[str] = None
    # image type
    source: Optional[Dict[str, Any]] = None
    # tool_use type
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    # tool_result type
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, List["AnthropicContentBlock"]]] = None
    is_error: Optional[bool] = None
    # thinking type
    thinking: Optional[str] = None
    signature: Optional[str] = None
    # redacted_thinking type
    data: Optional[str] = None


class AnthropicMessage(BaseModel):
    """A single message in the Anthropic Messages API conversation."""

    role: Literal["user", "assistant"]
    content: Union[str, List[AnthropicContentBlock]]


class AnthropicTool(BaseModel):
    """Tool definition for Anthropic Messages API."""

    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=lambda: {"type": "object"})


class AnthropicToolChoice(BaseModel):
    """Tool choice specification for Anthropic Messages API."""

    type: Literal["auto", "any", "tool", "none"]
    name: Optional[str] = None

    @model_validator(mode="after")
    def validate_tool_name(self):
        if self.type == "tool" and not self.name:
            raise ValueError("'name' is required when tool_choice type is 'tool'")
        return self


class AnthropicMessagesRequest(BaseModel):
    """Request body for POST /v1/messages (Anthropic Messages API)."""

    model: str
    messages: List[AnthropicMessage]
    max_tokens: int = Field(..., gt=0)
    metadata: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    system: Optional[Union[str, List[AnthropicContentBlock]]] = None
    temperature: Optional[float] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    tools: Optional[List[AnthropicTool]] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None


class AnthropicDelta(BaseModel):
    """Delta object for streaming content block updates."""

    type: Optional[str] = None  # text_delta, input_json_delta, thinking_delta, signature_delta
    text: Optional[str] = None
    thinking: Optional[str] = None
    partial_json: Optional[str] = None
    signature: Optional[str] = None
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None


class AnthropicStreamEvent(BaseModel):
    """Streaming event for Anthropic Messages API SSE responses."""

    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: Optional["AnthropicMessagesResponse"] = None
    delta: Optional[AnthropicDelta] = None
    content_block: Optional[AnthropicContentBlock] = None
    index: Optional[int] = None
    error: Optional[AnthropicError] = None
    usage: Optional[AnthropicUsage] = None


class AnthropicMessagesResponse(BaseModel):
    """Response body for POST /v1/messages (non-streaming)."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[AnthropicContentBlock] = Field(default_factory=list)
    model: str
    stop_reason: Optional[str] = None  # end_turn, max_tokens, stop_sequence, tool_use
    stop_sequence: Optional[str] = None
    usage: Optional[AnthropicUsage] = None
