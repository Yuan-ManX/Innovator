"""
Anthropic LLM client implementation
"""

import logging
from typing import Any, Iterable

import anthropic

from ..retry import RetryConfig, async_retry
from ..schema import (
    FunctionCall,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
)
from .base import LLMClientBase

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClientBase):
    """
    Anthropic-compatible LLM client.

    Design goals:
    - Adapter-layer friendly (Anthropic / OpenAI / Gemini interchangeable)
    - Innovator schema compatible
    - Optional thinking & tool calling
    - SDK-agnostic parsing logic
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/anthropic",
        model: str = "MiniMax-M2.1",
        retry_config: RetryConfig | None = None,
    ):
        super().__init__(api_key, api_base, model, retry_config)

        self.client = anthropic.AsyncAnthropic(
            base_url=api_base,
            api_key=api_key,
            default_headers={"Authorization": f"Bearer {api_key}"},
        )

    # ------------------------------------------------------------------
    # Core API request (retryable)
    # ------------------------------------------------------------------

    async def _make_api_request(
        self,
        system_message: str | None,
        api_messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ):
        """
        Low-level Anthropic API invocation.
        This method is retry-wrapped if retry is enabled.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16384,
            "messages": api_messages,
        }

        if system_message:
            params["system"] = system_message

        if tools:
            params["tools"] = self._convert_tools(tools)

        return await self.client.messages.create(**params)

    # ------------------------------------------------------------------
    # Tool schema conversion
    # ------------------------------------------------------------------

    def _convert_tools(self, tools: Iterable[Any]) -> list[dict[str, Any]]:
        """
        Convert tool definitions into Anthropic-compatible schemas.

        Supported formats:
        - Raw dict schema
        - Objects with to_schema() method
        """
        converted: list[dict[str, Any]] = []

        for tool in tools:
            if isinstance(tool, dict):
                converted.append(tool)
            elif hasattr(tool, "to_schema"):
                converted.append(tool.to_schema())
            else:
                raise TypeError(
                    f"Unsupported tool type for AnthropicClient: {type(tool)}"
                )

        return converted

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def _convert_messages(
        self,
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert internal Message objects into Anthropic API format.

        Returns:
            (system_message, api_messages)
        """
        system_message: str | None = None
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
                continue

            if msg.role in ("user", "assistant"):
                api_messages.append(self._convert_chat_message(msg))
            elif msg.role == "tool":
                api_messages.append(self._convert_tool_result(msg))

        return system_message, api_messages

    def _convert_chat_message(self, msg: Message) -> dict[str, Any]:
        """
        Convert a user or assistant message to Anthropic format.
        """
        # Assistant message with structured content
        if msg.role == "assistant" and (msg.thinking or msg.tool_calls):
            content_blocks: list[dict[str, Any]] = []

            if msg.thinking:
                content_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": msg.thinking,
                    }
                )

            if msg.content:
                content_blocks.append(
                    {
                        "type": "text",
                        "text": msg.content,
                    }
                )

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": tc.function.arguments,
                        }
                    )

            return {
                "role": "assistant",
                "content": content_blocks,
            }

        # Plain user / assistant message
        return {
            "role": msg.role,
            "content": msg.content,
        }

    def _convert_tool_result(self, msg: Message) -> dict[str, Any]:
        """
        Convert tool execution result into Anthropic tool_result block.
        """
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
            ],
        }

    # ------------------------------------------------------------------
    # Response parsing (SDK-agnostic)
    # ------------------------------------------------------------------

    def _parse_response(self, response) -> LLMResponse:
        """
        Parse Anthropic response into internal LLMResponse schema.
        """
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in getattr(response, "content", []):
            block_type = getattr(block, "type", None)

            if block_type == "text":
                text_parts.append(block.text)

            elif block_type == "thinking":
                thinking_parts.append(block.thinking)

            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=FunctionCall(
                            name=block.name,
                            arguments=block.input,
                        ),
                    )
                )

        usage: TokenUsage | None = None
        if getattr(response, "usage", None):
            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens or 0,
                completion_tokens=response.usage.output_tokens or 0,
                total_tokens=(
                    (response.usage.input_tokens or 0)
                    + (response.usage.output_tokens or 0)
                ),
            )

        return LLMResponse(
            content="".join(text_parts),
            thinking="".join(thinking_parts) or None,
            tool_calls=tool_calls or None,
            finish_reason=getattr(response, "stop_reason", "stop"),
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Public unified generate API (Adapter-compatible)
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        """
        Unified generation API.

        This method matches the LLMProvider interface and can be
        plugged directly into an Adapter Layer.
        """
        request = self._prepare_request(messages, tools)

        if self.retry_config.enabled:
            retry_wrapper = async_retry(
                config=self.retry_config,
                on_retry=self.retry_callback,
            )
            response = await retry_wrapper(self._make_api_request)(
                request["system_message"],
                request["api_messages"],
                request["tools"],
            )
        else:
            response = await self._make_api_request(
                request["system_message"],
                request["api_messages"],
                request["tools"],
            )

        return self._parse_response(response)
