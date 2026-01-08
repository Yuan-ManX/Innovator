"""
OpenAI LLM client implementation
"""

import logging
from typing import Any, Iterable

from openai import AsyncOpenAI

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


class OpenAIClient(LLMClientBase):
    """
    OpenAI-compatible LLM client.

    Design goals:
    - Adapter-layer friendly
    - Innovator schema compatible
    - Function calling / tool calling support
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        retry_config: RetryConfig | None = None,
    ):
        super().__init__(api_key, api_base, model, retry_config)

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    # ------------------------------------------------------------------
    # Core API request
    # ------------------------------------------------------------------

    async def _make_api_request(
        self,
        api_messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ):
        params: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }

        if tools:
            params["tools"] = self._convert_tools(tools)
            params["tool_choice"] = "auto"

        return await self.client.chat.completions.create(**params)

    # ------------------------------------------------------------------
    # Tool conversion
    # ------------------------------------------------------------------

    def _convert_tools(self, tools: Iterable[Any]) -> list[dict[str, Any]]:
        """
        Convert tools into OpenAI function-calling schema.
        """
        converted = []
        for tool in tools:
            if isinstance(tool, dict):
                converted.append(tool)
            elif hasattr(tool, "to_openai_schema"):
                converted.append(tool.to_openai_schema())
            elif hasattr(tool, "to_schema"):
                converted.append(tool.to_schema())
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")
        return converted

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "tool":
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
            else:
                api_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        return api_messages

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message

        text_content = message.content or ""
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        type="function",
                        function=FunctionCall(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                )

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return LLMResponse(
            content=text_content,
            thinking=None,  # OpenAI does not expose thinking
            tool_calls=tool_calls or None,
            finish_reason=choice.finish_reason,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        api_messages = self._convert_messages(messages)

        if self.retry_config.enabled:
            retry_wrapper = async_retry(
                config=self.retry_config,
                on_retry=self.retry_callback,
            )
            response = await retry_wrapper(self._make_api_request)(
                api_messages,
                tools,
            )
        else:
            response = await self._make_api_request(api_messages, tools)

        return self._parse_response(response)
