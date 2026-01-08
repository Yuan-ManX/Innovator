"""
Base class for LLM clients (OpenAI / Anthropic / Gemini).
"""

from abc import ABC, abstractmethod
from typing import Any, Iterable

from ..retry import RetryConfig
from ..schema import (
    LLMResponse,
    Message,
    ToolCall,
    Usage,
)


class LLMClientBase(ABC):
    """
    Abstract base class for all LLM clients.

    This class defines a unified interface for:
    - message-based generation
    - tool calling
    - thinking / reasoning
    - usage tracking
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        retry_config: RetryConfig | None = None,
        timeout: float | None = None,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.retry_config = retry_config or RetryConfig()
        self.timeout = timeout

        # Optional callback: on_retry(attempt: int, error: Exception)
        self.retry_callback = None

    # ========================
    # Public API
    # ========================

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Unified generate entry.

        All vendor-specific behavior MUST be handled internally.
        """
        payload = self._build_payload(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        raw_response = await self._call_api(payload, stream=stream)
        return self._parse_response(raw_response)

    # ========================
    # Required vendor hooks
    # ========================

    @abstractmethod
    async def _call_api(self, payload: dict[str, Any], *, stream: bool) -> Any:
        """Perform the actual HTTP / SDK call."""
        pass

    @abstractmethod
    def _parse_response(self, raw_response: Any) -> LLMResponse:
        """Convert raw vendor response to unified LLMResponse."""
        pass

    # ========================
    # Payload construction
    # ========================

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        """
        Build a vendor-agnostic request payload.

        Vendor-specific structure is handled in `_convert_messages`
        and `_convert_tools`.
        """
        system, api_messages = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }

        if system:
            payload["system"] = system

        if tools:
            payload["tools"] = self._convert_tools(tools)

        if temperature is not None:
            payload["temperature"] = temperature

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        payload["stream"] = stream
        return payload

    # ========================
    # Conversion hooks
    # ========================

    @abstractmethod
    def _convert_messages(
        self,
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert internal Message list to vendor-specific format.

        Returns:
            (system_message, api_messages)
        """
        pass

    @abstractmethod
    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """
        Convert tool definitions to vendor-specific schema.
        """
        pass
