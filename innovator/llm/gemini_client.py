"""
Gemini LLM client implementation
"""

import logging
from typing import Any

import google.generativeai as genai

from ..schema import LLMResponse, Message, TokenUsage
from .base import LLMClientBase

logger = logging.getLogger(__name__)


class GeminiClient(LLMClientBase):
    """
    Gemini-compatible LLM client.

    Notes:
    - No native tool calling (as of now)
    - No exposed chain-of-thought
    - Best suited for creative generation (storyboard / motion text)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-1.5-pro",
    ):
        super().__init__(api_key, api_base=None, model=model, retry_config=None)

        genai.configure(api_key=api_key)
        self.model_client = genai.GenerativeModel(model)

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[Message]) -> str:
        """
        Convert messages into a single Gemini prompt.
        """
        prompt_parts: list[str] = []

        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"[System]\n{msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"[User]\n{msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"[Assistant]\n{msg.content}")

        return "\n\n".join(prompt_parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,  # Ignored
    ) -> LLMResponse:
        prompt = self._convert_messages(messages)

        response = await self.model_client.generate_content_async(prompt)

        text = response.text or ""

        usage = None
        if hasattr(response, "usage_metadata"):
            usage = TokenUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count
                or 0,
                total_tokens=response.usage_metadata.total_token_count or 0,
            )

        return LLMResponse(
            content=text,
            thinking=None,
            tool_calls=None,
            finish_reason="stop",
            usage=usage,
        )
