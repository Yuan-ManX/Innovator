"""
Base Tool Abstractions for Innovator
A General AI Agent Framework for Animation, Film & Game Creation
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


# ============================================================
# Tool Execution Result
# ============================================================

class ToolResult(BaseModel):
    """
    Standardized tool execution result.

    Designed for:
    - LLM tool calling
    - Agent internal tool orchestration
    - Creative pipeline execution (animation / render / simulation)
    """

    success: bool = Field(..., description="Whether the tool execution succeeded")
    content: str = Field(
        default="",
        description="Human-readable output or summary"
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured result payload (optional)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )


# ============================================================
# Base Tool Interface
# ============================================================

class Tool:
    """
    Base class for all Innovator tools.

    Tools may represent:
    - File / system operations
    - Creative generation steps (animation, story, layout)
    - External model calls (LLM / video / audio / 3D)
    - World interaction or simulation
    """

    # ------------------------------
    # Tool Metadata
    # ------------------------------

    @property
    def name(self) -> str:
        """Unique tool name."""
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Short description for LLM or Agent selection."""
        raise NotImplementedError

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        JSON Schema describing input parameters.

        Used for:
        - LLM tool calling
        - Agent validation
        """
        raise NotImplementedError

    # ------------------------------
    # Execution
    # ------------------------------

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool.

        Should:
        - Be side-effect safe where possible
        - Return ToolResult for agent reasoning
        """
        raise NotImplementedError

    # ------------------------------
    # LLM Schema Conversion
    # ------------------------------

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """
        Convert tool to Anthropic-compatible schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Convert tool to OpenAI-compatible schema.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_llm_schema(self, provider: str) -> Dict[str, Any]:
        """
        Convert tool schema based on LLM provider.
        """
        if provider == "anthropic":
            return self.to_anthropic_schema()

        if provider == "openai":
            return self.to_openai_schema()

        # Default fallback (future providers)
        return self.to_openai_schema()

