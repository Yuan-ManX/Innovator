"""
Schema definitions for Animation Agent.
"""

from .schema import (
    # Enums
    LLMProvider,
    VideoProvider,
    AgentStage,

    # Core message & tool protocol
    FunctionCall,
    ToolCall,
    Message,

    # Animation domain schemas
    ScenePlan,
    StoryboardShot,
    MotionInstruction,
    RenderRequest,
    RenderResult,

    # Usage / response
    TokenUsage,
    AgentResponse,
)

__all__ = [
    # Enums
    "LLMProvider",
    "VideoProvider",
    "AgentStage",

    # Core protocol
    "FunctionCall",
    "ToolCall",
    "Message",

    # Animation structures
    "ScenePlan",
    "StoryboardShot",
    "MotionInstruction",
    "RenderRequest",
    "RenderResult",

    # Agent response
    "TokenUsage",
    "AgentResponse",
]
