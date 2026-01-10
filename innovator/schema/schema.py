"""
Unified schema definitions for an Automatic Animation Agent.

Covers:
- LLM interaction
- Planning / Storyboard / Motion / Render stages
- Tool & Action calls
- Video rendering abstraction
"""

from enum import Enum
from typing import Any, Literal, Optional, List, Dict
from pydantic import BaseModel, Field


# =========================
# Provider Enums
# =========================

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class VideoProvider(str, Enum):
    SORA = "sora"
    RUNWAY = "runway"
    PIKA = "pika"
    CUSTOM = "custom"


class AgentStage(str, Enum):
    PLANNING = "planning"
    STORYBOARD = "storyboard"
    MOTION = "motion"
    RENDER = "render"


# =========================
# Tool / Action Call
# =========================

class FunctionCall(BaseModel):
    """Structured function / action call."""
    name: str
    arguments: Dict[str, Any]


class ToolCall(BaseModel):
    """Tool invocation structure."""
    id: str
    type: Literal["function", "action"]
    function: FunctionCall


# =========================
# Core Message Protocol
# =========================

class Message(BaseModel):
    """
    Unified agent message format.

    Supports:
    - text
    - structured blocks
    - tool / action calls
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | List[Dict[str, Any]]

    # Extended reasoning / chain-of-thought (optional, model-dependent)
    thinking: Optional[str] = None

    # Tool or action calls emitted by assistant
    tool_calls: Optional[List[ToolCall]] = None

    # Tool execution feedback
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


# =========================
# Animation-Specific Outputs
# =========================

class ScenePlan(BaseModel):
    """High-level scene planning output."""
    scene_id: str
    description: str
    mood: Optional[str] = None
    duration_sec: Optional[float] = None


class StoryboardShot(BaseModel):
    """Storyboard shot definition."""
    shot_id: str
    camera: str
    composition: str
    description: str
    duration_sec: float


class MotionInstruction(BaseModel):
    """Character / camera motion instruction."""
    target: str  # character / camera / object
    action: str
    timing: Optional[str] = None
    intensity: Optional[str] = None


class RenderRequest(BaseModel):
    """Video render request abstraction."""
    provider: VideoProvider
    prompt: str
    duration_sec: float
    resolution: str = "1080p"
    fps: int = 24
    style: Optional[str] = None
    seed: Optional[int] = None


class RenderResult(BaseModel):
    """Render result returned by video model."""
    video_url: str
    provider: VideoProvider
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =========================
# Token / Cost Tracking
# =========================

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# =========================
# Unified Agent Response
# =========================

class AgentResponse(BaseModel):
    """
    Unified response for all Agent stages.

    One response may include:
    - natural language
    - structured animation data
    - tool / render calls
    """

    stage: AgentStage

    # Textual response (human-readable)
    content: str

    # Structured outputs (machine-readable)
    scenes: Optional[List[ScenePlan]] = None
    storyboard: Optional[List[StoryboardShot]] = None
    motions: Optional[List[MotionInstruction]] = None
    render_request: Optional[RenderRequest] = None
    render_result: Optional[RenderResult] = None

    # Tool calls
    tool_calls: Optional[List[ToolCall]] = None

    # Metadata
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None
