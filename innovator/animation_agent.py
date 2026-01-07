"""
animation_agent.py
统一动画生成 Agent 框架 + LLM Stage 接口
包括：
- 核心数据结构（Style, Character, Camera, Motion, Shot, Scene, Timeline, Context）
- LLM Stage 基类
- Planning / Storyboard / Motion Design Stage
- AnimationAgent 总调度器
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List
from abc import ABC, abstractmethod


# ===========================
# 核心数据结构
# ===========================

@dataclass
class GlobalStyle:
    visual_style: str              # anime, cinematic, realistic
    lighting: str                  # low-key, soft, dramatic
    color_palette: str             # cold blue, warm orange
    fps: int = 24
    resolution: str = "1920x1080"

    def to_prompt(self) -> str:
        return (
            f"Style: {self.visual_style}, "
            f"Lighting: {self.lighting}, "
            f"Color: {self.color_palette}, "
            f"FPS: {self.fps}"
        )


@dataclass
class CharacterProfile:
    name: str
    appearance: str
    costume: str
    personality: str

    def to_prompt(self) -> str:
        return (
            f"Character {self.name}: "
            f"{self.appearance}, wearing {self.costume}, "
            f"personality: {self.personality}"
        )


@dataclass
class Camera:
    shot_type: str      # close-up, medium, wide
    movement: str       # pan, dolly, static
    lens: str           # 35mm, 50mm
    angle: str          # high, low, eye-level

    def to_prompt(self) -> str:
        return (
            f"Camera: {self.shot_type}, "
            f"{self.movement}, {self.lens}, {self.angle}"
        )


@dataclass
class Motion:
    start_pose: str
    action: str
    end_pose: str

    def to_prompt(self) -> str:
        return (
            f"Motion from {self.start_pose}, "
            f"performing {self.action}, "
            f"ending at {self.end_pose}"
        )


@dataclass
class Shot:
    id: str
    duration: float
    subject: str
    environment: str
    camera: Camera
    motion: Motion

    def to_prompt(self) -> str:
        return (
            f"Shot {self.id}: {self.subject} in {self.environment}. "
            f"{self.camera.to_prompt()}. "
            f"{self.motion.to_prompt()}."
        )

@dataclass
class Scene:
    id: str
    description: str
    shots: List[Shot] = field(default_factory=list)

@dataclass
class Timeline:
    scenes: List[Scene] = field(default_factory=list)

@dataclass
class AnimationContext:
    style: GlobalStyle
    characters: Dict[str, CharacterProfile] = field(default_factory=dict)
    timeline: Timeline = field(default_factory=Timeline)

    def build_prompt_context(self) -> str:
        parts = [self.style.to_prompt()]
        for c in self.characters.values():
            parts.append(c.to_prompt())
        return "\n".join(parts)


# ===========================
# LLM Stage 接口
# ===========================

class AnimationStage(ABC):
    name: str

    @abstractmethod
    async def run(self, context: AnimationContext) -> AnimationContext:
        pass


# 基于 LLM 的 Stage 基类
class LLMStage(AnimationStage, ABC):
    name: str
    system_prompt: str
    user_prompt_template: str

    def __init__(self, llm):
        """
        llm: LLMClient 实例，需实现 generate(messages) async 方法
        """
        self.llm = llm

    async def run(self, context: AnimationContext) -> AnimationContext:
        prompt = self.build_prompt(context)
        response = await self.llm.generate(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        data = self.parse_response(response.content)
        self.apply(context, data)
        return context

    def build_prompt(self, context: AnimationContext) -> str:
        return self.user_prompt_template.format(
            context=context.build_prompt_context()
        )

    def parse_response(self, content: str) -> dict:
        try:
            return json.loads(content)
        except Exception as e:
            raise ValueError(f"{self.name} JSON parse failed: {e}")

    @abstractmethod
    def apply(self, context: AnimationContext, data: dict):
        pass


# ===========================
# Planning Stage（剧情规划）
# ===========================

class PlanningStage(LLMStage):
    name = "planning"
    system_prompt = "You are a professional animation director. Your job is to plan animation scenes."
    user_prompt_template = """
Given the following animation context:

{context}

Generate a high-level scene plan.

Output JSON only in the following format:
{{
  "scenes": [
    {{
      "id": "scene_1",
      "description": "..."
    }}
  ]
}}
"""

    def apply(self, context: AnimationContext, data: dict):
        for s in data.get("scenes", []):
            context.timeline.scenes.append(
                Scene(
                    id=s["id"],
                    description=s["description"],
                )
            )


# ===========================
# Storyboard Stage（分镜生成）
# ===========================

class StoryboardStage(LLMStage):
    name = "storyboard"
    system_prompt = "You are a storyboard artist specialized in cinematic animation."
    user_prompt_template = """
Context:
{context}

For each scene, generate shot breakdowns.

Output JSON only:
{{
  "shots": [
    {{
      "scene_id": "scene_1",
      "shot_id": "shot_1",
      "duration": 3.0,
      "subject": "Hero",
      "environment": "ruined city",
      "camera": {{
        "shot_type": "wide",
        "movement": "dolly in",
        "lens": "35mm",
        "angle": "eye-level"
      }}
    }}
  ]
}}
"""

    def apply(self, context: AnimationContext, data: dict):
        scene_map = {s.id: s for s in context.timeline.scenes}
        for s in data.get("shots", []):
            scene = scene_map[s["scene_id"]]
            shot = Shot(
                id=s["shot_id"],
                duration=s["duration"],
                subject=s["subject"],
                environment=s["environment"],
                camera=Camera(**s["camera"]),
                motion=Motion(start_pose="undefined", action="undefined", end_pose="undefined")
            )
            scene.shots.append(shot)


# ===========================
# Motion Design Stage（动作生成）
# ===========================

class MotionDesignStage(LLMStage):
    name = "motion"
    system_prompt = "You are an animation motion designer. Design physically plausible character motion."
    user_prompt_template = """
Shots:
{context}

Generate motion for each shot.

Output JSON only:
{{
  "motions": [
    {{
      "shot_id": "shot_1",
      "start_pose": "...",
      "action": "...",
      "end_pose": "..."
    }}
  ]
}}
"""

    def apply(self, context: AnimationContext, data: dict):
        for scene in context.timeline.scenes:
            for shot in scene.shots:
                for m in data.get("motions", []):
                    if m["shot_id"] == shot.id:
                        shot.motion.start_pose = m["start_pose"]
                        shot.motion.action = m["action"]
                        shot.motion.end_pose = m["end_pose"]


# ===========================
# AnimationAgent 总调度器
# ===========================

class AnimationAgent:
    def __init__(self, context: AnimationContext, stages: List[AnimationStage]):
        self.context = context
        self.stages = stages

    async def run(self) -> AnimationContext:
        for stage in self.stages:
            self.context = await stage.run(self.context)
        return self.context


# ===========================
# Render Stage（画面生成）
# ===========================

class RenderStage(AnimationStage):
    """
    RenderStage 将 Timeline 的 Shots 转换成图像序列或视频。
    """

    name = "render"

    def __init__(self, renderer):
        """
        renderer: 必须实现
            async render_shot(shot: Shot) -> str
        返回：生成图像/视频路径
        """
        self.renderer = renderer

    async def run(self, context: AnimationContext) -> AnimationContext:
        """
        遍历 Timeline 中的所有 Shots，生成视频帧或渲染图像，
        并在 Shot 中记录渲染结果路径。
        """
        for scene in context.timeline.scenes:
            for shot in scene.shots:
                # 调用渲染接口
                render_path = await self.renderer.render_shot(shot)
                # 写回 Shot
                shot.render_output = render_path

        return context


class AgentMemory:
    """LLM-facing compressed memory"""

    def __init__(self, system_prompt: str):
        self.messages: list[Message] = [
            Message(role="system", content=system_prompt)
        ]

    def add_user(self, content: str):
        self.messages.append(Message(role="user", content=content))

    def add_assistant(self, content: str, tool_calls=None):
        self.messages.append(
            Message(role="assistant", content=content, tool_calls=tool_calls)
        )

    def add_tool(self, name: str, content: str, tool_call_id: str):
        self.messages.append(
            Message(
                role="tool",
                name=name,
                tool_call_id=tool_call_id,
                content=content,
            )
        )

    def snapshot(self) -> list[Message]:
        return self.messages.copy()


class AnimationAgent:
    """
    Core Agent for animation / cinematic content generation
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: list[Tool],
        system_prompt: str,
        max_steps: int = 30,
    ):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

        self.memory = AgentMemory(system_prompt)
        self.context = AnimationContext()

    def add_task(self, user_prompt: str):
        self.memory.add_user(user_prompt)

    async def run(self) -> str:
        """
        Main animation generation loop:
        - plan → generate → refine
        """
        for step in range(self.max_steps):
            response = await self.llm.generate(
                messages=self.memory.snapshot(),
                tools=list(self.tools.values()),
            )

            self.memory.add_assistant(
                content=response.content,
                tool_calls=response.tool_calls,
            )

            # Stop condition: no more tool calls
            if not response.tool_calls:
                return response.content

            # Execute animation-related tools
            for call in response.tool_calls:
                result = await self._execute_tool(call)
                self.memory.add_tool(
                    name=call.function.name,
                    content=result,
                    tool_call_id=call.id,
                )

        return "Animation generation stopped: max steps reached."

    async def _execute_tool(self, tool_call) -> str:
        name = tool_call.function.name
        args = tool_call.function.arguments

        if name not in self.tools:
            return f"Error: unknown tool {name}"

        tool = self.tools[name]
        result: ToolResult = await tool.execute(**args)

        if not result.success:
            return f"Error: {result.error}"

        # Optional: update animation context
        self._update_context(name, result.content)
        return result.content

    def _update_context(self, tool_name: str, content: str):
        """
        Update animation world state based on tool output
        """
        if tool_name == "create_scene":
            self.context.add_scene(
                Scene(
                    id=f"scene_{len(self.context.scenes)+1}",
                    description=content,
                )
            )
