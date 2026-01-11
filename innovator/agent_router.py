"""
Agent Router for Innovator
A General AI Agent Framework for Animation, Film & Game Creation

Features:
- Multi-agent routing: Planner, Director, Animation, Film, Game, Render
- Confidence scoring per agent
- Automatic fallback to safe agent if confidence is low
- Explainable routing decisions with reasons
- Handles review outcomes for iterative pipelines
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================
# Agent Types
# ============================================================

class AgentType(str, Enum):
    PLANNER = "planner"
    DIRECTOR = "director"
    ANIMATION = "animation"
    FILM = "film"
    GAME = "game"
    RENDER = "render"
    END = "end"


# ============================================================
# Task & Routing Models
# ============================================================

class Task:
    """
    High-level task abstraction routed between agents.
    """
    def __init__(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        stage: Optional[str] = None,
    ):
        self.task_id = task_id
        self.task_type = task_type  # animation / film / game / mixed
        self.payload = payload
        self.stage = stage  # planning / execution / render / review


@dataclass
class AgentRouteResult:
    agent_name: AgentType
    confidence: float
    reason: str
    fallback_used: bool = False


@dataclass
class RoutingDecision:
    """
    Router output: where to send the task next.
    """
    next_agents: List[AgentType]
    reason: str

    def __repr__(self) -> str:
        agents = ", ".join(a.value for a in self.next_agents)
        return f"<RoutingDecision → [{agents}] | {self.reason}>"


# ============================================================
# Agent Router
# ============================================================

class AgentRouter:
    """
    Central routing brain for Innovator Agents with confidence + fallback.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        fallback_agent: AgentType = AgentType.PLANNER,
    ):
        self.confidence_threshold = confidence_threshold
        self.fallback_agent = fallback_agent
        # agent_name -> scoring_fn(task, context) -> (score, reason)
        self._agents: Dict[AgentType, Callable[[Task, dict], Tuple[float, str]]] = {}

    # --------------------------------------------------------
    # Registration
    # --------------------------------------------------------

    def register_agent(
        self,
        agent_name: AgentType,
        scoring_fn: Callable[[Task, dict], Tuple[float, str]],
    ):
        self._agents[agent_name] = scoring_fn

    # --------------------------------------------------------
    # Routing Entry Point
    # --------------------------------------------------------

    def route(
        self,
        current_agent: Optional[AgentType],
        task: Task,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        context = context or {}

        # Entry point
        if current_agent is None:
            return RoutingDecision(
                next_agents=[AgentType.PLANNER],
                reason="New task entry → Planner",
            )

        # Planner → Director
        if current_agent == AgentType.PLANNER:
            return RoutingDecision(
                next_agents=[AgentType.DIRECTOR],
                reason="Planner output ready → Director decision",
            )

        # Director → Domain Agent(s) with confidence scoring
        if current_agent == AgentType.DIRECTOR:
            return self._route_from_director(task, context)

        # Execution → Render
        if current_agent in (AgentType.ANIMATION, AgentType.FILM, AgentType.GAME):
            return RoutingDecision(
                next_agents=[AgentType.RENDER],
                reason="Execution output requires rendering",
            )

        # Render → Review or End
        if current_agent == AgentType.RENDER:
            return self._route_from_render(task, context)

        return RoutingDecision(
            next_agents=[AgentType.END],
            reason="No matching routing rule",
        )

    # --------------------------------------------------------
    # Director Routing with Confidence + Fallback
    # --------------------------------------------------------

    def _route_from_director(self, task: Task, context: dict) -> RoutingDecision:
        if not self._agents:
            return RoutingDecision(
                next_agents=[self.fallback_agent],
                reason="No agents registered → fallback",
            )

        results: List[AgentRouteResult] = []

        for agent_name, scoring_fn in self._agents.items():
            if agent_name in (AgentType.PLANNER, AgentType.DIRECTOR, AgentType.RENDER):
                continue  # Skip non-execution agents for scoring
            try:
                score, reason = scoring_fn(task, context)
            except Exception as e:
                score, reason = 0.0, f"Scoring error: {e}"
            results.append(
                AgentRouteResult(agent_name=agent_name, confidence=score, reason=reason)
            )

        # Sort by confidence
        results.sort(key=lambda r: r.confidence, reverse=True)
        best = results[0]

        # Fallback logic
        if best.confidence < self.confidence_threshold:
            return RoutingDecision(
                next_agents=[self.fallback_agent],
                reason=(
                    f"Confidence {best.confidence:.2f} below threshold "
                    f"{self.confidence_threshold:.2f}. "
                    f"Fallback to '{self.fallback_agent.value}'. "
                    f"Original reason: {best.reason}"
                ),
            )

        return RoutingDecision(
            next_agents=[best.agent_name],
            reason=f"Director selected {best.agent_name.value} | {best.reason}",
        )

    # --------------------------------------------------------
    # Render Routing
    # --------------------------------------------------------

    def _route_from_render(self, task: Task, context: dict) -> RoutingDecision:
        review_result = context.get("review_result", "accept")
        if review_result == "accept":
            return RoutingDecision(
                next_agents=[AgentType.END],
                reason="Render accepted → pipeline complete",
            )
        if review_result == "revise":
            return RoutingDecision(
                next_agents=[AgentType.DIRECTOR],
                reason="Render revision requested → Director",
            )
        if review_result == "redesign":
            return RoutingDecision(
                next_agents=[AgentType.PLANNER],
                reason="Major redesign required → Planner",
            )
        return RoutingDecision(
            next_agents=[AgentType.END],
            reason="Unknown review result → end",
        )


# ============================================================
# Built-in Agent Scoring Functions
# ============================================================

def animation_agent_score(task: Task, context: dict) -> Tuple[float, str]:
    keywords = [
        "animation", "animated", "character", "motion",
        "pose", "rig", "keyframe", "movement"
    ]
    score = sum(k in task.payload.get("prompt", "").lower() for k in keywords) / len(keywords)
    return score, "Animation-related intent detected"


def film_agent_score(task: Task, context: dict) -> Tuple[float, str]:
    keywords = [
        "film", "cinematic", "camera", "shot",
        "lighting", "scene", "storyboard", "montage"
    ]
    score = sum(k in task.payload.get("prompt", "").lower() for k in keywords) / len(keywords)
    return score, "Cinematic / film language detected"


def game_agent_score(task: Task, context: dict) -> Tuple[float, str]:
    keywords = [
        "game", "npc", "quest", "combat",
        "level", "interaction", "player", "skill"
    ]
    score = sum(k in task.payload.get("prompt", "").lower() for k in keywords) / len(keywords)
    return score, "Game mechanics and interaction detected"


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    router = AgentRouter(confidence_threshold=0.65, fallback_agent=AgentType.PLANNER)

    # Register execution agents
    router.register_agent(AgentType.ANIMATION, animation_agent_score)
    router.register_agent(AgentType.FILM, film_agent_score)
    router.register_agent(AgentType.GAME, game_agent_score)

    task = Task(
        task_id="task_001",
        task_type="animation",
        payload={"prompt": "Create a cinematic sword fight animation"},
    )

    current_agent = None
    context = {}

    while True:
        decision = router.route(current_agent, task, context)
        print(f"{current_agent} → {decision}")

        if decision.next_agents[0] == AgentType.END:
            break

        # Simulate single-agent step
        current_agent = decision.next_agents[0]

        # Simulate render review
        if current_agent == AgentType.RENDER:
            context["review_result"] = "accept"
