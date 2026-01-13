"""
Memory Tool - Persistent memory for Innovator agents.

This module provides tools that allow an agent to:
- Write important information into long-term memory
- Read and query previously stored memory
- Maintain continuity across agent runs, tasks, and execution chains
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List

from .base import Tool, ToolResult


# ============================================================
# Memory Write Tool
# ============================================================

class MemoryWriteTool(Tool):
    """
    Tool for writing information into agent memory.

    Typical usage by agent:
    - memory_write("User prefers concise technical explanations", category="user_preference")
    - memory_write("Project uses Python 3.12 with async-first design", category="project_info")
    - memory_write("Chose diffusion-based pipeline for animation generation", category="decision")
    """

    def __init__(self, memory_file: str = "./workspace/.agent_memory.json"):
        self.memory_file = Path(memory_file)
        # Lazy initialization: file is created only when first memory is written

    @property
    def name(self) -> str:
        return "memory_write"

    @property
    def description(self) -> str:
        return (
            "Write important information into the agent's persistent memory. "
            "Use this to store user preferences, project context, decisions, "
            "assumptions, or any knowledge that should be reused later. "
            "Each memory entry is timestamped."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store. Be concise, factual, and explicit.",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional category/tag for the memory "
                        "(e.g., 'user_preference', 'project_info', 'decision', 'constraint')."
                    ),
                },
            },
            "required": ["content"],
        }

    def _load_memory(self) -> List[dict]:
        """Load memory from file. Returns empty list if file does not exist."""
        if not self.memory_file.exists():
            return []

        try:
            return json.loads(self.memory_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_memory(self, memory: List[dict]):
        """Persist memory to file, creating directories if needed."""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    async def execute(self, content: str, category: str = "general") -> ToolResult:
        """Write a memory entry."""
        try:
            memory = self._load_memory()

            entry = {
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "content": content,
            }

            memory.append(entry)
            self._save_memory(memory)

            return ToolResult(
                success=True,
                content=f"Memory stored: {content} (category: {category})",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to write memory: {str(e)}",
            )


# ============================================================
# Memory Read Tool
# ============================================================

class MemoryReadTool(Tool):
    """
    Tool for reading and querying agent memory.

    Typical usage by agent:
    - memory_read()
    - memory_read(category="user_preference")
    - memory_read(category="decision")
    """

    def __init__(self, memory_file: str = "./workspace/.agent_memory.json"):
        self.memory_file = Path(memory_file)

    @property
    def name(self) -> str:
        return "memory_read"

    @property
    def description(self) -> str:
        return (
            "Read the agent's persistent memory. "
            "Optionally filter by category to retrieve specific knowledge, "
            "preferences, or past decisions."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category filter for memory entries",
                },
            },
        }

    async def execute(self, category: Optional[str] = None) -> ToolResult:
        """Read memory entries."""
        try:
            if not self.memory_file.exists():
                return ToolResult(
                    success=True,
                    content="No memory stored yet.",
                )

            memory = json.loads(self.memory_file.read_text(encoding="utf-8"))

            if not memory:
                return ToolResult(
                    success=True,
                    content="No memory stored yet.",
                )

            # Apply category filter if provided
            if category:
                memory = [m for m in memory if m.get("category") == category]
                if not memory:
                    return ToolResult(
                        success=True,
                        content=f"No memory found for category: {category}",
                    )

            # Format memory for agent consumption
            lines = []
            for idx, entry in enumerate(memory, 1):
                ts = entry.get("timestamp", "unknown time")
                cat = entry.get("category", "general")
                content = entry.get("content", "")
                lines.append(
                    f"{idx}. [{cat}] {content}\n   (stored at {ts})"
                )

            formatted = "Agent Memory:\n" + "\n".join(lines)

            return ToolResult(success=True, content=formatted)

        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to read memory: {str(e)}",
            )
