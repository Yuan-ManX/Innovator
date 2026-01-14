"""
Innovator Agent Run Logger

Records the full execution trace of an Innovator agent run, including:
- LLM requests & responses
- Tool calls & execution results
- Memory / planning events (extensible)
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .schema import Message, ToolCall


class AgentLogger:
    """
    Innovator Agent Run Logger

    Each agent run generates a structured, append-only log file
    for debugging, replay, evaluation, and analysis.
    """

    def __init__(
        self,
        agent_name: str = "Innovator",
        log_dir: Optional[Path] = None,
    ):
        self.agent_name = agent_name
        self.run_id = uuid.uuid4().hex[:8]

        self.log_dir = (
            log_dir
            if log_dir
            else Path.home() / ".innovator" / "logs"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file: Optional[Path] = None
        self.log_index: int = 0
        self.start_time: Optional[datetime] = None

    # ==========================================================
    # Run lifecycle
    # ==========================================================

    def start_run(
        self,
        model: str,
        provider: str,
        workspace: str,
    ):
        """Start a new agent run"""
        self.start_time = datetime.now()

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.agent_name}_{self.run_id}_{timestamp}.log"
        self.log_file = self.log_dir / filename
        self.log_index = 0

        header = {
            "run_id": self.run_id,
            "agent": self.agent_name,
            "model": model,
            "provider": provider,
            "workspace": workspace,
            "start_time": self.start_time.isoformat(),
        }

        self._write_entry("RUN_START", header)

    def end_run(self, status: str = "completed", error: str | None = None):
        """End agent run"""
        payload = {
            "status": status,
            "end_time": datetime.now().isoformat(),
        }
        if error:
            payload["error"] = error

        self._write_entry("RUN_END", payload)

    # ==========================================================
    # LLM logging
    # ==========================================================

    def log_llm_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ):
        data = {
            "messages": [],
            "tools": [tool.name for tool in tools] if tools else [],
        }

        for msg in messages:
            item = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.thinking:
                item["thinking"] = msg.thinking
            if msg.tool_calls:
                item["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            if msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            if msg.name:
                item["name"] = msg.name

            data["messages"].append(item)

        self._write_entry("LLM_REQUEST", data)

    def log_llm_response(
        self,
        content: str,
        thinking: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        finish_reason: str | None = None,
    ):
        data = {
            "content": content,
        }

        if thinking:
            data["thinking"] = thinking
        if tool_calls:
            data["tool_calls"] = [tc.model_dump() for tc in tool_calls]
        if finish_reason:
            data["finish_reason"] = finish_reason

        self._write_entry("LLM_RESPONSE", data)

    # ==========================================================
    # Tool logging
    # ==========================================================

    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ):
        self._write_entry(
            "TOOL_CALL",
            {
                "tool": tool_name,
                "arguments": arguments,
            },
        )

    def log_tool_result(
        self,
        tool_name: str,
        success: bool,
        result: str | None = None,
        error: str | None = None,
    ):
        payload = {
            "tool": tool_name,
            "success": success,
        }
        if success:
            payload["result"] = result
        else:
            payload["error"] = error

        self._write_entry("TOOL_RESULT", payload)

    # ==========================================================
    # Generic / extensible events
    # ==========================================================

    def log_event(self, event_type: str, payload: dict[str, Any]):
        """Log custom framework events (memory, planning, etc.)"""
        self._write_entry(event_type.upper(), payload)

    # ==========================================================
    # Internal
    # ==========================================================

    def _write_entry(self, entry_type: str, payload: dict[str, Any]):
        if not self.log_file:
            return

        self.log_index += 1

        entry = {
            "index": self.log_index,
            "type": entry_type,
            "timestamp": datetime.now().isoformat(),
            "payload": payload,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_log_file_path(self) -> Optional[Path]:
        return self.log_file
