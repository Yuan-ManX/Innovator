"""
System Execution Tools for Innovator
Shell command execution with background process management.

Supports:
- Unix/Linux/macOS (bash)
- Windows (PowerShell)

Part of Innovator — A General AI Agent Framework for Animation, Film & Game Creation
"""

from __future__ import annotations

import asyncio
import platform
import re
import time
import uuid
from typing import Any, Dict, Optional

from pydantic import Field, model_validator

from .base import Tool, ToolResult


# ============================================================
# Execution Result
# ============================================================

class ShellExecutionResult(ToolResult):
    """
    Standardized shell execution result for Innovator Agents.
    """

    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    exit_code: int = Field(default=0, description="Process exit code")
    process_id: Optional[str] = Field(default=None, description="Background process ID")

    @model_validator(mode="after")
    def auto_format_content(self) -> "ShellExecutionResult":
        """
        Generate LLM-friendly content automatically.
        """
        if self.content:
            return self

        parts = []
        if self.stdout:
            parts.append(self.stdout.strip())
        if self.stderr:
            parts.append(f"[stderr]\n{self.stderr.strip()}")
        if self.process_id:
            parts.append(f"[process_id]: {self.process_id}")
        if self.exit_code:
            parts.append(f"[exit_code]: {self.exit_code}")

        self.content = "\n\n".join(parts) if parts else "(no output)"
        return self


# ============================================================
# Background Process Model
# ============================================================

class BackgroundProcess:
    """
    Pure data container for a background shell process.
    """

    def __init__(
        self,
        process_id: str,
        command: str,
        process: asyncio.subprocess.Process,
        start_time: float,
    ):
        self.process_id = process_id
        self.command = command
        self.process = process
        self.start_time = start_time

        self.output_lines: list[str] = []
        self.last_read_index = 0
        self.status: str = "running"
        self.exit_code: Optional[int] = None

    def add_output(self, line: str):
        self.output_lines.append(line)

    def read_new_output(self, pattern: Optional[str] = None) -> list[str]:
        lines = self.output_lines[self.last_read_index :]
        self.last_read_index = len(self.output_lines)

        if pattern:
            try:
                regex = re.compile(pattern)
                lines = [l for l in lines if regex.search(l)]
            except re.error:
                pass

        return lines

    async def terminate(self):
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()

        self.exit_code = self.process.returncode
        self.status = "terminated"


# ============================================================
# Background Process Manager
# ============================================================

class BackgroundProcessManager:
    """
    Global registry for background shell processes.
    """

    _processes: Dict[str, BackgroundProcess] = {}
    _monitors: Dict[str, asyncio.Task] = {}

    @classmethod
    def add(cls, proc: BackgroundProcess):
        cls._processes[proc.process_id] = proc

    @classmethod
    def get(cls, pid: str) -> Optional[BackgroundProcess]:
        return cls._processes.get(pid)

    @classmethod
    def list_ids(cls) -> list[str]:
        return list(cls._processes.keys())

    @classmethod
    async def monitor(cls, pid: str):
        proc = cls.get(pid)
        if not proc:
            return

        async def _monitor():
            try:
                while proc.process.returncode is None:
                    line = await proc.process.stdout.readline()
                    if not line:
                        await asyncio.sleep(0.1)
                        continue
                    proc.add_output(line.decode("utf-8", errors="replace").rstrip())

                proc.exit_code = await proc.process.wait()
                proc.status = "completed" if proc.exit_code == 0 else "failed"
            finally:
                cls._monitors.pop(pid, None)

        cls._monitors[pid] = asyncio.create_task(_monitor())

    @classmethod
    async def terminate(cls, pid: str) -> BackgroundProcess:
        proc = cls._processes.pop(pid, None)
        if not proc:
            raise ValueError(f"Process not found: {pid}")

        await proc.terminate()
        return proc


# ============================================================
# Shell Execution Tool
# ============================================================

class ShellTool(Tool):
    """
    Execute system shell commands.

    Intended for:
    - Environment setup
    - Build / render orchestration
    - Engine / simulator launch
    """

    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.shell_name = "PowerShell" if self.is_windows else "bash"

    @property
    def name(self) -> str:
        return "shell_exec"

    @property
    def description(self) -> str:
        return f"Execute {self.shell_name} commands (foreground or background)."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 120},
                "background": {"type": "boolean", "default": False},
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        timeout: int = 120,
        background: bool = False,
    ) -> ShellExecutionResult:

        try:
            if self.is_windows:
                cmd = ["powershell.exe", "-NoProfile", "-Command", command]
                spawn = asyncio.create_subprocess_exec
            else:
                cmd = command
                spawn = asyncio.create_subprocess_shell

            if background:
                pid = str(uuid.uuid4())[:8]
                process = await spawn(
                    *cmd if isinstance(cmd, list) else cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )

                proc = BackgroundProcess(pid, command, process, time.time())
                BackgroundProcessManager.add(proc)
                await BackgroundProcessManager.monitor(pid)

                return ShellExecutionResult(
                    success=True,
                    process_id=pid,
                    data={"status": "running"},
                    content=f"Process started in background (id={pid})",
                )

            # Foreground
            process = await spawn(
                *cmd if isinstance(cmd, list) else cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            return ShellExecutionResult(
                success=process.returncode == 0,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                exit_code=process.returncode or 0,
            )

        except Exception as e:
            return ShellExecutionResult(success=False, error=str(e))


# ============================================================
# Background Output Tool
# ============================================================

class ShellOutputTool(Tool):
    """
    Retrieve output from a background shell process.
    """

    @property
    def name(self) -> str:
        return "shell_output"

    @property
    def description(self) -> str:
        return "Retrieve incremental output from a background shell process."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "process_id": {"type": "string"},
                "filter": {"type": "string"},
            },
            "required": ["process_id"],
        }

    async def execute(
        self,
        process_id: str,
        filter: Optional[str] = None,
    ) -> ShellExecutionResult:

        proc = BackgroundProcessManager.get(process_id)
        if not proc:
            return ShellExecutionResult(
                success=False,
                error=f"Process not found: {process_id}",
            )

        lines = proc.read_new_output(filter)
        return ShellExecutionResult(
            success=True,
            stdout="\n".join(lines),
            process_id=process_id,
            data={"status": proc.status},
        )


# ============================================================
# Background Termination Tool
# ============================================================

class ShellKillTool(Tool):
    """
    Terminate a background shell process.
    """

    @property
    def name(self) -> str:
        return "shell_kill"

    @property
    def description(self) -> str:
        return "Terminate a background shell process."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "process_id": {"type": "string"},
            },
            "required": ["process_id"],
        }

    async def execute(self, process_id: str) -> ShellExecutionResult:
        try:
            proc = await BackgroundProcessManager.terminate(process_id)
            return ShellExecutionResult(
                success=True,
                process_id=process_id,
                data={"final_status": proc.status},
            )
        except Exception as e:
            return ShellExecutionResult(success=False, error=str(e))
