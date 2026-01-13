"""
File operation tools for Innovator Agent Framework.

Includes:
- ReadTool: Read file content with line numbers and token-safe truncation
- WriteTool: Write full content to a file
- EditTool: Perform exact string replacement in a file

All tools:
- Are sandboxed to a workspace directory
- Follow Innovator Tool Base conventions
- Return ToolResult consistently
"""

from pathlib import Path
from typing import Any, Optional

import tiktoken

from .base import Tool, ToolResult


# ============================================================
# Token-safe truncation utility
# ============================================================

def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text by token count using a head+tail strategy.

    Keeps the beginning and end of the text, truncating the middle
    when the token limit is exceeded.

    Args:
        text: Input text
        max_tokens: Maximum allowed token count

    Returns:
        Truncated or original text
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Estimate char/token ratio
    char_count = len(text)
    ratio = len(tokens) / max(char_count, 1)

    # Allocate tokens to head and tail (with safety margin)
    chars_per_side = int((max_tokens / 2) / ratio * 0.95)

    head = text[:chars_per_side]
    tail = text[-chars_per_side:]

    # Align to line boundaries if possible
    if "\n" in head:
        head = head[: head.rfind("\n")]
    if "\n" in tail:
        tail = tail[tail.find("\n") + 1 :]

    note = (
        f"\n\n... [Content truncated: {len(tokens)} tokens → "
        f"~{max_tokens} token limit] ...\n\n"
    )

    return head + note + tail


# ============================================================
# Base class for workspace-aware file tools
# ============================================================

class WorkspaceTool(Tool):
    """Base class for file tools with workspace isolation."""

    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir).resolve()

    def _resolve_path(self, path: str) -> Path:
        """Resolve path safely within workspace."""
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.workspace_dir / file_path

        file_path = file_path.resolve()

        if not str(file_path).startswith(str(self.workspace_dir)):
            raise ValueError("Access denied: path outside workspace")

        return file_path


# ============================================================
# Read Tool
# ============================================================

class ReadTool(WorkspaceTool):
    """Read file contents with line numbers."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read a text file from the workspace. "
            "Output includes line numbers in the format 'LINE|CONTENT'. "
            "Supports reading partial content using offset and limit for large files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path to the file",
                },
                "offset": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                },
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> ToolResult:
        try:
            file_path = self._resolve_path(path)

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {path}",
                )

            lines = file_path.read_text(encoding="utf-8").splitlines()

            start = max((offset - 1), 0) if offset else 0
            end = start + limit if limit else len(lines)
            end = min(end, len(lines))

            numbered = [
                f"{i + 1:6d}|{lines[i]}"
                for i in range(start, end)
            ]

            content = "\n".join(numbered)

            content = truncate_text_by_tokens(content, max_tokens=32000)

            return ToolResult(success=True, content=content)

        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ============================================================
# Write Tool
# ============================================================

class WriteTool(WorkspaceTool):
    """Write full content to a file (overwrite)."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write complete content to a file. "
            "This operation overwrites existing files. "
            "Read the file first before overwriting when modifying existing content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path to the file",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str) -> ToolResult:
        try:
            file_path = self._resolve_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_path.write_text(content, encoding="utf-8")

            return ToolResult(
                success=True,
                content=f"File written successfully: {file_path}",
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ============================================================
# Edit Tool
# ============================================================

class EditTool(WorkspaceTool):
    """Edit a file by exact string replacement."""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Replace an exact string in a file. "
            "The old string must appear exactly once. "
            "You must read the file before editing."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative or absolute path to the file",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact string to be replaced (must be unique)",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string",
                },
            },
            "required": ["path", "old_str", "new_str"],
        }

    async def execute(self, path: str, old_str: str, new_str: str) -> ToolResult:
        try:
            file_path = self._resolve_path(path)

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {path}",
                )

            content = file_path.read_text(encoding="utf-8")

            count = content.count(old_str)
            if count == 0:
                return ToolResult(
                    success=False,
                    error="Target string not found in file",
                )
            if count > 1:
                return ToolResult(
                    success=False,
                    error="Target string is not unique; edit aborted",
                )

            new_content = content.replace(old_str, new_str)
            file_path.write_text(new_content, encoding="utf-8")

            return ToolResult(
                success=True,
                content=f"File edited successfully: {file_path}",
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))
