"""
Innovator Configuration Management

Unified configuration system for Innovator AI Agent Framework,
supporting creative agents for animation, film, and game pipelines.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ============================================================
# Retry & LLM
# ============================================================

class RetryConfig(BaseModel):
    """Retry policy for model calls"""

    enabled: bool = True
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


class LLMConfig(BaseModel):
    """LLM / Multimodal model configuration"""

    api_key: str
    api_base: str = ""
    model: str = "anthropic"
    provider: str = "openai"  # anthropic | openai | custom
    retry: RetryConfig = Field(default_factory=RetryConfig)


# ============================================================
# Agent Runtime
# ============================================================

class AgentConfig(BaseModel):
    """Agent behavior & execution limits"""

    max_steps: int = 50
    workspace_dir: str = "./workspace"
    system_prompt_path: str = "./prompts/system_prompt.md"


# ============================================================
# Memory
# ============================================================

class MemoryConfig(BaseModel):
    """Agent long-term memory configuration"""

    enabled: bool = True
    memory_file: str = "./workspace/.agent_memory.json"
    max_entries: Optional[int] = None  # None = unlimited
    auto_compact: bool = False


# ============================================================
# MCP
# ============================================================

class MCPConfig(BaseModel):
    """Model Context Protocol configuration"""

    connect_timeout: float = 10.0
    execute_timeout: float = 60.0
    sse_read_timeout: float = 120.0


# ============================================================
# Tools & Skills
# ============================================================

class ToolsConfig(BaseModel):
    """Tooling configuration"""

    # Core tools
    enable_file_tools: bool = True
    enable_bash: bool = True

    # Memory
    enable_memory: bool = True

    # Skills
    enable_skills: bool = True
    skills_dir: str = "./skills"

    # MCP
    enable_mcp: bool = True
    mcp_config_path: str = "mcp.json"
    mcp: MCPConfig = Field(default_factory=MCPConfig)


# ============================================================
# Main Config
# ============================================================

class Config(BaseModel):
    """Innovator main configuration"""

    llm: LLMConfig
    agent: AgentConfig
    memory: MemoryConfig
    tools: ToolsConfig

    # --------------------------
    # Loading
    # --------------------------

    @classmethod
    def load(cls) -> "Config":
        config_path = cls.get_default_config_path()
        if not config_path.exists():
            raise FileNotFoundError(
                "Configuration file not found. "
                "Place config.yaml in innovator/config/, ~/.innovator/config/, "
                "or the package config directory."
            )
        return cls.from_yaml(config_path)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        config_path = Path(config_path)

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # --------------------------
        # LLM
        # --------------------------
        if not data.get("api_key") or data["api_key"] == "YOUR_API_KEY_HERE":
            raise ValueError("Please configure a valid API key")

        retry_cfg = RetryConfig(**data.get("retry", {}))

        llm_cfg = LLMConfig(
            api_key=data["api_key"],
            api_base=data.get("api_base", llm_cfg_default("api_base")),
            model=data.get("model", llm_cfg_default("model")),
            provider=data.get("provider", "anthropic"),
            retry=retry_cfg,
        )

        # --------------------------
        # Agent
        # --------------------------
        agent_cfg = AgentConfig(
            max_steps=data.get("max_steps", 50),
            workspace_dir=data.get("workspace_dir", "./workspace"),
            system_prompt_path=data.get("system_prompt_path", "system_prompt.md"),
        )

        # --------------------------
        # Memory
        # --------------------------
        memory_data = data.get("memory", {})
        memory_cfg = MemoryConfig(
            enabled=memory_data.get("enabled", True),
            memory_file=memory_data.get(
                "memory_file", "./workspace/.agent_memory.json"
            ),
            max_entries=memory_data.get("max_entries"),
            auto_compact=memory_data.get("auto_compact", False),
        )

        # --------------------------
        # Tools
        # --------------------------
        tools_data = data.get("tools", {})
        mcp_cfg = MCPConfig(**tools_data.get("mcp", {}))

        tools_cfg = ToolsConfig(
            enable_file_tools=tools_data.get("enable_file_tools", True),
            enable_bash=tools_data.get("enable_bash", True),
            enable_memory=tools_data.get("enable_memory", True),
            enable_skills=tools_data.get("enable_skills", True),
            skills_dir=tools_data.get("skills_dir", "./skills"),
            enable_mcp=tools_data.get("enable_mcp", True),
            mcp_config_path=tools_data.get("mcp_config_path", "mcp.json"),
            mcp=mcp_cfg,
        )

        return cls(
            llm=llm_cfg,
            agent=agent_cfg,
            memory=memory_cfg,
            tools=tools_cfg,
        )

    # --------------------------
    # Config discovery
    # --------------------------

    @staticmethod
    def get_package_dir() -> Path:
        return Path(__file__).parent

    @classmethod
    def find_config_file(cls, filename: str) -> Optional[Path]:
        candidates = [
            Path.cwd() / "innovator" / "config" / filename,
            Path.home() / ".innovator" / "config" / filename,
            cls.get_package_dir() / "config" / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    @classmethod
    def get_default_config_path(cls) -> Path:
        return cls.find_config_file("config.yaml") or (
            cls.get_package_dir() / "config" / "config.yaml"
        )


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def llm_cfg_default(field: str) -> str:
    defaults = {
        "api_base": "",
        "model": "",
    }
    return defaults[field]
