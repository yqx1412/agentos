"""``agentos.toml``: runtime configuration, currently the MCP servers to connect to.

Example::

    [mcp_servers.filesystem]
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-filesystem", "{workspace}"]
    tools = ["read_text_file", "write_file", "list_directory"]   # optional allowlist

    [mcp_servers.textkit]
    command = "{python}"
    args = ["-m", "agentos.mcp_servers.textkit"]

Placeholders in ``command``, ``args``, ``env`` and ``cwd``: ``{workspace}`` (the resolved
workspace directory) and ``{python}`` (the interpreter running AgentOS).
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentos.sandbox import SandboxConfig

SERVER_NAME_PATTERN = r"^[A-Za-z0-9_-]+$"


class ConfigError(Exception):
    """The config file is missing, unreadable or invalid."""


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    enabled: bool = True
    tools: list[str] | None = Field(
        default=None, description="Only expose these tools from the server (default: all)"
    )
    timeout: float = Field(default=60.0, gt=0, description="Seconds per tool call")
    startup_timeout: float = Field(default=60.0, gt=0, description="Seconds to connect")
    # A7: one level for all of this server's tools. AgentOS cannot see what a remote tool
    # does, so the default assumes it may change things.
    permission: Literal["read", "write", "dangerous"] = "write"

    def expanded(self, workspace: Path) -> MCPServerConfig:
        subs = {"workspace": str(workspace), "python": sys.executable}

        def x(s: str) -> str:
            for key, value in subs.items():
                s = s.replace("{" + key + "}", value)
            return s

        return self.model_copy(
            update={
                "command": x(self.command),
                "args": [x(a) for a in self.args],
                "env": {k: x(v) for k, v in self.env.items()},
                "cwd": x(self.cwd) if self.cwd is not None else None,
            }
        )


class AgentOSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    def enabled_servers(self, workspace: Path) -> dict[str, MCPServerConfig]:
        return {
            name: cfg.expanded(workspace) for name, cfg in self.mcp_servers.items() if cfg.enabled
        }


def load_config(path: Path) -> AgentOSConfig:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    try:
        config = AgentOSConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    bad = [n for n in config.mcp_servers if not re.match(SERVER_NAME_PATTERN, n)]
    if bad:
        raise ConfigError(f"{path}: server names may only use letters, digits, _ and -: {bad}")
    return config
