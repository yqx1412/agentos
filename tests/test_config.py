import sys
from pathlib import Path

import pytest

from agentos.config import AgentOSConfig, ConfigError, MCPServerConfig, load_config


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "agentos.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_servers(tmp_path: Path) -> None:
    cfg = load_config(
        write(
            tmp_path,
            """
[mcp_servers.fs]
command = "npx"
args = ["-y", "server-fs", "{workspace}"]
tools = ["read_text_file"]

[mcp_servers.off]
command = "x"
enabled = false
""",
        )
    )
    assert set(cfg.mcp_servers) == {"fs", "off"}
    assert cfg.mcp_servers["fs"].tools == ["read_text_file"]
    enabled = cfg.enabled_servers(tmp_path)
    assert list(enabled) == ["fs"]
    assert enabled["fs"].args[-1] == str(tmp_path)


def test_python_placeholder() -> None:
    cfg = MCPServerConfig(command="{python}", args=["-m", "x"], env={"W": "{workspace}"})
    out = cfg.expanded(Path("/ws"))
    assert out.command == sys.executable
    assert out.env == {"W": str(Path("/ws"))}


def test_empty_config_is_valid(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, "")) == AgentOSConfig()


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("[mcp_servers.x]\nargs = []\n", "command"),  # missing command
        ("[mcp_servers.x]\ncommand = 'a'\ntypo = 1\n", "typo"),  # unknown key
        ("[mcp_servers.x]\ncommand = 'a'\ntimeout = 0\n", "timeout"),
        ("[mcp_servers.'bad name']\ncommand = 'a'\n", "server names"),
        ("[mcp_servers.x\n", "invalid TOML"),
        ("unknown_section = 1\n", "unknown_section"),
    ],
)
def test_invalid_config(tmp_path: Path, text: str, fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        load_config(write(tmp_path, text))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")
