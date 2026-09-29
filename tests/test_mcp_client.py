"""MCP client tests against real stdio servers (textkit and a flaky fixture), no Ollama."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_agent import ScriptedLLM, assistant  # tests/ is on sys.path under pytest

from agentos.agent import Agent
from agentos.builtin_tools import builtin_tools
from agentos.cli import main
from agentos.config import MCPServerConfig
from agentos.llm import Message, ToolCall
from agentos.mcp_client import MCPError, MCPManager
from agentos.tools import ToolRegistry

FIXTURES = Path(__file__).parent / "fixtures"
TEXTKIT = MCPServerConfig(command=sys.executable, args=["-m", "agentos.mcp_servers.textkit"])
FLAKY = MCPServerConfig(command=sys.executable, args=[str(FIXTURES / "flaky_server.py")], timeout=2)


@pytest.fixture(scope="module")
def textkit() -> Iterator[MCPManager]:
    with MCPManager({"textkit": TEXTKIT}) as m:
        yield m


def call(reg: ToolRegistry, name: str, **args: object):
    return reg.execute(ToolCall(name=name, arguments=args))


def test_tools_are_listed_with_prefix_and_schema(textkit: MCPManager) -> None:
    reg = ToolRegistry(textkit.tools())
    assert reg.names() == [
        "textkit__text_stats",
        "textkit__word_frequency",
        "textkit__find_lines",
    ]
    params = next(s for s in reg.schemas() if s["function"]["name"] == "textkit__find_lines")
    params = params["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"text", "pattern"}
    assert all(t.source == "mcp:textkit" for t in textkit.tools())


def test_successful_call(textkit: MCPManager) -> None:
    res = call(ToolRegistry(textkit.tools()), "textkit__text_stats", text="a b\nc")
    assert res.ok
    assert '"words": 3' in res.output


def test_server_side_errors_become_error_results(textkit: MCPManager) -> None:
    reg = ToolRegistry(textkit.tools())
    bad_regex = call(reg, "textkit__find_lines", text="x", pattern="(")
    assert not bad_regex.ok
    assert "invalid regex" in bad_regex.output
    missing_arg = call(reg, "textkit__text_stats")
    assert not missing_arg.ok
    assert "text" in missing_arg.output
    assert missing_arg.as_message_content().startswith("ERROR:")


def test_allowlist_filters_tools() -> None:
    cfg = TEXTKIT.model_copy(update={"tools": ["find_lines"]})
    with MCPManager({"tk": cfg}) as m:
        assert [t.name for t in m.tools()] == ["tk__find_lines"]


def test_allowlist_with_unknown_tool_fails_startup() -> None:
    cfg = TEXTKIT.model_copy(update={"tools": ["nope"]})
    with pytest.raises(MCPError, match="no tool"), MCPManager({"tk": cfg}):
        pass


def test_missing_command_fails_startup() -> None:
    cfg = MCPServerConfig(command="agentos-no-such-command-xyz")
    with pytest.raises(MCPError, match="'broken' failed to start"), MCPManager({"broken": cfg}):
        pass


def test_hung_server_times_out_at_startup() -> None:
    cfg = MCPServerConfig(
        command=sys.executable, args=["-c", "import time; time.sleep(60)"], startup_timeout=1
    )
    with pytest.raises(MCPError, match="did not start within"), MCPManager({"hung": cfg}):
        pass


def test_flaky_server_failures_never_raise() -> None:
    with MCPManager({"flaky": FLAKY}) as m:
        reg = ToolRegistry(m.tools())
        assert call(reg, "flaky__sleep", seconds=0).ok

        failed = call(reg, "flaky__fail", message="disk full")
        assert not failed.ok
        assert "disk full" in failed.output

        slow = call(reg, "flaky__sleep", seconds=10)
        assert not slow.ok
        assert "timed out" in slow.output.lower()

        died = call(reg, "flaky__exit_process")
        assert not died.ok
        after = call(reg, "flaky__sleep", seconds=0)
        assert not after.ok


def test_agent_uses_builtin_and_mcp_tools_together(textkit: MCPManager, tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("to be or not to be", encoding="utf-8")
    llm = ScriptedLLM(
        [
            assistant("", ("read_file", {"path": "notes.txt"})),
            assistant("", ("textkit__word_frequency", {"text": "to be or not to be", "top_n": 1})),
            assistant("The most frequent word is 'to'."),
        ]
    )
    reg = ToolRegistry([*builtin_tools(tmp_path), *textkit.tools()])
    result = Agent(llm, reg).run("most frequent word in notes.txt?")
    assert result.stop_reason == "final_answer"
    assert result.tool_errors == 0
    tool_msgs = [m for m in result.messages if isinstance(m, Message) and m.role == "tool"]
    assert '"word": "to"' in tool_msgs[-1].content


def test_cli_tools_lists_configured_servers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "agentos.toml"
    cfg.write_text(
        "[mcp_servers.tk]\n"
        'command = "{python}"\n'
        'args = ["-m", "agentos.mcp_servers.textkit"]\n'
        'tools = ["text_stats"]\n',
        encoding="utf-8",
    )
    assert main(["tools", "--config", str(cfg), "--workspace", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "read_file" in out
    assert "tk__text_stats" in out
    assert "mcp:tk" in out
    assert "word_frequency" not in out


def test_cli_reports_config_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "agentos.toml"
    cfg.write_text("[mcp_servers.x]\n", encoding="utf-8")
    assert main(["tools", "--config", str(cfg), "--workspace", str(tmp_path)]) == 2
    assert "config error" in capsys.readouterr().err
