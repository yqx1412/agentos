"""A7 'done when': common escape attempts are blocked.

Every test here is an escape attempt (or a check that a defence is in place). They run real
child processes, on Windows (Job Object) and on Linux in CI (rlimits + process groups).
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agentos.builtin_tools import builtin_tools
from agentos.llm import ToolCall
from agentos.sandbox import (
    RunResult,
    SandboxConfig,
    run_command,
    run_python,
    sandbox_tools,
    validate_command,
)
from agentos.tools import Policy, ToolError, ToolRegistry

CFG = SandboxConfig(timeout=8, memory_mb=256)


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    w = tmp_path / "ws"
    w.mkdir()
    (tmp_path / "secret.txt").write_text("TOPSECRET", encoding="utf-8")
    (w / "inside.txt").write_text("hello", encoding="utf-8")
    (w / ".git").mkdir()
    (w / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    return w


def py(ws: Path, code: str, cfg: SandboxConfig = CFG) -> RunResult:
    return run_python(ws, cfg, code)


def blocked(r: RunResult) -> bool:
    return r.returncode not in (0, None) and (
        "sandbox:" in r.stderr or "PermissionError" in r.stderr
    )


# -- run_python: what still works ------------------------------------------------------


def test_python_normal_work_inside_the_workspace(ws: Path) -> None:
    r = py(
        ws,
        "import json, math, collections, pathlib\n"
        "print(open('inside.txt').read())\n"
        "pathlib.Path('sub').mkdir()\n"
        "open('sub/out.json', 'w').write(json.dumps({'pi': round(math.pi, 2)}))\n"
        "print(sorted(p.name for p in pathlib.Path('.').iterdir()))\n",
    )
    assert r.returncode == 0, r.stderr
    assert "hello" in r.stdout and "sub" in r.stdout
    assert (ws / "sub" / "out.json").read_text() == '{"pi": 3.14}'


# -- run_python: file system escapes ---------------------------------------------------

FILE_ESCAPES = {
    "relative read": "print(open('../secret.txt').read())",
    "absolute read": "print(open({secret!r}).read())",
    "chdir then read": "import os; os.chdir('..'); print(open('secret.txt').read())",
    "list parent": "import os; print(os.listdir('..'))",
    "pathlib read": "import pathlib; print(pathlib.Path('..', 'secret.txt').read_text())",
    "write outside": "open('../planted.txt', 'w').write('x')",
    "delete outside": "import os; os.remove('../secret.txt')",
    "rename out": "import os; os.rename('inside.txt', '../moved.txt')",
    "write git config": "open('.git/config', 'a').write('[core]\\n fsmonitor = evil\\n')",
    "write git hook": "open('.git/hooks/pre-commit', 'w').write('evil')",
    "os.open write": "import os; os.open('../planted.txt', os.O_WRONLY | os.O_CREAT)",
    "shutil copy out": "import shutil; shutil.copyfile('inside.txt', '../copied.txt')",
}


@pytest.mark.parametrize("name", FILE_ESCAPES)
def test_python_file_escapes_are_blocked(ws: Path, name: str) -> None:
    code = FILE_ESCAPES[name].format(secret=str(ws.parent / "secret.txt"))
    r = py(ws, code)
    assert blocked(r), (name, r)
    assert "TOPSECRET" not in r.stdout
    assert (ws.parent / "secret.txt").exists()
    for planted in ("planted.txt", "moved.txt", "copied.txt"):
        assert not (ws.parent / planted).exists()
    assert "evil" not in (ws / ".git" / "config").read_text()


def test_python_symlink_out_of_the_workspace_is_followed_and_blocked(ws: Path) -> None:
    link = ws / "link.txt"
    try:
        link.symlink_to(ws.parent / "secret.txt")
    except OSError:
        pytest.skip("creating symlinks needs extra privileges on this machine")
    r = py(ws, "print(open('link.txt').read())")
    assert blocked(r) and "TOPSECRET" not in r.stdout
    r = py(ws, "import os; os.symlink('../secret.txt', 'l2.txt')")
    assert blocked(r)


# -- run_python: processes, network, native code ---------------------------------------

PROCESS_ESCAPES = {
    "subprocess": "import subprocess; subprocess.run(['whoami'])",
    "os.system": "import os; os.system('whoami')",
    "os.popen": "import os; os.popen('whoami').read()",
    "os.exec": "import os, sys; os.execv(sys.executable, [sys.executable, '-c', 'print(1)'])",
    "multiprocessing": "import multiprocessing",
    "ctypes": "import ctypes",
    "_ctypes via importlib": "import importlib; importlib.import_module('_ctypes')",
    "__import__": "__import__('ctypes')",
    "socket": "import socket; socket.create_connection(('1.1.1.1', 80))",
    "urllib": "import urllib.request; urllib.request.urlopen('http://example.com')",
    "sys.modules": "import sys; sys.modules['subprocess'].run(['whoami'])",
    "second hook": ("import sys, os\nsys.addaudithook(lambda *a: None)\nos.system('whoami')"),
}


@pytest.mark.parametrize("name", PROCESS_ESCAPES)
def test_python_process_network_and_native_escapes_are_blocked(ws: Path, name: str) -> None:
    r = py(ws, PROCESS_ESCAPES[name])
    assert r.returncode not in (0, None), (name, r)
    assert ("sandbox:" in r.stderr) or ("KeyError" in r.stderr), (name, r.stderr)


def test_python_does_not_see_the_parents_secrets(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTOS_TEST_API_KEY", "sk-should-not-leak")
    r = py(ws, "import os; print(os.environ.get('AGENTOS_TEST_API_KEY'))")
    assert r.returncode == 0 and r.stdout.strip() == "None"


# -- run_python: resources -------------------------------------------------------------


def test_python_infinite_loop_is_stopped_at_the_timeout(ws: Path) -> None:
    start = time.perf_counter()
    r = py(ws, "while True: pass", SandboxConfig(timeout=2))
    assert r.timed_out and r.returncode is None
    assert time.perf_counter() - start < 8
    assert "TIMED OUT" in r.render(1000)


def test_python_memory_bomb_is_capped(ws: Path) -> None:
    r = py(ws, "x = bytearray(2_000_000_000)\nprint('allocated')")
    assert "allocated" not in r.stdout
    assert r.returncode not in (0, None)


def test_python_output_flood_is_capped(ws: Path) -> None:
    r = py(ws, "import sys\nfor _ in range(200_000): sys.stdout.write('A' * 100)")
    assert len(r.render(5000)) < 6000
    assert "truncated" in r.stdout or "cut" in r.render(5000)


# -- run_command: rules ----------------------------------------------------------------

CMD = SandboxConfig(timeout=8, memory_mb=512, allowed_commands=["git"])
PATH = os.environ.get("PATH", "")

COMMAND_ESCAPES = {
    "shell": ["powershell", "-c", "whoami"],
    "cmd": ["cmd", "/c", "whoami"],
    "bash": ["bash", "-c", "whoami"],
    "curl": ["curl", "http://example.com"],
    "exe by path": [str(Path(sys.executable)), "-c", "print(1)"],
    "git -c": ["git", "-c", "core.pager=whoami", "log"],
    "git -C outside": ["git", "-C", "..", "status"],
    "git global first": ["git", "--git-dir=../x", "status"],
    "git config": ["git", "config", "alias.x", "!whoami"],
    "git remote": ["git", "fetch", "https://example.com/r.git"],
    "git ext-diff": ["git", "diff", "--ext-diff"],
    "git relative out": ["git", "add", "../secret.txt"],
    "git absolute out": [
        "git",
        "add",
        "C:\\Windows\\win.ini" if os.name == "nt" else "/etc/passwd",
    ],
    "git option path out": ["git", "diff", "--output=../planted.txt"],
    "git home": ["git", "add", "~/x"],
    "git into .git": ["git", "add", ".git/config"],
}


@pytest.mark.parametrize("name", COMMAND_ESCAPES)
def test_command_escapes_are_refused(ws: Path, name: str) -> None:
    with pytest.raises(ToolError):
        validate_command(COMMAND_ESCAPES[name], ws, CMD, PATH)
    assert not (ws.parent / "planted.txt").exists()


def test_command_refuses_batch_files(ws: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # cmd.exe re-parses a batch file's arguments, so argv quoting no longer protects them.
    monkeypatch.setattr(shutil, "which", lambda name, path=None: "C:\\tools\\git.cmd")
    with pytest.raises(ToolError, match="batch"):
        validate_command(["git", "status"], ws, CMD, PATH)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_command_git_works_inside_the_workspace(ws: Path) -> None:
    shutil.rmtree(ws / ".git")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "inside.txt"],
        ["git", "commit", "-qm", "x"],
    ):
        r = run_command(ws, CMD, argv)
        assert r.returncode == 0, (argv, r)
    r = run_command(ws, CMD, ["git", "log", "--format=%an %s"])
    assert r.stdout.strip() == "agentos x"  # the sandbox identity, not the user's config


# -- run_command: OS limits (python is allowlisted here ONLY to drive the tests) -------

UNSAFE = SandboxConfig(timeout=3, memory_mb=256, max_processes=4, allowed_commands=["python"])
PY_NAME = Path(getattr(sys, "_base_executable", sys.executable)).stem


@pytest.fixture
def py_on_path(monkeypatch: pytest.MonkeyPatch) -> SandboxConfig:
    exe_dir = str(Path(getattr(sys, "_base_executable", sys.executable)).parent)
    monkeypatch.setenv("PATH", exe_dir + os.pathsep + os.environ.get("PATH", ""))
    return UNSAFE.model_copy(update={"allowed_commands": [PY_NAME]})


def test_timeout_stops_the_whole_process_tree(ws: Path, py_on_path: SandboxConfig) -> None:
    # The child starts a grandchild that keeps writing a heartbeat file, then both hang.
    beat = ws / "beat.txt"
    grandchild = (
        "import time\nwhile True:\n    open('beat.txt', 'a').write('.')\n    time.sleep(0.1)\n"
    )
    child = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        "time.sleep(60)\n"
    )
    r = run_command(ws, py_on_path, [PY_NAME, "-c", child])
    assert r.timed_out
    assert beat.exists(), r.stderr  # the grandchild really ran
    size = beat.stat().st_size
    time.sleep(1.0)
    assert beat.stat().st_size == size  # ...and is gone now


def test_process_count_is_capped(ws: Path, py_on_path: SandboxConfig) -> None:
    code = (
        "import subprocess, sys\n"
        "ok = 0\n"
        "for _ in range(20):\n"
        "    try:\n"
        "        subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'])\n"
        "        ok += 1\n"
        "    except OSError:\n"
        "        break\n"
        "print(ok)\n"
    )
    r = run_command(ws, py_on_path, [PY_NAME, "-c", code])
    if sys.platform != "win32":
        pytest.skip("per-call process caps are a Job Object feature; POSIX uses a group kill")
    assert int(r.stdout.strip()) < py_on_path.max_processes


def test_command_does_not_see_the_parents_secrets(
    ws: Path, py_on_path: SandboxConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTOS_TEST_API_KEY", "sk-should-not-leak")
    r = run_command(
        ws, py_on_path, [PY_NAME, "-c", "import os; print(os.environ.get('AGENTOS_TEST_API_KEY'))"]
    )
    assert r.stdout.strip() == "None"


# -- permissions -----------------------------------------------------------------------


def _call(name: str, **args: object) -> ToolCall:
    return ToolCall(name=name, arguments=dict(args))


def test_dangerous_tools_are_denied_without_an_approver(ws: Path) -> None:
    reg = ToolRegistry([*builtin_tools(ws), *sandbox_tools(ws, CFG)])
    r = reg.execute(_call("run_python", code="open('pwned.txt','w').write('x')"))
    assert not r.ok and "permission denied" in r.output and "needs approval" in r.output
    assert not (ws / "pwned.txt").exists()
    assert reg.execute(_call("write_file", path="ok.txt", content="x")).ok  # write is auto


def test_the_approver_sees_the_validated_call_and_decides(ws: Path) -> None:
    seen: list[tuple[str, dict]] = []

    def approver(tool, args):
        seen.append((tool.name, args))
        return "fine" in args["code"]

    reg = ToolRegistry(sandbox_tools(ws, CFG), policy=Policy(approver=approver))
    assert reg.execute(_call("run_python", code="print('fine')")).ok
    denied = reg.execute(_call("run_python", code="print('nope')"))
    assert not denied.ok and "did not approve" in denied.output
    assert seen == [
        ("run_python", {"code": "print('fine')"}),
        ("run_python", {"code": "print('nope')"}),
    ]
    # Invalid arguments never reach the approver.
    assert not reg.execute(_call("run_python")).ok
    assert len(seen) == 2


def test_a_crashing_approver_denies(ws: Path) -> None:
    def approver(tool, args):
        raise RuntimeError("no tty")

    reg = ToolRegistry(sandbox_tools(ws, CFG), policy=Policy(approver=approver))
    r = reg.execute(_call("run_python", code="print(1)"))
    assert not r.ok and "approval failed" in r.output


def test_auto_approve_levels(ws: Path) -> None:
    read_only = ToolRegistry(builtin_tools(ws), policy=Policy(auto_approve="read"))
    assert read_only.execute(_call("read_file", path="inside.txt")).ok
    assert not read_only.execute(_call("write_file", path="x.txt", content="x")).ok
    everything = ToolRegistry(sandbox_tools(ws, CFG), policy=Policy(auto_approve="dangerous"))
    assert everything.execute(_call("run_python", code="print(1)")).ok
    with pytest.raises(ValueError):
        Policy(auto_approve="root")


def test_builtin_write_file_cannot_touch_git_internals(ws: Path) -> None:
    reg = ToolRegistry(builtin_tools(ws))
    for path in (".git/config", ".git/hooks/pre-commit", "sub/../.git/config", ".GIT/config"):
        r = reg.execute(_call("write_file", path=path, content="[core]\n fsmonitor = evil"))
        assert not r.ok and "protected" in r.output, path
    assert "evil" not in (ws / ".git" / "config").read_text()


def test_sandbox_tools_are_dangerous(ws: Path) -> None:
    assert {t.permission for t in sandbox_tools(ws)} == {"dangerous"}
    assert {t.name: t.permission for t in builtin_tools(ws)} == {
        "read_file": "read",
        "write_file": "write",
        "calculator": "read",
    }


def test_git_is_really_blocked_from_running_programs_via_config(ws: Path) -> None:
    """End to end: the classic 'git config core.fsmonitor' escape fails at every step."""
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    shutil.rmtree(ws / ".git")
    reg = ToolRegistry(
        [*builtin_tools(ws), *sandbox_tools(ws, CMD)], policy=Policy(auto_approve="dangerous")
    )
    assert reg.execute(_call("run_command", command=["git", "init", "-q"])).ok
    steps = [
        _call("run_command", command=["git", "config", "core.fsmonitor", "whoami"]),
        _call("run_command", command=["git", "-c", "core.fsmonitor=whoami", "status"]),
        _call("write_file", path=".git/config", content="[core]\n\tfsmonitor = whoami\n"),
        _call(
            "run_python", code="open('.git/config','a').write('[core]\\n\\tfsmonitor = whoami\\n')"
        ),
    ]
    for step in steps:
        r = reg.execute(step)
        assert ("ERROR" in r.as_message_content()) or "sandbox:" in r.output, step
    out = subprocess.run(
        ["git", "config", "--local", "--get", "core.fsmonitor"],
        cwd=ws,
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == ""


# -- CLI and config --------------------------------------------------------------------


def test_cli_approval_prompt_defaults_to_no(
    ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agentos.cli import ask_approval

    tool = sandbox_tools(ws, CFG)[0]
    for reply, expected in (("y", True), ("YES", True), ("", False), ("n", False)):
        monkeypatch.setattr("builtins.input", lambda _prompt, r=reply: r)
        assert ask_approval(tool, {"code": "print(1)"}) is expected
    assert '"code": "print(1)"' in capsys.readouterr().err  # the exact call is shown

    def eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert ask_approval(tool, {"code": "x"}) is False


def test_config_sandbox_and_mcp_permission(tmp_path: Path) -> None:
    from agentos.config import ConfigError, load_config

    cfg_file = tmp_path / "agentos.toml"
    cfg_file.write_text(
        '[sandbox]\nallowed_commands = ["git", "uv"]\ntimeout = 5\n\n'
        '[mcp_servers.textkit]\ncommand = "x"\npermission = "read"\n',
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.sandbox.allowed_commands == ["git", "uv"] and cfg.sandbox.timeout == 5
    assert cfg.mcp_servers["textkit"].permission == "read"
    cfg_file.write_text('[mcp_servers.x]\ncommand = "x"\npermission = "root"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg_file)
