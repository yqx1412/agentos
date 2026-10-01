"""Sandboxed execution (A7): ``run_python`` and ``run_command``, both permission ``dangerous``.

Layers, outermost first:

1. **Permission.** Both tools are ``dangerous``: by default a human has to approve each call
   (see :class:`agentos.tools.Policy`).
2. **OS limits.** Every child runs with a wall-clock timeout, a memory cap and a cap on how
   many processes it may create, and it is killed together with everything it started.
   Windows: a Job Object (kill-on-close, job memory limit, active-process limit).
   POSIX: rlimits plus a new session whose whole process group is killed.
3. **Python audit hook** (``run_python`` only). Before the agent's code runs, the child
   installs a ``sys.addaudithook`` hook that refuses process creation, network access,
   ``ctypes`` and other native escape hatches, and any file access outside the workspace
   (reads are also allowed inside the Python installation, which imports need).
4. **Command rules** (``run_command`` only). Commands are argv lists, never a shell string.
   Only allowlisted programs run (never ``.bat``/``.cmd`` files, which ``cmd.exe`` would
   re-parse); ``git`` is limited to a list of subcommands and global options such as ``-c``
   are refused; any argument that looks like a path must resolve inside the workspace.
5. **Clean environment.** Children get a minimal environment, so API keys and tokens in the
   parent's environment do not leak, and a private temp directory.

Paths under ``.git`` are protected from every tool, built-in file tools included: writing
``.git/config`` or a hook is the classic way to turn "may run git" into "may run anything".

Limits of this design, stated plainly: an audit hook is not a security boundary (CPython
documents that native code can bypass it, which is why ``ctypes`` and friends are blocked
rather than trusted), and nothing here stops a child from *reading* files inside the
Python installation. It contains a confused or sloppy model; it is not a container. The
human approval step is what stands between a hostile instruction and the machine.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agentos.tools import Tool, ToolError

PROTECTED_DIRS = frozenset({".git"})

# git subcommands that only work on the repository in the working directory. Anything that
# talks to remotes, edits config, or runs user-defined programs (config, submodule, filter
# drivers via -c ...) is left out.
GIT_SUBCOMMANDS = frozenset(
    [
        "init",
        "status",
        "log",
        "diff",
        "show",
        "add",
        "commit",
        "rm",
        "mv",
        "restore",
        "ls-files",
        "rev-parse",
        "branch",
        "tag",
        "shortlog",
        "blame",
        "grep",
    ]
)
GIT_FORBIDDEN_ARGS = ("-c", "--config-env", "--exec-path", "--ext-diff", "--textconv")


class SandboxConfig(BaseModel):
    """``[sandbox]`` in agentos.toml."""

    model_config = ConfigDict(extra="forbid")

    allowed_commands: list[str] = Field(default_factory=lambda: ["git"])
    timeout: float = Field(default=30.0, gt=0, le=600, description="Seconds per call")
    memory_mb: int = Field(default=512, ge=32, description="Memory cap for the whole call")
    max_processes: int = Field(default=8, ge=1, description="run_command only")
    max_output: int = Field(default=20_000, ge=100, description="Characters returned")


# -- paths -----------------------------------------------------------------------------


def _norm(p: str | Path) -> str:
    return os.path.normcase(os.path.realpath(p))


def within(path: str | Path, root: str | Path) -> bool:
    """Whether ``path`` (symlinks resolved) is ``root`` or inside it."""
    p, r = _norm(path), _norm(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def is_protected(path: str | Path, workspace: str | Path) -> bool:
    rel = os.path.relpath(_norm(path), _norm(workspace))
    return any(part.lower() in PROTECTED_DIRS for part in Path(rel).parts)


def check_path(workspace: Path, rel: str, *, write: bool) -> Path:
    """Resolve a workspace path or raise ToolError. ``write`` also refuses protected dirs."""
    target = Path(os.path.realpath(workspace / rel))
    if not within(target, workspace):
        raise ToolError(f"path {rel!r} is outside the workspace")
    if write and is_protected(target, workspace):
        raise ToolError(
            f"path {rel!r} is inside a protected directory ({', '.join(PROTECTED_DIRS)})"
        )
    return target


# -- limited subprocess ----------------------------------------------------------------


@dataclass
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float

    def render(self, limit: int) -> str:
        parts = []
        if self.timed_out:
            parts.append("TIMED OUT: the process and everything it started were stopped.")
        else:
            parts.append(f"exit code {self.returncode}")
        for name, text in (("stdout", self.stdout), ("stderr", self.stderr)):
            if text.strip():
                parts.append(f"--- {name} ---\n{_cap(text, limit)}")
        return "\n".join(parts)


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} more characters cut ...]"


def clean_env(tmp: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """A minimal environment: no tokens, keys or user config from the parent."""
    keep = ("PATH", "SYSTEMROOT", "WINDIR", "PATHEXT", "LANG", "LC_ALL", "NUMBER_OF_PROCESSORS")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(
        {
            "TEMP": str(tmp),
            "TMP": str(tmp),
            "TMPDIR": str(tmp),
            "HOME": str(tmp),
            "USERPROFILE": str(tmp),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    env.update(extra or {})
    return env


def run_limited(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    memory_mb: int,
    max_processes: int,
    out_dir: Path,
) -> RunResult:
    """Run ``argv`` with a timeout, a memory cap and a process cap; kill the whole tree.

    Output goes to files in ``out_dir`` rather than pipes, so a child that prints without
    end cannot fill the parent's memory.
    """
    out_path, err_path = out_dir / "stdout.txt", out_dir / "stderr.txt"
    start = time.perf_counter()
    timed_out = False
    with out_path.open("wb") as out, err_path.open("wb") as err:
        if sys.platform == "win32":
            from agentos import _winjob

            job = _winjob.Job(memory_mb=memory_mb, max_processes=max_processes)
            try:
                proc = subprocess.Popen(
                    list(argv),
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                job.assign(proc)
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    job.stop_all()
                    proc.wait(timeout=10)
            finally:
                # Whatever the call left running ends here, and is gone before the output
                # files are read and the temp dir is removed.
                job.stop_and_wait()
                job.close()
        else:
            proc = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
                preexec_fn=_posix_limits(memory_mb, timeout, out_bytes=50_000_000),
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                _posix_stop_group(proc.pid)
                proc.wait(timeout=10)
    seconds = time.perf_counter() - start
    return RunResult(
        returncode=None if timed_out else proc.returncode,
        stdout=_read_capped(out_path),
        stderr=_read_capped(err_path),
        timed_out=timed_out,
        seconds=seconds,
    )


def _read_capped(path: Path, limit: int = 200_000) -> str:
    with path.open("rb") as f:
        data = f.read(limit + 1)
    text = data[:limit].decode("utf-8", errors="replace")
    return text + ("\n[... output truncated ...]" if len(data) > limit else "")


def _posix_limits(memory_mb: int, timeout: float, *, out_bytes: int):
    def apply() -> None:  # runs in the child between fork and exec
        import resource

        mem = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        cpu = int(timeout) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (out_bytes, out_bytes))

    return apply


def _posix_stop_group(pid: int) -> None:
    import signal

    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)  # the child leads its own session/group


# -- tools -----------------------------------------------------------------------------

_BOOTSTRAP = Path(__file__).with_name("_sandbox_boot.py")


@contextlib.contextmanager
def _run_dir(prefix: str) -> Iterator[str]:
    """A private temp dir per call. Removal is retried briefly: on Windows a virus scanner
    can still hold a file the child just wrote for a moment after every process is gone."""
    tmp = tempfile.mkdtemp(prefix=prefix)
    try:
        yield tmp
    finally:
        for _ in range(25):
            try:
                shutil.rmtree(tmp)
                break
            except FileNotFoundError:
                break
            except OSError:
                time.sleep(0.1)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


class RunPythonArgs(BaseModel):
    code: str = Field(description="Python source to run. Print results to stdout.")


class RunCommandArgs(BaseModel):
    command: list[str] = Field(
        description='Program and arguments as a list, e.g. ["git", "status"]. No shell syntax.',
        min_length=1,
    )


def run_python(workspace: Path, cfg: SandboxConfig, code: str) -> RunResult:
    with _run_dir("agentos-py-") as tmp:
        run_dir = Path(tmp)
        (run_dir / "code.py").write_text(code, encoding="utf-8")
        argv = [
            # The base interpreter, not a venv's python.exe: on Windows that is a launcher
            # that starts the real interpreter as a second process, which the one-process
            # limit below forbids. The agent's code gets the standard library only.
            getattr(sys, "_base_executable", None) or sys.executable,
            "-I",  # isolated: ignore PYTHON* env vars and the user site directory
            "-X",
            "utf8",
            str(_BOOTSTRAP),
            str(workspace.resolve()),
            str(run_dir),
            str(run_dir / "code.py"),
        ]
        return run_limited(
            argv,
            cwd=workspace,
            env=clean_env(run_dir),
            timeout=cfg.timeout,
            memory_mb=cfg.memory_mb,
            max_processes=1,  # the interpreter itself; the OS refuses any child process
            out_dir=run_dir,
        )


def validate_command(argv: list[str], workspace: Path, cfg: SandboxConfig, path: str) -> str:
    """Check a command against the rules; return the resolved executable or raise."""
    name = argv[0]
    if os.sep in name or "/" in name or (os.altsep and os.altsep in name):
        raise ToolError("give the program by name (e.g. 'git'), not by path")
    base = name.lower().removesuffix(".exe")
    if base not in {c.lower() for c in cfg.allowed_commands}:
        raise ToolError(f"{name!r} is not an allowed command; allowed: {cfg.allowed_commands}")
    exe = shutil.which(name, path=path)
    if exe is None:
        raise ToolError(f"{name!r} is allowed but not installed")
    if Path(exe).suffix.lower() in (".bat", ".cmd"):
        raise ToolError(f"{name!r} resolves to a batch file, which this sandbox refuses to run")

    args = argv[1:]
    if base == "git":
        if not args or args[0].startswith("-"):
            raise ToolError("git global options are not allowed; start with the subcommand")
        if args[0] not in GIT_SUBCOMMANDS:
            raise ToolError(f"git {args[0]!r} is not allowed; allowed: {sorted(GIT_SUBCOMMANDS)}")
        for a in args:
            if a.split("=", 1)[0] in GIT_FORBIDDEN_ARGS:
                raise ToolError(f"git option {a!r} is not allowed")
    for a in args:
        value = a.split("=", 1)[1] if a.startswith("-") and "=" in a else a
        if _looks_like_path(value):
            target = workspace / os.path.expanduser(value)
            if not within(target, workspace):
                raise ToolError(f"argument {a!r} points outside the workspace")
            if is_protected(target, workspace):
                raise ToolError(f"argument {a!r} points into a protected directory")
    return exe


def _looks_like_path(s: str) -> bool:
    if not s or s.startswith("-"):
        return False
    return (
        "/" in s
        or "\\" in s
        or s.startswith((".", "~"))
        or (len(s) >= 2 and s[1] == ":")
        or os.path.isabs(s)
    )


def run_command(workspace: Path, cfg: SandboxConfig, argv: list[str]) -> RunResult:
    with _run_dir("agentos-cmd-") as tmp:
        run_dir = Path(tmp)
        env = clean_env(
            run_dir,
            {
                # git must not read the user's or the system's config (aliases, hooks paths)
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_AUTHOR_NAME": "agentos",
                "GIT_AUTHOR_EMAIL": "agentos@localhost",
                "GIT_COMMITTER_NAME": "agentos",
                "GIT_COMMITTER_EMAIL": "agentos@localhost",
            },
        )
        exe = validate_command(argv, workspace, cfg, env.get("PATH", ""))
        return run_limited(
            [exe, *argv[1:]],
            cwd=workspace,
            env=env,
            timeout=cfg.timeout,
            memory_mb=cfg.memory_mb,
            max_processes=cfg.max_processes,
            out_dir=run_dir,
        )


def sandbox_tools(workspace: Path, cfg: SandboxConfig | None = None) -> list[Tool]:
    cfg = cfg or SandboxConfig()
    ws = workspace.resolve()

    def py(args: RunPythonArgs) -> str:
        return run_python(ws, cfg, args.code).render(cfg.max_output)

    def cmd(args: RunCommandArgs) -> str:
        return run_command(ws, cfg, args.command).render(cfg.max_output)

    return [
        Tool(
            "run_python",
            "Run Python code in a sandbox with the workspace as the current directory. "
            f"Stdlib only, no network, no subprocesses, {cfg.timeout:g} s limit. "
            "Returns the exit code, stdout and stderr.",
            RunPythonArgs,
            py,
            source="sandbox",
            permission="dangerous",
        ),
        Tool(
            "run_command",
            f"Run an allowed program ({', '.join(cfg.allowed_commands)}) in the workspace. "
            "Pass the program and its arguments as a list; there is no shell.",
            RunCommandArgs,
            cmd,
            source="sandbox",
            permission="dangerous",
        ),
    ]
