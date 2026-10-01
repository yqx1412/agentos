"""Child-side bootstrap for ``run_python`` (A7). Runs as a script, stdlib only.

Usage: python -I _sandbox_boot.py WORKSPACE RUN_DIR CODE_FILE

Reads the agent's code, installs an audit hook that cannot be removed, changes into the
workspace and runs the code. The hook refuses process creation, network, native-code
escape hatches and file access outside the allowed roots. It runs before any agent code,
so everything the code does passes through it. See agentos/sandbox.py for what this does
and does not protect against.
"""

import os
import sys

WORKSPACE, RUN_DIR, CODE_FILE = (os.path.realpath(p) for p in sys.argv[1:4])
with open(CODE_FILE, encoding="utf-8") as _f:
    SOURCE = _f.read()


def _n(p):
    return os.path.normcase(os.path.realpath(p)).rstrip(os.sep)


WRITE_ROOTS = [_n(WORKSPACE), _n(RUN_DIR)]
# Imports read the standard library, so the Python installation is readable.
READ_ROOTS = WRITE_ROOTS + sorted(
    {_n(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)}
)
PROTECTED = {".git"}

BLOCKED_EVENTS = {
    "os.system",
    "os.fork",
    "os.forkpty",
    "os.posix_spawn",
    "os.spawn",
    "os.startfile",
    "os.kill",
    "os.killpg",
    "os.symlink",
    "os.link",
    "subprocess.Popen",
    "_winapi.CreateProcess",
    "_winapi.CreateNamedPipe",
    "socket.connect",
    "socket.bind",
    "socket.sendto",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "ctypes.dlopen",
    "ctypes.dlsym",
    "ctypes.call_function",
    "webbrowser.open",
    "sys.setprofile",
    "sys.settrace",
}
BLOCKED_PREFIXES = ("os.exec", "winreg.", "_winapi.", "msvcrt.")
# Modules that reach native code or the OS directly. Blocked at import, which is earlier
# and more reliable than catching each of their calls.
BLOCKED_MODULES = {
    "ctypes",
    "_ctypes",
    "_winapi",
    "_posixsubprocess",
    "subprocess",
    "multiprocessing",
    "_multiprocessing",
    "socket",
    "_socket",
    "ssl",
    "_ssl",
    "winreg",
    "msvcrt",
    "pty",
    "signal",
    "_signal",
    "mmap",
}

WRITE_PATH_EVENTS = {
    "os.remove": 1,
    "os.rmdir": 1,
    "os.mkdir": 1,
    "os.rename": 2,
    "os.replace": 2,
    "os.truncate": 1,
    "os.chmod": 1,
    "os.chown": 1,
    "os.utime": 1,
    "shutil.rmtree": 1,
    "shutil.copyfile": 2,
    "shutil.copytree": 2,
    "shutil.move": 2,
    "shutil.chown": 1,
}
READ_PATH_EVENTS = {"os.listdir": 1, "os.scandir": 1, "os.chdir": 1, "glob.glob": 1}


class SandboxViolation(PermissionError):
    pass


def _inside(path, roots):
    p = _n(path)
    return any(p == r or p.startswith(r + os.sep) for r in roots)


def _protected(path):
    p = _n(path)
    ws = WRITE_ROOTS[0]
    if not (p == ws or p.startswith(ws + os.sep)):
        return False
    return any(part.lower() in PROTECTED for part in p[len(ws) :].split(os.sep))


def _check(path, write, event):
    if isinstance(path, int) or path is None:  # an already-open file descriptor
        return
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    path = os.fspath(path)
    if write:
        if not _inside(path, WRITE_ROOTS):
            raise SandboxViolation(f"sandbox: {event} outside the workspace is not allowed: {path}")
        if _protected(path):
            raise SandboxViolation(f"sandbox: {event} in a protected directory: {path}")
    elif not _inside(path, READ_ROOTS):
        raise SandboxViolation(f"sandbox: reading outside the workspace is not allowed: {path}")


_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def _hook(event, args):
    if event == "import":
        top = str(args[0]).split(".")[0]
        if top in BLOCKED_MODULES:
            raise SandboxViolation(f"sandbox: importing {top!r} is not allowed")
        return
    if event in BLOCKED_EVENTS or event.startswith(BLOCKED_PREFIXES):
        raise SandboxViolation(f"sandbox: {event} is not allowed")
    if event == "open":
        path, mode, flags = args
        write = any(c in str(mode) for c in "wax+") if mode else bool((flags or 0) & _WRITE_FLAGS)
        _check(path, write, "writing")
        return
    n = WRITE_PATH_EVENTS.get(event)
    if n:
        for p in args[:n]:
            _check(p, True, event)
        return
    n = READ_PATH_EVENTS.get(event)
    if n:
        for p in args[:n]:
            _check(p if p is not None else ".", False, event)


# Drop modules the interpreter loaded at startup that would be blocked on import: an
# already-imported module would otherwise be handed out from sys.modules without an event.
for _name in list(sys.modules):
    if _name.split(".")[0] in BLOCKED_MODULES:
        del sys.modules[_name]

os.chdir(WORKSPACE)
sys.path[:] = [p for p in sys.path if _inside(p, READ_ROOTS[2:])] if len(READ_ROOTS) > 2 else []
sys.addaudithook(_hook)

_code = compile(SOURCE, "<agent code>", "exec")
del SOURCE
exec(_code, {"__name__": "__main__", "__builtins__": __builtins__})
