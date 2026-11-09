# -*- coding: utf-8 -*-
"""Per-role file, shell, GPU and web rules for one paper-reproducer expert.

The JiuwenSwarm port of ``framework/tools.py``: the role table lives in ``policy.json`` next to this file
(written by ``team/build.py`` from ``team/roles.json``), and this rail enforces it on every tool call.

- The run root is read from the flow's preamble ("All files of this run are under <root>. ..."). Paths inside it
  follow the role's read/write prefixes. Outside it, writes are refused and reads are allowed only in the shared
  skill library, the MCP server definitions, installed expert-team packages and the worker's own workspace, after
  resolving symlinks, and never for secret-looking names. JiuwenSwarm's config/ (model API keys) is not readable.
  Until the root is known, file, shell and GPU tools are refused.
- Shell commands: with ``"sandbox": "bwrap"`` in ``policy.json`` (``team/build.py --bwrap``) they run under
  bubblewrap with only the role's paths mounted (read-only, or read-write for its write prefixes), a cleared
  environment, and the network only for roles with web access; a missing ``bwrap`` then refuses every shell call.
  With ``"sandbox": "none"`` (the default build) they run as given, starting in the role's own directory (the bash
  tool's ``workdir``): the role's paths bind the file tools only, and the persona asks the role to stay there.
- GPU tools (the gpu_runner MCP, any host prefix) are allowed by level: ro (status, output, list), wait (also
  wait), run (also submit, cancel), rw (also download_job_files). Submitted dirs must be readable, download targets writable.
- Web tools need ``web``. Other tools (todo, skills, structured_output, tool_search) pass.

Progressive tool exposure wraps calls as ``tool_call {"name", "args"}``; the inner call is checked.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

try:  # the rules below are plain functions so tests can run them without an agent runtime
    from openjiuwen.core.single_agent.rail.base import AgentRail
except ImportError:  # pragma: no cover
    AgentRail = object

POLICY_FILE = Path(__file__).with_name("policy.json")
SANDBOXES = ("bwrap", "none")  # how shell commands run; team/build.py writes it into every policy.json
ROOT_RX = re.compile(r"All files of this run are under (/\S+?)\. Every relative path")
EXPERIMENT_RX = re.compile(r"You reproduce experiment `([a-z0-9][a-z0-9_-]*)`")

READ_TOOLS = {"read_file", "glob", "grep", "list_files", "view_page"}
WRITE_TOOLS = {"write_file", "edit_file", "search_replace", "write"}
SHELL_TOOLS = {"bash", "powershell", "mcp_exec_command", "create_terminal", "code"}
WEB_TOOLS = {"mcp_free_search", "mcp_paid_search", "mcp_fetch_webpage", "web_search", "web_fetch"}
GPU_RX = re.compile(r"(?:^|_)(gpu_submit|gpu_status|gpu_output|gpu_wait|gpu_cancel|gpu_list|download_job_files)$")
GPU_NEED = {"gpu_status": "ro", "gpu_output": "ro", "gpu_list": "ro", "gpu_submit": "run", "gpu_wait": "wait",
            "gpu_cancel": "run", "download_job_files": "rw"}
GPU_LEVEL = {None: 0, "ro": 1, "wait": 2, "run": 3, "rw": 4}
EXTRA_READ = (".pages",)  # view_page renders pages here
# view_page calls per turn (one flow call, or one session turn). The headless runner capped it at 6 after figure
# readers zoomed into ever thinner slivers 60+ times in one turn (2026-11-03); a policy may set its own max_views.
MAX_VIEWS = 12
def jiuwen_dirs() -> tuple[Path, Path]:
    """(agent workspace, data dir) of the running JiuwenSwarm, asked at call time from JiuwenSwarm itself - the same
    helpers its prompt uses to name these paths, so they follow JIUWENSWARM_DATA_DIR, named instances and tenants.
    Outside JiuwenSwarm (tests, the headless runner): JIUWENSWARM_DATA_DIR, else ~/.jiuwenswarm, as it resolves them.
    (JIUWENSWARM_HOME is JiuwenSwarm's override of the user's home, the parent of .jiuwenswarm, not the data dir.)"""
    try:
        from jiuwenswarm.common.utils import get_agent_workspace_dir, get_user_workspace_dir
        return Path(get_agent_workspace_dir()), Path(get_user_workspace_dir())
    except Exception:
        data = os.environ.get("JIUWENSWARM_DATA_DIR") or (
            Path(os.environ.get("JIUWENSWARM_HOME") or Path.home()) / ".jiuwenswarm")
        data = Path(data).expanduser()
        return data / "agent" / "workspace", data


def outside_read() -> tuple[Path, ...]:
    """Outside the run root a role may only read these (plus its own workspace, learned in init): the shared skill
    library, MCP server definitions, installed expert-team packages, agent templates and plugin packages. Never
    config/ (model API keys), auth/, .env."""
    ws, data = jiuwen_dirs()
    return (ws / "skills", ws / "mcp", data / ".agent_teams" / "agent_groups",
            ws / "plugins" / "agent_templates", ws / "plugins" / "plugin_packages")


def sandbox_read() -> tuple[Path, ...]:
    """What the sandbox mounts read-only outside the run root: the directories JiuwenSwarm's prompt names to every
    worker ("Installed Skill root", "Installed Agent groups", ...), so a path the model was told about exists in its
    shell too. Only mcp_hub of mcp/, whose credentials/ sits next to it (bwrap cannot filter by name the way
    check_outside_read does), and never config/ (model API keys). Plus, per worker, its own workspace (read-only) and
    its JiuwenSwarm temporary working directory (see WORK_DIR_RX)."""
    ws, data = jiuwen_dirs()
    return (ws / "skills", data / ".agent_teams" / "agent_groups", ws / "mcp" / "mcp_hub",
            ws / "plugins" / "agent_templates", ws / "plugins" / "plugin_packages")
# A team member's "Temporary working directory" (its shell cwd): <team>/team-workspace/artifacts/<date>/<chat>/work/
# <member>. Anything else offered as the cwd (the server's own directory, when none is set) is not mounted.
WORK_DIR_RX = re.compile(r"/\.agent_teams/[^/]+/team-workspace/artifacts/[^/]+/[^/]+/work/[^/]+$")
# The bubblewrap options of each (role, run root) live in a file here, so the command the model sees and keeps in its
# history is `bwrap --args 3 bash -c '<cmd>' 3< <file>`, not ~2.4k characters of mounts on every shell call
# (2026-11-09: 103k characters, 11% of a builder's 386k-token context). No sandbox mounts this directory, and the
# file tools cannot write outside the run root.
SANDBOX_ARGS_DIR = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "repro-sandbox"
# uv's cache, shared by every sandboxed shell that can write (builders, tuner, evaluator) across arms and runs, so torch
# is downloaded once, not once per arm. Its own directory, not ~/.cache/uv: a sandboxed role writes it, and nothing
# outside the team should install from what a role wrote. Installs copy out of it (UV_LINK_MODE=copy): hardlinks fail
# across bind mounts anyway, and a hardlinked venv file edited in one arm would change the cache for every other.
SANDBOX_UV_CACHE = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "repro-uv-cache"
# The Pythons uv installs from a sandboxed shell (UV_PYTHON_INSTALL_DIR). Inside the sandbox HOME is /tmp, so uv put
# them in /tmp/.local/share/uv/python, which is gone when the call ends: a venv made with `uv venv --python 3.12`
# then points at nothing from the next call on (2026-11-09: a builder rebuilt its venv on the system Python instead).
# Persistent and shared like the cache: read-write for shells that write, read-only for the others, which run the
# builders' venvs and so need the interpreter those venvs link to.
SANDBOX_UV_PYTHON = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "repro-uv-python"
SECRET_NAME_RX = re.compile(r"(^\.env(\..*)?$|\.(pem|key|p12|pfx)$|^id_(rsa|ed25519|ecdsa)|secret|credential|token)", re.I)


class Denied(Exception):
    pass


def load_policy(path: Path = POLICY_FILE) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("read", [])
    data.setdefault("write", [])
    data.setdefault("sandbox", "bwrap")  # a package built before the build chose it always sandboxed
    if data["sandbox"] not in SANDBOXES:
        raise ValueError(f"{path}: sandbox must be one of {', '.join(SANDBOXES)}, not {data['sandbox']!r}")
    return data


def _under(rel: str, prefix: str) -> bool:
    return prefix == "" or rel == prefix or rel.startswith(prefix + "/")


def prefixes(policy: dict, mode: str, experiment: str | None) -> list[str]:
    out = []
    for p in policy["read" if mode == "read" else "write"]:
        if "{experiment}" in p:
            if not experiment:
                continue  # an arm-scoped write needs the experiment from the instructions
            p = p.replace("{experiment}", experiment)
        out.append(p)
    if mode == "read" and policy["read"]:
        out += list(EXTRA_READ)
    return out


def _within(path: Path, top: Path) -> bool:
    return path == top or top in path.parents


def check_outside_read(path: Path, own_workspace: str | None) -> str:
    """`path` (outside the run root) if it is in an allowed read area, else raise Denied. Symlinks are resolved
    first, so a link inside an allowed area cannot point at config/ or anywhere else."""
    real = Path(os.path.realpath(path))
    tops = [Path(os.path.realpath(t)) for t in outside_read()]
    if own_workspace:
        tops.append(Path(os.path.realpath(own_workspace)))
    if not any(_within(real, t) for t in tops):
        raise Denied(f"{path} is outside the run root; outside it only the skill library, MCP definitions, "
                     "expert-team packages and your own workspace are readable")
    if any(SECRET_NAME_RX.search(part) for part in real.parts[len(Path(real.anchor).parts):]):
        raise Denied(f"{path} looks like a secret; it is not readable")
    return str(real)


def check_path(policy: dict, root: str | None, raw: Any, mode: str, experiment: str | None = None,
               own_workspace: str | None = None) -> str:
    """The absolute path for `raw` if this role may `mode` ("read" / "write") it, else raise Denied."""
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise Denied(f"bad path: {raw!r}")
    if not root:
        raise Denied("the run root is not known yet; paths must be under the run root named in your task")
    base = Path(root)
    path = Path(os.path.normpath(raw if os.path.isabs(raw) else os.path.join(root, raw)))
    if path != base and base not in path.parents:
        if mode == "read":
            return check_outside_read(path, own_workspace)
        raise Denied(f"{raw} is outside the run root {root}")
    rel = path.relative_to(base).as_posix()
    rel = "" if rel == "." else rel
    allowed = prefixes(policy, mode, experiment)
    if not any(_under(rel, a) for a in allowed) and not (mode == "read" and rel == "" and allowed):
        raise Denied(f"{policy.get('role', 'this role')} may not {mode} {rel or '.'} "
                     f"(allowed: {', '.join(a or '.' for a in allowed) or 'nothing'})")
    cur = base
    for part in Path(rel).parts:  # no component may be a symlink
        cur = cur / part
        if cur.is_symlink():
            raise Denied(f"symlink in path: {rel}")
    return str(path)


def work_dir_ok(work_dir: str | None, root: str) -> str | None:
    """`work_dir` (resolved) if it is a team member's JiuwenSwarm temporary working directory that neither holds nor
    sits inside the run root, else None."""
    if not work_dir:
        return None
    real = Path(os.path.realpath(work_dir))
    base = Path(os.path.realpath(root))
    if (not real.is_dir() or not WORK_DIR_RX.search(real.as_posix())
            or not _within(real, Path(os.path.realpath(jiuwen_dirs()[1])))
            or _within(base, real) or _within(real, base)):
        return None
    return str(real)


def outside_mounts(policy: dict, root: str, own_workspace: str | None, work_dir: str | None) -> list[str]:
    """bwrap options mounting what JiuwenSwarm tells the worker about outside the run root (see sandbox_read)."""
    a: list[str] = []
    base = Path(os.path.realpath(root))
    ro = [Path(p) for p in sandbox_read()] + ([Path(own_workspace)] if own_workspace else [])
    for p in ro:
        real = Path(os.path.realpath(p))
        if real.is_dir() and not _within(base, real) and not _within(real, base):
            a += ["--ro-bind", str(real), str(p)]
    wd = work_dir_ok(work_dir, root)
    if wd:
        a += ["--bind" if policy.get("shell") == "rw" else "--ro-bind", wd, wd]
    return a


def bwrap_args(policy: dict, root: str, experiment: str | None, own_workspace: str | None = None,
               work_dir: str | None = None) -> list[str]:
    """bubblewrap's options for this role: only its paths mounted (plus what JiuwenSwarm advertises to every worker,
    see sandbox_read), `root` as the working directory."""
    home = Path.home()
    # uv and the Pythons it installed (venvs symlink into them); not the rest of ~/.local/share/uv, which holds
    # uv's stored index credentials
    uv_bin, uv_share = home / ".local" / "bin", home / ".local" / "share" / "uv" / "python"
    path_env = ":".join([str(uv_bin), "/usr/local/bin", "/usr/bin", "/bin"])
    # No --die-with-parent: with it, bwrap's pid-1 init dies with the call and takes the whole pid namespace along,
    # so `cmd &` was killed the moment the shell call returned. Without it a background job outlives the call (as it
    # does unsandboxed) and keeps its sandbox; the pid namespace still hides and shields host processes.
    a = ["--unshare-all", "--clearenv",
         "--setenv", "PATH", path_env, "--setenv", "HOME", "/tmp", "--setenv", "LANG", "C.UTF-8",
         "--setenv", "UV_CACHE_DIR", "/tmp/uv-cache", "--setenv", "UV_LINK_MODE", "copy",
         "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
         "--symlink", "usr/lib64", "/lib64", "--ro-bind", "/etc", "/etc",
         "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    if policy.get("web"):
        a.insert(a.index("--unshare-all") + 1, "--share-net")  # bwrap applies these in order: after, or no net
        # /etc/resolv.conf is often a symlink out of /etc (WSL: /mnt/wsl/resolv.conf, systemd-resolved: /run/...);
        # mount its target too, or DNS fails inside the sandbox
        real = os.path.realpath("/etc/resolv.conf")
        if not real.startswith("/etc/") and os.path.isfile(real):
            a += ["--ro-bind", real, real]
    for p in (uv_bin, uv_share):
        if p.is_dir():
            a += ["--ro-bind", str(p), str(p)]
    a += outside_mounts(policy, root, own_workspace, work_dir)
    writes = policy.get("shell") == "rw"
    if writes:  # read-only shells keep the empty /tmp/uv-cache: they install nothing
        SANDBOX_UV_CACHE.mkdir(mode=0o700, parents=True, exist_ok=True)
        cache = str(SANDBOX_UV_CACHE)
        a[a.index("/tmp/uv-cache")] = cache
        a += ["--bind", cache, cache]
    SANDBOX_UV_PYTHON.mkdir(mode=0o700, parents=True, exist_ok=True)
    pythons = str(SANDBOX_UV_PYTHON)
    # UV_PYTHON_INSTALL_BIN=0: `uv python install` would also link executables into ~/.local/bin, mounted read-only
    a += ["--setenv", "UV_PYTHON_INSTALL_DIR", pythons, "--setenv", "UV_PYTHON_INSTALL_BIN", "0",
          "--bind" if writes else "--ro-bind", pythons, pythons]
    base = Path(root)
    for r in prefixes(policy, "read", experiment):
        if (base / r).exists():
            a += ["--ro-bind", str(base / r), str(base / r)]
    if policy.get("shell") == "rw":
        for r in prefixes(policy, "write", experiment):
            (base / r).mkdir(parents=True, exist_ok=True)
            a += ["--bind", str(base / r), str(base / r)]
    return a + ["--chdir", root]


def sandbox_args_file(args: list[str]) -> Path:
    """The file holding `args` NUL-separated (for `bwrap --args`), named by its content. It is rewritten whenever its
    content differs, so a stale or altered file is never used as found. bwrap closes the fd once it has read it, so
    the sandboxed command cannot reach the file through /proc/self/fd."""
    data = b"".join(a.encode() + b"\0" for a in args)
    path = SANDBOX_ARGS_DIR / f"{hashlib.sha256(data).hexdigest()[:32]}.args"
    try:
        if path.read_bytes() == data:
            return path
    except OSError:
        pass
    SANDBOX_ARGS_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(SANDBOX_ARGS_DIR, 0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return path


def bwrap_command(policy: dict, root: str, cmd: str, experiment: str | None, own_workspace: str | None = None,
                  work_dir: str | None = None) -> str:
    """`cmd` wrapped in bubblewrap with only this role's paths mounted, as a short shell string: the mounts are read
    from a file (`bwrap --args`), so they are not copied into the model's history on every call."""
    opts = sandbox_args_file(bwrap_args(policy, root, experiment, own_workspace, work_dir))
    return f"{shlex.join(['bwrap', '--args', '3', 'bash', '-c', cmd])} 3< {shlex.quote(str(opts))}"


def shell_home(policy: dict, root: str, experiment: str | None) -> Path:
    """Where an unsandboxed shell command starts: the role's own writable directory (a builder's
    repo/arms/<experiment>, the evaluator's repo/eval, ...), created if missing; the run root for a read-only shell."""
    base = Path(root)
    if policy.get("shell") == "rw":
        for w in prefixes(policy, "write", experiment):
            (base / w).mkdir(parents=True, exist_ok=True)
            return base / w
    return base


def unwrap(name: str, args: Any) -> tuple[str, Any, dict | None]:
    """(tool name, its args, the dict to rewrite in place) for a direct or a `tool_call`-wrapped call."""
    if name == "tool_call" and isinstance(args, dict) and isinstance(args.get("name"), str):
        inner = args.get("args")
        if not isinstance(inner, dict):
            inner = {}
            args["args"] = inner
        return args["name"], inner, inner
    return name, args, args if isinstance(args, dict) else None


def decide(policy: dict, root: str | None, experiment: str | None, name: str, args: Any,
           sandbox: str = "bwrap", own_workspace: str | None = None, work_dir: str | None = None) -> None:
    """Raise Denied, or return after rewriting `args` in place where needed (shell wrapping, default paths)."""
    name, args, target = unwrap(name, args)
    args = args if isinstance(args, dict) else {}
    short = name.rsplit("__", 1)[-1]
    if short in READ_TOOLS:
        if short in ("glob", "grep", "list_files") and not args.get("path"):
            if target is not None and root:
                target["path"] = root
            args = target or args
        # view_page writes its JPEG next to the PDF, so it only takes PDFs under the run root
        check_path(policy, root, args.get("file_path") or args.get("path"), "read", experiment,
                   None if short == "view_page" else own_workspace)
        if short == "view_page" and not _within(Path(os.path.normpath(os.path.join(root, args.get("file_path") or ""))),
                                                Path(root)):
            raise Denied("view_page only renders PDFs under the run root")
    elif short in WRITE_TOOLS:
        check_path(policy, root, args.get("file_path") or args.get("path"), "write", experiment)
    elif short in SHELL_TOOLS:
        if not policy.get("shell"):
            raise Denied(f"{policy.get('role')} has no shell")
        if short != "bash":
            raise Denied(f"use bash; {short} is not sandboxed for this team")
        if not root:
            raise Denied("the run root is not known yet")
        if sandbox == "none":  # nothing confines the command; start it in the role's own directory
            if target is not None and not args.get("workdir"):
                target["workdir"] = str(shell_home(policy, root, experiment))
            return
        if not shutil.which("bwrap"):
            raise Denied("this team was built with --bwrap but bubblewrap (bwrap) is not installed on this host; "
                         "install it, or rebuild the team without --bwrap to run shells unsandboxed")
        if target is None or not isinstance(args.get("command"), str):
            raise Denied("bash needs a `command` string")
        target["command"] = bwrap_command(policy, root, args["command"], experiment, own_workspace, work_dir)
    elif GPU_RX.search(short):
        op = GPU_RX.search(short).group(1)
        if GPU_LEVEL[policy.get("gpu")] < GPU_LEVEL[GPU_NEED[op]]:
            what = {"ro": "see", "wait": "wait for", "run": "submit", "rw": "download"}[GPU_NEED[op]]
            raise Denied(f"{policy.get('role')} may not {what} GPU jobs")
        if op == "gpu_submit":
            check_path(policy, root, args.get("local_dir"), "read", experiment)  # uploads: run root only
        if op == "download_job_files":
            files = args.get("files")
            if not isinstance(files, dict) or not files:
                raise Denied("download_job_files needs `files`")
            for dest in files.values():
                check_path(policy, root, dest, "write", experiment)
    elif short in WEB_TOOLS or short.startswith("browser"):
        if not policy.get("web"):
            raise Denied(f"{policy.get('role')} has no web access")


def unshow(value: Any, wrapped: str, original: str) -> Any:
    """`value` (a tool result or tool message) with every `wrapped` replaced by `original`. Strings, dicts and lists
    are rebuilt; other objects get their `content` / `data` attribute rewritten in place."""
    if isinstance(value, str):
        return value.replace(wrapped, original)
    if isinstance(value, dict):
        return {k: unshow(v, wrapped, original) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(unshow(v, wrapped, original) for v in value)
    for attr in ("content", "data"):
        inner = getattr(value, attr, None)
        if isinstance(inner, (str, dict, list)):
            try:
                setattr(value, attr, unshow(inner, wrapped, original))
            except (AttributeError, TypeError, ValueError):
                pass
    return value


def member_work_dir() -> str | None:
    """This worker's shell cwd as JiuwenSwarm set it (its "Temporary working directory"), or None when unset: not
    get_cwd(), which falls back to the server's own directory."""
    try:
        from openjiuwen.core.sys_operation import cwd as cwd_mod
        state = cwd_mod._cwd_state.get()
    except Exception:  # pragma: no cover - another runtime
        return None
    return (getattr(state, "original_cwd", None) or getattr(state, "cwd", None)) if state else None


def _text(x: Any) -> str:
    if isinstance(x, str):
        return x
    content = getattr(x, "content", None)
    if content is None and isinstance(x, dict):
        content = x.get("content")
    if isinstance(content, list):
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return content if isinstance(content, str) else str(x or "")


class RunPolicyRail(AgentRail):
    """Enforces policy.json on this expert's tool calls."""

    priority = 10

    def __init__(self) -> None:
        if AgentRail is not object:
            super().__init__()
        self.policy = load_policy()
        self.root: str | None = None
        self.experiment: str | None = None
        self.sandbox = self.policy.get("sandbox", "bwrap")
        self.own_workspace: str | None = None
        self.views = 0
        self.shown: dict[str, str] = {}  # bwrap-wrapped command -> the command the model wrote, until its result

    def init(self, agent) -> None:
        """Remember this worker's own workspace, the one directory outside the run root it may read besides the
        shared skill library, MCP definitions and expert-team packages."""
        if AgentRail is not object:
            super().init(agent)
        ws = getattr(getattr(agent, "deep_config", None), "workspace", None)
        root = getattr(ws, "root_path", None) or (ws if isinstance(ws, (str, Path)) else None)
        if root:
            self.own_workspace = str(root)

    def observe(self, text: str) -> None:
        if not self.root:
            m = ROOT_RX.search(text)
            if m:
                self.root = m.group(1).rstrip("/")
        if not self.experiment:
            m = EXPERIMENT_RX.search(text)
            if m:
                self.experiment = m.group(1)

    async def before_invoke(self, ctx) -> None:
        self.views = 0  # a new turn
        self.observe(_text(getattr(ctx.inputs, "query", "")))

    def count_view(self, name: str, args: Any) -> None:
        """Raise Denied past this turn's view_page limit."""
        inner, _, _ = unwrap(name, args)
        if inner.rsplit("__", 1)[-1] != "view_page":
            return
        limit = int(self.policy.get("max_views", MAX_VIEWS))
        if self.views >= limit:
            raise Denied(f"view_page limit reached ({limit} views this turn): answer from the pages you have seen, "
                         "with an uncertainty that reflects how well you could read them")
        self.views += 1

    async def before_model_call(self, ctx) -> None:
        if self.root and self.experiment:
            return
        for msg in getattr(ctx.inputs, "messages", None) or []:
            self.observe(_text(msg))

    async def before_tool_call(self, ctx) -> None:
        inputs = ctx.inputs
        name = str(getattr(inputs, "tool_name", "") or "")
        args = getattr(inputs, "tool_args", None)
        if isinstance(args, str):
            try:
                args = json.loads(args)
                inputs.tool_args = args
            except ValueError:
                args = {}
        _, inner, _ = unwrap(name, args)
        before = inner.get("command") if isinstance(inner, dict) else None
        try:
            decide(self.policy, self.root, self.experiment, name, args, self.sandbox, self.own_workspace,
                   member_work_dir())
            self.count_view(name, args)
        except Denied as exc:
            self._reject(ctx, f"[run policy] {exc}")
            return
        after = inner.get("command") if isinstance(inner, dict) else None
        if isinstance(before, str) and isinstance(after, str) and after != before:
            self.shown[after] = before  # after_tool_call puts `before` back where the result echoes `after`

    async def after_tool_call(self, ctx) -> None:
        """Show the model the command it wrote, not its bwrap form. A model that sees `bwrap --args 3 bash -c '...'
        3< <file>` copies it, which then fails inside the sandbox (the args file is not mounted there); 2026-11-09:
        56 of two builders' 90 shell calls. It sees it in two places: the bash tool's echo of the command it ran,
        and its own earlier call, since the agent writes the rewritten args back into the ToolCall that its
        history keeps. Both are put back here."""
        self._restore(ctx, rewrite_result=True)

    async def on_tool_exception(self, ctx) -> None:
        self._restore(ctx, rewrite_result=False)

    def _restore(self, ctx, rewrite_result: bool) -> None:
        inputs = ctx.inputs
        _, args, _ = unwrap(str(getattr(inputs, "tool_name", "") or ""), getattr(inputs, "tool_args", None))
        wrapped = args.get("command") if isinstance(args, dict) else None
        original = self.shown.pop(wrapped, None) if isinstance(wrapped, str) else None
        if original is None:
            return
        args["command"] = original  # the same dict as the history's ToolCall.arguments
        call = getattr(inputs, "tool_call", None)
        if call is not None and isinstance(getattr(call, "arguments", None), str):
            call.arguments = call.arguments.replace(json.dumps(wrapped)[1:-1], json.dumps(original)[1:-1])
        if rewrite_result:
            for field in ("tool_msg", "tool_result"):
                value = getattr(inputs, field, None)
                if value is not None:
                    setattr(inputs, field, unshow(value, wrapped, original))

    @staticmethod
    def _reject(ctx, message: str) -> None:
        from openjiuwen.core.foundation.llm.schema.message import ToolMessage

        inputs = ctx.inputs
        call_id = str(getattr(getattr(inputs, "tool_call", None), "id", "") or "unknown-tool-call")
        ctx.extra["_skip_tool"] = True
        inputs.tool_result = {"error": message}
        inputs.tool_msg = ToolMessage(content=message, tool_call_id=call_id)
