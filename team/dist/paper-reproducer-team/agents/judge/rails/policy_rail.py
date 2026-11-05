# -*- coding: utf-8 -*-
"""Per-role file, shell, GPU and web rules for one paper-reproducer expert.

The JiuwenSwarm port of ``framework/tools.py``: the role table lives in ``policy.json`` next to this file
(written by ``team/build.py`` from ``team/roles.json``), and this rail enforces it on every tool call.

- The run root is read from the flow's preamble ("All files of this run are under <root>. ..."). Paths inside it
  follow the role's read/write prefixes. Outside it, writes are refused and reads are allowed only in the shared
  skill library, the MCP server definitions, installed expert-team packages and the worker's own workspace, after
  resolving symlinks, and never for secret-looking names. JiuwenSwarm's config/ (model API keys) is not readable.
  Until the root is known, file, shell and GPU tools are refused.
- Shell commands run under bubblewrap with only the role's paths mounted (read-only, or read-write for its write
  prefixes), a cleared environment, and the network only for roles with web access.
- GPU tools (the gpu_runner MCP, any host prefix) are allowed by level: ro (status, output, list), run (also
  submit, wait, cancel), rw (also download_job_files). Submitted dirs must be readable, download targets writable.
- Web tools need ``web``. Other tools (todo, skills, structured_output, tool_search) pass.

Progressive tool exposure wraps calls as ``tool_call {"name", "args"}``; the inner call is checked.
"""
from __future__ import annotations

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
ROOT_RX = re.compile(r"All files of this run are under (/\S+?)\. Every relative path")
EXPERIMENT_RX = re.compile(r"You reproduce experiment `([a-z0-9][a-z0-9_-]*)`")

READ_TOOLS = {"read_file", "glob", "grep", "list_files", "view_page"}
WRITE_TOOLS = {"write_file", "edit_file", "search_replace", "write"}
SHELL_TOOLS = {"bash", "powershell", "mcp_exec_command", "create_terminal", "code"}
WEB_TOOLS = {"mcp_free_search", "mcp_paid_search", "mcp_fetch_webpage", "web_search", "web_fetch"}
GPU_RX = re.compile(r"(?:^|_)(gpu_submit|gpu_status|gpu_output|gpu_wait|gpu_cancel|gpu_list|download_job_files)$")
GPU_NEED = {"gpu_status": "ro", "gpu_output": "ro", "gpu_list": "ro", "gpu_submit": "run", "gpu_wait": "run",
            "gpu_cancel": "run", "download_job_files": "rw"}
GPU_LEVEL = {None: 0, "ro": 1, "run": 2, "rw": 3}
EXTRA_READ = (".pages",)  # view_page renders pages here
# view_page calls per turn (one flow call, or one session turn). The headless runner capped it at 6 after figure
# readers zoomed into ever thinner slivers 60+ times in one turn (2026-11-03); a policy may set its own max_views.
MAX_VIEWS = 12
JIUWENSWARM_HOME = Path(os.environ.get("JIUWENSWARM_HOME", "~/.jiuwenswarm")).expanduser()
# Outside the run root a role may only read these (plus its own workspace, learned in init): the shared skill
# library, MCP server definitions, and installed expert-team packages. Never config/ (model API keys), auth/, .env.
OUTSIDE_READ = (JIUWENSWARM_HOME / "agent" / "workspace" / "skills",
                JIUWENSWARM_HOME / "agent" / "workspace" / "mcp",
                JIUWENSWARM_HOME / ".agent_teams" / "agent_groups")
SECRET_NAME_RX = re.compile(r"(^\.env(\..*)?$|\.(pem|key|p12|pfx)$|^id_(rsa|ed25519|ecdsa)|secret|credential|token)", re.I)


class Denied(Exception):
    pass


def load_policy(path: Path = POLICY_FILE) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("read", [])
    data.setdefault("write", [])
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
    tops = [Path(os.path.realpath(t)) for t in OUTSIDE_READ]
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


def bwrap_command(policy: dict, root: str, cmd: str, experiment: str | None) -> str:
    """`cmd` wrapped in bubblewrap with only this role's paths mounted, as a shell string."""
    home = Path.home()
    # uv and the Pythons it installed (venvs symlink into them); not the rest of ~/.local/share/uv, which holds
    # uv's stored index credentials
    uv_bin, uv_share = home / ".local" / "bin", home / ".local" / "share" / "uv" / "python"
    path_env = ":".join([str(uv_bin), "/usr/local/bin", "/usr/bin", "/bin"])
    a = ["bwrap", "--die-with-parent", "--unshare-all", "--clearenv",
         "--setenv", "PATH", path_env, "--setenv", "HOME", "/tmp", "--setenv", "LANG", "C.UTF-8",
         "--setenv", "UV_CACHE_DIR", "/tmp/uv-cache",
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
    base = Path(root)
    for r in prefixes(policy, "read", experiment):
        if (base / r).exists():
            a += ["--ro-bind", str(base / r), str(base / r)]
    if policy.get("shell") == "rw":
        for r in prefixes(policy, "write", experiment):
            (base / r).mkdir(parents=True, exist_ok=True)
            a += ["--bind", str(base / r), str(base / r)]
    a += ["--chdir", root, "bash", "-c", cmd]
    return shlex.join(a)


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
           sandbox: str = "bwrap", own_workspace: str | None = None) -> None:
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
        if sandbox == "none":
            return
        if not shutil.which("bwrap"):
            raise Denied("bubblewrap (bwrap) is not installed; set REPRO_SANDBOX=none to accept unsandboxed shells")
        if target is None or not isinstance(args.get("command"), str):
            raise Denied("bash needs a `command` string")
        target["command"] = bwrap_command(policy, root, args["command"], experiment)
    elif GPU_RX.search(short):
        op = GPU_RX.search(short).group(1)
        if GPU_LEVEL[policy.get("gpu")] < GPU_LEVEL[GPU_NEED[op]]:
            what = {"ro": "see", "run": "submit or wait for", "rw": "download"}[GPU_NEED[op]]
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
        self.sandbox = os.environ.get("REPRO_SANDBOX", "bwrap")
        self.own_workspace: str | None = None
        self.views = 0

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
        try:
            decide(self.policy, self.root, self.experiment, name, args, self.sandbox, self.own_workspace)
            self.count_view(name, args)
        except Denied as exc:
            self._reject(ctx, f"[run policy] {exc}")

    @staticmethod
    def _reject(ctx, message: str) -> None:
        from openjiuwen.core.foundation.llm.schema.message import ToolMessage

        inputs = ctx.inputs
        call_id = str(getattr(getattr(inputs, "tool_call", None), "id", "") or "unknown-tool-call")
        ctx.extra["_skip_tool"] = True
        inputs.tool_result = {"error": message}
        inputs.tool_msg = ToolMessage(content=message, tool_call_id=call_id)
