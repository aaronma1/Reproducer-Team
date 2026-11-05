"""gpurun: the agents' client for the gpu_runner daemon (../gpu_runner), reached over ssh.

The agents' shell is sandboxed with no network and no ssh key, so GPU jobs go through these tools,
which the framework runs outside the sandbox. A job uploads one workspace directory the role may
read, runs shell commands in it on the GPU box (which has network, uv and a shared package cache),
and its files come back with `gpu_download` into a directory the role may write.

Every job a run submits is recorded in ``runs/<run>/gpu_jobs.jsonl``; agents can only see and
cancel their own run's jobs, and ``cancel_all`` stops whatever a run leaves behind.
"""
from __future__ import annotations

import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints",
                "node_modules", "wandb"}
MAX_UPLOAD_MB = 500
# Commands that only look around or only install: the agents' own shell does the first, and an environment
# is only worth installing in the job that uses it. A job made only of these holds the shared GPU for nothing.
PROBES = {"ls", "cat", "pwd", "echo", "which", "env", "printenv", "find", "head", "tail", "du", "df", "nvidia-smi",
          "whoami", "hostname", "true", "stat", "file", "tree", "wc"}
_SPLIT = re.compile(r"&&|\|\||;|\|")


class GpuError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuConfig:
    ssh_host: str | None = "gpu"  # ssh alias; None runs the CLI on this machine
    python: str = "/home/a84456917/Projects/gpu_runner/.venv/bin/python"
    runner: str = "/home/a84456917/Projects/gpu_runner/gpu_runner.py"
    hardware: str = "1x A100-PCIE-40GB"  # what the agents are told they get
    max_wait_s: int = 1800  # longest single gpu_wait call
    max_jobs_per_role: int = 30  # (12 until 2026-11-03: it stopped reppo's tuner with GPU budget left) jobs one role (e.g. builder:x, across its turns) may submit in a run
    budget_slack: float = 1.5  # GPU time may reach this multiple of the run's gpu_hours before submits stop
    # Off since 2026-11-04 (the user's call): the ledger has no real job times from the daemon, so a job running while
    # the laptop slept was charged the whole night (17 and 34 "GPU-hours" refused). gpu_hours stays a planning hint.
    enforce_budget: bool = False


def probe_only(commands: list[str]) -> bool:
    """True if every step of every command only inspects (ls, cat, ...) or only installs (`uv sync`, `pip
    install`), so the job would run nothing."""
    for cmd in commands:
        for part in _SPLIT.split(cmd):
            words = part.split()
            if not words or words[0] == "cd":
                continue
            if words[0] in PROBES or words[:2] in (["uv", "sync"], ["uv", "venv"], ["uv", "lock"]) or \
                    words[:2] == ["pip", "install"] or words[:3] == ["uv", "pip", "install"] or \
                    words[:3] == ["pip", "list"] or words[-1:] == ["--version"]:
                continue
            return False
    return True


def _role(label: str) -> str:
    return re.sub(r"#\d+$", "", label)


def _tar(path: Path) -> bytes:
    def skip(ti):
        parts = ti.name.split("/")
        if any(p in EXCLUDE_DIRS for p in parts) or ti.name.endswith((".pyc", ".pyo")) or ti.issym() or ti.islnk():
            return None
        return ti

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(str(path), arcname=".", filter=skip)
    data = buf.getvalue()
    if len(data) > MAX_UPLOAD_MB * 2**20:
        raise GpuError(f"upload is {len(data) / 2**20:.0f} MB (> {MAX_UPLOAD_MB} MB); download datasets inside "
                       "the job instead of uploading them")
    return data


def _status(line: str) -> dict:
    """`FAILED 7 (failed: exit 7)` / `COMPLETED 0 ` / `RUNNING  ` -> dict."""
    status, _, rest = line.strip().partition(" ")
    rest, code, failed = rest.strip(), None, None
    if rest:
        head, _, tail = rest.partition(" ")
        if head.lstrip("-").isdigit():
            code, rest = int(head), tail.strip()
        if rest.startswith("(failed: ") and rest.endswith(")"):
            failed = rest[len("(failed: "):-1]
    return {"status": status, "exit_code": code, "failed_command": failed}


def _tail(text: str, lines: int) -> str:
    parts = text.splitlines(keepends=True)
    if not lines or len(parts) <= lines:
        return text
    return f"[... {len(parts) - lines} earlier lines omitted ...]\n" + "".join(parts[-lines:])


class GpuRunner:
    """One run's view of the GPU queue."""

    def __init__(self, cfg: GpuConfig, journal: Path, cli=None, budget_hours: float | None = None, clock=time.time):
        self.cfg, self.journal = cfg, Path(journal)
        self._cli = cli or self._ssh_cli
        self._lock = threading.Lock()
        self.budget_hours, self._now = budget_hours, clock
        # When each job was first seen running and first seen finished (from status polls: the daemon does not
        # report times). Kept next to the journal so a resumed run keeps counting.
        self.times_file = self.journal.with_name("gpu_times.json")
        self._times = json.loads(self.times_file.read_text()) if self.times_file.is_file() else {}

    # ---- transport ---------------------------------------------------------------
    def _ssh_cli(self, args: list[str], stdin: bytes | None = None, timeout: int = 120) -> tuple[bytes, str]:
        cmd = [self.cfg.python, self.cfg.runner, *args]
        if self.cfg.ssh_host:
            cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
                   "-o", "ServerAliveCountMax=4", self.cfg.ssh_host, shlex.join(cmd)]
        try:
            p = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise GpuError(f"gpu_runner {args[0]} timed out after {timeout}s") from None
        err = p.stderr.decode(errors="replace")
        if p.returncode != 0:
            raise GpuError(err.strip() or p.stdout.decode(errors="replace").strip() or f"gpu_runner exited {p.returncode}")
        return p.stdout, err

    def _text(self, args, **kw) -> str:
        return self._cli(args, **kw)[0].decode(errors="replace")

    # ---- this run's jobs ---------------------------------------------------------
    def jobs(self) -> list[dict]:
        if not self.journal.is_file():
            return []
        return [json.loads(l) for l in self.journal.read_text().splitlines() if l.strip()]

    def _own(self, job: str, label: str | None = None) -> dict:
        for j in self.jobs():
            if j["job"] == job:
                return j
        raise GpuError(f"unknown job {job!r} (not submitted by this run)")

    # ---- GPU time ------------------------------------------------------------------
    def _seen(self, job: str, status: str) -> None:
        t = self._times.setdefault(job, {})
        key = "end" if status in TERMINAL else "start" if status == "RUNNING" else None
        if key and key not in t:
            t[key] = self._now()
            with self._lock:
                self.times_file.write_text(json.dumps(self._times))

    def hours_used(self) -> float:
        """GPU hours this run's jobs have held, as seen by status polls. A job never seen running is charged
        from submission (an over-estimate when it queued behind other teams); a running job up to now."""
        total = 0.0
        for j in self.jobs():
            t = self._times.get(j["job"], {})
            if not t:
                continue  # never polled: not known to have started
            total += max(0.0, t.get("end", self._now()) - t.get("start", j["submitted"]))
        return total / 3600

    def _refresh(self) -> None:
        """Poll jobs still open in the ledger, so one nobody polls any more stops counting as running."""
        for j in self.jobs():
            t = self._times.get(j["job"])
            if t and "end" not in t:
                try:
                    self.status(j["job"])
                except GpuError:
                    pass  # the box is unreachable: keep the old estimate

    # ---- operations --------------------------------------------------------------
    def submit(self, label: str, src: Path, rel: str, commands: list[str], tag: str = "") -> dict:
        if not commands or not all(isinstance(c, str) and c.strip() for c in commands):
            raise GpuError("commands must be a non-empty list of shell commands")
        if not src.is_dir():
            raise GpuError(f"not a directory: {rel}")
        if probe_only(commands):
            raise GpuError("this job only inspects files or installs packages, so it would hold the shared GPU for "
                           "nothing: look at files with your own shell, and chain `uv sync` with the command that "
                           "uses the environment in the same job")
        mine = [j for j in self.jobs() if _role(j["label"]) == _role(label)]
        if len(mine) >= self.cfg.max_jobs_per_role:
            raise GpuError(f"{_role(label)} has submitted {len(mine)} jobs, the limit for one role in a run; finish "
                           "with the results you have, or say in your answer what is still missing")
        if self.budget_hours is not None and self.cfg.enforce_budget:
            self._refresh()
            used, limit = self.hours_used(), self.budget_hours * self.cfg.budget_slack
            if used >= limit:
                raise GpuError(f"this run has used about {used:.2f} GPU-hours (budget {self.budget_hours:g}, hard "
                               f"limit {limit:g}); no more jobs can be submitted")
        data = _tar(src)
        message_id, last = str(uuid.uuid4()), None
        mb = len(data) / 2**20
        timeout = 600 + int(10 * mb)  # ~0.1 MB/s worst case; 600 s alone lost a 383 MB upload three times (2026-11-03)
        for attempt in range(3):  # retries reuse the message id, so the daemon never queues a duplicate
            try:
                out = self._text(["submit", "--message_id", message_id, "-", *commands], stdin=data, timeout=timeout)
                job = out.strip().splitlines()[-1].strip()
                break
            except GpuError as e:
                last = e
                time.sleep(3 * (attempt + 1))
        else:
            raise GpuError(f"submit failed: {last}. The upload was {mb:.0f} MB: submit a smaller directory, or move "
                           "scratch outputs and local dataset copies out of it (download data inside the job)")
        rec = {"job": job, "label": label, "dir": rel, "tag": tag, "commands": commands, "submitted": time.time(),
               "upload_kb": len(data) // 1024}
        with self._lock:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return {"job": job, "status": "QUEUED", "upload_kb": rec["upload_kb"]}

    def status(self, job: str) -> dict:
        self._own(job)
        out = self._text(["query", job, "--status"], timeout=60)
        st = {"job": job, **_status(out.splitlines()[0] if out else "")}
        self._seen(job, st["status"])
        return st

    def output(self, job: str, tail_lines: int = 200) -> dict:
        self.status(job)  # an agent may read a failure here and never poll the job again
        out, err = self._cli(["query", job, "--stdout", "--stderr"], timeout=120)
        return {"stdout": _tail(out.decode(errors="replace"), tail_lines), "stderr": _tail(err, tail_lines)}

    def wait(self, job: str, max_wait_s: int = 600, poll_s: int = 20) -> dict:
        deadline = time.time() + max(10, min(int(max_wait_s), self.cfg.max_wait_s))
        while True:
            st = self.status(job)
            if st["status"] in TERMINAL:
                return st
            if time.time() + poll_s > deadline:
                return {**st, "still_running": True}
            time.sleep(poll_s)

    def cancel(self, job: str, label: str | None = None) -> dict:
        """Cancel one of this run's jobs; with `label`, only one that role submitted."""
        rec = self._own(job)
        if label is not None and _role(rec["label"]) != _role(label):
            raise GpuError(f"job {job} was submitted by {rec['label']}, not by you")
        out = self._text(["cancel", job], timeout=180)
        st = {"job": job, **_status(out.splitlines()[0] if out else "")}
        self._seen(job, st["status"])
        return st

    def download(self, job: str, paths: list[str], dest: Path) -> dict:
        """Fetch `paths` (relative to the job dir) into the local directory `dest`, keeping their relative paths."""
        self._own(job)
        if not paths:
            raise GpuError("paths must name at least one file or directory in the job dir")
        for p in paths:
            if os.path.isabs(p) or ".." in Path(p).parts:
                raise GpuError(f"bad job path: {p!r}")
        before = self.status(job)["status"]
        data, _ = self._cli(["download", job, "--", *paths], timeout=1800)
        placed = []
        with tempfile.TemporaryDirectory(prefix="gpurun_dl_") as tmp:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                members = [m for m in tar.getmembers() if m.isfile() or m.isdir()]  # no links or devices
                tar.extractall(tmp, members=members, filter="data")
            for p in paths:
                src = Path(tmp) / os.path.normpath(p)
                if not src.exists():
                    raise GpuError(f"{p} is missing from the job dir")
            for p in paths:
                src, target = Path(tmp) / os.path.normpath(p), dest / os.path.normpath(p)
                if src.is_dir():
                    shutil.copytree(src, target, dirs_exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, target)
                placed.append(p)
        res = {"downloaded": placed, "bytes": len(data), "job_status": before}
        if before not in TERMINAL:
            res["warning"] = "job still running: files may be incomplete"
        return res

    def list(self) -> list[dict]:
        live = {}
        try:
            for line in self._text(["list"], timeout=60).splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    live[parts[0]] = parts[1]
        except GpuError:
            pass
        own = self.jobs()
        for j in own:
            if j["job"] in live:
                self._seen(j["job"], live[j["job"]])
        return [{"job": j["job"], "status": live.get(j["job"], "?"), "label": j["label"], "tag": j["tag"],
                 "dir": j["dir"], "first_command": j["commands"][0][:120]} for j in own]

    def cancel_all(self) -> list[str]:
        """Cancel every job of this run that has not finished. Returns the cancelled ids."""
        done = []
        for j in self.list():
            if j["status"] not in TERMINAL:
                try:
                    self.cancel(j["job"])
                    done.append(j["job"])
                except GpuError:
                    pass
        return done
