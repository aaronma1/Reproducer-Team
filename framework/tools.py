"""Scoped workspace and sandboxed tools for the role agents.

Every agent sees one directory, ``runs/<run>/work/``, and nothing else: not this repo, not
``.env``, not other runs. Per-role read/write rules are enforced here, not in prompts.

    work/
      paper.pdf, scope.txt      restricted to the paper-reading roles
      repo/data eval tests      owned by `contracts`
      repo/arms/<arm>/          owned by `builder:<arm>`
      report.md                 owned by `reporter`

GPU jobs (``gpu.py``) are the one way out of the sandbox: roles with ``gpu="rw"`` may upload a
directory they can read and download job files into a directory they can write.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

PAPER_FILES = ("paper.pdf", "scope.txt")
# Read-only mounts that let the sandboxed shell run this project's Python (numpy etc.), and its PATH.
def _py_binds() -> tuple[tuple[str, str], ...]:
    """(source, destination) read-only mounts for the venv and the interpreter it points at."""
    pairs = [(sys.prefix, sys.prefix), (sys.base_prefix, sys.base_prefix)]
    link = Path(sys.prefix) / "bin" / "python"
    if link.is_symlink():  # uv points at an alias dir (cpython-3.13-...) that is itself a symlink
        alias = Path(os.readlink(link)).parent.parent
        pairs.append((str(alias.resolve()), str(alias)))
    return tuple(dict.fromkeys((s, d) for s, d in pairs if Path(s).is_dir()))


PY_BINDS = _py_binds()
SANDBOX_PATH = ":".join([str(Path(sys.prefix) / "bin"), "/usr/local/bin", "/usr/bin", "/bin"])


def use_sandbox_venv(venv: str | Path) -> None:
    """Give the sandboxed shell another venv (e.g. one with CPU torch/jax for tiny checks), first on PATH."""
    global PY_BINDS, SANDBOX_PATH
    venv = Path(venv).expanduser()
    if not (venv / "bin" / "python").exists():
        raise SystemExit(f"sandbox venv not found: {venv}")
    binds = [(str(venv), str(venv))]
    link = venv / "bin" / "python"
    if link.is_symlink():  # the interpreter the venv points at, and the dir that holds it
        real = Path(os.readlink(link))
        real = real if real.is_absolute() else (link.parent / real)
        binds.append((str(real.parent.parent.resolve()), str(real.parent.parent)))
        binds.append((str(real.resolve().parent.parent), str(real.resolve().parent.parent)))
    PY_BINDS = tuple(dict.fromkeys(b for b in binds + list(PY_BINDS) if Path(b[0]).is_dir()))
    SANDBOX_PATH = str(venv / "bin") + ":" + SANDBOX_PATH
_pdf_cache: dict = {}


# pdfium is not thread-safe: two figure readers rendering at once killed whole runs with no traceback (2026-11-04,
# 12-18 readers in parallel). Every pdfium call takes this lock.
PDFIUM_LOCK = threading.Lock()


def pdf_text(path: Path) -> str:
    """Text of a PDF, one `--- page n ---` section per page (cached per file version)."""
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _pdf_cache:
        import pypdfium2 as pdfium  # imported lazily: only the paper-reading roles need it
        with PDFIUM_LOCK:
            doc = pdfium.PdfDocument(str(path))
            try:
                _pdf_cache[key] = "\n\n".join(
                    f"--- page {i + 1} ---\n{page.get_textpage().get_text_range()}" for i, page in enumerate(doc))
            finally:
                doc.close()
    return _pdf_cache[key]


class AccessDenied(PermissionError):
    pass


@dataclass(frozen=True)
class Policy:
    read: tuple[str, ...]  # workspace-relative prefixes ("" = everything not denied)
    write: tuple[str, ...] = ()
    shell: str | None = None  # None | "ro" | "rw"
    gpu: str | None = None  # None | "ro" (status, output, list) | "wait" (also wait: the job watcher) | "run" (also
    #                         submit, cancel its own jobs: a reviewer's short check) | "rw" (also download, cancel any)
    web: bool = False  # web_search / web_fetch (libraries, environments and references the paper cites)


def policy_for(label: str) -> tuple[str, Policy]:
    """Map a flow call label to (role, policy). Unknown labels get no access."""
    head, _, arm = re.sub(r"#\d+$", "", label).partition(":")  # retries are labelled `gate:a#2`
    repo = ("repo",)
    table = {
        "lead": Policy(read=PAPER_FILES, web=True),
        "figure": Policy(read=PAPER_FILES),  # reads paper numbers off figures (view_page)
        "figure-check": Policy(read=PAPER_FILES),  # the second, independent reading on the other tier
        "builder": Policy(read=PAPER_FILES + repo, write=(f"repo/arms/{arm}",) if arm else (), shell="rw", gpu="rw",
                          web=True),
        "reviewer": Policy(read=PAPER_FILES + repo, shell="ro", gpu="run", web=True),
        "watch": Policy(read=(), gpu="wait"),  # waits on one job and reports it; nothing else
        "evaluator": Policy(read=PAPER_FILES + repo, write=("repo/eval",), shell="rw"),
        "audit": Policy(read=PAPER_FILES + repo, shell="ro"),
        "tuner": Policy(read=PAPER_FILES + repo, write=("repo/tuned",), shell="rw", gpu="rw", web=True),
        "judge": Policy(read=PAPER_FILES + repo, shell="ro"),
        "reporter": Policy(read=("",), write=("report.md",)),
    }
    if head == "builder" and (not arm or "/" in arm or arm.startswith(".")):
        return head, Policy(read=())  # malformed arm name: no access
    return head, table.get(head, Policy(read=()))


def _under(rel: str, prefix: str) -> bool:
    return prefix == "" or rel == prefix or rel.startswith(prefix + "/")


class Workspace:
    def __init__(self, root: Path, gpu=None):
        self.root = Path(root).resolve()
        self.gpu = gpu  # a gpu.GpuRunner, or None when this run has no GPU

    @classmethod
    def create(cls, run_dir: Path, paper_dir: Path) -> "Workspace":
        root = Path(run_dir) / "work"
        for d in ("repo/data", "repo/eval", "repo/tests", "repo/arms", "repo/tuned"):
            (root / d).mkdir(parents=True, exist_ok=True)
        for name in PAPER_FILES:
            src = Path(paper_dir) / name
            if src.is_file() and not src.is_symlink():
                shutil.copyfile(src, root / name)
        return cls(root)

    def for_label(self, label: str) -> "Scope":
        role, pol = policy_for(label)
        return Scope(self, role, pol)


class Scope:
    """One role's view of the workspace."""

    def __init__(self, ws: Workspace, role: str, policy: Policy):
        self.ws, self.role, self.policy = ws, role, policy

    # ---- path checks -------------------------------------------------------
    def _resolve(self, rel: str, mode: str) -> Path:
        if not rel or os.path.isabs(rel) or "\x00" in rel:
            raise AccessDenied(f"bad path: {rel!r}")
        parts = Path(rel).parts
        if ".." in parts:
            raise AccessDenied("'..' is not allowed")
        norm = "/".join(p for p in parts if p != ".")
        allowed = self.policy.read if mode == "read" else self.policy.write
        if not any(_under(norm, a) for a in allowed):
            raise AccessDenied(f"{self.role} may not {mode} {norm}")
        # Walk from the root: no component may be a symlink.
        cur = self.ws.root
        for p in norm.split("/"):
            cur = cur / p
            if cur.is_symlink():
                raise AccessDenied(f"symlink in path: {norm}")
        if self.ws.root not in cur.resolve().parents and cur.resolve() != self.ws.root:
            raise AccessDenied("path escapes workspace")
        return cur

    # ---- file tools --------------------------------------------------------
    def read_file(self, rel: str, limit: int = 200_000) -> bytes:
        path = self._resolve(rel, "read")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as f:
            return f.read(limit)

    def read_text(self, rel: str, limit: int = 150_000) -> str:
        """Like read_file, but returns extracted text for a PDF and decoded text otherwise."""
        path = self._resolve(rel, "read")
        if path.suffix.lower() == ".pdf":
            if path.is_symlink() or not path.is_file():
                raise AccessDenied("not a regular file")
            return pdf_text(path)[:limit]
        return self.read_file(rel, limit).decode(errors="replace")

    def render_page(self, rel: str, page: int, crop: str = "", scale: float = 2.0) -> bytes:
        """JPEG of one PDF page (1-based), optionally cropped to `crop` = "x0,y0,x1,y1" as fractions of the page
        (0,0 top left). Lets a model look at a figure: text extraction keeps a plot's tick labels but not its curves."""
        path = self._resolve(rel, "read")
        if path.suffix.lower() != ".pdf" or path.is_symlink() or not path.is_file():
            raise AccessDenied("view_page only renders a PDF you may read")
        import io

        import pypdfium2 as pdfium
        with PDFIUM_LOCK:
            doc = pdfium.PdfDocument(str(path))
            try:
                if not 1 <= int(page) <= len(doc):
                    raise ValueError(f"page {page} is out of range 1..{len(doc)}")
                img = doc[int(page) - 1].render(scale=scale).to_pil()
            finally:
                doc.close()
        if crop:
            x0, y0, x1, y1 = (float(v) for v in crop.split(","))
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("crop must be x0,y0,x1,y1 with 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
            w, h = img.size
            img = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
            if img.width < 1200:  # a cropped panel is enlarged so small tick labels stay legible
                k = 1200 / img.width
                img = img.resize((1200, int(img.height * k)))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)  # several times smaller than PNG; legible
        return buf.getvalue()

    def write_file(self, rel: str, data: bytes | str) -> None:
        path = self._resolve(rel, "write")
        if path.exists() and path.stat().st_nlink > 1:
            raise AccessDenied("refusing to write through a hard link")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._resolve(rel, "write")  # re-check after mkdir
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, "wb") as f:
            f.write(data.encode() if isinstance(data, str) else data)

    def edit_file(self, rel: str, old: str, new: str) -> None:
        """Replace the single occurrence of `old` in a writable file."""
        text = self.read_file_for_edit(rel)
        n = text.count(old)
        if n != 1:
            raise ValueError(f"`old` occurs {n} times in {rel}; it must occur exactly once")
        self.write_file(rel, text.replace(old, new))

    def read_file_for_edit(self, rel: str) -> str:
        path = self._resolve(rel, "write")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as f:
            return f.read().decode()

    def grep(self, pattern: str, rel: str = "", max_lines: int = 300) -> list[str]:
        """`path:line: text` for lines matching a regex, in files this role may read (PDFs as text)."""
        rx = re.compile(pattern)
        roots = [rel] if rel else [p or "." for p in self.policy.read]
        hits: list[str] = []
        for top in roots:
            base = self.ws.root if top in ("", ".") else self._resolve(top, "read")
            files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
            for f in files:
                r = str(f.relative_to(self.ws.root))
                try:
                    text = self.read_text(r)
                except Exception:  # noqa: BLE001 - unreadable for this role, a symlink, binary or a broken PDF
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{r}:{i}: {line[:300]}")
                        if len(hits) >= max_lines:
                            return hits + [f"[stopped at {max_lines} matches]"]
        return hits

    def list_dir(self, rel: str = "repo") -> list[str]:
        if rel in ("", ".", "./"):  # the root: show only what this role can reach
            reach = [a for a in self.policy.read]
            return sorted(p.name + ("/" if p.is_dir() else "") for p in self.ws.root.iterdir()
                          if not p.is_symlink() and any(_under(p.name, a) or _under(a, p.name) for a in reach))
        path = self._resolve(rel, "read")
        return sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir() if not p.is_symlink())

    # ---- gpu ---------------------------------------------------------------
    _GPU_LEVEL = {None: 0, "ro": 1, "wait": 2, "run": 3, "rw": 4}

    def gpu(self, need: str = "ro"):
        if self.ws.gpu is None:
            raise AccessDenied("this run has no GPU")
        if self._GPU_LEVEL[self.policy.gpu] < self._GPU_LEVEL[need]:
            what = {"ro": "see", "wait": "wait for", "run": "submit", "rw": "download or change"}[need]
            raise AccessDenied(f"{self.role} may not {what} GPU jobs")
        return self.ws.gpu

    def gpu_submit(self, label: str, rel: str, commands: list[str], tag: str = "") -> dict:
        src = self._resolve(rel, "read")
        return self.gpu("run").submit(label, src, rel, commands, tag)

    def gpu_cancel(self, label: str, job: str) -> dict:
        """`rw` roles may cancel any of the run's jobs; `run` roles only their own."""
        return self.gpu("run").cancel(job, None if self.policy.gpu == "rw" else label)

    def gpu_download(self, job: str, paths: list[str], dest: str) -> dict:
        target = self._resolve(dest, "write")
        target.mkdir(parents=True, exist_ok=True)
        return self.gpu("rw").download(job, paths, target)

    # ---- shell -------------------------------------------------------------
    def shell(self, cmd: str, *, timeout: int = 600, max_out: int = 20_000, sandbox: str = "bwrap") -> str:
        """Run `cmd` with only this role's files mounted and an empty environment."""
        if self.policy.shell is None:
            raise AccessDenied(f"{self.role} has no shell")
        if sandbox == "bwrap":
            if not shutil.which("bwrap"):
                raise AccessDenied("bwrap not available; pass sandbox='none' to accept a weaker guarantee")
            argv = self._bwrap_argv(cmd)
        elif sandbox == "none":  # weaker: no filesystem isolation, role paths are not enforced
            argv = ["bash", "-c", cmd]
        else:
            raise ValueError(sandbox)
        env = {"PATH": SANDBOX_PATH, "HOME": "/tmp", "LANG": "C.UTF-8"}  # never inherit os.environ
        try:
            p = subprocess.run(argv, cwd=self.ws.root, env=env, capture_output=True, timeout=timeout)
            out = (p.stdout + p.stderr).decode(errors="replace") + f"\n[exit {p.returncode}]"
        except subprocess.TimeoutExpired:
            out = f"[timed out after {timeout}s]"
        return out if len(out) <= max_out else out[:max_out] + "\n[truncated]"

    def _bwrap_argv(self, cmd: str) -> list[str]:
        root, w = self.ws.root, self.policy.write
        a = ["bwrap", "--unshare-all", "--die-with-parent", "--clearenv",
             "--setenv", "PATH", SANDBOX_PATH, "--setenv", "XLA_FLAGS", "--xla_force_host_platform_device_count=1", "--setenv", "HOME", "/tmp",
             "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
             "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
             "--dir", "/work", "--chdir", "/work"]
        for src, dst in PY_BINDS:  # read at call time: use_sandbox_venv may have replaced them  # same absolute paths inside, so the venv's interpreter symlinks resolve
            a += ["--ro-bind", src, dst]
        # Mount only what this role may read; paper files appear only for roles allowed them.
        for r in self.policy.read:
            if r and (root / r).exists():
                a += ["--ro-bind", str(root / r), f"/work/{r}"]
        if self.policy.shell == "rw":
            for r in w:
                (root / r).mkdir(parents=True, exist_ok=True)
                a += ["--bind", str(root / r), f"/work/{r}"]
        return a + ["bash", "-c", cmd]
