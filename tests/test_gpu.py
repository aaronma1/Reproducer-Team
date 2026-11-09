"""gpurun tools against a fake gpu_runner CLI: role rules, job ownership, downloads into writable dirs only."""
import io
import tarfile

import pytest

from framework.gpu import GpuConfig, GpuError, GpuRunner
from framework.tools import AccessDenied, Workspace


class FakeCli:
    def __init__(self):
        self.jobs, self.uploads = {}, {}

    def __call__(self, args, stdin=None, timeout=0):
        op = args[0]
        if op == "submit":
            job = f"job{len(self.jobs)}"
            self.jobs[job], self.uploads[job] = "COMPLETED", stdin
            return f"{job}\n".encode(), ""
        if op == "query" and "--status" in args:
            return f"{self.jobs[args[1]]} 0 ".encode(), ""
        if op == "query":
            return b"hello\n", ""
        if op == "cancel":
            self.jobs[args[1]] = "CANCELLED"
            return b"CANCELLED  ", ""
        if op == "list":
            return "\n".join(f"{j}  {s}  cmd" for j, s in self.jobs.items()).encode(), ""
        if op == "download":
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                data = b'{"acc": 0.5}'
                ti = tarfile.TarInfo("out/metrics.json")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
            return buf.getvalue(), ""
        raise AssertionError(args)


@pytest.fixture
def ws(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    w = Workspace.create(tmp_path / "run", paper)
    w.gpu = GpuRunner(GpuConfig(), tmp_path / "run" / "gpu_jobs.jsonl", cli=FakeCli())
    return w


def test_builder_submits_and_downloads_into_own_arm(ws):
    b = ws.for_label("builder:a")
    b.write_file("repo/arms/a/train.py", "print(1)")
    job = b.gpu_submit("builder:a", "repo/arms/a", ["python train.py"], "c1/a")["job"]
    names = tarfile.open(fileobj=io.BytesIO(ws.gpu._cli.uploads[job]), mode="r:gz").getnames()
    assert "./train.py" in names
    b.gpu_download(job, ["out"], "repo/arms/a/out/" + job)
    assert (ws.root / "repo/arms/a/out" / job / "out/metrics.json").read_text() == '{"acc": 0.5}'
    with pytest.raises(AccessDenied):
        b.gpu_download(job, ["out"], "repo/arms/b")
    assert [j["tag"] for j in ws.for_label("reviewer:a").gpu().list()] == ["c1/a"]


def test_gate_may_run_a_check_but_not_download_or_cancel_others(ws):
    (ws.root / "repo/arms/a").mkdir(parents=True)
    built = ws.for_label("builder:a").gpu_submit("builder:a", "repo/arms/a", ["python train.py"])["job"]
    g = ws.for_label("reviewer:a")
    mine = g.gpu_submit("reviewer:a", "repo", ["python check.py"])["job"]
    with pytest.raises(AccessDenied):
        g.gpu_download(mine, ["out"], "repo/arms/a")
    with pytest.raises(GpuError):
        g.gpu_cancel("reviewer:a", built)
    assert g.gpu_cancel("reviewer:a#2", mine)["status"] == "CANCELLED"  # a later turn of the same role
    assert ws.for_label("builder:b").gpu_cancel("builder:b", built)["status"] == "CANCELLED"  # rw: any job


def test_audit_cannot_submit_and_paper_roles_have_no_gpu(ws):
    with pytest.raises(AccessDenied):
        ws.for_label("audit:c1").gpu_submit("audit:c1", "repo", ["python x.py"])
    with pytest.raises(AccessDenied):
        ws.for_label("lead").gpu()
    with pytest.raises(AccessDenied):  # a builder may not upload the paper
        ws.for_label("builder:a").gpu_submit("builder:a", ".", ["python x.py"])


def test_the_watcher_may_wait_but_not_submit_cancel_or_read_files(ws):
    from framework.agents import gpu_tool_specs
    w = ws.for_label("watch:j1#2")
    assert w.gpu("wait") is ws.gpu                       # a later check of the same job
    with pytest.raises(AccessDenied):
        w.gpu_submit("watch:j1", "repo", ["python x.py"])
    with pytest.raises(AccessDenied):
        w.read_file("paper.pdf")
    assert {s["function"]["name"] if "function" in s else s["name"] for s in gpu_tool_specs("wait", "")} == \
        {"gpu_status", "gpu_output", "gpu_list", "gpu_wait"}


def test_probe_only_jobs_are_refused():
    from framework.gpu import probe_only
    assert probe_only(["ls arms"]) and probe_only(["cd arms/a && uv sync"]) and probe_only(["uv sync", "nvidia-smi"])
    assert probe_only(["cd a && ls -la | head", "python --version"])
    assert not probe_only(["cd a && uv sync && uv run python train.py"])
    assert not probe_only(["uv sync", "uv run python -c 'import jax; print(jax.devices())'"])


def test_job_cap_per_role(ws):
    ws.gpu.cfg = GpuConfig(max_jobs_per_role=2)
    b = ws.for_label("builder:a")
    (ws.root / "repo/arms/a").mkdir(parents=True)
    b.gpu_submit("builder:a", "repo/arms/a", ["python t.py"])
    b.gpu_submit("builder:a#2", "repo/arms/a", ["python t.py"])
    with pytest.raises(GpuError, match="limit"):
        b.gpu_submit("builder:a#3", "repo/arms/a", ["python t.py"])
    with pytest.raises(GpuError, match="only inspects"):
        b.gpu_submit("builder:a", "repo/arms/a", ["ls"])


def test_gpu_hour_budget_stops_submits(tmp_path):
    now = [1000.0]
    paper = tmp_path / "paper"
    paper.mkdir()
    w = Workspace.create(tmp_path / "run", paper)
    w.gpu = GpuRunner(GpuConfig(budget_slack=1.0, enforce_budget=True), tmp_path / "run" / "gpu_jobs.jsonl", cli=FakeCli(),
                      budget_hours=1.0, clock=lambda: now[0])
    b = w.for_label("builder:a")
    (w.root / "repo/arms/a").mkdir(parents=True)
    w.gpu._cli.jobs["job0"] = "RUNNING"
    job = b.gpu_submit("builder:a", "repo/arms/a", ["python t.py"])["job"]
    w.gpu._cli.jobs[job] = "RUNNING"
    w.gpu.status(job)                      # seen running at t=1000
    now[0] += 1800
    assert abs(w.gpu.hours_used() - 0.5) < 1e-9
    b.gpu_submit("builder:a", "repo/arms/a", ["python t.py"])  # 0.5 h used of 1
    now[0] += 1800
    w.gpu._cli.jobs[job] = "COMPLETED"
    w.gpu.status(job)                      # finished at t=4600: one hour
    with pytest.raises(GpuError, match="GPU-hours"):
        b.gpu_submit("builder:a", "repo/arms/a", ["python t.py"])
    # a resumed run keeps the times
    again = GpuRunner(GpuConfig(budget_slack=1.0, enforce_budget=True), tmp_path / "run" / "gpu_jobs.jsonl", cli=FakeCli(),
                      budget_hours=1.0, clock=lambda: now[0])
    assert abs(again.hours_used() - 1.0) < 1e-9


def _clocked(tmp_path, now):
    paper = tmp_path / "paper"
    paper.mkdir()
    w = Workspace.create(tmp_path / "run", paper)
    w.gpu = GpuRunner(GpuConfig(budget_slack=1.0, enforce_budget=True), tmp_path / "run" / "gpu_jobs.jsonl", cli=FakeCli(),
                      budget_hours=1.0, clock=lambda: now[0])
    (w.root / "repo/arms/a").mkdir(parents=True)
    return w, w.for_label("builder:a")


def test_failure_read_through_output_stops_the_clock(tmp_path):
    now = [1000.0]
    w, b = _clocked(tmp_path, now)
    job = b.gpu_submit("builder:a", "repo/arms/a", ["python t.py"])["job"]
    w.gpu._cli.jobs[job] = "RUNNING"
    w.gpu.status(job)
    now[0] += 60
    w.gpu._cli.jobs[job] = "FAILED"
    w.gpu.output(job)                      # the agent sees the traceback and never polls this job again
    now[0] += 7200
    assert abs(w.gpu.hours_used() - 60 / 3600) < 1e-9


def test_submit_closes_jobs_nobody_polls(tmp_path):
    now = [1000.0]
    w, b = _clocked(tmp_path, now)
    job = b.gpu_submit("builder:a", "repo/arms/a", ["python t.py"])["job"]
    w.gpu._cli.jobs[job] = "RUNNING"
    w.gpu.status(job)
    w.gpu._cli.jobs[job] = "FAILED"        # ended unseen; charged up to the next submit, not for ever
    now[0] += 1800
    b.gpu_submit("builder:a", "repo/arms/a", ["python t.py"])
    now[0] += 7200
    assert abs(w.gpu.hours_used() - 0.5) < 1e-9


def test_only_own_jobs_and_bad_paths(ws):
    with pytest.raises(GpuError):
        ws.gpu.status("someone-elses-job")
    b = ws.for_label("builder:a")
    (ws.root / "repo/arms/a").mkdir(parents=True)
    job = b.gpu_submit("builder:a", "repo/arms/a", ["python t.py"])["job"]
    with pytest.raises(GpuError):
        b.gpu_download(job, ["../x"], "repo/arms/a")


def test_cancel_all_skips_finished(ws):
    b = ws.for_label("builder:a")
    (ws.root / "repo/arms/a").mkdir(parents=True)
    j1 = b.gpu_submit("builder:a", "repo/arms/a", ["python t.py"])["job"]
    j2 = b.gpu_submit("builder:a", "repo/arms/a", ["sleep 9"])["job"]
    ws.gpu._cli.jobs[j2] = "RUNNING"
    assert ws.gpu.cancel_all() == [j2] and ws.gpu._cli.jobs[j1] == "COMPLETED"


def test_gpu_hours_are_not_enforced_by_default(tmp_path):
    from framework.gpu import GpuConfig
    assert GpuConfig().enforce_budget is False
