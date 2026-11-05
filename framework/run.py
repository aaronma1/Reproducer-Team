"""CLI entry point: ``python -m framework.run --backend mock|live ...``."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path

from openjiuwen.agent_teams.workflow.engine import MockBackend, run_workflow
from openjiuwen.agent_teams.workflow.engine import BudgetExhausted
from openjiuwen.agent_teams.workflow.engine.budget import BudgetLedger

from .config import DEFAULT_FLOW, ROOT, RUNS_DIR, Config, openrouter_credentials

RESULTS_DIR = ROOT / "results"


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="framework.run")
    p.add_argument("--backend", choices=["mock", "live"], required=True)
    p.add_argument("--flow", type=Path, default=DEFAULT_FLOW)
    p.add_argument("--args", type=Path, required=True, help="JSON file passed to run(args)")
    p.add_argument("--fixtures", type=Path, help="MockBackend fixtures JSON (mock only)")
    p.add_argument("--tokens", type=int, default=None, help="workflow token ceiling (counts cached input too)")
    p.add_argument("--usd", type=float, default=None, help="stop the run once OpenRouter reports this much spend")
    p.add_argument("--resume", type=Path, help="existing runs/<id> dir to resume")
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--shell-timeout", type=int, help="hard limit in seconds for one sandboxed shell command")
    return p.parse_args(argv)


def _jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj if isinstance(obj, (str, int, float, bool, type(None))) else str(obj)


def _cancel_gpu_on_signal(gpu) -> None:
    """SIGTERM/SIGINT: cancel this run's GPU jobs, then exit. An agent blocked in gpu_wait (a worker thread)
    would otherwise keep the process from reaching the cleanup in `run`, leaving jobs on the shared queue."""
    def handler(signum, frame):
        print(f"signal {signum}: cancelling this run's GPU jobs", flush=True)
        try:
            print(f"cancelled: {gpu.cancel_all()}", flush=True)
        finally:
            os._exit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)


def make_backend(ns: argparse.Namespace, args, run_dir: Path):
    if ns.backend == "mock":
        from .fixtures import load_fixtures

        fixtures = load_fixtures(ns.fixtures) if ns.fixtures else {}
        return MockBackend(fixtures=fixtures)
    from .agents import Client
    from .backend import ReproBackend
    from .tools import Workspace

    cfg = Config.load()
    if cfg.sandbox_venv:
        from .tools import use_sandbox_venv
        use_sandbox_venv(cfg.sandbox_venv)
    if ns.shell_timeout:
        cfg.tool_timeout_s = ns.shell_timeout
    key, base = openrouter_credentials()
    paper_dir = cfg.paper_dir(str(args["paper"]))
    if not (paper_dir / "paper.pdf").is_file():
        raise SystemExit(f"no paper.pdf in {paper_dir}")
    ws = Workspace.create(run_dir, paper_dir)
    if args.get("compute", "remote") == "remote":
        from .gpu import GpuRunner
        hours = args.get("gpu_hours")
        ws.gpu = GpuRunner(cfg.gpu, run_dir / "gpu_jobs.jsonl", budget_hours=float(hours) if hours else None)
        _cancel_gpu_on_signal(ws.gpu)
    print(f"workspace: {ws.root}", flush=True)
    return ReproBackend(ws, cfg, Client(key, base), secrets=[key], say=lambda m: print(m, flush=True),
                        usd_limit=ns.usd)


def _verification(result: dict, gpu_jobs: list[dict]) -> str:
    """VERIFICATION.md: the independent checks behind each verdict (audits, reviews, caps, GPU run ids)."""
    out = ["# Verification", "", f"Paper verdict: **{result.get('verdict')}**", "", result.get("table") or "", ""]
    out += ["## Independent recomputation (audit:<claim>)", ""]
    for cid, a in (result.get("audits") or {}).items():
        out += [f"### {cid}", "```json", json.dumps(a, indent=1, ensure_ascii=False), "```", ""]
    if result.get("tuned_audits"):
        out += ["## Independent recomputation of tuned results", ""]
        for cid, a in result["tuned_audits"].items():
            out += [f"### {cid}", "```json", json.dumps(a, indent=1, ensure_ascii=False), "```", ""]
    out += ["## Experiment reviews (builder <-> reviewer)", ""]
    for e in result.get("experiments") or []:
        state = "passed review" if e.get("reviewed") else ("ran, review open" if e.get("ran") else "never finished")
        out.append(f"- `{e.get('experiment')}`: {state} after {e.get('rounds')} rounds")
        out += [f"  - open: {o}" for o in e.get("open") or []]
    out += ["", "## Caps", ""]
    for c in result.get("caps") or []:
        out.append(f"- {c.get('experiment') or 'all'}: at most `{c['ceiling']}` ({c['reason']})")
        out += [f"  - {d}" for d in c.get("detail") or []]
    if result.get("deferred"):
        out += ["", "## Claims the lead deferred", ""] + [f"- {d['claim_id']}: {d['reason']}" for d in result["deferred"]]
    if gpu_jobs:
        out += ["", "## GPU runs (gpurun job ids)", "", "| job | tag | submitted by | dir | first command |", "|---|---|---|---|---|"]
        out += [f"| {j['job']} | {j.get('tag') or '-'} | {j['label']} | {j['dir']} | `{j['commands'][0][:80]}` |"
                for j in gpu_jobs]
    return "\n".join(out) + "\n"


def export(run_dir: Path, result: dict, paper: str, dest_root: Path = RESULTS_DIR) -> Path:
    """Copy a finished run to results/<paper>/ in the cluster layout: REPORT.md, VERIFICATION.md, code, out/."""
    work, dest = run_dir / "work", dest_root / paper
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    report = work / "report.md"
    body = report.read_text() if report.is_file() else (result.get("table") or "")
    if result.get("caps_md"):  # written by the flow, not the reporter, so the reasons are exact
        body = body.rstrip() + "\n\n" + result["caps_md"]
    (dest / "REPORT.md").write_text(body)
    jobs_file = run_dir / "gpu_jobs.jsonl"
    jobs = [json.loads(l) for l in jobs_file.read_text().splitlines() if l.strip()] if jobs_file.is_file() else []
    (dest / "VERIFICATION.md").write_text(_verification(result, jobs))
    skip = shutil.ignore_patterns("__pycache__", ".venv", "*.pyc", ".pytest_cache")
    for sub in ("eval", "arms", "tuned"):
        if (work / "repo" / sub).is_dir():
            shutil.copytree(work / "repo" / sub, dest / sub, ignore=skip)
    (dest / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (dest / "RUN").write_text(f"{run_dir}\n")
    return dest


async def run(ns: argparse.Namespace, backend=None) -> object:
    args = json.loads(ns.args.read_text())
    compute = args.get("compute", "remote") if isinstance(args, dict) else "remote"
    if compute not in ("local", "remote"):  # "local": the agents' sandboxed shell; "remote": the GPU box via gpurun
        raise SystemExit(f'args.compute must be "local" or "remote", not {compute!r}')
    if ns.backend == "live":  # the flow asks for tiers; the live backend resolves them via config.toml
        args.setdefault("models", {"cheap": "cheap", "strong": "strong"})
    if ns.resume:
        run_dir = ns.resume
    else:
        paper = Path(str(args.get("paper", "paper"))).name if isinstance(args, dict) else "paper"
        run_dir = ns.runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{paper}"
        run_dir.mkdir(parents=True)
        (run_dir / "args.json").write_text(json.dumps(args, indent=2))
    journal = run_dir / "journal.jsonl"

    events = (run_dir / "events.jsonl").open("a", encoding="utf-8")
    wf_budget = BudgetLedger(total=ns.tokens) if ns.tokens else BudgetLedger()

    def progress(ev) -> None:
        events.write(json.dumps(_jsonable(ev), ensure_ascii=False) + "\n")
        events.flush()
        if ev.kind in ("phase", "agent_started", "agent_completed", "agent_failed", "workflow_failed", "log"):
            print(f"[{ev.kind}] {ev.phase or ''} {ev.label or ''} {ev.message or ''}".rstrip(), flush=True)

    try:
        backend = backend or make_backend(ns, args, run_dir)
        result = await run_workflow(
            str(ns.flow),
            args=args,
            backend=backend,
            # `resume` is the journal that is loaded; `journal_path` is where it is finalized.
            resume=str(journal) if journal.exists() else None,
            journal_path=str(journal),
            strict=True,
            log_sink=lambda m: print(m, file=sys.stderr, flush=True),
            progress_sink=progress,
            workflow_budget=wf_budget,
        )
    except BudgetExhausted as e:
        print(f"stopped: {e}", flush=True)
        (run_dir / "result.json").write_text(json.dumps({"error": str(e), "tokens": wf_budget.spent}, indent=2))
        raise SystemExit(2)
    finally:
        events.close()
        gpu = getattr(getattr(backend, "ws", None), "gpu", None)
        if gpu is not None:  # never leave this run's jobs on the shared queue
            left = await asyncio.to_thread(gpu.cancel_all)
            if left:
                print(f"cancelled leftover GPU jobs: {left}", flush=True)
        cost = getattr(backend, "cost", None)
        print(f"tokens spent: {wf_budget.spent}" + (f", cost ${cost:.4f}" if cost is not None else ""), flush=True)
    limit, cost = getattr(backend, "usd_limit", None), getattr(backend, "cost", None)
    if isinstance(result, dict) and limit is not None and cost is not None and cost >= limit:
        # the flow only sees calls being skipped, so the real cause is added here
        result["reason"] = f"run budget reached (${cost:.4f} of ${limit:g})" + (
            f"; {result['reason']}" if result.get("reason") else "")
    (run_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))
    print(f"run dir: {run_dir}")
    if isinstance(result, dict) and result.get("reason"):
        print(f"not exported, the run stopped early: {result['reason']}")  # results/ holds finished runs only
    elif ns.backend == "live" and isinstance(result, dict):
        print(f"exported: {export(run_dir, _jsonable(result), Path(str(args.get('paper'))).name)}")
    return result


def main(argv: list[str] | None = None) -> None:
    asyncio.run(run(_parse(argv)))


if __name__ == "__main__":
    main()
