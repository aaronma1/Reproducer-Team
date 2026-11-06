---
name: paper-reproducer
description: |
  Reproduces an ML paper's claims with a SwarmFlow pipeline: lead specs experiments, builders run them with a reviewer, results are computed twice and judged reproduced / partial / not_reproduced / could_not_test.
  Use when the user wants a paper's reported numbers checked by re-running its experiments from scratch.
  Do NOT use for summarizing or reviewing a paper without running experiments.
version: "0.1"
kind: swarm-skill
roles: []
---

# Paper Reproducer

A fixed pipeline, run by `scripts/workflow.py`: lead (claims, then one spec per experiment) -> builder <-> reviewer
rounds (pilot, then full runs) -> evaluator computes every claim + audit recomputes each one -> optional tuner
rounds -> rule verdicts -> judge -> reporter. The script owns all orchestration; do not orchestrate it yourself.

The build loop escalates to the lead instead of looping: a builder that cannot go on asks (`questions` in its build
answer), a reviewer that finds the experiment description wrong marks the issue `owner: "spec"`, and one lead turn
answers both — from the paper with a citation, or as a decision that is recorded in the report. A spec-owned review
does not use up a `build_rounds` round.

## Workflow

1. Make a run directory (absolute path) holding `paper.pdf` and `scope.txt` (what to reproduce, compute budget,
   rules such as "do not use the authors' code"). Every role reads and writes only under it.
2. Call `swarmflow(script_path=<this skill's dir>/scripts/workflow.py, args=<JSON string>)`. Do not edit the script.

args (JSON): `{"root": "<run dir>", "compute": "auto", "experts": true}` plus optional
`"smoke": true, "max_claims": 1` (a short check run), `"build_rounds": 5, "max_tries": 3, "max_questions": 2,
"max_spec_fixes": 2, "job_wait_s": 1800, "job_waits": 6, "tune": false, "tune_rounds": 4, "tune_trials": 4, "gpu_hours": 8,
"models": {"strong": <pool model>, "cheap": <pool model>}`.

- `experts: true` hands each role to the matching member of the **paper-reproducer-team** expert team, which
  enforces each role's file, shell, GPU and web rules. Bind that team to the session first. Without it, every role
  runs as a generic worker with no such rules.
- `compute: "auto"` uses the GPU through the gpu_runner MCP server
  (https://github.com/aaronma1/gpu_runner) when its `gpu_list` answers, and local CPU otherwise.
- `models`: workers run on `strong` and checkers on `cheap`, so no agent checks its own model's work. Give two
  different models from the team's model pool; leave it out and every role uses the default model.

The final result (and `report.md` in the run directory) has the verdict table, caps, and every deviation. A run
that stops before a verdict still writes `report.md`, with each experiment's open findings.

## Files

- `scripts/workflow.py`: the executable SwarmFlow definition (prompts, schemas, judging rules).
