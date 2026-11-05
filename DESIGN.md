# Paper Reproducer — headless SwarmFlow project

Constraints the code follows, and the current design in brief. The full description, with diagrams, is
[`docs/architecture.html`](docs/architecture.html); the reasons behind each change are dated in `NOTES.md`.

## Constraints (unchanged since the repo was created)

- Run `flows/reproducer_flow.py` on the SwarmFlow engine as a library (own runner, own `AgentBackend`, own role
  agents). No JiuwenSwarm server involved.
- The flow script may import only `swarmflow` and the stdlib, and must run unchanged inside JiuwenSwarm via
  `swarmflow(script_path=...)`.
- Dev loop: `MockBackend` + per-scenario fixtures under pytest. Live: tool-using agents on a paper dir.
- No orchestration logic in Python outside the flow; the backend only executes calls. No changes to openjiuwen;
  engine gaps go in `NOTES.md`.
- The role -> tools/paths table is **enforced in `framework/tools.py`**, not only in prompts.

## Current design (2026-11-04)

- Pipeline: lead (claims, figure readings, experiments) -> per experiment builder <-> reviewer rounds (pilot, then
  full runs) -> evaluator + independent audit per claim -> optional tuner rounds -> rule verdicts -> judge -> reporter.
- Workers (lead, figure, builder, evaluator, tuner) on the `strong` tier; checkers (figure-check, reviewer, audit,
  judge) on `cheap`, so a checker is never the worker's own model.
- Every role may read the paper and render its pages (`view_page`). Builds are not blind to the paper's numbers;
  integrity comes from the fixed run-file format, the evaluator/audit recompute from raw files, and the reviewer.
- Prompts state goals and five shared rules; no per-role procedure guides.
- Hard gates are few and listed in `docs/architecture.html` (section 07): structure checks with repairs, the unit
  check, review-open and runs-never-finished caps, the 2% evaluator/audit agreement cap, rule verdicts at a 15%
  tolerance floor, and the judge's answer format.
- No GPU-hour limit (`GpuConfig.enforce_budget = False`); 30 GPU jobs per role per run.
- Runs are journaled; `--resume` replays every call whose signature (prompt, label, phase, model, schema, history)
  still matches.

## History

- Milestones: 1 scaffold + test_judge, 2 fixtures + flow mock tests + resume, 3 tools confinement, 4 agents/backend,
  5 gpurun client (2026-10).
- Until 2026-11-03 the flow had thirteen roles (reader, spec, expert, planner, contracts, contracts-check, builder,
  challenger, gate, audit, tuner, judge, reporter) with a job graph, arm families and a leak check on specs.
- 2026-11-03: simplified to the pipeline above. 2026-11-04: open-ended prompts and paper access for every role.
