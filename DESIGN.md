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
- The build loop escalates through one channel (`confer`): a builder's blocking `questions` and a reviewer's
  `owner: "spec"` issues are answered in one lead turn — with a citation when the paper answers, or as a recorded
  `decision` that lands in the experiment's `open_choices` with `basis: "lead"` when it does not. That turn may patch
  `acceptance` / `procedure` / `arms` / `open_choices`. A spec-owned review does not consume a `build_rounds` round
  (the builder cannot fix what only the lead can fix); `max_questions` and `max_spec_fixes` cap the channel.
- A review issue carries a stable `key`, an `owner`, and optionally a measured `sensitivity_pct`: a builder-owned
  guess that moves the claim by at most `SENSITIVITY_TOL` is a note, not a blocker.
- Workers (lead, figure, builder, evaluator, tuner) on the `strong` tier; checkers (figure-check, reviewer, audit,
  judge) on `cheap`, so a checker is never the worker's own model.
- Figure numbers are read twice (one per tier); a reading outside the metric's range is dropped, a disagreement
  triggers one tie-break read and may widen the tolerance only to `MAX_READ_TOL`, never to the raw gap.
- Every role may read the paper and render its pages (`view_page`). Builds are not blind to the paper's numbers;
  integrity comes from the fixed run-file format, the evaluator/audit recompute from raw files, and the reviewer.
- Prompts state goals and five shared rules; no per-role procedure guides.
- Hard gates are few and listed in `docs/architecture.html` (section 07): structure checks with repairs, the unit
  check, the build shape check (reported run files exist and parse), review-open and runs-never-finished caps, the 2%
  evaluator/audit agreement cap, rule verdicts at a 15% tolerance floor, and the judge's answer format.
- No GPU-hour limit (`GpuConfig.enforce_budget = False`); 30 GPU jobs per role per run.
- The builder submits and ends its turn (`stage: submitted`); the flow waits for the job with a cheap `watch`
  role and wakes the same session with the tail, so a queue wait costs cheap calls, not a long builder turn.
- A limit the paper does not state that binds past its tolerance is a scope decision owned by `spec` (guard
  limits are tolerated to 10%); the lead's acceptance is recorded and stops it blocking. A stopped run still
  writes `report.md` with its open findings.
- Runs are journaled; `--resume` replays every call whose signature (prompt, label, phase, model, schema, history)
  still matches.

## History

- Milestones: 1 scaffold + test_judge, 2 fixtures + flow mock tests + resume, 3 tools confinement, 4 agents/backend,
  5 gpurun client (2026-10).
- Until 2026-11-03 the flow had thirteen roles (reader, spec, expert, planner, contracts, contracts-check, builder,
  challenger, gate, audit, tuner, judge, reporter) with a job graph, arm families and a leak check on specs.
- 2026-11-03: simplified to the pipeline above. 2026-11-04: open-ended prompts and paper access for every role.
- 2026-11-06: the build loop escalates to the lead instead of re-prompting the builder with a problem it
  cannot fix (a 20-hour Build phase ended with no report: an acceptance line read `d1=d2=26` off a superscript
  and three rounds re-reported it).
