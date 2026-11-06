# NOTES — engine quirks found while building

A dated log, oldest first. Entries before "2026-11-03: simplified flow" describe the retired thirteen-role flow
(reader, spec, expert, planner, contracts, challenger, gate, ...) and the files that went with it (resume.sh, the
frozen round-1/round-2 flows); they stay for the reasons they record. The current design is in `DESIGN.md` and
`docs/architecture.html`; the last sections of this file record how it got there.

Engine: `openjiuwen` 0.1.19, git rev `07b9632391c0…` (same as JiuwenSwarm's `uv.lock`).
The version string is not a sufficient pin: it is a git dependency, so the pin is the rev.

## Resume
- `run_workflow(resume=<path>)` is the journal that is **loaded**; `journal_path` is only where it is
  **finalized** on normal completion. Passing only `journal_path` does not replay anything from a
  finalized journal. `framework/run.py` passes both (resume only if the file already exists).
- Records are also appended to a WAL sidecar `<journal_path>.wal` as calls complete, and the WAL is
  replayed on load, so a crashed run resumes even without a finalized journal.

## Backend contract (beyond the design doc)
- The backend is the **only writer of the token ledger** (`self.budget.add`, and `self.workflow_budget`
  after `bind_workflow_budget`). `MockBackend` bills only the session ledger, so a `--tokens`
  workflow ceiling is not exercised by mock runs.
- Extra hooks: `bind_budget`, `bind_workflow_budget`, `bind_progress_sink`, `model_pool_names()`
  (if it returns a list, `model` hints outside it fail fast), `describe_model`.
- `MockBackend` matches fixtures by exact `opts["label"]` only.
- Option keys the engine accepts: `label phase schema model timeout isolation agent_type builtin_model
  effort` plus `KNOWN_OPTIONS`.

## Loader
- Scripts are imported as real modules; `swarmflow` resolves via `sys.modules`. Lint rejects
  `time`/`random`/`datetime`/`uuid` use; `strict=True` turns that into `LintError`.

## Environment
- `uv run --project ../jiuwenswarm …` re-syncs the JiuwenSwarm venv as a side effect (it printed
  "Uninstalled 1 package / Installed 1 package"). Prefer reading `uv.lock` for the pin.
- The design doc says Python >= 3.11; JiuwenSwarm runs 3.13 and this repo is pinned to 3.13 via
  `.python-version`.

## Agent loop (hand-written; ReActAgent not used)
- `framework/agents.py` is a small tool-calling loop over OpenRouter, so the `submit` tool, schema
  validation and per-call cost accounting are under our control.
- Lessons from live runs on double-q-learning:
  - Truncating a paper read made the reader re-read it 25 times (fixed: whole-file reads).
  - Tight per-turn caps (300k, then 400k tokens) stopped every long worker mid-job. These models need
    many short steps over a large context.
  - Compacting old messages breaks the provider's prompt cache. It now runs only near the context window.
- Context: 1M tokens for both tiers. Spend is capped per run in dollars (`--usd`), from OpenRouter's
  reported `usage.cost`. The token ledger counts cached input too, so it no longer tracks cost.
- Prompt caching (measured 2026-10-27): deepseek-v4.1-flash-1017 served 3968/4036 tokens from cache on an
  identical repeat (~15x cheaper). glm-5.3-flash showed no cache hit on the same test.

## Ideas taken from elsewhere
- JiuwenSwarm cluster runs (`../paper-reproducer/cluster`, deepseek-v4.1-flash):
  - Median 82k / max 178k input tokens per call, ~96% cache hits, no `max_tokens` cap.
  - Workers used bash, read_file and edit_file most, so `edit_file` and `grep` were added.
  - Their member rules ("never report a number you did not read from an artifact", validate numerical
    methods on a small exact case, fix root causes instead of changing the protocol) are now in the
    shared system prompt.
- Parity (github.com/KenC2006/Parity):
  - A rejected plan goes back to the same agent with the deterministic check's reasons. Here, spec and
    planner retries are turns of one session whose label is set per turn (`spec:c1`, `spec:c1#2`). The
    engine drops a `None` session label and takes `label` from the turn's options.
    Parity's own `member` option is not usable: JiuwenSwarm's and MockBackend's KNOWN_OPTIONS reject it.
  - Reviewers run on a different model from workers: gate, challenger and contracts-check are on
    `cheap` (glm); contracts and builders are on `strong` (deepseek).

## How the cluster team communicated (adopted 2026-10-27)
Source: `../paper-reproducer/cluster/edge-of-stability/web_*/agent_teams/.../sessions/*/{tasks,messages}`.
- Task specs had GOAL / setup values ("treat as target, list every deviation") / COMPUTE / DELIVERABLES
  at exact paths / checkable ACCEPTANCE. Here: specs now carry `acceptance` criteria, and the gate checks
  them criterion by criterion.
- The verifier validated the numerical core on a tiny exact case IN PARALLEL with the build. It found a
  440% error trap, and the fix reached the engineer as a concrete action before any GPU run. Here: the
  challenger reviews each arm's code against the paper before its real runs.
- Findings were graded ("does NOT change the verdict") and came with evidence and an exact fix. Here:
  REVIEW = issues with evidence, action and `blocking`, and the pass is decided in the flow
  (`blocking()`), not by the reviewer.
- The final audit re-derived every headline number from raw artifacts. Here: `audit:<claim>`, on a
  different model from contracts, recomputes from raw outputs, and a disagreement caps the claim.
- Every multi-turn role's turns are labelled base, base#2, ..., so JSON fixtures can address each turn
  (fixtures/*.json, with `$extends`).

## gpurun (added 2026-10-27)
- `framework/gpu.py` drives the gpu_runner daemon (`../gpu_runner`) over `ssh gpu` with the same CLI the
  MCP server uses. The agents' bwrap shell has no network and no ssh key, so the GPU is reached only through
  the gpu_* tools, run by the framework: builder/contracts may submit (upload a dir they can read) and
  download (into a dir they can write); gate/audit/contracts-check may only see status/output/list.
- A job gets ONE A100-PCIE-40GB (CUDA_VISIBLE_DEVICES set by the daemon), 56 CPU cores, network, `uv` and a
  shared uv cache; system python is 3.12. nvidia-smi lists all 8 GPUs of the box.
- Every job is journaled in `runs/<id>/gpu_jobs.jsonl`; agents only see their own run's jobs, and
  `framework/run.py` cancels any still-queued/running job of the run when it ends.
- Finished live runs are exported to `results/<paper>/` in the cluster layout
  (REPORT.md, VERIFICATION.md, data/ eval/ tests/ arms/<arm>/..., result.json, RUN).
- The single GPU queue was the bottleneck of the first 4-paper round (one job at a time, most of them
  `uv sync` + CPU-only tests). The agents' shell now has CPU torch 2.8 / jax / scipy / pandas from the project
  `.venv` (installed with `uv pip install --index-url https://download.pytorch.org/whl/cpu ... torch==2.8.0+cpu
  "jax[cpu]" scipy pandas`; not in uv.lock, and `uv run` keeps them since it syncs inexactly), so tiny checks
  run locally and only real runs go to the GPU. `sandbox_venv` in config.toml can point the shell elsewhere.
- Agents are told to submit 15 calls before `max_steps` (a contracts-check spent all 200 calls and never
  answered, which the flow then counted as a blocking review).
- A killed run must not leave jobs on the queue: `framework/run.py` cancels the run's GPU jobs on
  SIGTERM/SIGINT (a SIGINT alone did not stop a run whose agent was blocked in gpu_wait).
- Overnight round 1: lottery-ticket-small was stopped at 05:25 after ~8 h in Contracts. Its data code asked HF
  for a CIFAR-10 file that does not exist (404), fell back to the slow UofT mirror and re-downloaded the
  dataset in every job, holding the single GPU ~1 h per job while the other three runs waited.
- Paused 2026-10-28 05:35 (laptop offline): dlinear, edge, reppo stopped mid-Build. GPU jobs still running on
  the box were left running; their ids are in runs/<id>/gpu_jobs.jsonl (agents see them with gpu_list after a
  resume). `./resume.sh` resumes the three on `flows/reproducer_flow_round1.py`, a frozen copy with the
  prompts they started with, so their journals replay (call signatures hash prompt content, not the path).

## 2026-10-28: resumed sessions sent dict history (fixed)
On `--resume`, `ReproBackend.send_turn` rebuilt a session from the engine's history mirror, which stores a
structured reply as a dict. OpenRouter rejected it (`HTTP 400 messages.2.content: Invalid input`), the turn was
skipped, and the flow carried on without it: reppo's Compute returned nothing (verdict could_not_test with all
runs on disk), and dlinear lost builder:nlinear#2/#3. `_text()` now JSON-encodes non-string history content.
Skipped calls are not journaled, so a re-resume re-runs them live.

## 2026-10-28 ~11:15: stopped, GPU given up
reppo-retrying finished (partial: RePPO 0.918 vs PPO-V+Ent 0.581 normalized, capped by contracts review).
dlinear-ltsf (Build), edge-of-stability (Build, gd-small-lr at gate) and lottery-ticket-small (Contracts) were
stopped with SIGTERM; the gpu_runner daemon was already down, so no job cancels went through. `./resume.sh`
resumes all three once a GPU daemon is back. The laptop sleeping also stalled runs (09:37-11:10): in-flight
ssh downloads hung with no keepalive; gpu.py now sets ServerAliveInterval.

## 2026-10-29: fixes from the reppo-retrying run
- Root cause of its integrity caveat: `SLUG` rejects `.`, so the reader's arm `reppo-1.2` was dropped
  silently in `run()`. Spec retries then went back and forth between "unknown arm reppo-1.2" and "needs role a
  and b", and contracts built RePPO itself. Now: `arm_problems` sends a bad arm name back to the reader with a
  suggested slug, `claim_problems` requires an ordering claim's better/worse to be listed arms, and a run whose
  arm list stays unusable stops.
- Retries (reader, lead, spec, planner) go through `retry_prompt`: this try's problems, the earlier tries'
  problems ("do not bring them back") and the full rule list. Spec rules are also in the first prompt.
- Calls the flow cannot go on without: expert and Compute are retried once, then the run stops with a reason
  (before, a failed Compute was judged as `could_not_test`, and a failed expert kept every spec with no risks).
  Compute is also re-asked for claims missing from its results.
- Contracts get `max_rounds + 1` review rounds and `CONTRACT_RULES` (complete run set, one artifact per run,
  paired diff, never build an arm). Unresolved blocking issues ride on the cap as `detail`.
- Caps section of REPORT.md is written by the flow (`caps_md`, appended in `export`), not by the reporter.
- Audit has `problems` (contracts' outputs contradicting the raw files); any found get contracts one fix turn,
  and they are listed in VERIFICATION.md.
- Ordering claims: optional `diff` (paired a - b CI over shared seeds). It decides the verdict only when the
  auditor's diff agrees (`checked_diff`); corrected protocols are not audited, so their diff is not used.
- Ordering claims: the reader gives `paper_a`/`paper_b` (secret, leak-linted); the table's "Scale vs paper"
  column shows measured / paper per arm. Not judged.
- `source_problems`: a compared arm whose audited results were not read from `repo/arms/<arm>/` caps the claim.
- GPU: jobs made only of probes / installs (`ls`, `cat`, `uv sync` alone, ...) are refused; at most
  `max_jobs_per_role` (12) jobs per role; submits stop once observed GPU time reaches `gpu_hours * budget_slack`
  (1.5). The daemon reports no times, so time is taken from status polls (`gpu_times.json`); a job never seen
  running is charged from submission. Gate and contracts-check have GPU level `run`: submit a short check,
  wait, cancel their own jobs, no download.
- `flows/reproducer_flow_round1.py` is unchanged (journal replay of the 2026-10-27 runs).
- `args.compute` is now `"local"` (jobs in the agents' sandboxed shell) or `"remote"` (the GPU box via gpurun,
  the default); was `"cpu"` / `"gpu"`. `framework/run.py` rejects anything else. The frozen round-1 flow still
  compares with `"cpu"`, so `"remote"` args resume on it unchanged.
- Final verdicts are now set by the `judge` agent on the `cheap` tier (GLM), by the user's decision on
  2026-10-30. This reverses "no agent chooses a verdict": the pure functions give a rule verdict, the judge may
  override it and caps, with a reason. Table and caps section show both. An invalid answer is retried; if there is
  still none, the rule verdicts stand and a cap entry says so. The judge can read the paper and repo/ (shell ro).
- Lead and spec writer are one role since 2026-10-30: `spec` (cheap tier, so the expert on `strong` is a different model; labels `spec`, `spec#2`, ...) selects claims and
  writes all their specs in one answer, so claims on one experiment share a setup (the 2026-10-29 run had three
  different roulette bet tables). A claim whose spec stays bad gets `no clean spec`; the others go on.
- `flows/reproducer_flow_round2.py` is a frozen copy with separate lead/spec and the judge, for resuming
  runs/20261029-161314-double-q-learning (`--flow flows/reproducer_flow_round2.py`). The `lead` policy and guide stay
  for it and for round 1.
- Gate issues carry `owner`. Blocking issues with owner `contracts` no longer go back to the builder (who cannot
  write repo/eval): after Build, contracts get one fix turn and `contracts-check:post-build` re-checks it; if it stays
  open, a cap says so. Found live on 2026-10-30: grid_eval.py bounded Q in [-12, +10] (one reward's range, not
  the return's) and rejected ~97% of valid runs, and every gate sent that to a builder who could not fix it.
- Arm families (2026-10-30): the 2026-10-29 double-q run rebuilt one experiment four times (8-17% shared lines, 59.7M
  builder tokens). The planner's `families` now groups compared arms that can share a harness, even if their update rule differs: one `builder:<family>` and one
  `challenger:<family>`, code in repo/arms/<family>/, outputs in repo/arms/<family>/<arm>/, a `gate:<arm>` per arm.
  A family of one keeps repo/arms/<arm>/. A bug in shared code reaches every arm, so the challenger carries more weight.
- `impl_rounds` (default 4) bounds builder <-> challenger; `max_rounds` still bounds the gate loop.
- GLM sent placeholder specs on 2026-10-30 (`procedure: "p"`, `metric: "m"`, adding one claim per turn) in the merged
  spec turn. They passed every structural check. `stub_problems` now rejects them (MIN_SPEC, MIN_CRITERION).
- Contracts-check blocks only on: a wrong result on honest outputs, rejecting valid runs, crashing on them, a leak, or
  not matching the specs (2026-10-30). Gaming issues are non-blocking: three rounds kept finding new ones and never
  converged, and auditors recompute from raw files. They are listed in VERIFICATION.md (`eval_notes`).
- Prompts never mention fabrication or gaming, at the user's request (2026-10-30): naming it could give an agent the idea.
  `test_prompts_never_raise_the_idea_of_faking_results` guards this. The frozen round 1 and 2 flows keep their old text.
- A gate's finished review was lost on 2026-10-30 to one extra field (`not_checked`): three invalid submits, then skipped,
  which the flow counts as a blocking issue. `_parse` now drops keys the schema forbids if the rest is valid.
- First live judge (2026-10-30) lifted a `contracts failed review` cap saying no defect was recorded: the flow had passed it
  the cap's name but not its findings, which live only in the flow. It also cited the builder's own REVIEW.md as
  independent. The judge now gets `caps` with `detail`, and is told repo/arms/ is the builder's own evidence.
- Leak lint false positive (edge-of-stability, 2026-11-02): it compares numbers only, so a target of 200 matched "width-200" and "200-example subset",
  and a target of 400 matched "~400 sharpness measurements". Three spec tries failed and both `eos-hover` claims, the paper's main result, became
  `no clean spec`. Now: the reader lists such setup values in the claim's optional `setup_numbers` (exempt from the lint for that claim, so the
  exemption also covers a real mention of that number), and a rejection quotes where each number sits, since the retry could not find them before.
- GPU ledger (same run): `gpu_output` did not record status, so a job whose failure was read there stayed "running" and charged 1.9 h of phantom time; the
  run passed its 2.25 h limit after about 1 h of real use. `output` now polls status, and `submit` polls open jobs first. A job's end is still the time it was
  first seen ended.
- Claim encoding (edge-of-stability, 2026-11-02): the reader turned "sharpness rises to 2/eta for both step sizes" into an ordering between the two arms, which
  fails by construction (the smaller step size has the higher ceiling), and "loss keeps falling" into a value read off a plot, which only passed on a judge
  override. The reader prompt now keeps `ordering` for method comparisons and asks for per-arm predicates on trends. This changes the reader's prompt, so a run
  started before it no longer replays its reader on --resume.
- `source_problems` capped single-arm claims for not reading the experiment's other arm, because specs of one experiment list all its arms. A non-ordering claim
  now only needs one of its arms read from an arm directory. Ordering claims still need both.
- Setup values (2026-11-02): spec writer, expert and planner now start seeds and hyperparameters from scope.txt, then the paper; other values are marked `assumed`
  and a budget cut must be stated. Prompts only, no deterministic check.
- Tune phase (`"tune": true`, `tune_trials`, by the user's decision on 2026-11-02): after the blind run is audited, `tuner` (strong tier) sees the paper's numbers and
  may change any hyperparameter, including stated ones, in repo/tuned/. Builders stay blind. A tuned match needs its own `audit:<claim>:tuned` and is shown as
  `reproduced (tuned)`. The blind number stays in Measured, and the flow writes the changes and trial count to the Tuning section. It shares the run's GPU budget and
  job cap, so a run that spent its budget cannot tune.
- Seeds (edge-of-stability, 2026-11-02): the spec chose seed 0 alone with no stated reason, and the prompt-only `assumed` rule did not catch it. Specs now carry `seeds`
  and `seeds_reason`; `spec_problems` rejects fewer than 3 without a reason. The spec writer must go through the whole paper for hyperparameters before marking one
  `assumed`, and take stop rules from the paper's protocol, not a figure's axis (a 4000-iteration cap from Fig. 4 missed the crossing at 4100).
- Spec answers as broken JSON (lottery-ticket and dlinear-ltsf, 2026-11-02/03): GLM sent 6 of 9 spec answers as strings that did not parse
  (`Expecting property name...`, `Expecting ',' delimiter` at char ~3.6k), so both runs stopped at Spec. `submit` now declares `result` as an object
  when the answer has an object schema, so the model sends the JSON itself; a string still parses (with raw newlines allowed). Rejected submits are
  kept in <run>/rejected/<label>.<n>.txt, and a schema rejection lists up to 6 errors, not just the first. A skipped turn leaves nothing in its
  session, so `_Turns` now resends the original request until one turn has answered (spec#2 got only the rejection list and selected nothing).
- GPU budget (edge-of-stability, 2026-11-03): 1.5 h (hard limit 2.25) was spent before Tune; the tuner then ran ~2.5 h of CPU trials in its sandbox,
  outside the ledger. lottery-ticket and dlinear-ltsf now get `gpu_hours: 4`.
- Looser guardrails (2026-11-03, at the user's request: too many runs stopped at a check). Deterministic checks now repair what they can instead of
  sending the agent back: non-slug arm names are renamed everywhere (`repair_arms`) and unusable claims are dropped and logged once some claims are
  usable; a spec selection drops unknown ids and repeats, defers claims it does not mention and is cut to `max_claims` (`repair_selection`); a plan's
  deps on unknown jobs or on post-arm contracts jobs are dropped (`repair_plan`), and over-budget totals or missing gpu_hours estimates are logged,
  not rejected (the GPU ledger still enforces the budget). The `seeds` rule is advisory. A spec that still leaks on the last try has the numbers
  replaced by "[withheld]" (`redact`) instead of losing the claim. An agent gets 5 invalid submits, not 3. lottery, dlinear and reppo args now have
  gpu_hours 6 and 5 spec/plan tries. Unchanged: the leak check on earlier tries, placeholder specs, plan cycles and missing arms, and every review gate.
- `repair_plan` first dropped an arm job's dep on a post-arm contracts job. On lottery-ticket (2026-11-03) that cut reinit's wait on the ticket arm's
  masks (`stage-masks`), so reinit could start first. The dep is now rewired to the arm jobs behind that contracts job: a cross-arm dep, which
  `family_waves` orders. Builders read all of repo/, so the later arm can read the earlier one's outputs.
- Upload timeouts (lottery-ticket, 2026-11-03): the builder submitted `repo` at 383 MB (177 MB local CIFAR-10 copy in repo/data plus 206 MB of its own
  CPU smoke outputs in arms/conv2-convnet/tmp). Each submit tried 3 x 600 s and failed, about an hour lost. The live run's tmp/ was moved to
  <run>/moved-builder-tmp by hand. The submit timeout now grows with the upload (600 s + 10 s/MB), and a failure tells the agent the size and what to move.
- Job cap (reppo-retrying, 2026-11-03): the tuner hit `max_jobs_per_role` = 12 after 0.4 of 9 GPU-hours; the cap, not the budget, ended tuning.
  Raised to 30 for later runs (the GPU-hour ledger is the real limit).
- The method's own score (reppo-retrying, 2026-11-03, the user's call): the spec writer deferred `reppo12-breakout-score` (a value claim, paper 0.72)
  as "redundant with the ordering" and because "its acceptance would embed the paper's figure value". The spec prompt now says to select the method's
  score claim whenever the paper reports it, and that a spec never states the number (the flow compares it). The tuner's target is the method's
  number (`paper_value`, or `paper_a` for an ordering), not the ordering; `judge_tuned` caps a tuned ordering by `judge_value(a, paper_a)`, so beating
  a weak baseline is not a tuned match. The tuner is also told to look for a bug when an arm scores near random.
- The original reppo-retrying run (2026-10-27) never ran RePPO: `reppo-1.2` was not a slug, so only ppo-q-ent and ppo-v-ent were built, and its
  "a=0.92" was ppo-q-ent. The 2026-11-03 run's ppo-q-ent (same spec values) scored 0.015. Its shared harness reuses the same RNG keys every
  iteration (`per_step = split_n(train_keys, R)`), but a GPU test with fresh keys per iteration changed nothing (ppo-q-ent raw ~3, reppo ~0.3),
  so that is not the cause. The old run's ppo-q-ent did not learn either (normalized 0.0-0.10, 5 of 10 seeds at 0); its 0.918 for RePPO came from
  a reference RePPO the contracts agent wrote in eval/reppo12_ref (now against CONTRACT_RULES). GPU tests of the new RePPO with 512-wide layers,
  a Q(s_t,a_t) GAE baseline, or both, all stayed at random level (raw < 1). Not yet tested: running the old reference trainer today.
- Tolerance (2026-11-03, the user's call): value, predicate and tuned matches use at least REL_TOL = 0.15 relative tolerance (was 0.05 by default);
  a claim's own `rel_tol` can only loosen it. Partial is still within 3x. Audit agreement (2%) is unchanged.
- Bracket repair (reppo, 2026-11-03): with `result` typed as an object, GLM still sent the spec answer as a string, and dropped the `}` that ends
  each selected item before the next (`...}, {"claim_id"`), then padded the end with extra closers. 3 spec turns (15 submits) failed. `_close_braces`
  rebalances brackets with a stack (close an object followed by `, {`, drop closers with nothing open, close inner objects at `]`, close what is
  open at the end). The result is used only if it validates against the schema. It recovered all 9 rejected answers of that run.
- Prompt/code review (2026-11-03): the reppo reader gave `paper_a` 200 (a raw score) for a "normalized return" metric, so no measured value
  could match. `unit_problems` rejects a paper number above 1.5 on a metric named normalized/fraction/probability/proportion, and the reader
  prompt says paper numbers are in the metric's units. Reader claims with problems are now retried, and dropped only on the last try.
- Algorithmic papers (the user's call): reader and spec writer are told that a method's benchmark numbers (its score, return, accuracy or error)
  are the main claims: one `value` claim each, selected first, never deferred as redundant with an ordering. The spec role guide no longer
  says "not redundant with each other", which had contradicted this.
- Tune now picks claims by `judge_tuned`, so an ordering that holds while the method's score is off the paper's is tuned too.
- No-learning check: the builder guide asks for a short run that shows learning before the full runs, and a gate marks blocking a trained
  arm still at untrained/random level (reppo's RePPO passed every review at random level).
- Text fixes: TEAM names builder/challenger per family and the tuner; the planner guide allows a direct dep on another arm's job; the GPU
  guide says to submit the smallest directory, never one with datasets or old outputs; the judge has its own `max_judge_tries` (default 3).
- Leak check removed (2026-11-03, the user's call): it kept flagging hyperparameters that equal a result (reppo: GAE lambda 0.8 and clip 0.5
  vs scores 0.8 and 0.5; edge: width 200), and agents got round it by writing numbers in words. Gone: leak_lint, redact, claim_secrets, the
  spec rule and the reader's `secrets`/`setup_numbers` instructions (both stay optional in the schema for old fixtures). The spec writer is
  still asked not to write the paper's reported results into specs, and builders still cannot read the paper.

## 2026-11-03: simplified flow (replaces everything above that names reader/spec/expert/planner/contracts/challenger/gate)
- Roles: lead (claims, then one spec per experiment, one session), builder:<experiment> <-> reviewer:<experiment> (up to
  `build_rounds`: a pilot that must show learning, then the full runs), evaluator (all claims from the run files, own code
  in repo/eval), audit:<claim> (independent recompute; >2% disagreement caps partial), tuner (up to `tune_rounds` rounds,
  `tune_trials` settings each, fed the measured-vs-paper gap each round; its result gets audit:<claim>:tuned), judge,
  reporter. Workers on `strong`, checkers on `cheap`.
- Builders write one JSON per run in a fixed format (repo/arms/<experiment>/runs/<arm>/seed<k>.json) instead of
  following a contracts interface. No planner/job graph, families, contracts, contracts-check, expert, protocol risks
  or leak check. Gates left: claim/experiment structure (retried), unit check, review still open -> partial, full runs
  never finished -> could_not_test, audit disagreement -> partial, rule verdicts (15% tolerance), judge.
- Removed files: flows/lead_impl_verify.py (+ test, fixture), flows/reproducer_flow_round1.py and _round2.py, resume.sh,
  three old fixtures. args use `max_tries`, `build_rounds`, `tune_rounds`, `tune_trials`.
- Figure reading (2026-11-03): read_file gives a PDF's text only, so numbers that exist only in a plot were guessed (RePPO Breakout:
  0.8, 0.72, 0.55 in three runs). The lead now marks each claim's `source` (table/text/figure) and names the figure; claims from a
  figure get two independent readings (figure:<claim> on strong, figure-check:<claim> on cheap; both models accept images) with a
  `view_page` tool that renders a PDF page or crop. Readings are averaged; the uncertainty (larger reported half-width, or half
  the gap) widens the claim's tolerance; a reading that fails the unit check is not used. Readers zoomed 60+ times without
  answering until view_page was capped at 6 per turn and the figure roles at 25 calls. Live check on Fig. 6a: both read RePPO
  ~170 raw -> 0.68 +- 0.045 and PPO-Q+Ent ~95-100 -> 0.39 +- 0.05, for $0.005.
- GPU-hour limit off (2026-11-04, the user's call): the ledger charges a job from first seen running to first seen ended, with no
  real times from the daemon, so jobs running while the laptop slept were charged the night (reppo 17.2 h, dlinear 34.5 h) and
  every submit was refused. `GpuConfig.enforce_budget` (default False) now gates the check; gpu_hours stays a planning hint.
  The three runs were stopped and resumed with --resume (only the in-progress builder turns re-ran). dlinear-ltsf started before
  figure reading, so it resumes on flows/reproducer_flow_dlinear_resume.py (lead prompt, example and claim schema as at its start;
  its lead signature was checked against the journal). Delete that file once the run is done.
- pause_runs.sh / unpause_runs.sh freeze and continue live runs (SIGSTOP/SIGCONT to each run's process group) for offline periods.
- Open-ended prompts (2026-11-04, the user's call): every role may read paper.pdf and scope.txt and has view_page, builders
  included, so builds are no longer blind to the paper's numbers (the report says so). Prompts state goals, not procedures:
  gone are the learning/random-policy checks, the seed and `assumed` rules, the lead's full method re-description and the
  placeholder/acceptance spec checks; ROLE_GUIDE is empty, TEAM/RULES are short. Kept: the run-file format, evaluator +
  independent audit with a disagreement cap, builder <-> reviewer rounds (open review caps partial), the scope rules, the
  rule verdicts and the judge. Why: RePPO failed in three builds partly because the spec left the Q critic's bootstrap open
  (the 11-04 builder used max_a Q(s',a), 'assumed'; every Q-critic arm, incl. plain PPO-Q, stayed at random level while the
  V-critic arms learned); a builder that reads Alg. 1 itself can check that.
- The dlinear resume flow was deleted with the run (all runs killed 2026-11-04 to switch to this flow).
- Superseded on 2026-11-04 by the open-ended prompts above: the builder no longer has to show learning in its pilot, and
  builders can read the paper (the 2026-11-03 entries say otherwise).
- Figure uncertainties in raw units (2026-11-03): one reader gave "± 10" (raw score) for a normalized value of 0.66, which
  made the claim's tolerance ~1500%. `_unc` converts an uncertainty larger than the value with the reader's own
  raw/value ratio, and caps a single reading's relative uncertainty at 50%.
- Claim ids are renamed like arm names (2026-11-04): edge's lead used `sharpness-ratio-lr-0.02` and all seven claims
  were sent back. `repair_arms` now slugifies claim ids too (`ratio-lr-0-02`).
- pdfium lock (2026-11-04): reppo and lottery died silently in their figure reads (12-18 readers rendering at once; no
  traceback, 13 GB free). pdfium is not thread-safe; `tools.PDFIUM_LOCK` now wraps every pdfium call (page text and
  renders), and pages are sent as JPEG (several times smaller than PNG). Both runs resumed from their journals; only the
  in-flight figure reads re-ran. A second watch now reports a run process that disappears without finishing.
- First result on the open-ended flow (edge-of-stability, 2026-11-04): reproduced, 9 of 9 claims, one experiment, review
  passed in round 2, no tuning, 23M tokens ($1.29). The builder took the stopping rule from the paper (train to 99%
  accuracy, runs ended at 6,672 and 9,830 iterations) instead of last week's 4,000-step cap, which had cost the hover claim.
- Open question: the lead's own `rel_tol` can only loosen the 15% floor, with no limit. On edge it set 30-60% (the figure
  reads justified 4-8% on the ratios); on lottery it set 3%. Measured values sat within ~3-15% of the paper, so edge's
  verdicts would very likely hold at 15%, but a lead can make a claim easy. Not yet changed.
- With the paper open, reviewers judge results against the paper's numbers (lottery, round 1: the ticket was 66.05 vs 68.75
  and the reinit baseline far below the paper's curve, both marked blocking). The build now aims at the paper's numbers,
  which overlaps with the tuner's job.

- Unstated choices (2026-11-05, drafted, not yet run): the 10-27 contracts agent's reference RePPO reached ~275 raw (1.09
  normalized; rerun today gives identical numbers) because its spec settled the paper's gaps (wide net, no advantage
  standardization, env as shipped with no episode cap). The 11-04 builder filled them from the pgx example and added a
  2000-step eval episode cap from an unchecked belief ("Breakout lives 1000 steps"); the reference's greedy eval episodes
  run 14k-27k steps. Now: the lead lists `open_choices` per experiment (what the paper leaves open that could move a
  number, with a suggested setting and its source); the builder follows the paper's text first, adds no unstated limits
  (or records their bind rate), measures beliefs about the env/data, and reports `choices` (setting, value, source
  paper/reference/measured/guess, where in the code); the reviewer checks those first, and an untested guess that could
  move a number is blocking. Both fields are optional in the schemas so old fixtures still validate.
- Cap check (2026-11-05, jobs 1826b617 / c6e0a1ad): the reference trainer reruns bit-identical (1.0948). The 11-04
  builder's own best config with only the eval episode cap lifted (2000 -> 200000 steps): RePPO 256.5 raw (1.02 norm,
  was 124 capped), PPO-V 155.8 (0.62, was 115). So the cap explains the whole RePPO gap, and RePPO > PPO-V by ~100 raw.
  Both now sit above the paper's curves (RePPO ~0.69, PPO-V ~0.30), so the paper's eval protocol is still unknown
  (end-of-training sampled training returns here: RePPO ~69, PPO-V ~100, which do not match either).
- Paper's eval protocol (2026-11-05, read from the official code github.com/nissymori/remax-rl, agents/minatar/reppo.py;
  never shown to agents): cap OFF. evaluate() runs 100 envs with argmax actions in a while_loop until all are terminated
  (pgx returns zero reward after termination), no step limit; pgx-minatar Breakout has no time limit and
  sticky_action_prob 0.1. Logged as final_eval_R after 10M steps. Its RePPO otherwise differs from our builds:
  Conv32 -> Dense64 trunk, 64-64 tanh heads, flax default init, Adam eps 1e-5, normalize_advantage=True (the EI
  advantage IS standardized, unlike the App. D listing), use_current_probs=True (v2), ent_coef 0.0 for the main runs.
- Build checks (2026-11-05, implemented): run files carry `limits` ({name: {value, bound_fraction}}); the build answer
  carries `choices`, `limits` and `changes` (reason paper/reference/bug/result, `tested`). `build_problems` adds
  blocking issues mechanically, before the reviewer's: a limit the paper does not state that is unmeasured or cut more
  than LIMIT_TOL = 1%, and an untested `result` change; left open after the last round they cap at partial like any
  open review. The report gets a "Build settings" section (choices and limits). Builder rules: safety bounds far above
  need, diagnose a failing check before changing it and never pass it by changing what it measures (11-04: the
  builder's own "evaluation incomplete" check fired because good episodes outran a 5000-step eval budget; it added a
  2000-step per-episode cap so the check passed, then widened the net 64 -> 256/128 to chase the capped score). The GPU
  note asks for a wall-clock `timeout` instead of step caps.
- Reasoning is logged (2026-11-05): each model call's reasoning (`reasoning` / `reasoning_details`) and any text next
  to tool calls go to events.jsonl as activity_type "reasoning" (tool_name "thinking" / "says"), up to 20k chars.
- The 11-04 reppo run was stopped (2026-11-05) and restarted on these prompts. lottery-ticket-small keeps running on
  the old flow; if it needs --resume, use --flow flows/reproducer_flow_lottery_resume.py (a copy of the flow at 076aaf7),
  and delete that file once the run is done. Deleted 2026-11-07: the run is over, so the copy is gone.
- Web tools (2026-11-05, the user's call; JiuwenSwarm's cluster and agent modes had web access and did not use the
  authors' code): lead, builder, reviewer and tuner get `web_search` (openjiuwen free search, DuckDuckGo/Bing, no key)
  and `web_fetch` (non-HTML responses such as raw GitHub files come back verbatim; HTML goes through openjiuwen's page
  extractor). RULES: use the web for the libraries, environments, datasets and references the paper uses or cites,
  never the paper's own repository or other reproductions, and name the URL of every fact taken from it. Every call
  is in events.jsonl. Why: the 11-05 lead's spec said "pgx minatar with the default 1000-step episode limit and no
  sticky actions"; pgx-minatar Breakout has no limit and sticky_action_prob 0.1 (its breakout.py, one fetch away).
  The 11-05 11:12 reppo run (no web) was stopped after its lead turns and restarted with the tools.
- Method overshoot (2026-11-05, the user's call): the paper's own method may beat the paper's number. Claims carry an
  optional `method` flag (the lead sets it: true for the proposed method's own result, false for a baseline's);
  `judge_value(..., overshoot_ok)` reproduces a method claim that is better than the paper in the claim's direction,
  and `judge_tuned` lets an ordering's method arm exceed `paper_a`. Baselines still have to match within tolerance.
  Value claims now show measured/paper in 'scale'. Reviewer, tuner and judge prompts say the same. Why: with the
  paper's own uncapped eval, a faithful RePPO build scored 256 raw (1.02 normalized) against the paper's ~0.69.
- Prompt check on the 11:38 reppo run (stopped after its lead turns): the lead set `method` on entropy claims too, so
  higher entropy than the paper would have passed; `method` is now only for the method's benchmark performance, not
  diagnostics. Its spec suggested a pgx net "1x1 conv 16 + 3x3 conv 32, Dense 256, single-layer heads", contradicting
  App. E.1 (Conv 2x2 + avg-pool, two hidden layers per head), and it made no web call; the spec prompt now says a
  suggestion never contradicts the paper and library/reference defaults are checked on the web (URL) or marked
  'unverified'. Its environment suggestion no longer claimed a 1000-step limit.
- Spec checks (2026-11-05): every `open_choices` suggestion has a `basis`: `paper` (section), `web` (fetched code or
  docs, URL) or `guess`; the lead must confirm suggestions before giving them, and builders must check every `guess`.
  `spec_only: true` stops the flow after the lead's specs (no builds, nothing exported) to test spec prompts cheaply;
  args/reppo-retrying-spec.json runs it for reppo.
- Build escalation (2026-11-06, after the 20-hour run `web_1a1ad703422_f961592f77f7` ended with no report): the
  builder <-> reviewer loop had no route for a problem the builder cannot fix. The lead's e3 spec said the MLP width
  was `d1=d2=26` (a superscript "6" lost to flat text extraction); the reviewer proved it was `2^6=64` in round 3 and
  said so in rounds 3, 4 and 5, each time to a builder whose code was already right, and the fifth round ran out.
  Two more experiments were blocked on unstated guesses ("resolve this constant" with no bound on the search) when
  the run was stopped. Now: a build answer may carry `questions` (each with `what_i_checked`), a review issue may name
  `owner: "spec"`, and both go to one `confer` turn of the lead that answers with a `source` (`paper`/`reference`
  with a citation, or `decision`) and may return a `spec_patch` for `acceptance`/`procedure`/`arms`/`open_choices`.
  A `decision` is appended to the experiment's `open_choices` with `basis: "lead"` and printed in "Decisions taken
  during the build", so an unstated setting stays visible instead of being absorbed. A spec-owned review does not
  spend a `build_rounds` round, since no builder turn can fix it; `max_questions` (2) and `max_spec_fixes` (2) bound
  the channel, and the `lead:conf:<exp>` turns are serialized by a lock because the experiments run in parallel on one
  lead session.
- Measured guesses (2026-11-06): an issue may carry `sensitivity_pct`; a builder-owned blocking issue at or below
  `SENSITIVITY_TOL` (5%) is downgraded to a note. `REVIEW_RULES` said only that a guess which "could move" a claim is
  blocking, which nothing can satisfy: both e1 and e2 were blocked that way for hours, and each round of "test it"
  is an open-ended search. The measurement is what makes the complaint falsifiable.
- Build shape check (2026-11-06): a build answer that reports no run files, or files that are missing or do not parse,
  is rejected mechanically before the results are believed. It is skipped when the flow cannot see the run root
  (headless), and only what a builder can always satisfy blocks - coverage mismatches are logged, not blocking,
  because a mechanical check that cannot be satisfied is the trap this change exists to remove.

- Limit gate escape hatch (2026-11-07): a limit the paper does not state that binds past `LIMIT_TOL` (1%) used to be a
  permanent builder loop. e3's `steps_budget` cut 90.5% of what it applied to, e2's beta floor 47% and its alpha*beta
  band 83%, and "raise or remove it and re-run" is not something a builder can do when the GPU budget is what set the
  limit. Now: `LIMIT` may declare `guard: true` (a deliberate safety bound, tolerated up to `GUARD_TOL` 10% and still
  reported), and any limit that binds past its tolerance is owned by `spec`, so it goes to the lead with its evidence.
  The lead either says "raise it" (the builder can act) or `source: "decision"` accepts the narrowing, which records
  it in `lead_decisions`, prints it in the report, and stops the same limit blocking again. A reported run file
  outside the run root is also a builder-fixable issue now, instead of a "not found" that cannot be acted on.
- A stopped run reports (2026-11-07): `_stop` returned `claims: []` and no `report.md`, so a stop left nothing behind
  (the 2026-11-06 run ended after twenty hours with an empty answer). It now writes `report.md` under the run root with
  the reason, one row per experiment (full runs yes/no, review open, rounds, open findings) and the lead's decisions,
  and returns the same text as `report`. There is no agent to ask at that point, so the flow writes it itself.

- Figure readings that disagree (2026-11-07): the d-mask-ar1 read of the live run came out as `0.305 ± 0.7`, a 70%
  tolerance, because one reader reported the raw weight `-0.39` for a metric that reads `1 - d_Jac` (the strong reader
  correctly returned 1.0) and `combine_reads` averaged two readings in different units. Three changes: a reading
  outside the metric's range is dropped before combining (when the lead's own value is a fraction, a negative or
  >1.5 reading came from another quantity), the disagreement may widen the band but never past `MAX_READ_TOL` (0.5),
  and two readings that disagree trigger one tie-break read from the strong tier, after which the reading farthest
  from the median is dropped. The figure prompt now says a derived metric must be computed and reported in its own
  units, and that a figure that does not print what the metric needs gets `null`, not a nearby number. Widening the
  tolerance was never making the number more accurate: at 70% every measurement matched.
