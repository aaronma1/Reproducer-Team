# paper-reproducer (headless SwarmFlow)

Tries to reproduce the claims of an ML paper with a team of tool-using agents, and judges each claim
`reproduced`, `partial`, `not_reproduced` or `could_not_test`. The orchestration is one SwarmFlow script,
[`flows/reproducer_flow.py`](flows/reproducer_flow.py), run on the openjiuwen engine as a library: no JiuwenSwarm
server is needed, and the same script also runs unchanged inside JiuwenSwarm.

```
lead (claims + one spec per experiment)
  -> builder <-> reviewer rounds (pilot, then full runs)
  -> evaluator computes every claim  +  audit recomputes each one independently
  -> tuner rounds on claims that fell short (optional)
  -> rule verdicts -> judge -> reporter
```

Workers (lead, builder, evaluator, tuner) run on the `strong` model tier; the checkers (reviewer, audit, judge) run
on `cheap`, so no agent checks its own model's work. See [`docs/architecture.html`](docs/architecture.html) for the
full picture and [`DESIGN.md`](DESIGN.md) for the constraints the code follows.

Design choices worth knowing before a run:

- **Not blind.** Every role, builders included, may read `paper.pdf` and `scope.txt`. Results are trusted because the
  evaluator and an independent auditor both compute every claim from the raw run files, not because builders are
  kept from the paper's numbers.
- **Open-ended prompts.** Prompts state the goal and a few hard rules (cite numbers from files, follow `scope.txt`, no
  paper or third-party code, report deviations). The builder <-> reviewer rounds catch the rest.
- **Figures are read, not guessed.** Numbers that exist only in a plot are read twice from the rendered page
  (`view_page`), once per model tier; the reading uncertainty widens that claim's tolerance.
- **Verdicts.** A value matches within 15% of the paper's number (partial within 3x that); the judge sets final
  verdicts next to the rule verdicts.
- **GPU.** One shared queue on the GPU box runs one job at a time across all runs. `gpu_hours` is a planning hint:
  the GPU-hour limit is off because the ledger cannot see real job times.

## Setup

```sh
uv sync                       # Python 3.13, pinned openjiuwen git rev (see pyproject.toml)
cp .env.example .env          # then set OPENROUTER_API_KEY
```

[`config.toml`](config.toml) holds the model tiers (OpenRouter ids), tool limits, per-role step caps, the papers
directory and the GPU box (`[gpu]`, reached with `ssh <ssh_host>` running `../gpu_runner`).

## Papers and args

A paper is a directory with `paper.pdf` and `scope.txt` (what to reproduce). It is named either by path or by name
under `papers_dir` from `config.toml`. Each run takes an args JSON from [`args/`](args/):

```json
{"paper": "../paper-reproducer/dataset/reppo-retrying", "compute": "remote", "gpu_hours": 6,
 "tune": true, "max_tries": 3, "build_rounds": 5, "tune_rounds": 4, "tune_trials": 4}
```

`compute` is `remote` (GPU box) or `local`; `smoke: true` and `max_claims` make a short run; `models` can override
the tier ids. The full list is in the flow's docstring.

## Running

```sh
# mock backend, no API calls
uv run python -m framework.run --backend mock --args args/<name>.json --fixtures fixtures/reproduced.json

# live run in the foreground
uv run python -m framework.run --backend live --args args/<name>.json --usd 50

# live run detached (survives closing the terminal); prints the log path under logs/
./launch.sh <name> --usd 50

# resume a stopped or crashed run
./launch.sh <name> --resume runs/<id> --usd 50
```

Other flags: `--tokens` (token ceiling), `--shell-timeout` (seconds per agent shell command), `--flow`, `--runs-dir`.
Stopping a run with SIGINT/SIGTERM cancels its GPU jobs.

`./pause_runs.sh` / `./unpause_runs.sh` freeze and continue all live runs (for example before going offline); jobs
already on the GPU box keep running.

## Outputs

- `runs/<time>-<paper>/`: everything for one run: `args.json`, `journal.jsonl` (+ `.wal`, used by `--resume`),
  `events.jsonl`, `gpu_jobs.jsonl`, `result.json` and the agents' workspace in `work/`.
- `results/<paper>/`: a finished run exported as `REPORT.md`, `VERIFICATION.md`, code and outputs. Runs that stop
  early are not exported.

`runs/`, `results/` and `logs/` are gitignored.

## Layout

| Path | What |
| --- | --- |
| `flows/reproducer_flow.py` | the flow: roles, prompts, rule verdicts (imports only `swarmflow` + stdlib) |
| `framework/run.py` | CLI, journal/resume, export to `results/` |
| `framework/backend.py` | `AgentBackend` that executes the flow's agent calls |
| `framework/agents.py` | tool-calling loop over OpenRouter, `submit` + schema validation, cost accounting |
| `framework/tools.py` | sandboxed tools; the role -> tools/paths rules are enforced here |
| `framework/gpu.py` | client for the gpu_runner daemon on the GPU box |
| `framework/config.py` | `config.toml` and `.env` loading |
| `fixtures/` | MockBackend fixtures for tests and mock runs |
| `NOTES.md` | engine quirks and lessons from live runs |

## Tests

```sh
uv run pytest
```
