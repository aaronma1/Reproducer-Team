# paper-reproducer-team (JiuwenSwarm expert team)

An expert team (AgentGroup) that runs the paper-reproducer SwarmFlow inside JiuwenSwarm with the role rules of the
headless runner. Each pipeline role is one expert; the flow hands a role's calls to the expert of the same name
(`agent_template`), and that expert's rail enforces what `framework/tools.py` enforces headless:

| Role | Reads | Writes | Shell | GPU | Web |
| --- | --- | --- | --- | --- | --- |
| lead | paper.pdf, scope.txt | - | - | - | yes |
| figure, figure-check | paper.pdf, scope.txt | - | - | - | - |
| builder | paper, repo/ | repo/arms/&lt;experiment&gt;/ | rw | rw | yes |
| reviewer | paper, repo/ | - | ro | run | yes |
| evaluator | paper, repo/ | repo/eval/ | rw | - | - |
| audit, judge | paper, repo/ | - | ro | - | - |
| tuner | paper, repo/ | repo/tuned/ | rw | rw | yes |
| reporter | everything | report.md | - | - | - |

Paths are relative to the run directory. Shell commands run under bubblewrap with only those paths mounted, a
cleared environment, and the network only for roles with web access. GPU levels: `ro` status/output/list, `run`
also submit/wait/cancel, `rw` also download. Every role also has `view_page`, which renders a PDF page or a cropped
panel to a JPEG to open with `read_file`. The leader only stages the paper and launches the workflow.

## Layout

| Path | What |
| --- | --- |
| `roles.json` | the role table: names, descriptions, policies |
| `personas/` | one persona per role, plus `_team.md` (team and shared rules) appended to each |
| `common/policy_rail.py` | `RunPolicyRail`: reads the role's `policy.json` and checks every tool call |
| `common/view_page.py` | the `view_page` tool |
| `skill/` | the `paper-reproducer` Swarm Skill; its `scripts/workflow.py` is `flows/reproducer_flow.py` |
| `build.py` | renders `dist/paper-reproducer-team/` and checks it against the publish rules |

## Build, install, run

```sh
python team/build.py --zip          # -> team/dist/paper-reproducer-team/ (+ .zip)
```

1. Import it in JiuwenSwarm: Web, Agent management, Expert Teams, import from the path
   `team/dist/paper-reproducer-team` (or the zip).
2. Optional GPU: install https://github.com/aaronma1/gpu_runner (`make install` and `gpu_runner daemon` on the GPU
   machine, `make install-mcp` on this one), then add it as a stdio MCP server in JiuwenSwarm (Connectors): command
   = the gpu_runner venv's `python`, argument = `gpu_runner_mcp.py`. Without it, runs use local CPU.
3. Turn on SwarmFlow (`/swarmflow on`, then `/new`), bind the team to the session, and ask the leader to reproduce
   a paper. It makes a run directory with `paper.pdf` and `scope.txt` and runs the skill with
   `{"root": <dir>, "compute": "auto", "experts": true}`.

The host needs `bwrap` (bubblewrap) for the sandboxed shell and `uv` in `~/.local/bin` for Python environments.
`REPRO_SANDBOX=none` in the backend's environment turns the shell sandbox off (role paths are then enforced for
file tools only).

## Publish

Open the imported team's detail page and click Publish (sign in with GitHub or GitCode; it goes through Hub
moderation). The skill can also be published on its own from `dist/paper-reproducer-team/skills/paper-reproducer`.

## Not verified yet

- That expert members built for SwarmFlow workers load their rails and tools, and see the gpu_runner MCP server.
- That `model` and `agent_template` combine, so `models` still splits the tiers per role.
- That the group-shared skill is visible to the leader (if not, install the skill separately).
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
