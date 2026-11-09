# paper-reproducer-team (JiuwenSwarm expert team)

**Shell sandbox: bubblewrap.** Every member's shell runs in a bubblewrap sandbox with only its role's paths mounted. The host needs `bwrap` installed; without it, shell calls are refused.

## Results so far

Agent = JiuwenSwarm agent mode, one deepseek-v4.1-flash agent. Cluster = JiuwenSwarm team mode, a leader with engineer and verifier members, deepseek-v4.1-flash. SwarmFlow = the scripted flow, deepseek-v4.1-flash-1017 for authors and glm-5.3-flash for most checkers.

| Paper | | Agent | Cluster | SwarmFlow | Best |
|---|---|---|---|---|---|
| reppo-retrying | result | partial: RePPO behind PPO at the paper's settings | partial: baseline did not train (PPO 9.5 raw) | partial: **RePPO ahead of PPO by the paper's ratio**, capped only by an open review issue | **SwarmFlow**: only system to reproduce the finding |
| | cost | 15.7M tok · $0.17 · 1.0 h | 48.5M tok · $1.09 · 1.7 h | 31.9M tok journaled (+~7.5M re-run) · ≈$0.82 · ≈8.5 h | — |
| edge-of-stability | result | reproduced (1 seed) | reproduced (1 seed) | **reproduced, 9/9 numeric claims within tolerance, 3 seeds, every number recomputed by an independent audit** | **SwarmFlow** on rigour; agent ran more side experiments |
| | cost | 10.0M tok · $0.21 · 0.6 h | 15.0M tok · $0.25 · 0.4 h | 23.2M tok · ≈$0.47 · 0.8 h | — |
| lottery-ticket-small | result | reproduced, pruned to 33% only | reproduced, pruned to 33% only (accuracy result overclaimed) | **pruned to 1.9%, the paper's full range**: all orderings reproduce; after tuning, deep-pruning accuracy matches the paper | **SwarmFlow**: only system to test the paper's headline region |
| | cost | 27.5M tok · $0.58 | 19.5M tok · $0.79 | 26.6M tok to audit (tuner still running) · ≈$0.55 so far | — |

SwarmFlow dollars are estimated at the rates measured on the reppo run ($0.020 per million tokens for the strong model, $0.026 for the cheap one); the flow did not log dollars on these runs. Agent and cluster figures are summed from JiuwenSwarm's traces, from their runs under `../paper-reproducer/{agent,cluster}/`.

An expert team (AgentGroup) that reproduces an ML paper's claims from scratch inside JiuwenSwarm, and judges each
claim `reproduced`, `partial`, `not_reproduced` or `could_not_test`. It runs the same SwarmFlow script as the
headless runner (`flows/reproducer_flow.py`, shipped as the team's `paper-reproducer` skill) with the same role
rules: every pipeline role is one expert, and each expert can touch only the files, shell, GPU and web access its
role needs.

## What the team does

The **leader** does no reproduction work itself. Asked to reproduce a paper, it makes a run directory holding
`paper.pdf` and `scope.txt` (what to reproduce, compute budget, rules), runs the `paper-reproducer` skill with
`{"root": <run dir>, "compute": "auto", "experts": true}`, relays progress and finally summarizes the verdict table
from `report.md`. The skill's script owns the orchestration and hands each step to the member of the same name:

```
lead (claims + one spec per experiment; figure / figure-check read plot-only numbers)
  -> builder <-> reviewer rounds (pilot, then full runs)
  -> evaluator computes every claim  +  audit recomputes each one independently
  -> tuner rounds on claims that fell short (optional)
  -> rule verdicts -> judge -> reporter (report.md)
```

| Member | Job |
| --- | --- |
| leader | stages the paper into a run directory and launches the workflow |
| lead | reads the paper and scope, lists the claims and writes one runnable spec per experiment |
| figure, figure-check | read numbers that exist only in plots, twice, independently, from the rendered page |
| builder | implements and runs one experiment in `repo/arms/<experiment>/`, one JSON file per run |
| watch | waits on one submitted GPU job and reports its status and last output |
| reviewer | checks each builder round against the paper and lists what must change |
| evaluator | computes every claim from the raw run files with its own code in `repo/eval/` |
| audit | recomputes each claim independently; a disagreement with the evaluator caps the claim |
| tuner | tunes settings in `repo/tuned/` towards the paper's numbers, reporting every change |
| judge | sets the final verdict per claim next to the rule verdict, explaining every departure |
| reporter | writes `report.md` from the verdict table, caps and run files |

The build loop escalates to the lead instead of looping: a builder that cannot go on asks a question, a reviewer
that finds the experiment description wrong marks the issue `owner: "spec"`, and one lead turn answers both, from
the paper with a citation or as a decision recorded in the report.

Shared rules (`personas/_team.md`, appended to every persona): never state a number not read from a file, command
output or fetched page; follow `scope.txt`; never use the paper's own code or a third-party reproduction of it.

## Role rules

Each member except the leader carries `RunPolicyRail`, which checks every tool call against its `policy.json`:

| Role | Reads | Writes | Shell | GPU | Web |
| --- | --- | --- | --- | --- | --- |
| lead | paper.pdf, scope.txt | - | - | - | yes |
| figure, figure-check | paper.pdf, scope.txt | - | - | - | - |
| builder | paper, repo/ | repo/arms/&lt;experiment&gt;/ | rw | rw | yes |
| watch | - | - | - | wait | - |
| reviewer | paper, repo/ | - | ro | run | yes |
| evaluator | paper, repo/ | repo/eval/ | rw | - | - |
| audit, judge | paper, repo/ | - | ro | - | - |
| tuner | paper, repo/ | repo/tuned/ | rw | rw | yes |
| reporter | everything | report.md | - | - | - |

Paths are relative to the run directory. In a team built with `--bwrap`, shell commands run under bubblewrap with
only those paths mounted, a cleared environment, and the network only for roles with web access; otherwise they run
unsandboxed and the paths bind the file tools only. GPU levels: `ro` status/output/list, `wait`
also wait, `run` also submit/cancel, `rw` also download. Every member also has `view_page`, which renders a PDF page or a
cropped panel to a JPEG to open with `read_file`.

## Layout

| Path | What |
| --- | --- |
| `roles.json` | the role table: names, descriptions, policies |
| `personas/` | one persona per role, plus `_team.md` (team and shared rules) appended to each |
| `common/policy_rail.py` | `RunPolicyRail`: reads the role's `policy.json` and checks every tool call |
| `common/view_page.py` | the `view_page` tool |
| `skill/` | the `paper-reproducer` Swarm Skill; its `scripts/workflow.py` is `flows/reproducer_flow.py` |
| `build.py` | renders the package, checks it against the publish rules and zips it |

## Build, install, run

```sh
python team/build.py                # -> team/dist/paper-reproducer-team/ and team/dist/paper-reproducer-team.zip
python team/build.py --no-zip       # directory only
python team/build.py --bwrap        # members' shells run in a bubblewrap sandbox (host needs bwrap)
```

1. Import it in JiuwenSwarm: Web, Agent management, Expert Teams, import `team/dist/paper-reproducer-team.zip`
   (or the directory).
2. Optional GPU: install https://github.com/aaronma1/gpu_runner (`make install` and `gpu_runner daemon` on the GPU
   machine, `make install-mcp` on this one), then add it as a stdio MCP server in JiuwenSwarm (Connectors): command
   = the gpu_runner venv's `python`, argument = `gpu_runner_mcp.py`. Set its timeout (`timeout_s`) to 1900 s:
   JiuwenSwarm cuts a stdio MCP call off at 60 s by default, so a longer `gpu_wait` fails and the job watcher falls
   back to a model call every ~45 s. Without gpu_runner, runs use local CPU.
3. Turn on SwarmFlow (`/swarmflow on`, then `/new`), bind the team to the session, and ask the leader to reproduce
   a paper.

Optional skill args: `"smoke": true, "max_claims": 1` for a short check run, `build_rounds`, `max_tries`,
`max_questions`, `max_spec_fixes`, `tune`, `tune_rounds`, `tune_trials`, `gpu_hours`, and
`"models": {"strong": ..., "cheap": ...}` (workers on `strong`, checkers on `cheap`, so no agent checks its own
model's work). See `skill/SKILL.md`.

The host needs `uv` in `~/.local/bin` for Python environments.

### Shell sandbox: `--bwrap`

The shell sandbox is chosen when the team is built and written into every member's `rails/policy.json`
(`"sandbox": "bwrap"` or `"none"`). The build prints which one it used, and the package README says it in its first
line, so an imported team shows its mode.

| | `python team/build.py --bwrap` | `python team/build.py` (default) |
| --- | --- | --- |
| Host needs | `bwrap` (bubblewrap); without it every shell call is refused, with a message saying why | nothing extra |
| A shell command | runs in bubblewrap with only the role's paths mounted (read-only, its write directory read-write), a cleared environment, network only for roles with web access | runs on the host as written |
| Starts in | the run directory | the role's own directory: a builder's `repo/arms/<experiment>/`, the evaluator's `repo/eval/`, the tuner's `repo/tuned/`; the run directory for read-only shells |
| Role paths bind | file tools and shell | file tools only |
| Personas say | "your file, shell and GPU tools only reach the paths your role owns"; "your shell is read-only" | the shell is not confined; a "Your shell" section names the role's directory, what it may write and read, and asks it to stay there, keep away from JiuwenSwarm's own files (its config holds API keys) and use the network only if its role has web access |

Without `--bwrap` those instructions are all that keeps a shell to its role: use it only on a host where that is
acceptable. JiuwenSwarm itself does not sandbox shells unless its own jiuwenbox sandbox is turned on
(`/sandbox enable`), whose default policy mounts the whole filesystem read-only, makes the agent workspace writable,
and keeps the host network. With `--bwrap`, the mounts of each role live in a private file under
`~/.cache/repro-sandbox/` (`bwrap --args`), so a shell call in the model's history is the command plus about 150
characters, not 2.4k characters of mounts.

## Publish

Open the imported team's detail page and click Publish (sign in with GitHub or GitCode; it goes through Hub
moderation). The skill can also be published on its own from `dist/paper-reproducer-team/skills/paper-reproducer`.

## Not verified yet

- That expert members built for SwarmFlow workers load their rails and tools, and see the gpu_runner MCP server.
- That `model` and `agent_template` combine, so `models` still splits the tiers per role.
- That the group-shared skill is visible to the leader (if not, install the skill separately).
