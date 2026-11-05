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
