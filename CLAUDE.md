# CLAUDE.md

Paper reproducer: one SwarmFlow script (`flows/reproducer_flow.py`) runs a team of agents that reproduce an ML
paper's claims. It runs two ways: headless on the openjiuwen engine (`framework/`), and inside JiuwenSwarm as the
**paper-reproducer-team** expert team (`team/`). README.md has the overview, DESIGN.md the constraints, NOTES.md the
dated lessons from live runs, HANDOFF.md the current state of work.

## Commands

```sh
uv run pytest                                   # everything; ~10 s, real-bwrap tests skip if bwrap is missing
uv run pytest tests/test_team.py                # team package + policy rail
uv run python team/build.py --bwrap             # rebuild team/dist/ (the zip you import into JiuwenSwarm)
uv run python team/build.py                     # same, shells unsandboxed
```

The installed team uses `--bwrap`. Rebuild with the same flag you mean to import, then import
`team/dist/paper-reproducer-team.zip` in JiuwenSwarm (Agent management, Expert Teams). A running session keeps the
version it started with.

## Where to edit

- `team/dist/` is generated: never edit it. Sources: `team/personas/*.md` (`_team.md` is appended to every role),
  `team/roles.json` (each role's read/write paths, shell, GPU, web), `team/common/policy_rail.py`,
  `team/skill/SKILL.md`. The skill's `scripts/workflow.py` is a copy of `flows/reproducer_flow.py`.
- Commit the rebuilt `team/dist/` together with its sources.
- The flow imports only `swarmflow` and the stdlib. The engine lints `time.*`, `random`, `uuid` and `.now()` (resume
  needs a pure script); pass values through args or agent results instead.
- Personas use `{{placeholders}}` filled per sandbox mode by `team/build.py`; the build fails on one left over.

## The policy rail (`team/common/policy_rail.py`)

Runs inside JiuwenSwarm on every tool call of a team member. It enforces `roles.json` on file, shell, GPU and web
tools and, with `--bwrap`, rewrites each shell command into `bwrap --args 3 bash -c '<cmd>' 3< <file>`.
Things that bit us (details in NOTES.md, 2026-11-09):

- JiuwenSwarm echoes the command that ran and writes rewritten args back into the history's ToolCall; the rail
  restores both after the call, or models copy the wrapper and it fails.
- JiuwenSwarm's own paths come from `jiuwen_dirs()` (its `get_agent_workspace_dir` / `get_user_workspace_dir`), not
  hard-coded. `JIUWENSWARM_HOME` is the user's home override, `JIUWENSWARM_DATA_DIR` the data dir.
- Never mount `~/.jiuwenswarm/config` (API keys) or `mcp/credentials`. Keep the pid namespace (without it a sandbox can
  kill host processes); do not add `--die-with-parent` back (it kills background jobs at the end of each call).
- Shared across sandboxes: `~/.cache/repro-uv-cache`, `~/.cache/repro-uv-python`; the flow's run setup uses the same.

## Looking at a live JiuwenSwarm run

- Session log: `~/.jiuwenswarm/agent/sessions/<session_id>/history.jsonl` (events, tool calls and results; members
  by `member_name`). Run files: `~/.jiuwenswarm/.agent_teams/jiuwen_team_<session_id>/team-workspace/...`.
- Token counts per call: `~/.jiuwenswarm/.trace/traces-<date>.jsonl` (OTel, ~2 GB/day; stream it, filter by
  session id first).
- `scripts/inspect/` has the scripts used so far (status, context per member, cross-run comparison, bash-call
  classification, wrapper copies, missing paths, DeepSeek reasoning probe). Each takes paths as arguments.
- Do not print anything from `~/.jiuwenswarm/config/config.yaml` that could be a key.
