# Handoff: 2026-11-09

Work on the JiuwenSwarm expert team (sandbox, context size, builder overhead). Lessons and measurements are in
NOTES.md under 2026-11-09; this file is the state and what to do next.

## State

| What | Where | Committed | Live-tested |
| --- | --- | --- | --- |
| Leader told to use `skill_directory` from `skill_tool` | personas, SKILL.md | `0f5da6d` | yes: a stray look at the wrong path remains, but harmless |
| bwrap wrapper hidden from shell output | rail `after_tool_call` | `0f5da6d` | yes: results show plain commands |
| ...and from the history's ToolCall args | rail `_restore` | `d409f2d` | yes: 0 of 102 calls copied it (was 53 of 113) |
| JiuwenSwarm's advertised dirs mounted in the sandbox | rail `outside_mounts` | `d409f2d` | yes |
| Shared uv cache `~/.cache/repro-uv-cache` | rail | `d409f2d` | yes: 1.1 GB used |
| Paths from JiuwenSwarm (`jiuwen_dirs`) | rail | `690236f` | no |
| Persistent uv Pythons `~/.cache/repro-uv-python` | rail | `690236f` | no (smoke-tested on the host) |
| Background jobs survive a call (`--die-with-parent` dropped) | rail | `690236f` | no (real-bwrap test) |
| Run setup: shared `repo/.venv`, `paper.txt`, facts in the builder's prompt | flow `prepare_run`, `run_facts` | **no** | no (smoke-tested on the host: 37 s) |
| Shells start in the role's own directory | rail `--chdir` | **no** | no |
| read_file / no `cd` / keep work in the arm | `_team.md`, `builder.md`, flow `OUTPUT_RULES` | **no** | no |
| `scripts/inspect/`, CLAUDE.md, this file | | **no** | |

The uncommitted work passes `uv run pytest` (158 tests) and is built into `team/dist/` with `--bwrap`, but is not
imported into JiuwenSwarm yet. The last live run (`web_1a1c19ea3ba_b733bf15a4ec`) used the build from 13:42.

JiuwenSwarm config changed today: `enable_context_debug: true` in `~/.jiuwenswarm/config/config.yaml` (line 791). It
writes full prompts to `{workspace}/context/{session_id}_context/context_debug/`; turn it off after one run.

## Measured today

- Builder context at the same call count is about half of this morning's (`scripts/inspect/daycmp.py`): ~158k ->
  ~87k tokens at call 40; growth per call ~3.5k -> ~1.7k.
- DeepSeek v4.1 flash on OpenRouter and on Ollama both ignore reasoning sent back in history (prompt tokens unchanged
  with 20k characters of it; `scripts/inspect/thinkprobe.py`). openjiuwen sends it for every model named "deepseek".
  The builders' growth is tool traffic, not reasoning.
- Latest builders' bash calls: 105, of which 23 ran code (`scripts/inspect/bashkinds.py`). The uncommitted run
  setup targets the rest.

## Next

1. Import the new zip, start a run, and check in Build: bash calls per builder and their kinds (`bashkinds.py`),
   no `cd` prefixes, no `ls`/`which`/`pip` checks, builders using `repo/.venv`, no copied wrapper (`nested.py`),
   no missing paths (`missing.py`), context per call (`ctxpeak.py`).
2. Read one builder request from the context-debug dump to confirm what is sent; then set `enable_context_debug`
   back to false.
3. Both DeepSeek entries set `context_window: 1048576`, so JiuwenSwarm's compressor (80% of the window) never acts.
   Check the model's real window and set it.
4. `ENV_PACKAGES` in the flow (numpy, scipy, pandas, matplotlib, pypdf, plus CPU torch) is a guess; adjust to what
   builders actually install.
5. Open questions: a background job outlives a stopped run (the rail could record what it started and offer to kill
   it); builders' file tools cannot write to JiuwenSwarm's temporary working directory although their shell can;
   leftover `/tmp/smoke`, `/tmp/p1.log` from an unsandboxed run (harmless).
6. Possible upstream PR to JiuwenSwarm: its prompt's "Installed Skill root" line leaves out team skills
   (`runtime_prompt_rail.py:774`).
