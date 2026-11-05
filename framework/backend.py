"""ReproBackend: runs the flow's calls with real tool-using agents. It only executes calls; all
orchestration (retries, rework, verdicts) stays in the flow script."""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from openjiuwen.agent_teams.workflow.engine import AgentBackend, AgentResult
from openjiuwen.agent_teams.workflow.engine.progress import ProgressKind, WorkflowProgressEvent

from .agents import Agent, Client, Turn
from .config import Config
from .tools import Workspace

_KEY = re.compile(r"sk-[A-Za-z0-9_-]{16,}")


def _text(content) -> str:
    """The engine mirrors a structured reply as a dict; the chat API only takes text (and no empty turns)."""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return content or "(empty reply)"


@dataclass
class _Session:
    agent: Agent | None
    label: str
    human: bool = False


class ReproBackend(AgentBackend):
    # Engine-owned keys (label / phase / schema / model), plus `agent_template`, which names the JiuwenSwarm
    # expert for a role (`experts: true`); here the label already picks the role, so it is ignored.
    KNOWN_OPTIONS: frozenset[str] = frozenset({"agent_template"})

    def __init__(self, workspace: Workspace, cfg: Config, client: Client, secrets: Sequence[str] = (),
                 say: Callable[[str], None] = lambda m: None, usd_limit: float | None = None):
        super().__init__()
        self.usd_limit, self.cost = usd_limit, 0.0
        self.ws, self.cfg, self.client, self._say = workspace, cfg, client, say
        self._secrets = [s for s in secrets if s]
        self._sessions: dict[str, _Session] = {}

    # ---- engine hooks ----------------------------------------------------------
    def model_pool_names(self) -> list[str]:
        return sorted(self.cfg.tiers)

    def describe_model(self, opts: dict) -> str | None:
        return self._tier(opts)[1].id

    def _tier(self, opts: dict):
        name = opts.get("model") or self.cfg.default_tier
        return name, self.cfg.tiers[name]

    def _redact(self, text: str) -> str:
        for s in self._secrets:
            text = text.replace(s, "***")
        return _KEY.sub("***", text)

    def _activity(self, label: str, tool: str, content: str) -> None:
        content = self._redact(content)
        self._say(f"  [{label}] {tool}: {content[:110]!r}")
        if self.progress_sink is not None:
            self.progress_sink(WorkflowProgressEvent(
                kind=ProgressKind.AGENT_ACTIVITY, label=label, tool_name=tool, content=content,
                activity_type=("reasoning" if tool in ("thinking", "says") else
                               "tool_result" if tool.endswith("->") else "tool_call"), message=f"tool: {tool}"))

    def _bill(self, tokens: int, cost: float = 0.0) -> None:
        self.cost += cost
        # The backend is the only writer of the ledgers: add to both as each model call returns.
        self.budget.add(tokens)
        if self.workflow_budget is not None:
            self.workflow_budget.add(tokens)

    def _stop(self) -> bool:
        return (self.budget.exhausted or bool(self.workflow_budget and self.workflow_budget.exhausted)
                or (self.usd_limit is not None and self.cost >= self.usd_limit))

    def _agent(self, opts: dict, instructions: str | None) -> Agent:
        label = str(opts.get("label") or "agent")
        scope = self.ws.for_label(label)
        return Agent(label=label, role=scope.role, scope=scope, client=self.client, tier=self._tier(opts)[1],
                     instructions=instructions, timeout_s=self.cfg.tool_timeout_s, out_limit=self.cfg.tool_output_limit,
                     max_steps=self.cfg.role_max_steps.get(scope.role, self.cfg.max_steps), read_limit=self.cfg.read_limit,
                     turn_token_limit=self.cfg.turn_token_limit, on_usage=self._bill, should_stop=self._stop, on_activity=self._activity)

    @staticmethod
    def _result(t: Turn) -> AgentResult:
        if t.skipped:  # never raise for a model failure; the engine decides whether to retry
            return AgentResult(skipped=True, tokens=t.input_tokens + t.output_tokens,
                               input_tokens=t.input_tokens, output_tokens=t.output_tokens)
        return AgentResult(text=t.text, structured=t.structured, tokens=t.input_tokens + t.output_tokens,
                           input_tokens=t.input_tokens, output_tokens=t.output_tokens)

    # ---- single shot -----------------------------------------------------------
    async def run(self, prompt: str, opts: dict, schema_json: dict | None, *, call_key: str | None = None) -> AgentResult:
        return self._result(await self._agent(opts, None).turn(prompt, schema_json))

    # ---- sessions --------------------------------------------------------------
    async def ensure_member_name(self, *, kind: str, opts: dict) -> str:
        return "repro-" + re.sub(r"[^a-z0-9]+", "-", str(opts.get("label") or kind).lower()).strip("-")

    async def open_session(self, *, kind: str, instructions: str | None, opts: dict,
                           fork_data: dict | None = None, member_name: str | None = None) -> str:
        sid = member_name or await self.ensure_member_name(kind=kind, opts=opts)
        label = str(opts.get("label") or sid)
        self._sessions[sid] = _Session(None if kind == "human" else self._agent(opts, instructions), label, kind == "human")
        return sid

    async def send_turn(self, session_id: str, prompt: str, opts: dict, schema_json: dict | None, *,
                        history: Sequence[dict] = (), correlation_id: str | None = None) -> AgentResult:
        sess = self._sessions[session_id]
        if sess.human:
            answer = await asyncio.to_thread(input, f"\n[{sess.label}] {prompt}\n> ")
            return AgentResult(text=answer) if schema_json is None else AgentResult(structured={"answer": answer})
        if sess.agent.turns == 0 and history:  # resumed: rebuild the conversation once from the engine's mirror
            sess.agent.messages += [{"role": m["role"], "content": _text(m["content"])} for m in history]
        return self._result(await sess.agent.turn(prompt, schema_json))

    async def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def aclose(self) -> None:
        self._sessions.clear()
        await self.client.aclose()
