"""Agent loop and ReproBackend against a scripted fake model (no network)."""
import json

import httpx
import pytest
from openjiuwen.agent_teams.workflow.engine import run_workflow
from openjiuwen.agent_teams.workflow.engine.budget import BudgetLedger

from framework.agents import Client
from framework.backend import ReproBackend
from framework.config import Config, Tier
from framework.tools import Workspace

SCHEMA = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}, "additionalProperties": False}


def call(name, **args):
    return {"id": f"c-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def reply(*calls, content="", prompt=10, completion=5):
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": list(calls)}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def fake_model(script, seen):
    it = iter(script)

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=next(it))
    return httpx.MockTransport(handler)


@pytest.fixture
def setup(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "paper.pdf").write_bytes(b"x")
    (paper / "scope.txt").write_text("scope text")
    ws = Workspace.create(tmp_path / "run", paper)
    cfg = Config(tiers={"cheap": Tier("m/cheap", "low"), "strong": Tier("m/strong")})

    def make(script):
        seen = []
        client = Client("sk-or-SECRET-SECRET-SECRET", "http://fake", transport=fake_model(script, seen))
        return ReproBackend(ws, cfg, client, secrets=["sk-or-SECRET-SECRET-SECRET"]), seen
    return make, ws


async def test_tool_then_bad_submit_then_good(setup):
    make, ws = setup
    be, seen = make([
        reply(call("read_file", path="scope.txt")),
        reply(call("submit", result='{"n": "not an int"}')),
        reply(call("submit", result='{"n": 3}')),
    ])
    r = await be.run("go", {"label": "lead", "model": "cheap"}, SCHEMA)
    assert r.structured == {"n": 3} and not r.skipped
    assert r.tokens == 45                         # three calls x (10 + 5)
    assert be.budget.spent == 45
    tool_msgs = [m for m in seen[-1]["messages"] if m["role"] == "tool"]
    assert "scope text" in tool_msgs[0]["content"]
    assert tool_msgs[1]["content"].startswith("invalid:")
    assert seen[0]["model"] == "m/cheap" and seen[0]["reasoning"] == {"effort": "low"}
    assert {t["function"]["name"] for t in seen[0]["tools"]} == {"read_file", "list_dir", "grep", "view_page", "web_search", "web_fetch",
                                                         "submit"}  # no write/shell
    assert "usage" in seen[0] and "max_tokens" not in seen[0]                   # cost reported, no output cap
    system = seen[0]["messages"][0]["content"]
    assert "lead" in system and "builder:<experiment>" in system and "Never state a number" in system


async def test_denied_path_is_a_tool_error_not_a_crash(setup):
    make, ws = setup
    be, seen = make([reply(call("read_file", path="paper.pdf")), reply(call("submit", result='{"n": 1}'))])
    r = await be.run("go", {"label": "mystery"}, SCHEMA)          # an unknown role may not read anything
    assert r.structured == {"n": 1}
    denied = [m for m in seen[-1]["messages"] if m["role"] == "tool"][0]["content"]
    assert denied.startswith("error:") and "may not read" in denied


async def test_gives_up_with_skipped_not_exception(setup):
    make, _ = setup
    from framework.agents import MAX_BAD
    bad = reply(call("submit", result="nope"))
    be, _ = make([bad] * MAX_BAD)
    r = await be.run("go", {"label": "lead"}, SCHEMA)
    assert r.skipped and r.structured is None


async def test_model_outage_is_skipped(setup, monkeypatch):
    make, _ = setup
    async def no_sleep(_): pass
    monkeypatch.setattr("framework.agents.asyncio.sleep", no_sleep)
    be, _ = make([])  # the fake raises StopIteration -> 500 from the transport handler path
    be.client._http = httpx.AsyncClient(base_url="http://fake", transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    r = await be.run("go", {"label": "lead"}, SCHEMA)
    assert r.skipped


async def test_key_never_reaches_events(setup):
    make, _ = setup
    be, _ = make([reply(call("read_file", path="scope.txt")), reply(call("submit", result='{"n": 1}'))])
    seen_events = []
    be.bind_progress_sink(seen_events.append)
    be._activity("lead", "shell", "echo sk-or-SECRET-SECRET-SECRET and sk-or-v1-abcdefghijklmnop1234")
    assert all("SECRET" not in (e.content or "") and "abcdefgh" not in (e.content or "") for e in seen_events)


async def test_session_keeps_context_and_bills_workflow_ledger(setup):
    make, _ = setup
    be, seen = make([reply(call("submit", result='{"n": 1}')), reply(call("submit", result='{"n": 2}'))])
    be.bind_workflow_budget(BudgetLedger(total=1000))
    sid = await be.open_session(kind="agent", instructions="be brief", opts={"label": "evaluator"})
    a = await be.send_turn(sid, "first", {"label": "evaluator"}, SCHEMA)
    b = await be.send_turn(sid, "second", {"label": "evaluator"}, SCHEMA)
    assert (a.structured, b.structured) == ({"n": 1}, {"n": 2})
    assert any("first" in str(m["content"]) for m in seen[1]["messages"])   # turn 2 still sees turn 1
    assert be.workflow_budget.spent == 30
    assert "be brief" in seen[0]["messages"][0]["content"]


async def test_resumed_session_sends_structured_history_as_text(setup):
    make, _ = setup
    be, seen = make([reply(call("submit", result='{"n": 2}'))])
    sid = await be.open_session(kind="agent", instructions=None, opts={"label": "evaluator"})
    history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": {"n": 1}},
               {"role": "user", "content": "again"}, {"role": "assistant", "content": ""}]
    r = await be.send_turn(sid, "second", {"label": "evaluator"}, SCHEMA, history=history)
    assert r.structured == {"n": 2}
    assert all(isinstance(m["content"], str) and m["content"] for m in seen[0]["messages"])
    assert '{"n": 1}' in seen[0]["messages"][2]["content"]


async def test_budget_ceiling_stops_the_loop(setup):
    make, _ = setup
    be, seen = make([reply(call("list_dir", path="repo"), prompt=500, completion=500)] * 5)
    be.bind_budget(BudgetLedger(total=100))
    r = await be.run("go", {"label": "evaluator"}, SCHEMA)
    assert r.skipped and len(seen) == 1                    # stopped after the call that blew the ceiling


async def test_unknown_model_hint_is_rejected_by_the_engine(setup, tmp_path):
    make, _ = setup
    be, _ = make([])
    flow = tmp_path / "f.py"
    flow.write_text('from swarmflow import agent\nMETA={"name":"x"}\nasync def run(args):\n    return await agent("hi", label="lead", options={"model": "gpt-9"})\n')
    with pytest.raises(Exception, match="gpt-9"):
        await run_workflow(str(flow), backend=be, strict=True)


async def test_whole_paper_is_readable_and_not_resent(setup):
    make, ws = setup
    (ws.root / "scope.txt").write_text("x" * 50_000)           # longer than the shell output limit
    be, seen = make([reply(call("read_file", path="scope.txt")), reply(call("read_file", path="scope.txt")),
                     reply(call("submit", result='{"n": 1}'))])
    r = await be.run("go", {"label": "lead"}, SCHEMA)
    tools = [m["content"] for m in seen[-1]["messages"] if m["role"] == "tool"]
    assert len(tools[0]) == 50_000                              # not cut at the 20k shell limit
    assert tools[1].startswith("You already read scope.txt") and r.structured == {"n": 1}


async def test_turn_token_cap(setup):
    make, _ = setup
    be, seen = make([reply(call("list_dir", path="repo"), prompt=400, completion=100)] * 10)
    be.cfg.turn_token_limit = 1000
    r = await be.run("go", {"label": "evaluator"}, SCHEMA)
    assert r.skipped and len(seen) == 2


async def test_old_context_is_compacted_but_paper_reads_kept(setup):
    make, ws = setup
    (ws.root / "scope.txt").write_text("S" * 5000)
    big = "x" * 5000
    script = [reply(call("read_file", path="scope.txt"))]
    script += [reply({"id": f"w{i}", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "repo"}'}})
               for i in range(12)]
    script += [reply({"id": "w-big", "type": "function", "function": {"name": "list_dir",
                                                                        "arguments": json.dumps({"path": "repo", "content": big})}})]
    script += [reply({"id": f"v{i}", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "repo"}'}})
               for i in range(10)]
    script += [reply(call("submit", result='{"n": 1}'))]
    be, seen = make(script)
    be.cfg.tiers["cheap"] = Tier("m/cheap", "low", context=10)   # every call is "near the window"
    r = await be.run("go", {"label": "reporter"}, SCHEMA)
    assert r.structured == {"n": 1}
    last = seen[-1]["messages"]
    tool_msgs = [m for m in last if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "S" * 5000                 # the paper-file read is never shrunk
    old_call = next(m for m in last if m["role"] == "assistant" and m.get("tool_calls")
                    and m["tool_calls"][0]["id"] == "w-big")
    assert "elided" in old_call["tool_calls"][0]["function"]["arguments"]   # old file content dropped
    assert len(json.dumps(last)) < 30_000


async def test_give_up_reason_is_reported(setup):
    make, _ = setup
    from framework.agents import MAX_BAD
    bad = reply(call("submit", result="nope"))
    be, _ = make([bad] * MAX_BAD)
    events = []
    be.bind_progress_sink(events.append)
    r = await be.run("go", {"label": "lead"}, SCHEMA)
    assert r.skipped
    gave_up = [e for e in events if e.tool_name == "gave up"]
    assert gave_up and f"{MAX_BAD} invalid submits" in gave_up[0].content


async def test_no_compaction_far_from_the_window_keeps_the_cache_prefix(setup):
    make, _ = setup
    big = json.dumps({"path": "repo", "content": "x" * 5000})
    script = [reply({"id": "w-big", "type": "function", "function": {"name": "list_dir", "arguments": big}})]
    script += [reply({"id": f"v{i}", "type": "function", "function": {"name": "list_dir", "arguments": '{"path": "repo"}'}})
               for i in range(12)]
    script += [reply(call("submit", result='{"n": 1}'))]
    be, seen = make(script)
    await be.run("go", {"label": "reporter"}, SCHEMA)
    first, last = seen[1]["messages"], seen[-1]["messages"]
    assert last[: len(first)] == first                      # append-only: earlier requests are a prefix


async def test_usd_ceiling_stops_the_run(setup):
    make, _ = setup
    r_ = reply(call("list_dir", path="repo"))
    r_["usage"]["cost"] = 0.6
    be, seen = make([r_] * 5)
    be.usd_limit = 1.0
    r = await be.run("go", {"label": "evaluator"}, SCHEMA)
    assert r.skipped and len(seen) == 2 and abs(be.cost - 1.2) < 1e-9


async def test_told_to_wrap_up_before_the_step_limit(setup):
    make, ws = setup
    from framework.agents import WRAP_UP_STEPS
    be, seen = make([reply(call("list_dir", path="."))] * WRAP_UP_STEPS + [reply(call("submit", result='{"n": 2}'))])
    be.cfg.max_steps = WRAP_UP_STEPS + 1
    r = await be.run("go", {"label": "lead"}, SCHEMA)
    assert r.structured == {"n": 2}
    warned = [m for m in seen[1]["messages"] if m["role"] == "user" and "model calls left" in m["content"]]
    assert warned and seen[0]["messages"][-1]["content"].startswith("go")


def test_prompts_never_raise_the_idea_of_faking_results():
    """Agents read these prompts: naming the misconduct could suggest it."""
    from pathlib import Path
    import re
    root = Path(__file__).resolve().parent.parent
    for f in ("flows/reproducer_flow.py", "framework/agents.py"):
        text = re.sub(r"\s+", " ", (root / f).read_text().lower())
        for word in ("fabricat", "dishonest", "cheat", "gamed", "hard-coded", "fake arm", "adversarial"):
            assert word not in text, f"{word!r} in {f}"


def test_extra_fields_do_not_lose_an_answer():
    from jsonschema import Draft202012Validator
    from framework.agents import _parse
    schema = {"type": "object", "additionalProperties": False, "required": ["issues"], "properties": {
        "issues": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["issue"],
                                              "properties": {"issue": {"type": "string"}}}}}}
    v = Draft202012Validator(schema)
    assert _parse('{"issues": [{"issue": "x", "extra": 1}], "not_checked": "y"}', v) == (True, {"issues": [{"issue": "x"}]})
    assert _parse('{"not_checked": "y"}', v)[0] is False                       # a required field is still required


async def test_submit_takes_an_object_and_rejected_submits_are_kept(setup):
    make, ws = setup
    be, seen = make([
        reply(call("submit", result='{"n": 1,, }')),               # a broken hand-escaped string
        reply(call("submit", result={"n": 4})),                    # the object itself
    ])
    r = await be.run("go", {"label": "spec#2"}, SCHEMA)
    assert r.structured == {"n": 4}
    submit = next(t for t in seen[0]["tools"] if t["function"]["name"] == "submit")
    assert submit["function"]["parameters"]["properties"]["result"]["type"] == "object"
    assert "JSON object (not a string)" in seen[0]["messages"][-1]["content"]
    saved = list((ws.root.parent / "rejected").glob("*.txt"))
    assert [p.name for p in saved] == ["spec#2.1.txt"]
    assert "not valid JSON" in saved[0].read_text() and '{"n": 1,, }' in saved[0].read_text()


async def test_text_answer_keeps_a_string_submit(setup):
    make, _ = setup
    be, seen = make([reply(call("submit", result="all done"))])
    r = await be.run("go", {"label": "reporter"}, None)
    assert r.text == "all done"
    submit = next(t for t in seen[0]["tools"] if t["function"]["name"] == "submit")
    assert submit["function"]["parameters"]["properties"]["result"]["type"] == "string"


def test_schema_rejection_lists_every_missing_field_and_allows_raw_newlines():
    from jsonschema import Draft202012Validator
    from framework.agents import _parse
    v = Draft202012Validator({"type": "object", "required": ["a", "b"],
                              "properties": {"a": {"type": "string"}, "b": {"type": "string"}}})
    ok, why = _parse("{}", v)
    assert not ok and "'a' is a required property" in why and "'b' is a required property" in why
    assert _parse('{"a": "line1\nline2", "b": "x"}', v) == (True, {"a": "line1\nline2", "b": "x"})


def test_a_dropped_closing_brace_between_items_is_repaired_only_if_valid():
    from jsonschema import Draft202012Validator
    from framework.agents import _parse
    schema = {"type": "object", "required": ["selected"], "properties": {"selected": {"type": "array", "items": {
        "type": "object", "required": ["claim_id", "spec"], "additionalProperties": False,
        "properties": {"claim_id": {"type": "string"}, "spec": {"type": "object"}}}}}}
    v = Draft202012Validator(schema)
    broken = '{"selected": [{"claim_id": "a", "spec": {"x": 1}, {"claim_id": "b", "spec": {"x": 2}}]}'
    ok, val = _parse(broken, v)
    assert ok and [s["claim_id"] for s in val["selected"]] == ["a", "b"]
    assert _parse('{"selected": [{"claim_id": "a", "spec": {"x": 1}, {"oops": 1}]}', v)[0] is False


def test_trailing_comma_is_repaired():
    from jsonschema import Draft202012Validator
    from framework.agents import _parse
    v = Draft202012Validator({"type": "object", "properties": {"a": {"type": "array"}}})
    assert _parse('{"a": [1, 2,], }', v) == (True, {"a": [1, 2]})


def _pdf(path):
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 100)
    doc.save(str(path))
    doc.close()


def test_render_page_and_crop(setup):
    _, ws = setup
    _pdf(ws.root / "paper.pdf")
    lead = ws.for_label("lead")
    png = lead.render_page("paper.pdf", 1)
    assert png[:3] == b"\xff\xd8\xff"                                          # JPEG
    assert lead.render_page("paper.pdf", 1, "0,0,0.5,0.5")[:3] == b"\xff\xd8\xff"
    with pytest.raises(ValueError):
        lead.render_page("paper.pdf", 2)
    with pytest.raises(Exception):
        ws.for_label("mystery").render_page("paper.pdf", 1)          # an unknown role may not see the paper


async def test_view_page_sends_the_image_after_the_tool_results(setup):
    make, ws = setup
    _pdf(ws.root / "paper.pdf")
    be, seen = make([reply(call("view_page", page=1, crop="")), reply(call("submit", result={"n": 1}))])
    r = await be.run("go", {"label": "figure:c1"}, SCHEMA)
    assert r.structured == {"n": 1}
    msgs = seen[1]["messages"]
    assert msgs[-2]["role"] == "tool" and "image follows" in msgs[-2]["content"]
    img = msgs[-1]["content"]
    assert msgs[-1]["role"] == "user" and img[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_concurrent_renders_do_not_crash(setup):
    import concurrent.futures as cf
    _, ws = setup
    _pdf(ws.root / "paper.pdf")
    lead = ws.for_label("lead")
    with cf.ThreadPoolExecutor(16) as ex:
        outs = list(ex.map(lambda i: lead.render_page("paper.pdf", 1, "" if i % 2 else "0,0,0.5,0.5"), range(64)))
    assert all(o[:3] == b"\xff\xd8\xff" for o in outs)


async def test_reasoning_and_text_are_logged(setup):
    make, ws = setup
    first = reply(call("read_file", path="scope.txt"), content="Reading the scope first.")
    first["choices"][0]["message"]["reasoning"] = "The scope names Breakout; check it."
    second = reply(call("submit", result='{"n": 2}'))
    second["choices"][0]["message"]["reasoning_details"] = [{"type": "reasoning.text", "text": "Done, n is 2."}]
    be, seen = make([first, second])
    events = []
    be.bind_progress_sink(events.append)
    r = await be.run("go", {"label": "lead"}, SCHEMA)
    assert r.structured == {"n": 2}
    got = [(e.tool_name, e.content, e.activity_type) for e in events if e.activity_type == "reasoning"]
    assert got == [("thinking", "The scope names Breakout; check it.", "reasoning"),
                   ("says", "Reading the scope first.", "reasoning"),
                   ("thinking", "Done, n is 2.", "reasoning")]


async def test_web_tools_dispatch_and_are_role_limited(setup, monkeypatch):
    import framework.agents as A

    class Stub:
        def __init__(self, tag):
            self.tag, self.seen = tag, []

        async def invoke(self, inputs):
            self.seen.append(inputs)
            return f"{self.tag}: {inputs}"
    stubs = {"web_search": Stub("S"), "web_fetch": Stub("F")}
    monkeypatch.setattr(A, "_web_tools", stubs)
    make, ws = setup
    be, seen = make([reply(call("web_search", query="pgx breakout sticky"), call("web_fetch", url="https://x/y.py")),
                     reply(call("submit", result='{"n": 1}'))])
    r = await be.run("go", {"label": "builder:main"}, SCHEMA)
    assert r.structured == {"n": 1}
    outs = [m["content"] for m in seen[-1]["messages"] if m["role"] == "tool"]
    assert outs[0].startswith("S: ") and "pgx breakout sticky" in outs[0] and outs[1].startswith("F: ")
    be2, seen2 = make([reply(call("submit", result='{"n": 1}'))])
    await be2.run("go", {"label": "audit:c1"}, SCHEMA)
    assert not {"web_search", "web_fetch"} & {t["function"]["name"] for t in seen2[0]["tools"]}
