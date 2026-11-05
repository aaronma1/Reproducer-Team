"""Whole-flow runs of flows/reproducer_flow.py on the real engine with MockBackend, strict lint."""
import copy
import json
from pathlib import Path

from openjiuwen.agent_teams.workflow.engine import MockBackend, run_workflow
from openjiuwen.agent_teams.workflow.engine.backends.mock import SKIP

from framework.fixtures import load_fixtures

ROOT = Path(__file__).resolve().parent.parent
FLOW = str(ROOT / "flows" / "reproducer_flow.py")
ARGS = {"paper": "mock-paper", "build_rounds": 4}
BLOCK = {"issues": [{"issue": "arm m never updates its weights", "evidence": "train.py:12", "action": "call step()",
                     "blocking": True}]}
OK = {"issues": []}


class Counting(MockBackend):
    """MockBackend that fails on any label without a fixture and counts every backend call."""

    def __init__(self, fixtures):
        super().__init__(fixtures=fixtures)
        self.calls = []

    def _pick_raw(self, prompt, opts, schema_json, rng):
        label = str(opts.get("label"))
        if label.startswith("judge") and label not in self.fixtures:
            self.calls.append(label)  # by default the judge agrees with the rules
            self.judge_prompt = prompt
            rows = json.loads(prompt.split("Rule results (JSON):\n")[1].split("\nEnd of rule results.")[0])
            return {"verdicts": [{"claim_id": r["id"], "verdict": r["verdict"], "reason": "agrees"} for r in rows]}
        assert label in self.fixtures, f"no fixture for {label!r}"
        self.calls.append(label)
        self.prompts = getattr(self, "prompts", {})
        self.prompts[label] = prompt
        return super()._pick_raw(prompt, opts, schema_json, rng)


def _base():
    return copy.deepcopy(load_fixtures(ROOT / "fixtures" / "reproduced.json"))


async def _run(fx, args=None, journal=None):
    be = Counting(fx)
    kw = {"resume": str(journal) if journal and journal.exists() else None, "journal_path": str(journal)} if journal else {}
    r = await run_workflow(FLOW, args={**ARGS, **(args or {})}, backend=be, strict=True, **kw)
    return r, be


async def test_reproduced():
    r, be = await _run(_base())
    assert r["verdict"] == "reproduced"
    assert [x["verdict"] for x in r["claims"]] == ["reproduced", "reproduced"]
    assert be.calls[:6] == ["lead", "lead#2", "builder:main", "reviewer:main", "builder:main#2", "reviewer:main#2"]
    assert be.calls.index("evaluator") < be.calls.index("audit:c1") < be.calls.index("judge") < be.calls.index("reporter")
    assert "paper.pdf" in be.prompts["builder:main"] and '"c1"' in be.prompts["builder:main"]   # reads the paper, sees its claims
    exp = r["experiments"][0]
    assert {k: exp[k] for k in ("experiment", "ran", "reviewed", "rounds", "open")} == {
        "experiment": "main", "ran": True, "reviewed": True, "rounds": 2, "open": []}
    assert exp["choices"][0]["source"] == "guess"
    assert "Settings the paper does not state: main" in r["caps_md"]       # the report lists the builder's choices
    assert "far above what the run needs" in be.prompts["builder:main"]     # build rules reach the builder


async def test_reviewer_sends_the_pilot_back_until_it_learns():
    fx = _base()
    fx["reviewer:main"] = BLOCK
    fx["builder:main#2"] = fx["builder:main"]                    # fixed pilot
    fx["reviewer:main#2"] = OK
    fx["builder:main#3"] = _base()["builder:main#2"]              # full runs
    fx["reviewer:main#3"] = OK
    r, be = await _run(fx)
    assert r["verdict"] == "reproduced"
    assert "never updates its weights" in be.prompts["builder:main#2"]
    assert be.calls.count("reviewer:main#3") == 1 and "builder:main#4" not in be.calls


async def test_review_still_open_after_the_last_round_caps_partial():
    fx = _base()
    for n in ("", "#2", "#3"):
        fx[f"reviewer:main{n}"] = BLOCK
    fx["builder:main#3"] = _base()["builder:main#2"]              # it ran the full runs anyway
    r, _ = await _run(fx, {"build_rounds": 3})
    assert r["verdict"] == "partial"
    assert all("main: review still open" in x["caps"] for x in r["claims"])
    assert "never updates its weights" in r["caps_md"]


async def test_no_full_runs_stops_the_run():
    fx = _base()
    fx["reviewer:main"] = fx["reviewer:main#2"] = BLOCK
    fx["builder:main#2"] = fx["builder:main"]
    r, be = await _run(fx, {"build_rounds": 2})
    assert r["verdict"] == "could_not_test" and "no experiment finished" in r["reason"]
    assert "evaluator" not in be.calls


async def test_audit_disagreement_caps():
    fx = _base()
    fx["audit:c1"] = {**fx["audit:c1"], "value": 0.40}
    r, _ = await _run(fx)
    row = r["claims"][0]
    assert row["verdict"] == "partial" and any("independent recompute disagrees" in c for c in row["caps"])


async def test_unit_mismatch_is_sent_back_to_the_lead():
    fx = _base()
    good = fx["lead"]
    bad = copy.deepcopy(good)
    bad["claims"][1].update(metric="final normalized return", paper_a=200, paper_b=150)
    fx["lead"], fx["lead#2"], fx["lead#3"] = bad, good, fx["lead#2"]
    r, be = await _run(fx)
    assert r["verdict"] == "reproduced" and be.calls[:3] == ["lead", "lead#2", "lead#3"]
    assert "paper_a 200 cannot be a value of the metric" in be.prompts["lead#2"]


async def test_non_slug_arm_is_renamed_everywhere():
    fx = _base()
    for a in fx["lead"]["arms"]:
        a["name"] = {"m": "m-1.2", "b": "b"}[a["name"]]
    fx["lead"]["claims"][1]["better"] = "m-1.2"
    for a in fx["lead#2"]["experiments"][0]["arms"]:
        a["name"] = {"m": "m-1-2", "b": "b"}[a["name"]]
    r, be = await _run(fx)
    assert r["verdict"] == "reproduced" and be.calls[1] == "lead#2"


async def test_spec_that_misses_a_compared_arm_is_retried():
    fx = _base()
    bad = copy.deepcopy(fx["lead#2"])
    bad["experiments"][0]["arms"] = bad["experiments"][0]["arms"][:1]
    fx["lead#3"] = fx["lead#2"]
    fx["lead#2"] = bad
    r, be = await _run(fx)
    assert r["verdict"] == "reproduced"
    assert "compares 'b', which the experiment does not run" in be.prompts["lead#3"]


async def test_unaddressed_claim_is_deferred():
    fx = _base()
    fx["lead#2"]["experiments"][0]["claims"] = [{"claim_id": "c1", "metric": "acc"}]
    del fx["audit:c2"]
    r, _ = await _run(fx)
    assert [x["id"] for x in r["claims"]] == ["c1"]
    assert r["deferred"] == [{"claim_id": "c2", "reason": "not addressed by the lead"}]


def _short():
    fx = _base()
    fx["evaluator"]["results"][0]["value"] = 0.30
    fx["audit:c1"] = {**fx["audit:c1"], "value": 0.30}
    tuned = {"claim_id": "c1", "value": 0.35, "a": None, "b": None, "diff": None, "trials": 4, "note": "",
             "sources": ["repo/tuned/main/runs/m/seed0.json"], "changes": [{"name": "lr", "blind": "1e-3",
                                                                           "tuned": "3e-4", "stated_by_paper": True}]}
    fx["tuner"] = {"results": [tuned]}
    fx["tuner#2"] = {"results": [{**tuned, "value": 0.49, "trials": 7}]}
    fx["audit:c1:tuned"] = {**fx["audit:c1"], "value": 0.49}
    return fx


async def test_tuner_gets_rounds_until_it_matches_then_the_tuned_result_is_audited():
    r, be = await _run(_short(), {"tune": True, "tune_rounds": 4})
    row = r["claims"][0]
    assert row["rule_verdict"] == "reproduced" and row["tuned_match"] and row["tuned"]["rounds"] == 2
    assert "| reproduced (tuned) |" in r["table"] and "lr: 1e-3 -> 3e-4" in r["caps_md"]
    assert "tuner#3" not in be.calls and "measured 0.35, paper 0.5" in be.prompts["tuner#2"]
    assert be.calls.index("tuner#2") < be.calls.index("audit:c1:tuned") < be.calls.index("judge")


async def test_tuned_result_the_audit_disputes_does_not_count():
    fx = _short()
    fx["audit:c1:tuned"] = {**fx["audit:c1"], "value": 0.36}
    r, _ = await _run(fx, {"tune": True})
    row = r["claims"][0]
    assert not row["tuned_match"] and row["verdict"] == "partial"      # 0.30 vs 0.5: within 3x the 15% tolerance


async def test_no_tuning_unless_asked():
    r, be = await _run(_short())
    assert "tuner" not in be.calls and r["claims"][0]["verdict"] == "partial"


async def test_judge_agent_sets_the_final_verdict():
    fx = _base()
    fx["judge"] = {"verdicts": [{"claim_id": "c1", "verdict": "partial", "reason": "seed4.json is missing"},
                                {"claim_id": "c2", "verdict": "reproduced", "reason": "agrees"}]}
    r, _ = await _run(fx)
    row = r["claims"][0]
    assert r["verdict"] == "partial" and row["verdict"] == "partial" and row["rule_verdict"] == "reproduced"
    assert "seed4.json is missing" in r["caps_md"]


async def test_invalid_judge_answer_is_retried_then_rules_stand():
    fx = _base()
    fx["judge"] = {"verdicts": []}
    fx["judge#2"] = SKIP
    r, _ = await _run(fx, {"max_tries": 2})
    assert r["verdict"] == "reproduced" and "judge_reason" not in r["claims"][0]
    assert any("rule verdicts stand" in c["reason"] for c in r["caps"])


async def test_smoke_caps_partial():
    r, _ = await _run(_base(), {"smoke": True})
    assert r["verdict"] == "partial"


async def test_resume_makes_no_backend_calls(tmp_path):
    journal = tmp_path / "journal.jsonl"
    first, be1 = await _run(_base(), journal=journal)
    assert be1.calls
    second, be2 = await _run(_base(), journal=journal)
    assert be2.calls == [] and second["table"] == first["table"]


def _read(a, ua, b, ub):
    return {"claim_id": "c2", "page": 6, "notes": "", "readings": [
        {"field": "paper_a", "raw": a * 100, "value": a, "uncertainty": ua, "how": "curve m at the last tick"},
        {"field": "paper_b", "raw": b * 100, "value": b, "uncertainty": ub, "how": "curve b at the last tick"}]}


async def test_figure_numbers_are_read_twice_and_widen_the_tolerance():
    fx = _base()
    fx["lead"]["claims"][1].update(source="figure", figure="Fig. 2 (page 6): curves m and b at the end", paper_a=0.9)
    fx["figure:c2"] = _read(0.40, 0.02, 0.30, 0.02)
    fx["figure-check:c2"] = _read(0.60, 0.03, 0.30, 0.02)         # the two readings of a disagree
    r, be = await _run(fx)
    assert {"figure:c2", "figure-check:c2"} <= set(be.calls) and be.calls.index("figure:c2") < be.calls.index("lead#2")
    assert "0.9" not in be.prompts["figure:c2"]                     # the readers do not see the lead's number
    row = r["claims"][1]
    assert row["paper"].endswith("(tol 20%)")                       # paper_a 0.5 +- 0.1: 20% tolerance
    assert "0.5 ± 0.1" in r["caps_md"] and "readings disagree" in r["caps_md"]
    assert [x["field"] for x in r["figure_reads"]] == ["paper_a", "paper_b"]


async def test_binding_unstated_limit_blocks_without_the_reviewer():
    fx = _base()
    cap = {"name": "eval episode cap", "value": "2000 steps", "stated_by_paper": False, "bound_fraction": 0.4,
           "evidence": "runs/m/seed0.json"}
    fx["builder:main#2"] = {**fx["builder:main#2"], "limits": [cap]}
    fx["reviewer:main#2"] = OK                          # the reviewer misses it; the build check does not
    fixed = {**fx["builder:main#2"], "limits": [{**cap, "value": "200000 steps", "bound_fraction": 0.0}]}
    fx["builder:main#3"], fx["reviewer:main#3"] = fixed, OK
    r, be = await _run(fx)
    assert "builder:main#3" in be.calls and "cut 40.0%" in be.prompts["builder:main#3"]
    assert r["experiments"][0]["reviewed"] and r["experiments"][0]["rounds"] == 3


async def test_untested_result_change_left_open_caps_partial():
    fx = _base()
    change = {"change": "widen layers 64 -> 256", "reason": "result", "tested": False, "evidence": "nets.py"}
    for k in ("builder:main#2", "builder:main#3", "builder:main#4"):
        fx[k] = {**fx["builder:main#2"], "changes": [change]}
    for k in ("reviewer:main#3", "reviewer:main#4"):
        fx[k] = OK
    r, be = await _run(fx)
    assert not r["experiments"][0]["reviewed"]
    assert all(x["verdict"] == "partial" for x in r["claims"])


async def test_spec_only_stops_before_the_builders():
    r, be = await _run(_base(), {"spec_only": True})
    assert r["verdict"] == "could_not_test" and r["experiments"][0]["id"] == "main"
    assert not any(c.startswith(("builder", "reviewer", "evaluator", "judge", "reporter")) for c in be.calls)


async def test_auto_compute_falls_back_to_local_when_the_probe_finds_no_gpu():
    fx = _base()
    fx["gpu-probe"] = {"available": False, "detail": "no gpu_runner tools"}
    r, be = await _run(fx, {"compute": "auto"})
    assert be.calls[0] == "gpu-probe"
    assert "cpu-hours" in be.prompts["lead#2"]
    assert r["verdict"] == "reproduced"


async def test_auto_compute_uses_the_gpu_when_the_probe_answers():
    fx = _base()
    fx["gpu-probe"] = {"available": True, "detail": "gpu_list answered"}
    _, be = await _run(fx, {"compute": "auto"})
    assert "gpu-hours" in be.prompts["lead#2"]


async def test_explicit_compute_skips_the_probe():
    _, be = await _run(_base(), {"compute": "local"})
    assert "gpu-probe" not in be.calls


async def test_root_puts_every_role_in_one_file_tree():
    _, be = await _run(_base(), {"root": "/srv/run1/"})
    for label in ("reviewer:main", "audit:c1", "reporter"):
        assert be.prompts[label].startswith("All files of this run are under /srv/run1."), label


async def test_no_root_leaves_prompts_unchanged():
    _, be = await _run(_base())
    assert not any("All files of this run" in p for p in be.prompts.values())


async def test_string_args_are_parsed():
    be = Counting(_base())
    r = await run_workflow(FLOW, args=json.dumps(ARGS), backend=be, strict=True)
    assert r["verdict"] == "reproduced"


async def test_experts_hand_each_role_to_its_member():
    fx = _base()
    fx["gpu-probe"] = {"available": False, "detail": "none"}
    be = Counting(fx)
    seen = {}
    orig = be._pick_raw

    def pick(prompt, opts, schema_json, rng):
        seen[str(opts.get("label"))] = opts.get("agent_template")
        return orig(prompt, opts, schema_json, rng)
    be._pick_raw = pick
    await run_workflow(FLOW, args={**ARGS, "experts": True, "compute": "auto"}, backend=be, strict=True)
    assert seen["gpu-probe"] == "builder"
    assert seen["lead"] == "lead" and seen["builder:main"] == "builder" and seen["reviewer:main"] == "reviewer"
    assert seen["audit:c1"] == "audit" and seen["reporter"] == "reporter"
async def test_with_a_claim_limit_only_the_selected_claims_are_read_off_figures():
    fx = _base()
    fx["lead"]["claims"][1].update(source="figure", figure="Fig. 2 (page 6): curves m and b at the end", paper_a=0.9)
    fx["figure:c2"] = _read(0.40, 0.02, 0.30, 0.02)
    fx["figure-check:c2"] = _read(0.60, 0.03, 0.30, 0.02)
    _, be = await _run(fx, {"max_claims": 2})
    assert be.calls.index("lead#2") < be.calls.index("figure:c2")      # specs first, then the chosen figures
    fx2 = _base()
    fx2["lead"]["claims"][1].update(source="figure", figure="Fig. 2", paper_a=0.9)
    fx2["lead#2"]["experiments"][0]["claims"] = [c for c in fx2["lead#2"]["experiments"][0]["claims"]
                                                 if c["claim_id"] == "c1"]
    fx2["lead#2"]["deferred"] = [{"claim_id": "c2", "reason": "limit"}]
    _, be2 = await _run(fx2, {"max_claims": 1})
    assert not any(c.startswith("figure") for c in be2.calls)           # c2 was deferred: no figure reads
