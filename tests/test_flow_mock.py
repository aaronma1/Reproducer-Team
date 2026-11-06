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
    assert "Decisions the lead took" not in be.judge_prompt          # nothing was decided: no empty section
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


async def test_figure_numbers_are_read_twice_and_a_tie_break_decides():
    fx = _base()
    fx["lead"]["claims"][1].update(source="figure", figure="Fig. 2 (page 6): curves m and b at the end", paper_a=0.9)
    fx["figure:c2"] = _read(0.40, 0.02, 0.30, 0.02)
    fx["figure-check:c2"] = _read(0.60, 0.03, 0.30, 0.02)         # the two readings of a disagree
    fx["figure:c2#2"] = _read(0.99, 0.05, 0.30, 0.02)             # a third opinion, also an outlier
    r, be = await _run(fx)
    assert {"figure:c2", "figure-check:c2"} <= set(be.calls) and be.calls.index("figure:c2") < be.calls.index("lead#2")
    assert "0.9" not in be.prompts["figure:c2"]                     # the readers do not see the lead's number
    assert "figure:c2#2" in be.calls                                # the disagreement triggered a tie-break read
    row = r["claims"][1]
    assert row["paper"].endswith("(tol 20%)")                       # 0.99 dropped, paper_a 0.5 +- 0.1
    assert "0.5 ± 0.1" in r["caps_md"] and "readings disagree" in r["caps_md"]
    assert "out of the metric's range dropped (0.99)" in r["caps_md"]
    assert [x["field"] for x in r["figure_reads"]] == ["paper_a", "paper_b"]


async def test_a_reading_outside_the_metric_range_is_not_averaged_in():
    """d-mask-ar1, 2026-11-07: a metric that reads `1 - d_Jac` (=1) and a reader that reported the raw weight -0.39."""
    fx = _base()
    fx["lead"]["claims"][0].update(source="figure", figure="Fig. 1 (page 3), panel a", paper_value=1)
    one = {"claim_id": "c1", "page": 3, "notes": "", "readings": [
        {"field": "paper_value", "raw": 1, "value": 1.0, "uncertainty": 0.0, "how": "masks match at 1e-1"}]}
    wrong = {"claim_id": "c1", "page": 3, "notes": "the panel prints weights, not the metric", "readings": [
        {"field": "paper_value", "raw": -0.39, "value": -0.39, "uncertainty": 0.005, "how": "the printed cell"}]}
    fx["figure:c1"], fx["figure-check:c1"] = one, wrong
    r, be = await _run(fx)
    row = r["claims"][0]
    assert row["paper"] == "1"                                      # not the old 0.305 +- 0.7
    assert "readings disagree" not in r["caps_md"]                  # one usable reading is not a disagreement
    assert "out of the metric's range dropped (-0.39)" in r["caps_md"]
    assert "figure:c1#2" not in be.calls                            # nothing to tie-break: the bad read is dropped


async def test_a_binding_limit_goes_to_the_lead_with_its_evidence():
    """A limit the paper does not state that cuts 40% is not a builder defect: raising it is a budget question."""
    fx = _base()
    cap = {"name": "eval episode cap", "value": "2000 steps", "stated_by_paper": False, "bound_fraction": 0.4,
           "evidence": "runs/m/seed0.json"}
    fx["builder:main#2"] = {**fx["builder:main#2"], "limits": [cap]}
    fx["reviewer:main#2"] = OK                          # the reviewer misses it; the build check does not
    fixed = {**fx["builder:main#2"], "limits": [{**cap, "value": "200000 steps", "bound_fraction": 0.0}]}
    fx["builder:main#3"], fx["reviewer:main#3"] = fixed, OK
    fx["lead:conf:main"] = {"answers": [{"key": "limit:eval episode cap", "source": "reference",
                                         "citation": "scope.txt compute budget",
                                         "answer": "raise it to 200000 steps and re-run; the budget covers it"}]}
    r, be = await _run(fx)
    assert "cut 40.0%" in be.prompts["lead:conf:main"]                      # the evidence reaches the lead
    assert "raise it to 200000 steps" in be.prompts["builder:main#3"]       # the instruction reaches the builder
    assert be.calls.index("reviewer:main#2") < be.calls.index("lead:conf:main")
    assert r["experiments"][0]["reviewed"] and r["experiments"][0]["rounds"] == 2   # the escalated round was void


async def test_a_narrowing_the_lead_accepts_stops_blocking():
    fx = _base()
    cap = {"name": "steps budget", "value": "10000 steps", "stated_by_paper": False, "bound_fraction": 0.905,
           "evidence": "every curve ends at 10000 steps"}
    for n in ("", "#2"):
        fx[f"builder:main{n}"] = {**fx["builder:main"], "limits": [cap]}   # unchanged: the lead accepted it
    fx["builder:main#3"] = _base()["builder:main#2"]
    fx["reviewer:main#3"] = OK
    fx["lead:conf:main"] = {"answers": [{"key": "limit:steps budget", "source": "decision",
                                         "answer": "accept the narrowing: the budget cannot cover the full sweep"}]}
    r, be = await _run(fx)
    assert r["verdict"] == "reproduced"
    assert be.prompts["builder:main#2"].startswith("The lead answered")     # told once, not five times
    assert "cut 90.5%" not in be.prompts["builder:main#2"]                  # no longer a builder problem
    assert r["lead_decisions"][0]["source"] == "decision"
    assert "accept the narrowing" in r["caps_md"]                           # and it is in the report


async def test_a_declared_guard_may_bind_a_little():
    fx = _base()
    guard = {"name": "safety steps", "value": "2000000 steps", "stated_by_paper": False, "bound_fraction": 0.05,
             "evidence": "only the slow seed reaches it, far above the paper's run length", "guard": True}
    fx["builder:main"] = {**fx["builder:main"], "limits": [guard]}
    r, be = await _run(fx)
    assert r["verdict"] == "reproduced"
    assert not any(c.startswith("lead:conf") for c in be.calls)             # 5% is a guard, not a narrowing
    assert "cut 5.0%" not in be.prompts["builder:main#2"]


async def test_a_stopped_run_still_writes_a_report(tmp_path):
    fx = _base()
    for n in ("", "#2"):
        fx[f"reviewer:main{n}"] = BLOCK
    fx["builder:main#2"] = _base()["builder:main"]
    r, be = await _run(fx, {"build_rounds": 2, "root": str(tmp_path)})
    assert r["verdict"] == "could_not_test"
    assert "never updates its weights" in r["report"]        # the open finding survives the stop
    assert "builder:main#3" not in be.calls                   # nothing runs after it
    on_disk = (tmp_path / "report.md").read_text()
    assert on_disk.startswith("# Reproduction stopped") and "never updates its weights" in on_disk


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


# --- the build escalation channel: builder questions and spec-owned review issues (2026-11-06) ---

QUESTION = {"key": "d-width", "question": "Is the MLP width d1 = d2 = 26 or 2^6 = 64? The acceptance says 26.",
            "what_i_checked": "paper.pdf p.64 is a raster line whose text extraction gives '26'",
            "blocking": True, "impact_if_wrong": "the hidden width, W1's fan-in and every norm in the pilot"}
SPEC_ISSUE = {"issue": "The acceptance line says d1=d2=26; the paper says d1=d2=2^6=64",
              "evidence": "paper.pdf p.64: the 6 is a superscript, size 6.97 on a raised baseline",
              "action": "fix the spec to d1=d2=2^6=64", "blocking": True, "owner": "spec", "key": "d-width"}


async def test_builder_question_is_answered_without_costing_a_review_round():
    fx = _base()
    fx["builder:main"] = {**fx["builder:main"], "questions": [QUESTION]}
    fx["builder:main#2"] = _base()["builder:main"]            # pilot, now that it has the answer
    fx["builder:main#3"] = _base()["builder:main#2"]          # full runs
    fx["lead:conf:main"] = {"answers": [{"key": "d-width", "answer": "d1 = d2 = 2^6 = 64, not 26",
                                         "source": "paper", "citation": "App. H.3.1 p.64", "blocks": True}]}
    r, be = await _run(fx, {"build_rounds": 2})
    assert r["verdict"] == "reproduced"
    assert "lead:conf:main" in be.calls
    assert "d1 = d2 = 2^6 = 64, not 26" in be.prompts["builder:main#2"]      # the answer reaches the builder
    assert "Is the MLP width" in be.prompts["lead:conf:main"]                # the question reaches the lead
    exp = r["experiments"][0]
    assert (exp["ran"], exp["reviewed"], exp["rounds"]) == (True, True, 2)   # the asking turn cost no review
    assert [(d["key"], d["source"]) for d in r["lead_decisions"]] == [("d-width", "paper")]


async def test_spec_owned_issue_is_patched_by_the_lead_not_sent_back_to_the_builder():
    fx = _base()
    fx["reviewer:main"] = {"issues": [SPEC_ISSUE]}
    fx["builder:main#2"] = _base()["builder:main"]
    fx["builder:main#3"] = _base()["builder:main#2"]
    fx["reviewer:main#2"] = OK
    fx["reviewer:main#3"] = OK
    fx["lead:conf:main"] = {"answers": [{"key": "d-width", "answer": "the spec is wrong: d1=d2=2^6=64",
                                         "source": "paper", "citation": "App. H.3.1 p.64"}],
                            "spec_patch": {"acceptance": ["5 run files per arm, seeds 0..4", "d1=d2=2^6=64"]}}
    r, be = await _run(fx, {"build_rounds": 2})
    assert r["verdict"] == "reproduced"
    assert "d1=d2=2^6=64" in be.prompts["builder:main#2"]                   # the corrected spec reaches the builder
    assert "spec was patched" in be.prompts["builder:main#2"]
    assert "The reviewer found blocking problems" not in be.prompts["builder:main#2"]  # not a builder defect
    exp = r["experiments"][0]
    assert (exp["ran"], exp["reviewed"], exp["rounds"]) == (True, True, 2)   # 3 reviews, 2 rounds: the void one free
    assert be.calls.count("reviewer:main") == 1 and "reviewer:main#3" in be.calls
    assert "d1=d2=2^6=64" in be.prompts["evaluator"]        # the patched spec reaches the evaluator too


async def test_spec_amendments_are_capped_and_the_loop_still_ends():
    fx = _base()
    for n in ("", "#2", "#3", "#4"):
        fx[f"reviewer:main{n}"] = {"issues": [SPEC_ISSUE]}
    for n in ("#2", "#3"):
        fx[f"builder:main{n}"] = _base()["builder:main"]
    fx["builder:main#4"] = _base()["builder:main#2"]
    patch = {"answers": [{"key": "d-width", "answer": "patched", "source": "paper"}],
             "spec_patch": {"acceptance": ["d1=d2=2^6=64"]}}
    fx["lead:conf:main"], fx["lead:conf:main#2"] = patch, patch
    r, be = await _run(fx, {"build_rounds": 2, "max_spec_fixes": 2})
    assert sum(c.startswith("lead:conf:main") for c in be.calls) == 2         # the cap holds
    assert r["verdict"] == "partial"                                         # full runs done, review still open


async def test_a_measured_small_sensitivity_is_a_note_not_a_blocker():
    fx = _base()
    fx["reviewer:main"] = {"issues": [{
        "issue": "init scale is an unstated guess", "evidence": "train.py:44 init_scale=1.0",
        "action": "test it", "blocking": True, "key": "init-scale", "owner": "builder",
        "sensitivity_pct": 1.2, "sensitivity_note": "scanned 1.0-10.0: the claim moves 1.2%"}]}
    r, be = await _run(fx)
    assert r["verdict"] == "reproduced"
    assert "pilot passed review" in be.prompts["builder:main#2"]             # it went on to the full runs
    assert "init scale is an unstated guess" in be.prompts["builder:main#2"]  # the note is still reported
    assert not any(c.startswith("lead:conf") for c in be.calls)              # nothing was escalated


async def test_a_build_with_no_run_files_is_sent_back():
    fx = _base()
    fx["builder:main"] = {**fx["builder:main"], "run_files": []}
    fx["builder:main#2"] = _base()["builder:main"]
    fx["builder:main#3"] = _base()["builder:main#2"]
    fx["reviewer:main#3"] = OK
    r, be = await _run(fx)
    assert "no run files were reported" in be.prompts["builder:main#2"]   # the empty answer is not accepted
    assert "builder:main#2" in be.calls
    assert r["verdict"] == "reproduced"


async def test_a_lead_decision_stays_visible_in_the_report():
    fx = _base()
    fx["builder:main"] = {**fx["builder:main"], "questions": [QUESTION]}
    fx["builder:main#2"] = _base()["builder:main"]
    fx["builder:main#3"] = _base()["builder:main#2"]
    fx["lead:conf:main"] = {"answers": [{"key": "d-width", "answer": "use width 64; it does not move the claim",
                                         "source": "decision"}]}
    r, be = await _run(fx)
    _judge_prompt = be.judge_prompt
    assert r["lead_decisions"][0]["source"] == "decision"
    assert "Decisions taken during the build: main" in r["caps_md"]
    assert "use width 64; it does not move the claim" in r["caps_md"]
    assert "Decisions the lead took during the build" in _judge_prompt  # the judge sees the ruling, not just the report


async def test_the_shape_check_reads_the_run_files_when_it_can_see_them(tmp_path):
    fx = _base()
    fx["builder:main"] = {**fx["builder:main"], "run_files": ["repo/arms/main/runs/m/seed0.json"]}
    fx["builder:main#2"] = _base()["builder:main"]
    fx["builder:main#3"] = _base()["builder:main#2"]
    fx["reviewer:main#3"] = OK
    _, be = await _run(fx, {"root": str(tmp_path)})
    assert "reported run files are missing" in be.prompts["builder:main#2"]      # nothing on disk
    seed = tmp_path / "repo" / "arms" / "main" / "runs" / "m" / "seed0.json"
    seed.parent.mkdir(parents=True)
    seed.write_text('{"arm": "m", "config": {"lr": 0.001}, "metrics": {"acc": 0.5}}')
    _, be2 = await _run(_base_with_one_run_file(fx), {"root": str(tmp_path)})
    assert "reported run files are missing" not in be2.prompts["builder:main#2"]  # the file is there and parses
    fx3 = _base()
    fx3["builder:main"] = {**fx3["builder:main"], "run_files": ["~/data/.gpu_runner/j1/runs/seed0.json"]}
    fx3["builder:main#2"] = _base()["builder:main"]
    _, be3 = await _run(fx3, {"root": str(tmp_path)})
    assert "outside the run root" in be3.prompts["builder:main#2"]  # a job-side path is not counted as work


def _base_with_one_run_file(fx):
    out = dict(fx)
    out["builder:main"] = {**out["builder:main"], "run_files": ["repo/arms/main/runs/m/seed0.json"]}
    return out


# --- L1: the builder submits and ends its turn; the flow waits (2026-11-07) -------------------------------------

async def test_a_submitted_job_is_waited_on_by_the_flow_not_the_builder(tmp_path):
    fx = _base()
    fx["builder:main"] = {**fx["builder:main"], "stage": "submitted", "run_files": []}
    fx["watch:j1"] = {"job_uuid": "j1", "status": "COMPLETED", "exit_code": 0,
                      "tail": "wrote runs/m/seed0.json", "done": True}
    fx["builder:main#2"] = _base()["builder:main"]
    fx["builder:main#3"] = _base()["builder:main#2"]
    seed = tmp_path / "repo" / "arms" / "main" / "runs" / "m" / "seed0.json"
    seed.parent.mkdir(parents=True)
    seed.write_text('{"arm": "m", "config": {}, "metrics": {"acc": 0.5}}')  # what the job wrote
    r, be = await _run(fx, {"root": str(tmp_path)})
    assert "END YOUR TURN" in be.prompts["builder:main"]                    # told to hand the wait over
    assert be.calls.index("watch:j1") < be.calls.index("reviewer:main")     # a submission is not reviewed
    assert "wrote runs/m/seed0.json" in be.prompts["builder:main#2"]        # woken with the job's tail
    assert "j1" in (tmp_path / "jobs.json").read_text()                     # the handle is on disk for a resume
    assert r["verdict"] == "reproduced"
    assert r["experiments"][0]["rounds"] == 2                               # the wait cost no review round


async def test_a_job_that_never_finishes_is_given_back_to_the_builder():
    fx = _base()
    fx["builder:main"] = {**fx["builder:main"], "stage": "submitted", "run_files": []}
    for n in ("", "#2"):
        fx[f"watch:j1{n}"] = {"job_uuid": "j1", "status": "QUEUED", "exit_code": None, "tail": "", "done": False}
    fx["builder:main#2"] = _base()["builder:main"]
    _, be = await _run(fx, {"job_waits": 2})
    assert be.calls.count("watch:j1") == 1 and be.calls.count("watch:j1#2") == 1   # two waits, then it stops
    assert "is still QUEUED after 2 wait(s)" in be.prompts["builder:main#2"]
    assert "do not resubmit it" in be.prompts["builder:main#2"]
