"""Pure functions of flows/reproducer_flow.py: verdict rules, caps, checks and repairs."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

FLOW = Path(__file__).resolve().parent.parent / "flows" / "reproducer_flow.py"


@pytest.fixture(scope="module")
def f():
    stub = types.ModuleType("swarmflow")
    for name in ("agent", "agent_session", "log", "parallel", "phase"):
        setattr(stub, name, lambda *a, **k: None)
    saved = sys.modules.get("swarmflow")
    sys.modules["swarmflow"] = stub
    try:
        spec = importlib.util.spec_from_file_location("reproducer_flow_under_test", FLOW)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if saved is not None:
            sys.modules["swarmflow"] = saved
        else:
            del sys.modules["swarmflow"]


def test_value_tolerance(f):
    assert f.judge_value(0.44, 0.5) == "reproduced"         # 12% off, inside 15%
    assert f.judge_value(0.30, 0.5) == "partial"            # 40% off, inside 3 x 15%
    assert f.judge_value(0.20, 0.5) == "not_reproduced"
    assert f.judge_value(None, 0.5) == "could_not_test"
    c = {"kind": "value", "paper_value": 0.5, "rel_tol": 0.05}
    assert f.judge_item(c, {"value": 0.44}) == "reproduced"  # a claim's tighter rel_tol never tightens the floor
    assert f.judge_item({**c, "rel_tol": 0.5}, {"value": 0.3}) == "reproduced"


def test_ordering(f):
    a, b = {"mean": 0.6, "lo": 0.55, "hi": 0.65}, {"mean": 0.4, "lo": 0.35, "hi": 0.45}
    assert f.judge_ordering(a, b) == "reproduced"
    assert f.judge_ordering(b, a) == "not_reproduced"
    assert f.judge_ordering(a, b, higher_is_better=False) == "not_reproduced"
    overlap = {"mean": 0.58, "lo": 0.3, "hi": 0.7}
    assert f.judge_ordering(overlap, b) == "partial"
    assert f.judge_ordering(overlap, b, diff={"mean": 0.1, "lo": 0.02, "hi": 0.2}) == "reproduced"
    assert f.judge_ordering(None, b) == "could_not_test"


def test_predicate(f):
    assert f.judge_predicate(0.1, ">=", 0) == "reproduced"
    assert f.judge_predicate(-1, ">=", 0) == "not_reproduced"
    assert f.judge_predicate(0.5, "~=", 0.5) == "reproduced"


def test_tuned_ordering_needs_the_methods_own_number(f):
    c = {"id": "x", "kind": "ordering", "better": "m", "worse": "b", "paper_a": 0.72, "paper_b": 0.6}
    weak = {"a": {"mean": 0.30, "lo": 0.28, "hi": 0.32}, "b": {"mean": 0.10, "lo": 0.08, "hi": 0.12}}
    match = {"a": {"mean": 0.71, "lo": 0.69, "hi": 0.73}, "b": {"mean": 0.55, "lo": 0.53, "hi": 0.57}}
    assert f.judge_item(c, weak) == "reproduced" and f.judge_tuned(c, weak) == "not_reproduced"
    assert f.judge_tuned(c, match) == "reproduced"


def test_caps_and_aggregate(f):
    assert f.apply_caps("reproduced", [{"ceiling": "partial"}]) == "partial"
    assert f.apply_caps("not_reproduced", [{"ceiling": "partial"}]) == "not_reproduced"
    assert f.aggregate(["reproduced", "reproduced"]) == "reproduced"
    assert f.aggregate(["reproduced", "not_reproduced"]) == "partial"
    assert f.aggregate(["not_reproduced", "could_not_test"]) == "not_reproduced"
    assert f.aggregate([]) == "could_not_test"


def test_audit_agreement_and_paired_diff(f):
    v = {"kind": "value"}
    assert f.audit_agrees(v, {"value": 0.5}, {"value": 0.505})[0]
    ok, why = f.audit_agrees(v, {"value": 0.5}, {"value": 0.4})
    assert not ok and "disagrees on value" in why
    assert f.audit_agrees(v, {"value": 0.5}, None) == (False, "no independent recompute")
    o = {"kind": "ordering"}
    assert f.audit_agrees(o, {"a": {"mean": 1}, "b": {"mean": 2}}, {"a": {"mean": 1}, "b": {"mean": 2}})[0]
    d = {"mean": 0.1, "lo": 0.0, "hi": 0.2}
    assert f.checked_diff({"diff": d}, {"diff": d}) == d
    assert f.checked_diff({"diff": d}, {"diff": {**d, "mean": 0.3}}) is None


def test_unit_problems(f):
    c = {"metric": "final normalized return", "paper_a": 200, "paper_b": 0.5}
    probs = f.unit_problems(c)
    assert len(probs) == 1 and "paper_a 200" in probs[0]
    assert f.unit_problems({"metric": "raw return", "paper_value": 200}) == []


def test_claim_problems(f):
    names = {"m", "b"}
    assert f.claim_problems({"id": "ok", "kind": "value", "paper_value": 1, "metric": "x"}, names) == []
    assert f.claim_problems({"id": "Bad Id", "kind": "value", "paper_value": 1}, names)
    assert f.claim_problems({"id": "v", "kind": "value", "paper_value": None}, names)
    assert any("'z' is not in the arm list" in p
               for p in f.claim_problems({"id": "o", "kind": "ordering", "better": "m", "worse": "z"}, names))


def test_arm_rename(f):
    read = {"arms": [{"name": "reppo-1.2"}, {"name": "PPO"}, {"name": "ppo"}],
            "claims": [{"id": "c", "kind": "ordering", "better": "reppo-1.2", "worse": "PPO"}]}
    fixed, renamed = f.repair_arms(read)
    assert [a["name"] for a in fixed["arms"]] == ["reppo-1-2", "ppo"]
    assert (fixed["claims"][0]["better"], fixed["claims"][0]["worse"]) == ("reppo-1-2", "ppo")
    assert renamed == {"reppo-1.2": "reppo-1-2", "PPO": "ppo"}


def test_selection_repair(f):
    ans = {"experiments": [{"id": "e", "claims": [{"claim_id": "a"}, {"claim_id": "zz"}]},
                           {"id": "e2", "claims": [{"claim_id": "a"}]}], "deferred": []}
    fixed = f.repair_selection(ans, ["a", "b"])
    assert [e["id"] for e in fixed["experiments"]] == ["e"] and fixed["experiments"][0]["claims"] == [{"claim_id": "a"}]
    assert fixed["deferred"] == [{"claim_id": "b", "reason": "not addressed by the lead"}]


def test_experiment_problems(f):
    by_id = {"c": {"id": "c", "kind": "ordering", "better": "m", "worse": "b"}}
    exp = {"id": "e", "arms": [{"name": "m"}, {"name": "b"}], "procedure": "x" * 100, "acceptance": ["5 runs"],
           "metrics": [{"name": "acc"}], "claims": [{"claim_id": "c", "metric": "acc"}]}
    assert f.experiment_problems(exp, by_id, {"m", "b"}) == []
    assert any("does not define" in p
               for p in f.experiment_problems({**exp, "claims": [{"claim_id": "c", "metric": "loss"}]}, by_id, {"m", "b"}))
    assert any("does not run" in p for p in f.experiment_problems({**exp, "arms": [{"name": "m"}]}, by_id, {"m", "b"}))
    assert any("unknown arm" in p for p in f.experiment_problems(exp, by_id, {"m"}))


def test_blocking_and_tables(f):
    assert f.blocking(None)[0]["blocking"]
    assert f.blocking({"issues": [{"issue": "x", "blocking": False}]}) == []
    rows = [{"id": "c", "paper": "0.5", "measured": "0.49", "verdict": "reproduced", "rule_verdict": "reproduced",
             "caps": [], "scale": "-", "tuned_match": True,
             "tuned": {"measured": "0.49", "verdict": "reproduced", "audited": True, "why": "", "rounds": 2,
                       "trials": 3, "changes": [{"name": "lr", "blind": "1", "tuned": "2", "stated_by_paper": False}]}}]
    assert "| reproduced (tuned) |" in f.build_table(rows)
    assert "after 2 rounds, 3 trials" in f.tuning_section(rows)


def test_combine_figure_reads(f):
    c = {"id": "c", "kind": "ordering", "metric": "normalized return", "paper_a": 0.9, "paper_b": None}
    r1 = {"readings": [{"field": "paper_a", "value": 0.60, "uncertainty": 0.02}]}
    r2 = {"readings": [{"field": "paper_a", "value": 0.62, "uncertainty": 0.03}]}
    out, recs = f.combine_reads(c, [r1, r2])
    assert abs(out["paper_a"] - 0.61) < 1e-9 and recs[0]["agree"] and abs(out["rel_tol"] - 0.03 / 0.61) < 1e-9
    assert recs[1]["field"] == "paper_b" and recs[1]["uncertainty"] is None and out["paper_b"] is None   # unread
    far = {"readings": [{"field": "paper_a", "value": 0.80, "uncertainty": 0.02}]}
    out, recs = f.combine_reads(c, [r1, far])
    assert abs(out["paper_a"] - 0.70) < 1e-9 and not recs[0]["agree"] and abs(out["rel_tol"] - 0.1 / 0.7) < 1e-9
    raw = {"readings": [{"field": "paper_a", "value": 150.0, "uncertainty": 5.0}]}
    out, recs = f.combine_reads(c, [raw, None])                       # a raw score on a normalized metric
    assert out["paper_a"] == 0.9 and "units" in recs[0]["note"]


def test_which_claims_need_a_figure_read(f):
    assert f.needs_figure_read({"kind": "value", "source": "figure"})
    assert not f.needs_figure_read({"kind": "value", "source": "table", "anchors": ["Fig. 3"]})
    assert f.needs_figure_read({"kind": "ordering", "anchors": ["Fig. 3"]})         # no source given: the anchor decides
    assert not f.needs_figure_read({"kind": "predicate", "source": "figure"})


def test_reading_uncertainty_in_raw_units_is_converted_and_capped(f):
    assert abs(f._unc({"value": 0.66, "uncertainty": 10, "raw": 166}) - 10 * 0.66 / 166) < 1e-9
    assert f._unc({"value": 0.66, "uncertainty": 0.05, "raw": 166}) == 0.05
    assert f._unc({"value": 0.66, "uncertainty": 5, "raw": None}) == 0.5 * 0.66          # capped at 50%
    c = {"id": "c", "kind": "value", "metric": "normalized return", "paper_value": 0.6}
    out, _ = f.combine_reads(c, [{"readings": [{"field": "paper_value", "value": 0.66, "uncertainty": 10, "raw": 166}]}])
    assert out["rel_tol"] < 0.1


def test_claim_ids_are_renamed_too(f):
    fixed, renamed = f.repair_arms({"arms": [{"name": "gd"}], "claims": [{"id": "ratio-lr-0.02", "kind": "value"}]})
    assert fixed["claims"][0]["id"] == "ratio-lr-0-02" and renamed == {"ratio-lr-0.02": "ratio-lr-0-02"}


def test_build_problems(f):
    lim = {"name": "cap", "value": "2000", "evidence": "e"}
    assert f.build_problems({"limits": [{**lim, "stated_by_paper": False, "bound_fraction": 0.005}]}) == []
    assert f.build_problems({"limits": [{**lim, "stated_by_paper": True, "bound_fraction": 0.9}]}) == []
    out = f.build_problems({"limits": [{**lim, "stated_by_paper": False, "bound_fraction": 0.4},
                                       {**lim, "stated_by_paper": False, "bound_fraction": None}]})
    assert len(out) == 2 and all(i["blocking"] for i in out)
    assert "40.0%" in out[0]["issue"] and "not measured" in out[1]["issue"]
    ch = {"change": "wider", "evidence": "e"}
    assert f.build_problems({"changes": [{**ch, "reason": "result", "tested": True},
                                         {**ch, "reason": "bug", "tested": False}]}) == []
    assert len(f.build_problems({"changes": [{**ch, "reason": "result", "tested": False}]})) == 1
    assert f.build_problems({}) == []


def test_build_record_section(f):
    assert f.build_record_section([{"experiment": "e"}]) == ""
    md = f.build_record_section([{"experiment": "e", "choices": [{"choice": "init", "value": "he", "source": "guess",
                                                                  "where": "a.py"}],
                                  "limits": [{"name": "cap", "value": "2000", "stated_by_paper": False,
                                              "bound_fraction": 0.4}]}])
    assert "| init | he | guess | a.py |" in md and "| cap | 2000 | no | 40.00% |" in md


def test_method_may_overshoot_the_paper(f):
    assert f.judge_value(1.02, 0.69) == "not_reproduced"                       # a baseline must match
    assert f.judge_value(1.02, 0.69, overshoot_ok=True) == "reproduced"         # the method may do better
    assert f.judge_value(0.30, 0.69, overshoot_ok=True) == "not_reproduced"     # but not worse
    assert f.judge_value(50, 87.5, higher_is_better=False, overshoot_ok=True) == "reproduced"   # fewer steps is better
    assert f.judge_value(200, 87.5, higher_is_better=False, overshoot_ok=True) == "not_reproduced"
    value = {"kind": "value", "paper_value": 0.69, "higher_is_better": True}
    assert f.judge_item({**value, "method": True}, {"value": 1.02}) == "reproduced"
    assert f.judge_item(value, {"value": 1.02}) == "not_reproduced"
    assert f.scale_note(value, {"value": 1.02}) == "x1.48"
    order = {"kind": "ordering", "higher_is_better": True, "paper_a": 0.69, "paper_b": 0.30}
    hi = {"a": {"mean": 1.02, "lo": 0.95, "hi": 1.1}, "b": {"mean": 0.62, "lo": 0.55, "hi": 0.7}}
    assert f.judge_tuned(order, hi) == "reproduced"                             # method arm above paper_a is fine
    lo = {"a": {"mean": 0.40, "lo": 0.35, "hi": 0.45}, "b": {"mean": 0.20, "lo": 0.15, "hi": 0.25}}
    assert f.judge_tuned(order, lo) != "reproduced"
    assert f.judge_tuned({**order, "method": False}, hi) != "reproduced"         # a diagnostic must match
