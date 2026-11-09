"""Paper reproducer (simplified, 2026-11-03).

lead (claims, then one spec per experiment) -> per experiment: builder <-> reviewer rounds (pilot, then full runs)
-> evaluator computes every claim from the run files -> audit recomputes each claim independently
-> tuner rounds on the claims that fell short (optional) -> rule verdicts -> judge -> reporter.

The build loop has one escalation channel, `confer`, with two producers: a builder that cannot go on without an
answer (`questions` in its build answer) and a reviewer that finds the spec itself wrong (`owner: "spec"` on an
issue). One lead turn answers every open item at once - from the paper when the paper says it, as a recorded decision
when it does not - and may patch the spec in the same turn. A spec-owned review does not spend a `build_rounds`
round: the builder must not be asked to fix what only the lead can fix (2026-11-06: an acceptance line read
`d1=d2=26` off a superscript, three builder rounds re-reported it, none could fix it, and the last round ran out).

Every agent may read the paper (view_page renders a page as an image), so builds are not blind to the paper's numbers;
the results are still computed twice from the raw run files, by the evaluator and an independent auditor, and a
disagreement caps the claim. The rule verdicts are pure python; the judge agent sets the final verdicts with them, the
caps and the audits, and every departure from a rule verdict is shown. Prompts state goals, not procedures (2026-11-04).

args: {"paper": str, "compute": "remote" | "local", "gpu_hours": 8, "max_tries": 3, "build_rounds": 5,
       "max_questions": 2, "max_spec_fixes": 2,
       "job_wait_start_s": 60, "job_wait_s": 1800, "job_budget_s": 10800,
       "tune": false, "tune_rounds": 4, "tune_trials": 4, "smoke": false, "max_claims": null, "spec_only": false,
       "models": {"cheap"?, "strong"?}, "root": null}

`compute` may also be "auto": a probe agent checks for the gpu_runner MCP tools (https://github.com/aaronma1/gpu_runner)
and the run uses "remote" if gpu_list answers, else "local". `root` (absolute path) is needed when the host gives each
worker its own workspace, as JiuwenSwarm does: paper.pdf and scope.txt must be in it, and every relative path in the
prompts is resolved under it, so all roles share one file tree. Headless runs leave it unset.
`experts: true` hands each role to the member of the same name in the bound expert team (team/ builds it), whose
rails enforce the role's file, shell, GPU and web rules.

Workers (lead, builder, evaluator, tuner) run on the `strong` tier and the agents that check them (reviewer, audit,
judge) on `cheap`, so a checker is never the same model as the worker it checks.

Imports only `swarmflow` and the standard library, so it also runs inside JiuwenSwarm.
"""
import asyncio
import json
import os
import re

from swarmflow import agent, agent_session, log, parallel, phase

META = {
    "name": "paper-reproducer",
    "description": "Lead specs the experiments, builders run them with a reviewer, results are computed twice, judged",
    "phases": [
        {"title": "Lead", "detail": "claims, then one runnable spec per experiment"},
        {"title": "Build", "detail": "builder <-> reviewer rounds: pilot, then the full runs"},
        {"title": "Evaluate", "detail": "evaluator computes every claim; audit recomputes each one"},
        {"title": "Tune", "detail": "tuner rounds towards the paper's numbers (optional)"},
        {"title": "Judge", "detail": "rule verdicts, then the judge sets the final ones"},
        {"title": "Report", "detail": "report.md"},
    ],
}

# ============================================================== pure judging ==

RANK = {"could_not_test": 0, "not_reproduced": 1, "partial": 2, "reproduced": 3}
PARTIAL_FACTOR = 3  # within 3x the tolerance counts as partial
REL_TOL = 0.15  # least relative tolerance for a match (a claim's own rel_tol can only loosen it)
AUDIT_TOL = 0.02  # evaluator and audit must agree this closely
SENSITIVITY_TOL = 5.0  # a guess the reviewer measured to move a claim by at most this percent is not blocking
SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_UNIT_WORDS = ("normalized", "normalised", "fraction", "probability", "proportion")


def _tol(claim):
    return max(float(claim.get("rel_tol") or 0.0), REL_TOL)


def cap(verdict, ceiling):
    """Lower `verdict` to `ceiling` if it is above it."""
    return verdict if RANK[verdict] <= RANK[ceiling] else ceiling


def apply_caps(verdict, caps):
    for c in caps:
        verdict = cap(verdict, c["ceiling"])
    return verdict


def judge_value(measured, paper, rel_tol=REL_TOL, higher_is_better=True, overshoot_ok=False):
    """Within `rel_tol` of the paper: reproduced; within PARTIAL_FACTOR x: partial. With `overshoot_ok` (the paper's
    own method), doing better than the paper also reproduces it: details the paper does not state can make a
    faithful build stronger than the paper's runs."""
    if measured is None or paper is None:
        return "could_not_test"
    if overshoot_ok and (measured > paper if higher_is_better else measured < paper):
        return "reproduced"
    allowed = rel_tol * abs(paper)
    d = abs(measured - paper)
    if d <= allowed:
        return "reproduced"
    return "partial" if d <= PARTIAL_FACTOR * allowed else "not_reproduced"


def _ci(x):
    if not x or x.get("mean") is None:
        return None
    m = x["mean"]
    return (m if x.get("lo") is None else x["lo"], m if x.get("hi") is None else x["hi"], m)


def judge_ordering(a, b, higher_is_better=True, diff=None):
    """`a` beats `b`: a paired-difference CI (when given) or separated CIs in the claimed direction reproduce it;
    the other way refutes it; otherwise partial if the means point the claimed way."""
    ca, cb = _ci(a), _ci(b)
    if ca is None or cb is None:
        return "could_not_test"
    cd = _ci(diff)
    if cd is not None and diff.get("lo") is not None and diff.get("hi") is not None:
        lo, hi, m = cd if higher_is_better else (-cd[1], -cd[0], -cd[2])
        if lo > 0:
            return "reproduced"
        if hi < 0:
            return "not_reproduced"
        return "partial" if m > 0 else "not_reproduced"
    if not higher_is_better:
        ca, cb = (-ca[1], -ca[0], -ca[2]), (-cb[1], -cb[0], -cb[2])
    if ca[0] > cb[1]:
        return "reproduced"
    if cb[0] > ca[1]:
        return "not_reproduced"
    return "partial" if ca[2] > cb[2] else "not_reproduced"


def judge_predicate(measured, op, target, tol=REL_TOL):
    if measured is None or target is None:
        return "could_not_test"
    if op == "~=":
        return judge_value(measured, target, tol)
    if op == ">=":
        if measured >= target:
            return "reproduced"
        return "partial" if target - measured <= tol * abs(target) else "not_reproduced"
    raise ValueError(f"unknown predicate op {op!r}")


def judge_item(claim, item):
    item = item or {}
    if claim["kind"] == "value":
        return judge_value(item.get("value"), claim.get("paper_value"), _tol(claim),
                           claim.get("higher_is_better", True), bool(claim.get("method")))
    if claim["kind"] == "predicate":
        return judge_predicate(item.get("value"), claim.get("op"), claim.get("target"), _tol(claim))
    return judge_ordering(item.get("a"), item.get("b"), claim.get("higher_is_better", True), item.get("diff"))


def judge_tuned(claim, item):
    """A tuned result aims at the paper's own number for the method: an ordering with `paper_a` also needs the
    method arm's mean within tolerance of it, or better, so beating a weak baseline is not a match."""
    v = judge_item(claim, item)
    if claim["kind"] == "ordering" and claim.get("paper_a") is not None:
        a = (item or {}).get("a") or {}
        v = cap(v, judge_value(a.get("mean"), claim["paper_a"], _tol(claim), claim.get("higher_is_better", True),
                               overshoot_ok=claim.get("method", True)))  # not for a diagnostic (method false)
    return v


def _close(x, y, rel_tol=AUDIT_TOL):
    return abs(x - y) <= rel_tol * max(abs(x), abs(y)) + 1e-9


def audit_agrees(claim, mine, theirs):
    """(agrees, why): does the independent recompute match the evaluator's numbers?"""
    if not theirs:
        return False, "no independent recompute"
    mine = mine or {}
    keys = ("a", "b") if claim["kind"] == "ordering" else ("value",)
    for k in keys:
        x = (mine.get(k) or {}).get("mean") if k in ("a", "b") else mine.get(k)
        y = (theirs.get(k) or {}).get("mean") if k in ("a", "b") else theirs.get(k)
        if x is None and y is None:
            continue
        if x is None or y is None or not _close(float(x), float(y)):
            return False, f"independent recompute disagrees on {k}: evaluator {x}, audit {y}"
    return True, ""


def checked_diff(mine, theirs):
    """The evaluator's paired difference, only if the auditor's agrees; else None (judged on the arms' CIs)."""
    d, e = (mine or {}).get("diff"), (theirs or {}).get("diff")
    if not d or not e or d.get("mean") is None or e.get("mean") is None:
        return None
    return d if _close(float(d["mean"]), float(e["mean"])) else None


def aggregate(verdicts):
    vs = [v for v in verdicts if v is not None]
    if not vs:
        return "could_not_test"
    if all(v == "reproduced" for v in vs):
        return "reproduced"
    if not any(v in ("reproduced", "partial") for v in vs):
        return "not_reproduced" if "not_reproduced" in vs else "could_not_test"
    return "partial"


def scale_note(claim, result):
    """measured / paper (not judged): per compared arm of an ordering, so a setup gap that moves both arms shows,
    and for a value claim."""
    if claim["kind"] == "value":
        m, paper = (result or {}).get("value"), claim.get("paper_value")
        return f"x{m / paper:.2f}" if paper and m is not None else "-"
    if claim["kind"] != "ordering":
        return "-"
    out = []
    for k, paper in (("a", claim.get("paper_a")), ("b", claim.get("paper_b"))):
        m = ((result or {}).get(k) or {}).get("mean")
        if paper and m is not None:
            out.append(f"{k} x{m / paper:.2f}")
    return ", ".join(out) or "-"


def _fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, dict):
        return "n/a" if x.get("mean") is None else f"{x['mean']:.4g}"
    return f"{x:.4g}"


def _measured(claim, res):
    if claim["kind"] == "ordering":
        return f"a={_fmt((res or {}).get('a'))} b={_fmt((res or {}).get('b'))}"
    return _fmt((res or {}).get("value"))


def build_table(rows):
    out = ["| Claim | Paper | Measured | Rule verdict | Verdict | Caps | Scale vs paper |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        tuned = " (tuned)" if r.get("tuned_match") and r["verdict"] == "reproduced" else ""
        out.append(f"| {r['id']} | {r['paper']} | {r['measured']} | {r.get('rule_verdict', r['verdict'])} | "
                   f"{r['verdict']}{tuned} | {', '.join(r['caps']) or '-'} | {r.get('scale') or '-'} |")
    return "\n".join(out)


def caps_section(rows, caps):
    out = ["## Caps (written by the flow)", ""]
    for r in rows:
        out.append(f"- `{r['id']}`: verdict `{r['verdict']}`, rule verdict `{r.get('rule_verdict', r['verdict'])}`"
                   + (" - caps: " + "; ".join(r["caps"]) if r["caps"] else " - no caps")
                   + (f" - judge: {r['judge_reason']}" if r.get("judge_reason") else ""))
    for c in caps:
        if c.get("detail"):
            out += ["", f"### {c['reason']}", "", "Open findings of the last review:", ""] + [f"- {d}" for d in c["detail"]]
    return "\n".join(out) + "\n"


def tuning_section(rows):
    out = []
    for r in rows:
        t = r.get("tuned")
        if not t:
            continue
        out += [f"- `{r['id']}`: blind {r['measured']}, tuned {t['measured']} after {t['rounds']} rounds, "
                f"{t['trials']} trials: {t['verdict']}" + ("" if t["audited"] else f" (not counted: {t['why']})")]
        out += [f"  - {c['name']}: {c['blind']} -> {c['tuned']}"
                + (" (differs from the paper's stated value)" if c["stated_by_paper"] else "") for c in t["changes"]]
    return "## Tuning (written by the flow)\n\n" + "\n".join(out) + "\n" if out else ""


# ============================================================ checks/repairs ==

def _norm(key):
    """A comparable form of a ledger key (issue keys come back from a model with stray case and spaces)."""
    return str(key or "").strip().lower()


def _slugify(name):
    return re.sub(r"[^a-z0-9_-]+", "-", str(name).lower()).strip("-_") or "x"


def repair_arms(read):
    """Rename non-slug arm names everywhere they appear (e.g. 'reppo-1.2' -> 'reppo-1-2'), drop repeated arms, and
    rename non-slug claim ids the same way."""
    ren, arms, seen = {}, [], set()
    for a in (read or {}).get("arms") or []:
        old = str(a.get("name", ""))
        new = old if SLUG.match(old) else _slugify(old)
        ren[old] = new
        if new not in seen:
            seen.add(new)
            arms.append({**a, "name": new})
    claims = []
    for c in (read or {}).get("claims") or []:
        c = {**c, **{k: ren.get(c[k], c[k]) for k in ("better", "worse") if isinstance(c.get(k), str)}}
        cid = str(c.get("id", ""))
        if cid and not SLUG.match(cid):  # e.g. 'sharpness-ratio-lr-0.02' (edge, 2026-11-04): rename, do not send back
            ren[cid] = c["id"] = _slugify(cid)
        claims.append(c)
    return {**(read or {}), "arms": arms, "claims": claims}, {o: n for o, n in ren.items() if o != n}


def unit_problems(claim):
    """A paper number that cannot be a value of a normalized/fractional metric (a raw score of 200 on 'normalized
    return', reppo 2026-11-03: no measured value could ever match)."""
    metric = str(claim.get("metric") or "").lower()
    if not any(w in metric for w in _UNIT_WORDS):
        return []
    return [f"{k} {claim[k]:g} cannot be a value of the metric ({claim.get('metric')}): give it in the metric's own "
            f"units, or change the metric to the raw quantity"
            for k in ("paper_value", "paper_a", "paper_b", "target")
            if isinstance(claim.get(k), (int, float)) and abs(claim[k]) > 1.5]


def claim_problems(claim, arm_names):
    p = unit_problems(claim)
    if not SLUG.match(str(claim.get("id", ""))):
        p.append("id must be a lowercase slug")
    kind = claim.get("kind")
    if kind == "value" and claim.get("paper_value") is None:
        p.append("a value claim needs paper_value")
    elif kind == "predicate" and (claim.get("op") not in ("~=", ">=") or claim.get("target") is None):
        p.append("a predicate claim needs op (~= or >=) and target")
    elif kind == "ordering":
        b, w = claim.get("better"), claim.get("worse")
        if not b or not w or b == w:
            p.append("an ordering claim needs two different arms as better and worse")
        for side, n in (("better", b), ("worse", w)):
            if n and n not in arm_names:
                p.append(f"{side} {n!r} is not in the arm list ({', '.join(sorted(arm_names))})")
    elif kind not in ("value", "predicate", "ordering"):
        p.append(f"unknown kind {kind!r}")
    return p


def repair_selection(ans, ids):
    """Drop unknown claim ids and repeats across experiments; defer every claim no experiment covers."""
    if not ans:
        return ans
    seen, exps = set(), []
    for e in ans.get("experiments") or []:
        cl = []
        for c in e.get("claims") or []:
            if c.get("claim_id") in ids and c["claim_id"] not in seen:
                seen.add(c["claim_id"])
                cl.append(c)
        if cl:
            exps.append({**e, "claims": cl})
    deferred = [d for d in ans.get("deferred") or [] if d.get("claim_id") in ids and d["claim_id"] not in seen]
    named = seen | {d["claim_id"] for d in deferred}
    deferred += [{"claim_id": c, "reason": "not addressed by the lead"} for c in ids if c not in named]
    return {**ans, "experiments": exps, "deferred": deferred}


def experiment_problems(exp, by_id, arm_names):
    """Reasons an experiment spec cannot be built."""
    p = []
    eid = exp.get("id", "")
    if not SLUG.match(str(eid)):
        p.append(f"experiment id {eid!r} must be a lowercase slug")
    arms = [a.get("name") for a in exp.get("arms") or []]
    if not arms:
        p.append(f"{eid}: no arms")
    p += [f"{eid}: unknown arm {a!r}; use only: {', '.join(sorted(arm_names))}" for a in arms if a not in arm_names]
    metrics = {m.get("name") for m in exp.get("metrics") or []}
    for c in exp.get("claims") or []:
        claim = by_id[c["claim_id"]]
        if c.get("metric") not in metrics:
            p.append(f"{eid}: claim {c['claim_id']} uses metric {c.get('metric')!r}, which `metrics` does not define")
        for side in ("better", "worse"):
            if claim["kind"] == "ordering" and claim.get(side) not in arms:
                p.append(f"{eid}: claim {c['claim_id']} compares {claim.get(side)!r}, which the experiment does not run")
    return p


MAX_READ_TOL = 0.5  # a reading's relative uncertainty is capped here: a wider band would make any result a match


def _unc(r):
    """A reading's uncertainty in the metric's units. A reader that gives it in the plot's raw units (reppo,
    2026-11-03: value 0.66, uncertainty 10, raw ~170) is converted with the reading's own raw/value ratio; the
    result is capped at MAX_READ_TOL of the value."""
    v, u, raw = float(r["value"]), abs(float(r["uncertainty"])), r.get("raw")
    if v and u > abs(v) and raw:  # wider than the value itself: it is in raw units
        u = u * abs(v / float(raw))
    return min(u, MAX_READ_TOL * abs(v)) if v else u


def figure_fields(claim):
    """The paper numbers of a claim that can be read off a figure."""
    return {"value": ["paper_value"], "ordering": ["paper_a", "paper_b"]}.get(claim["kind"], [])


def needs_figure_read(claim):
    return bool(figure_fields(claim)) and (claim.get("source") == "figure" or (
        claim.get("source") is None and any("fig" in str(a).lower() for a in claim.get("anchors") or [])))


def _field_reads(claim, field, reads):
    """(usable readings, dropped values) for one field. A reading outside the metric's range is not a reading of it:
    when the lead's own value is a fraction (0..1), a negative or far larger reading came from a different quantity
    (2026-11-07, d-mask-ar1: one reader reported the raw weight -0.39 for a metric that reads `1 - d_Jac`, i.e. 1)."""
    lead = claim.get(field)
    bounded = isinstance(lead, (int, float)) and 0 <= lead <= 1
    got, dropped = [], []
    for read in reads:
        for r in (read or {}).get("readings") or []:
            if r.get("field") != field or r.get("value") is None:
                continue
            v = float(r["value"])
            if bounded and not (-1e-6 <= v <= 1.5):
                dropped.append(v)
                continue
            got.append((v, _unc(r)))
    return got, dropped


def reads_disagree(claim, reads):
    """True when two usable readings of a field are further apart than their own uncertainties."""
    for field in figure_fields(claim):
        got, _ = _field_reads(claim, field, reads)
        if len(got) >= 2 and abs(got[0][0] - got[1][0]) > got[0][1] + got[1][1] + 1e-12:
            return True
    return False


def combine_reads(claim, reads):
    """(claim with its paper numbers set from the figure readings, one record per field). Two readings are averaged;
    the uncertainty is the larger reported one, or half their gap if that is larger, and it widens the claim's
    tolerance (a number read off a plot is only as exact as the reading). A field nobody read keeps the lead's value."""
    out, records = dict(claim), []
    for field in figure_fields(claim):
        got, dropped = _field_reads(claim, field, reads)
        rec = {"claim_id": claim["id"], "field": field, "lead": claim.get(field), "reads": got, "dropped": dropped}
        if not got:
            records.append({**rec, "value": claim.get(field), "uncertainty": None, "agree": None,
                            "note": (f"every reading was out of the metric's range ({', '.join(f'{d:g}' for d in dropped)});"
                                     " kept the lead's value") if dropped else ""})
            continue
        if len(got) >= 3:  # a tie-break read decides: drop the one farthest from the median
            med = sorted(v for v, _ in got)[len(got) // 2]
            k = max(range(len(got)), key=lambda i: abs(got[i][0] - med))
            dropped.append(got[k][0])
            got = [g for i, g in enumerate(got) if i != k]
        vals = [v for v, _ in got]
        value = sum(vals) / len(vals)
        gap = (max(vals) - min(vals)) / 2
        agree = len(got) < 2 or abs(vals[0] - vals[1]) <= got[0][1] + got[1][1] + 1e-12
        # The disagreement may widen the band, but never past MAX_READ_TOL: a gap that big is a wrong reading, not a
        # wide reading, and a 70% band makes every measurement a match (2026-11-07, d-mask-ar1).
        unc = max([u for _, u in got] + [min(gap, MAX_READ_TOL * abs(value))])
        trial = {**out, field: value}
        if unit_problems(trial):  # a reading in the wrong units is not used
            records.append({**rec, "value": claim.get(field), "uncertainty": None, "agree": False,
                            "note": "readings not in the metric's units; kept the lead's value"})
            continue
        out = trial
        if value:
            out["rel_tol"] = max(float(out.get("rel_tol") or 0.0), unc / abs(value))
        note = ""
        if dropped:
            note = (f"{len(dropped)} reading(s) out of the metric's range dropped "
                    f"({', '.join(f'{d:g}' for d in dropped)})")
        records.append({**rec, "value": value, "uncertainty": unc, "agree": agree, "note": note})
    return out, records


def figure_section(records):
    if not records:
        return ""
    out = ["## Paper numbers read off figures (written by the flow)", ""]
    for r in records:
        reads = ", ".join(f"{v:.4g} ± {u:.2g}" for v, u in r["reads"]) or "none"
        val = "kept the lead's value" if r["uncertainty"] is None else f"{r['value']:.4g} ± {r['uncertainty']:.2g}"
        out.append(f"- `{r['claim_id']}` {r['field']}: lead {r['lead']}, readings {reads} -> {val}"
                   + ("" if r["agree"] in (True, None) else " (the readings disagree; tolerance widened to cover both)")
                   + (f" ({r['note']})" if r.get("note") else ""))
    return "\n".join(out) + "\n"


def retry_prompt(what, attempt, probs, rules):
    return (f"A deterministic check rejected that {what} (try {attempt}):\n{_bullets(probs)}\n"
            + (f"Every {what} must meet all of these rules:\n{_bullets(rules)}\n" if rules else "")
            + f"Send the whole corrected {what}.")


def _issue(i, owner="builder"):
    """A review issue with the fields the loop routes on: a stable `key` and an `owner`."""
    out = dict(i or {})
    out["owner"] = out.get("owner") or owner
    out["key"] = out.get("key") or _slugify(out.get("issue", ""))[:64] or "issue"
    return out


def measured_immaterial(i):
    """A builder-owned guess whose own measured effect on the claim is small: a note, not a blocker."""
    if (i.get("owner") or "builder") != "builder" or i.get("sensitivity_pct") is None:
        return False
    return abs(float(i["sensitivity_pct"])) <= SENSITIVITY_TOL


def blocking(review):
    """Blocking issues of a review, each with a `key` and an `owner`. No usable review counts as one blocking issue.
    A guess the reviewer measured to move a claim by at most SENSITIVITY_TOL is downgraded to a note (its text is
    still reported): otherwise "this could move the number" blocks every build and no measurement can settle it."""
    if not review:
        return [{"issue": "the reviewer gave no usable answer", "evidence": "", "action": "review again",
                 "blocking": True, "key": "no-review", "owner": "builder"}]
    return [_issue(i) for i in review.get("issues") or [] if i.get("blocking") and not measured_immaterial(i)]


def limit_key(name):
    return _norm(f"limit:{name}")


def build_problems(built, accepted=()):
    """Blocking issues found mechanically in a build answer: an unmeasured or binding limit the paper does not
    state, and a change made to move a number toward the paper's that was not compared with the faithful setting.
    `accepted` holds the keys of limit issues the lead has already ruled on (a narrowing it decided to live with), so
    an accepted limit stops blocking and shows up in the report instead of looping through the builder forever.
    A limit that binds past its tolerance is a scope decision, not a code defect: it is owned by `spec`, which routes
    it to the lead, because a builder told to "raise it" by a compute budget it does not control can only spin."""
    out, accepted = [], {_norm(a) for a in accepted}
    for lim in built.get("limits") or []:
        name = str(lim.get("name"))
        key = limit_key(name)
        if lim.get("stated_by_paper") or key in accepted:
            continue
        frac = lim.get("bound_fraction")
        tol = GUARD_TOL if lim.get("guard") else LIMIT_TOL
        if frac is None:
            out.append({"issue": f"limit {name!r} ({lim.get('value')}) is not stated by the paper and its bind rate "
                                 f"was not measured", "evidence": lim.get("evidence") or "",
                        "action": "record in every run file how often it cut an episode/step/sample", "blocking": True,
                        "key": key, "owner": "builder"})
        elif frac > tol:
            out.append({"issue": f"limit {name!r} ({lim.get('value')}) is not stated by the paper and cut {frac:.1%} "
                                 f"of what it applies to (allowed {tol:.0%}"
                                 f"{', as a guard' if lim.get('guard') else ''})",
                        "evidence": lim.get("evidence") or "",
                        "action": "raise or remove it and re-run what it affected; if the compute budget cannot cover "
                                  "that, the lead decides whether to narrow the experiment and record it",
                        "blocking": True, "key": key, "owner": "spec"})
    for ch in built.get("changes") or []:
        if ch.get("reason") == "result" and not ch.get("tested"):
            out.append({"issue": f"change {ch.get('change')!r} was made to move a result toward the paper's but was "
                                 f"not compared with the faithful setting", "evidence": ch.get("evidence") or "",
                        "action": "compare both under the same conditions, or revert it", "blocking": True})
    return out


def build_shape_problems(built, exp):
    """Blocking problems with the shape of a build answer: it reports run files, and they exist and parse. Only what a
    builder can always satisfy is blocking; a coverage mismatch is logged, because a mechanical check that cannot be
    satisfied is the trap this flow was in on 2026-11-06. Skipped when the flow cannot see the run root."""
    files = built.get("run_files") or []
    if not files:
        return [{"issue": "no run files were reported for a pilot or a full run", "evidence": built.get("notes") or "",
                 "action": "write one run file per seed per arm in the format given and report their paths",
                 "blocking": True, "key": "no-run-files", "owner": "builder"}]
    root = _HOST["root"]
    if not root:  # headless runs keep this open: the flow has no path to the workers' tree
        return []
    root_abs, outside, bad, parsed = os.path.abspath(root), [], [], []
    for f in [f for f in files if f.rstrip().endswith(".json")]:  # the run format is one JSON per run
        f = os.path.expanduser(f)
        path = os.path.abspath(f if os.path.isabs(f) else os.path.join(root, f))
        if not path.startswith(root_abs + os.sep):  # a job-side path the flow cannot read is not evidence of work
            outside.append(f)
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except OSError:
            bad.append(f"{f} (not found)")
            continue
        except ValueError as e:
            bad.append(f"{f} (not JSON: {e})")
            continue
        if not isinstance(doc, dict) or not {"metrics", "config"} <= set(doc):
            bad.append(f"{f} (no `metrics` or `config`)")
        else:
            parsed.append(doc)
    out = []
    if outside:
        out.append({"issue": f"{len(outside)} run file(s) are reported outside the run root",
                    "evidence": "; ".join(outside[:4]),
                    "action": f"download them under {root} in the run format and report those paths",
                    "blocking": True, "key": "outside-run-files", "owner": "builder"})
    if bad:
        out.append({"issue": f"{len(bad)} of {len(parsed) + len(bad)} reported run files are missing or unreadable",
                    "evidence": "; ".join(bad[:6]), "action": "report only files that exist, in the run format",
                    "blocking": True, "key": "bad-run-files", "owner": "builder"})
    arms = {a.get("name") for a in exp.get("arms") or []}
    seen = {d.get("arm") for d in parsed if d.get("arm")}
    if arms and seen and not arms <= seen:
        log(f"{exp.get('id')}: run files cover {', '.join(sorted(seen))} of the arms {', '.join(sorted(arms))}")
    return out


def build_record_section(states):
    """Markdown: each experiment's unstated choices and limits, as the builder last reported them."""
    lines = []
    for s in states:
        ch, lim, dec = s.get("choices") or [], s.get("limits") or [], s.get("decisions") or []
        if not ch and not lim and not dec:
            continue
        if ch or lim:
            lines.append(f"\n### Settings the paper does not state: {s['experiment']}\n")
        if ch:
            lines += ["| Choice | Value | Source | Where |", "|---|---|---|---|"]
            lines += [f"| {c.get('choice')} | {c.get('value')} | {c.get('source')} | {c.get('where')} |" for c in ch]
        if lim:
            lines += ["", "| Limit | Value | Stated by paper | Bound fraction |", "|---|---|---|---|"]
            lines += [f"| {x.get('name')} | {x.get('value')} | {'yes' if x.get('stated_by_paper') else 'no'} | "
                      f"{'not measured' if x.get('bound_fraction') is None else format(x['bound_fraction'], '.2%')} |"
                      for x in lim]
        if dec:
            lines += [f"\n### Decisions taken during the build: {s['experiment']}\n",
                      "| Item | Answer | Source |", "|---|---|---|"]
            lines += [f"| {d.get('key')} | {str(d.get('answer', '')).replace('|', '/')} | {d.get('source')}"
                      + (f" ({d.get('citation')})" if d.get("citation") else "") + " |" for d in dec]
    return ("\n## Build settings\n" + "\n".join(lines) + "\n") if lines else ""


def format_issues(issues):
    return "\n".join(f"- {i['issue']}\n  evidence: {i.get('evidence') or '-'}\n  action: {i.get('action') or '-'}"
                     for i in issues) or "- (none)"


# ================================================================== schemas ==

S = {"type": "string"}
N = {"type": ["number", "null"]}
B = {"type": "boolean"}


def obj(props, required=None):
    return {"type": "object", "additionalProperties": False,
            "required": list(props) if required is None else required, "properties": props}


def arr(items):
    return {"type": "array", "items": items}


CI = {"anyOf": [obj({"mean": {"type": "number"}, "lo": N, "hi": N}), {"type": "null"}]}
_CLAIM = {"id": S, "text": S, "kind": {"enum": ["value", "ordering", "predicate"]}, "metric": S,
          "paper_value": N, "rel_tol": {"type": "number"}, "op": {"enum": ["~=", ">=", None]}, "target": N,
          "higher_is_better": B, "better": {"type": ["string", "null"]}, "worse": {"type": ["string", "null"]},
          "anchors": arr(S), "paper_a": N, "paper_b": N,
          "source": {"enum": ["table", "text", "figure"]}, "figure": S, "method": B}
CLAIMS = obj({"arms": arr(obj({"name": S, "description": S})),
              "claims": arr(obj(_CLAIM, [k for k in _CLAIM if k not in ("paper_a", "paper_b", "rel_tol", "source", "figure",
                                                                         "method")]))})
FIELDS = ["paper_value", "paper_a", "paper_b"]
FIGREAD = obj({"claim_id": S, "page": {"type": "integer"},
               "readings": arr(obj({"field": {"enum": FIELDS}, "raw": N, "value": {"type": "number"},
                                    "uncertainty": {"type": "number"}, "how": S})), "notes": S})
# Details the paper leaves open that could move a metric (the lead names them, the builder settles each one).
CHOICE_BASIS = ["paper", "web", "guess", "lead"]  # how a setting was set; `lead` is written by the flow
_OC = {"choice": S, "paper_says": S, "why_it_matters": S, "suggested": S, "basis": {"enum": CHOICE_BASIS}}
OPEN_CHOICE = obj(_OC, [k for k in _OC if k != "basis"])  # optional: older fixtures have none
_EXP = {"id": S, "arms": arr(obj({"name": S, "description": S})), "procedure": S,
        "seeds": {"type": "integer"}, "seeds_reason": S,
        "metrics": arr(obj({"name": S, "definition": S})),
        "claims": arr(obj({"claim_id": S, "metric": S})),
        "acceptance": arr(S), "open_choices": arr(OPEN_CHOICE)}
EXPERIMENT = obj(_EXP, [k for k in _EXP if k != "open_choices"])  # optional: older fixtures have none
SPECS = obj({"experiments": arr(EXPERIMENT), "deferred": arr(obj({"claim_id": S, "reason": S}))})
# A review issue names its owner: `builder` for something the code can fix, `spec` for the experiment description
# itself (an acceptance line that contradicts the paper). `sensitivity_pct` is the measured effect on the claim, in
# percent: a builder-owned guess at or below SENSITIVITY_TOL is a note.
_ISSUE = {"issue": S, "evidence": S, "action": S, "blocking": B, "key": S,
          "owner": {"enum": ["builder", "spec"]}, "sensitivity_pct": N, "sensitivity_note": S}
REVIEW = obj({"issues": arr(obj(_ISSUE, ["issue", "evidence", "action", "blocking"]))})
# Every setting the code uses that the paper does not state, and where its value came from.
CHOICE_SOURCES = ["paper", "reference", "measured", "guess"]
CHOICE = obj({"choice": S, "value": S, "source": {"enum": CHOICE_SOURCES}, "where": S})
# Every cap, clip, budget or truncation in the code, and the largest fraction of episodes/steps/samples it cut in any
# run file (null if not measured).
# `guard` marks a deliberate safety bound set far above what the run needs (BUILD_RULES allows those). A guard may
# cut a little; past GUARD_TOL it is not a guard but a narrowing of the experiment, which is the lead's call.
LIMIT = obj({"name": S, "value": S, "stated_by_paper": B, "bound_fraction": N, "evidence": S, "guard": B},
            ["name", "value", "stated_by_paper", "bound_fraction", "evidence"])
# Every setting changed since the builder's previous report, and why.
CHANGE_REASONS = ["paper", "reference", "bug", "result"]
CHANGE = obj({"change": S, "reason": {"enum": CHANGE_REASONS}, "tested": B, "evidence": S})
# A builder that cannot go on without an answer asks in `questions`, showing what it already read: a question the
# paper answers is answered by reading, not by the lead.
QUESTION = obj({"key": S, "question": S, "what_i_checked": S, "options": arr(S), "blocking": B, "impact_if_wrong": S},
               ["key", "question", "what_i_checked", "blocking"])
# `submitted` is an answer with job ids and no run files: the builder ends its turn there and the flow waits for
# the job, which can sit in a shared queue for hours (2026-11-07).
_BUILD = {"stage": {"enum": ["pilot", "full", "submitted", "failed"]}, "job_ids": arr(S), "run_files": arr(S), "notes": S,
          "choices": arr(CHOICE), "limits": arr(LIMIT), "changes": arr(CHANGE), "questions": arr(QUESTION)}
BUILD = obj(_BUILD, [k for k in _BUILD if k != "questions"])  # questions are optional: most turns have none
# A watcher's whole turn: wait on one job and say what happened. Nothing else.
WATCH = obj({"job_uuid": S, "status": S, "exit_code": N, "tail": S, "done": B}, ["job_uuid", "status", "done"])
# The lead's one answer turn. `source` says where an answer comes from: `paper`/`reference` (with a citation) answers
# it; `decision` records a call the paper does not make, which the flow writes into the experiment's open choices.
_ANSWER = {"key": S, "answer": S, "source": {"enum": ["paper", "reference", "decision"]}, "citation": S, "blocks": B}
_PATCH = {"acceptance": arr(S), "procedure": S, "arms": arr(obj({"name": S, "description": S})),
          "open_choices": arr(OPEN_CHOICE)}
CONF = obj({"answers": arr(obj(_ANSWER, ["key", "answer", "source"])),
            "spec_patch": obj(_PATCH, []), "notes": S}, ["answers"])
LIMIT_TOL = 0.01  # a limit the paper does not state may cut at most this fraction of episodes/steps/samples
GUARD_TOL = 0.10  # a declared safety guard may cut this much; more than this narrows the experiment itself
_ITEM = {"value": N, "a": CI, "b": CI, "diff": CI}
_REQ = ["value", "a", "b"]
RESULTS = obj({"results": arr(obj({"claim_id": S, **_ITEM, "sources": arr(S)}, ["claim_id", *_REQ, "sources"]))})
AUDIT = obj({"claim_id": S, **_ITEM, "sources": arr(S), "notes": S}, ["claim_id", *_REQ, "sources", "notes"])
TUNED = obj({"results": arr(obj({
    "claim_id": S, **_ITEM, "trials": {"type": "integer"}, "sources": arr(S), "note": S,
    "changes": arr(obj({"name": S, "blind": S, "tuned": S, "stated_by_paper": B}))},
    ["claim_id", *_REQ, "trials", "sources", "note", "changes"]))})
JUDGE = obj({"verdicts": arr(obj({"claim_id": S, "verdict": {"enum": sorted(RANK)}, "reason": S}))})

EXAMPLE_CLAIMS = {
    "arms": [{"name": "gd", "description": "plain gradient descent"},
             {"name": "heavy-ball", "description": "gradient descent with Polyak momentum"}],
    "claims": [
        {"id": "hb-steps", "text": "Heavy-ball reaches loss 1e-6 in 87.5 steps (median)", "kind": "value",
         "metric": "median steps until loss < 1e-6", "paper_value": 87.5, "op": None, "target": None,
         "higher_is_better": False, "better": None, "worse": None, "anchors": ["Table 1"], "source": "table",
         "method": True},
        {"id": "hb-beats-gd", "text": "Heavy-ball needs fewer steps than GD", "kind": "ordering",
         "metric": "median steps until loss < 1e-6", "paper_value": None, "op": None, "target": None,
         "higher_is_better": False, "better": "heavy-ball", "worse": "gd", "anchors": ["Sec. 4", "Fig. 2"],
         "paper_a": 87.5, "paper_b": 412.0, "source": "figure",
         "figure": "Fig. 2 (page 6), left panel: curves 'heavy-ball' and 'GD', steps at loss 1e-6"}]}
EXAMPLE_SPECS = {"experiments": [{
    "id": "quadratic", "seeds": 20, "seeds_reason": "Sec. 4 averages over 20 random starts",
    "arms": [{"name": "heavy-ball", "description": "x <- x - eta*grad + beta*(x - x_prev), eta 0.01, beta 0.9"},
             {"name": "gd", "description": "x <- x - eta*grad, eta 0.01"}],
    "procedure": "f(x) = 0.5 x^T A x with A = diag(1, 100) (Sec. 4). Start points x0 ~ N(0, I) with seeds 0..19, the "
                 "same for both arms; x_prev = x0 at the first step. Stop when f < 1e-6 or after 100000 steps.",
    "metrics": [{"name": "steps", "definition": "first step t with f(x_t) < 1e-6 (100000 if never)"}],
    "claims": [{"claim_id": "hb-steps", "metric": "steps"}, {"claim_id": "hb-beats-gd", "metric": "steps"}],
    "acceptance": ["20 run files per arm, seeds 0..19", "the update rules in the code match the arms' descriptions"],
    "open_choices": [
        {"choice": "step cap for a run that never reaches f < 1e-6", "paper_says": "nothing",
         "why_it_matters": "a cap below the slow arm's true step count shrinks its median and the gap between arms",
         "suggested": "a cap far above any run's steps; record how many runs hit it", "basis": "guess"}]}],
    "deferred": []}


def _example(x):
    return "Example of the format, for a different, made-up paper (do not copy its content):\n" + _j(x)


# =================================================================== helpers ==

ROLE_TIER = {"lead": "strong", "figure": "strong", "figure-check": "cheap", "builder": "strong", "reviewer": "cheap", "watch": "cheap", "evaluator": "strong", "audit": "cheap",
             "tuner": "strong", "judge": "cheap", "reporter": "cheap"}


_HOST = {"root": ""}  # set once at the start of run(); "" = the backend already shares one workspace


def _where():
    root = _HOST["root"]
    if not root:
        return ""
    return (f"All files of this run are under {root}. Every relative path in this task (paper.pdf, scope.txt, "
            f"repo/..., report.md) means {root}/<path>: give your file tools and shell commands absolute paths. "
            "Do not write outside it.\n\n")


async def ask(prompt, **kw):
    """agent() with the run-root preamble."""
    return await agent(_where() + prompt, **kw)


def session(**kw):
    """agent_session() with the run-root preamble in its instructions."""
    if _where():
        kw["instructions"] = _where() + (kw.get("instructions") or "")
    return agent_session(**kw)


GPU_PROBE = {"type": "object", "required": ["available", "detail"], "additionalProperties": False,
             "properties": {"available": {"type": "boolean"}, "detail": {"type": "string"}}}


async def resolve_compute(args):
    """'local' / 'remote' as given; 'auto' asks one agent whether the gpu_runner MCP tools answer."""
    if args.get("compute") != "auto":
        return args.get("compute")
    probe = await ask(
        "Check whether you have the gpu_runner tools (gpu_submit, gpu_list, gpu_wait, gpu_output, "
        "download_job_files; the host may prefix their names). If you do, call gpu_list once. Answer "
        "available=true only if gpu_list answered without an error; say in `detail` what you found. Do not submit "
        "anything.", label="gpu-probe", phase="Lead", schema=GPU_PROBE,
        options={"agent_template": "watch"} if args.get("experts") else None)  # GPU tools, nothing else
    ok = bool(probe and probe.get("available"))
    log(f"compute: {'remote (gpu_runner)' if ok else 'local'}: {(probe or {}).get('detail', 'probe gave no answer')}")
    return "remote" if ok else "local"




def _opts(args, role):
    """Per-call options: the role's model tier, and with `experts` the expert-team member that owns the role."""
    model = (args.get("models") or {}).get(ROLE_TIER[role])
    opts = {"model": model} if model else {}
    if args.get("experts"):
        opts["agent_template"] = role
    return opts or None


def _try(label, n):
    return label if n == 1 else f"{label}#{n}"


def _j(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, indent=1)


def _bullets(items):
    return "\n".join(f"- {i}" for i in items) or "- (none)"


def _local(args):
    return args.get("compute") == "local"


def run_format(root):
    return (f"One JSON file per run at {root}/<experiment>/runs/<arm>/seed<seed>.json: "
            '{"experiment": str, "arm": str, "seed": int, "job_id": str (the gpurun job id that produced it, or '
            '"local"), "config": {every hyperparameter the run used}, "metrics": {<metric name>: number, for every '
            'metric the spec names, at the end of the run}, "curves": {<metric name>: [[step, value], ...], the '
            'learning curve of each metric that changes during training}, "limits": {<name>: {"value": ..., '
            '"bound_fraction": fraction of episodes/steps/samples it cut in this run}, for every cap, clip, budget or '
            'truncation in the code ({} if none)}}. A run that did not finish has no file.')


def _compute_note(args):
    if _local(args):
        note = ("There is no GPU and no gpurun. Run jobs with python in your shell, each command finishing within "
                "about 8 minutes (split long work into chunks and keep partial results on disk).")
        if _HOST["root"]:
            note += (" If the system python lacks a package you need, make a project environment with `uv` under "
                     "the run's directory rather than installing globally.")
    else:
        tools = ("gpu_submit, gpu_wait, gpu_output, download_job_files" if _HOST["root"]
                 else "gpu_submit, gpu_wait, gpu_output, gpu_download")
        note = (f"Run experiments on the GPU with the gpurun tools ({tools}). "
                "Your own shell has no network and no GPU, but its python has CPU torch and jax for tiny checks: debug "
                "there first. Put a pyproject.toml with pinned versions next to your code, submit the smallest "
                "directory a job needs, batch seeds into as few jobs as possible (jobs run one at a time on a queue "
                "shared with other teams), and " + ("download_job_files" if _HOST["root"] else "gpu_download")
                + " each job's result files into your own directory. To make a job end, wrap its commands in a "
                "wall-clock `timeout`, which fails loudly, rather than adding step caps inside the experiment, which "
                "silently change its results.\n"
                "Once you have submitted, report stage `submitted` with the `job_ids` and END YOUR TURN: the flow "
                "waits for the job itself and wakes you with its last output. Do not call gpu_wait or gpu_output "
                "yourself, and never resubmit a command that is still QUEUED - the queue is shared with other teams "
                "and a queued job can take hours. If you must look at a job by hand, use one gpu_wait with "
                "max_wait_s=1800, never a short poll loop.")
    if args.get("smoke"):
        note += (" SMOKE TEST: cut every experiment to at most 3 seeds and at most 2% of the paper's steps; every shell "
                 "command must finish within 60 seconds. Record the cut.")
    return note


class _Turns:
    """A session whose turns are labelled base, base#2, ...; until one turn has answered, a retry carries the
    original request too (a skipped turn leaves nothing in the session)."""

    def __init__(self, base, **kw):
        self.base, self.n, self.sess = base, 0, session(**kw)
        self.first, self.answered = None, False

    async def send(self, prompt, schema, label=None):
        """One turn. `label` names a turn of another kind on the same session (the build conference); it does not
        advance the base counter, so `lead`, `lead#2`, ... keep meaning what they say."""
        if label is None:
            self.n += 1
            if self.first is None:
                self.first = prompt
            elif not self.answered:
                prompt = f"{self.first}\n\n{prompt}"
            label = _try(self.base, self.n)
        ans = await self.sess.send(prompt, schema=schema, options={"label": label})
        self.answered = self.answered or ans is not None
        return ans


def public_claim(c):
    """What blind agents may see of a claim: no paper numbers."""
    return {k: c.get(k) for k in ("id", "text", "kind", "metric", "higher_is_better", "better", "worse", "op", "method")}


def _result_fields(c, metric):
    if c["kind"] == "ordering":
        return (f"{c['id']} (ordering, metric `{metric}`): `a` = mean over seeds and 95% CI (lo, hi) for arm "
                f"{c.get('better')}, `b` = the same for {c.get('worse')}, `diff` = mean and 95% CI of the paired "
                f"difference a - b over the seeds both ran (null if they share none); `value` null")
    return f"{c['id']} ({c['kind']}, metric `{metric}`): `value` = the mean over seeds; `a`, `b`, `diff` null"


def _write_report(md):
    """Write report.md under the run root. The reporter agent writes it on a finished run; on a stopped run there is no
    agent left to ask, so the flow writes what it knows."""
    root = _HOST["root"]
    if not root:
        return ""
    path = os.path.join(root, "report.md")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(md)
        return path
    except OSError as e:  # a stopped run must not fail on the way out
        log(f"could not write {path}: {e}")
        return ""


def _stop(reason, states=None, **extra):
    """A stopped run still explains itself. 2026-11-06: twenty hours of Build ended with an empty answer because a
    stop produced no artifact at all - the open findings existed only in the workflow state."""
    rows = states or []
    md = [f"# Reproduction stopped: {reason}", "",
          "No verdict was reached: the workflow stopped before the results were evaluated.", ""]
    if rows:
        md += ["## Experiments", "",
               "| Experiment | Full runs | Review | Rounds | Open findings |", "|---|---|---|---|---|"]
        md += [f"| {s.get('experiment')} | {'yes' if s.get('ran') else 'no'} | "
               f"{'passed' if s.get('reviewed') else 'open'} | {s.get('rounds', 0)} | "
               f"{'; '.join(s.get('open') or []) or '-'} |" for s in rows]
        md += ["", "The code and the run files the builders wrote are under `repo/`: one JSON per seed per arm."]
        dec = [d for s in rows for d in (s.get("decisions") or [])]
        if dec:
            md += ["", "## Decisions taken during the build", "", "| Item | Answer | Source |", "|---|---|---|"]
            md += [f"| {d.get('key')} | {str(d.get('answer', '')).replace('|', '/')} | {d.get('source')} |"
                   for d in dec]
    body = "\n".join(md) + "\n"
    path = _write_report(body)
    if path:
        log(f"stopped report: {path}")
    out = {"verdict": "could_not_test", "reason": reason, "claims": [], "table": "", "caps": [],
           "caps_md": f"Stopped: {reason}\n", "report": body, **extra}
    if rows and "experiments" not in out:
        out["experiments"] = rows
    return out


# ===================================================================== steps ==

OUTPUT_RULES = (
    "Keep every tool result small. What a tool returns is copied into the run's record and its history, so an "
    "unbounded dump costs tokens twice and can overrun the progress stream. Print the fields you need - `jq "
    "'.metrics, .config'` on a run file, not its `curves` array; `tail -n 40` or `grep` a log; `head` a file, or read "
    "it with `limit`/`offset`/`pages` - and never dump a whole log, dataset or source tree. `glob` a specific "
    "directory, not a run root. When a check turns on a specific number, read that number rather than the file around "
    "it: a review that quotes no numbers is not a review. Call a tool by its name - `gpu_submit`, `gpu_wait`, "
    "`gpu_output` with `tail_lines`, `download_job_files` - instead of `tool_search`, which returns the whole "
    "catalogue every time."
)

BUILD_RULES = (
    "Follow the paper's own text first: its equations, algorithm boxes, code listings and tables. Fill only what it "
    "leaves open, from a reference implementation it points to or your own judgment, and settle each of the "
    "experiment's `open_choices` (check each one marked `guess` before relying on it). Check beliefs about the environment or data by measuring them, not from memory. "
    "Add no cap, clip, budget, truncation, wrapper or early stop the paper does not state, except a safety bound set "
    "far above what the run needs; record every limit in each run file with the fraction of episodes/steps/samples it "
    "cut. A limit the paper does not state that cuts more than " + f"{LIMIT_TOL:.0%}" + " makes the run invalid "
    "until it is raised. When a check fails, yours or the reviewer's, find out why before changing anything, and "
    "never make it pass by changing what it measures (for example by counting truncated episodes as finished). "
    "In your answer: `choices` lists every setting the paper does not state, its value, its source (`paper` if the "
    "paper does state it after all, `reference` naming the implementation and file, `measured`, or `guess`) and "
    "where it is set in the code; `limits` lists every limit with the largest fraction it cut in any run file (null "
    "if you did not measure it); `changes` lists every setting you changed since your last report and why: `paper`, "
    "`reference`, `bug`, or `result` when the change is meant to move a number toward the paper's, which counts "
    "only if you compared it with the faithful setting under the same conditions (`tested`).")
REVIEW_RULES = (
    "Start with the builder's choices, limits and changes, and with anything the code adds that the paper does not "
    "state. Check the limits against the run files and the code: a missing or understated limit is blocking. A guess "
    "that could move a claim's number is blocking unless the builder tested or justified it; for a `reference` "
    "choice, check the named source for the one that matters most. A comparison run while a limit was binding or "
    "unmeasured is not evidence. Compare the code with the paper's equations and listings before comparing the "
    "results with the paper's numbers. The paper's method scoring better than the paper reports is not a problem by "
    "itself (unstated details can make a faithful build stronger); a baseline far from the paper's number, in "
    "either direction, points to a setup difference.")


READ_PROMPT = (
    "Read scope.txt, then paper.pdf. List the arms: every method or baseline the experiments in scope run, each with "
    "a lowercase slug name (e.g. `ppo`, `double-q`) and a one-line description. Then extract the claims the scope "
    "asks to reproduce. If the paper proposes a method or algorithm, its main claims are the benchmark numbers that "
    "method produces (its score, return, accuracy or error on each benchmark in scope): a `value` claim for each, "
    "then any comparison with baselines. Per claim: a lowercase slug id; `method` true when the claim's number is "
    "the paper's proposed method's own performance on a benchmark (its score, return, accuracy or error), false "
    "for a baseline's number and for a diagnostic such as entropy, a loss curve or a count (the method may perform "
    "better than the paper reports; a baseline and a diagnostic should match it); kind `value` (a number), `ordering` (X "
    "beats Y; `better` and `worse` are arm names) or `predicate` (op `~=` or `>=` against a target, for a trend such "
    "as a loss that keeps falling); the metric; where it appears (anchors). For an ordering, also give `paper_a` and "
    "`paper_b`, the paper's value for `better` and for `worse` (from a table, or read off a plot; null if not "
    "reported). Every paper number is in the units of the claim's own metric: if the metric is normalized, "
    "normalize the paper's number the same way. Say where each claim's numbers come from in `source`: `table`, "
    "`text`, or `figure` when they are only in a plot; for a figure, `figure` names it exactly (figure and panel, "
    "page, which curves, and where to read them, e.g. at the end of training). Numbers only in a plot are read off "
    "it again afterwards by looking at the figure; you can look at a page yourself with view_page.\n\n"
    + _example(EXAMPLE_CLAIMS))
# Every prompt that reads files, runs commands or touches the GPU carries the output rule: a tool result is copied
# into the run's record and the run's history, so an unbounded dump is paid for twice (2026-11-07: 400 tool results
# over 5 KB were 77% of a 6.5 MB activity trail and overran the 6 MB progress budget).
BUILD_RULES = BUILD_RULES + "\n" + OUTPUT_RULES
REVIEW_RULES = REVIEW_RULES + "\n" + OUTPUT_RULES
READ_PROMPT = READ_PROMPT + "\n" + OUTPUT_RULES
READ_RULES = ["arm names are lowercase slugs; an ordering's better and worse are two different listed arms",
              "claim ids are unique lowercase slugs; a value claim has paper_value, a predicate has op and target",
              "every paper number is in the units of the claim's metric (a normalized metric has normalized numbers)"]


async def read_claims(args, lead):
    """(arms, claims, rejected) from the lead's first turn; bad claims are retried, dropped on the last try."""
    prompt, tries = READ_PROMPT, int(args.get("max_tries", 3))
    arms, claims, rejected = [], [], []
    for attempt in range(1, tries + 1):
        read, renamed = repair_arms(await lead.send(prompt, schema=CLAIMS))
        if renamed:
            log(f"arms renamed: {renamed}")
        arms = read.get("arms") or []
        names = {a["name"] for a in arms}
        claims, rejected, seen = [], [], set()
        for c in read.get("claims") or []:
            probs = claim_problems(c, names) + (["duplicate id"] if c.get("id") in seen else [])
            if probs:
                rejected.append({"id": c.get("id"), "problems": probs})
            else:
                seen.add(c["id"])
                claims.append(c)
        if arms and claims and (not rejected or attempt == tries):
            if rejected:
                log(f"claims dropped: {rejected}")
            break
        probs = [f"{r['id']}: {'; '.join(r['problems'])}" for r in rejected] + ([] if claims else ["no usable claims"])
        log(f"claims try {attempt} rejected: {probs}")
        prompt = retry_prompt("answer (arms and every claim)", attempt, probs, READ_RULES)
    return arms, claims, rejected


async def read_figures(args, claims):
    """(claims with paper numbers read off their figures, records): two independent readings per claim, one per
    model tier, by agents that look at the rendered page and do not see the lead's numbers."""
    todo = [c for c in claims if needs_figure_read(c)]
    if not todo:
        return claims, []

    def read_one(c, role, n=1):
        fields = {"paper_value": "the number for the method in `text`",
                  "paper_a": f"the value of arm `{c.get('better')}`", "paper_b": f"the value of arm `{c.get('worse')}`"}
        return ask(
            f"Read the paper's own number(s) for this claim off its figure. Find the figure's page (read_file "
            f"paper.pdf or grep for the caption), then look at it with view_page, cropping to the panel. Read each "
            f"curve where the claim says (e.g. the end of training), using the axis ticks. For each field give the raw "
            f"reading in the plot's units, the value in the claim's metric units (apply the normalization the paper "
            f"states, if the metric is normalized), and an honest uncertainty (half-width, in the metric's units) from "
            f"the tick spacing, line width and any shaded band. A derived metric (a mask similarity, a distance, a "
            f"ratio, a difference) must be computed from the numbers you read and given in its own units: never report "
            f"the raw quantity you read it from. If the figure does not print what the metric needs, send null for "
            f"that field and say in `how` what is missing - a value outside the metric's range is worse than no value. "
            f"Explain in `how` which curve and point you read.\n"
            f"Fields: {_j({k: fields[k] for k in figure_fields(c)})}\nWhere: {c.get('figure') or c.get('anchors')}\n"
            f"Claim:\n{_j(public_claim(c))}\n\n{OUTPUT_RULES}",
            label=_try(f"{role}:{c['id']}", n), schema=FIGREAD, options=_opts(args, role))

    reads = await parallel([lambda c=c, role=role: read_one(c, role) for c in todo
                            for role in ("figure", "figure-check")])
    per = {c["id"]: list(reads[2 * i:2 * i + 2]) for i, c in enumerate(todo)}
    tie = [c for c in todo if reads_disagree(c, per[c["id"]])]
    if tie:
        # A third reading decides: two readings that disagree cannot say which one is wrong, and averaging them is
        # how a 70% tolerance got written (2026-11-07). The tie-break is the strong tier, because the cheap check is
        # the reader that most often reports the wrong quantity.
        log(f"figure: {len(tie)} claim(s) with disagreeing readings, reading again: "
            + ", ".join(c["id"] for c in tie))
        extra = await parallel([lambda c=c: read_one(c, "figure", 2) for c in tie])
        for c, r in zip(tie, extra):
            per[c["id"]].append(r)
    out, records = {c["id"]: c for c in claims}, []
    for c in todo:
        out[c["id"]], recs = combine_reads(c, per[c["id"]])
        records += recs
    for r in records:
        if r["uncertainty"] is not None:
            log(f"figure {r['claim_id']}.{r['field']}: lead {r['lead']} -> {r['value']:.4g} ± {r['uncertainty']:.2g}"
                + ("" if r["agree"] else " (readings disagree)"))
    return [out[c["id"]] for c in claims], records


async def write_specs(args, lead, arms, claims):
    """(experiments, deferred) from the lead's second turn, or None."""
    hours = float(args.get("gpu_hours", 8))
    limit = args.get("max_claims") or (2 if args.get("smoke") else None)
    ids, by_id, names = [c["id"] for c in claims], {c["id"]: c for c in claims}, {a["name"] for a in arms}
    rules = ["every claim id is in one experiment's `claims` or in `deferred`, once",
             "experiment ids are lowercase slugs; arms come only from the arm list",
             "every experiment runs the arms its claims compare and defines each metric its claims use"]
    prompt = (
        f"Now decide what the team reproduces within about {hours:g} "
        f"{'cpu-hours on a laptop CPU' if _local(args) else 'gpu-hours'}, and group the claims into experiments. "
        + (f"Select at most {limit} claims. " if limit else "")
        + "If the paper proposes a method, its benchmark numbers come first. Defer the rest with a one-line reason. "
        "One experiment is one set of runs that several claims can be computed from. For each, give its arms, a "
        "short `procedure` (what to run, in your own words, pointing to the paper's sections), the number of seeds, "
        "the metrics (name and exact definition) and which metric each claim uses, and what a reviewer should check "
        "in `acceptance`. The builders read the paper themselves. In `open_choices`, list what the paper leaves "
        "open that could move a claim's number: for example network sizes, normalization, how the method's update "
        "is computed where the text is ambiguous, environment or dataset version and options, and how evaluation "
        "runs (episode or step limits, greedy or sampled actions, which checkpoint). For each, say what the paper "
        "does say (or 'nothing'), why it matters, and a suggested setting. A suggestion never contradicts what the "
        "paper does say. Confirm every suggestion before you give it, and set `basis`: `paper` (cite the section), "
        "`web` (a library's or reference implementation's code or docs you fetched with web_search/web_fetch; cite "
        "the URL), or `guess` for anything you did not confirm, including your own reasoning. Do not present a "
        "guess as a fact: the builder must check every `guess` before relying on it.\n\nClaims:\n" + _j(claims)
        + "\n\n" + _example(EXAMPLE_SPECS))
    tries = int(args.get("max_tries", 3))
    for attempt in range(1, tries + 1):
        ans = repair_selection(await lead.send(prompt, schema=SPECS), ids)
        if not ans:
            probs = ["empty reply"]
        else:
            exps = ans["experiments"]
            if limit:
                kept = 0
                for e in exps:
                    e["claims"] = e["claims"][:max(0, int(limit) - kept)]
                    kept += len(e["claims"])
                exps = [e for e in exps if e["claims"]]
                ans = {**ans, "experiments": exps}
            probs = [] if exps else ["no experiment selected"]
            seen_ids = set()
            for e in exps:
                probs += experiment_problems(e, by_id, names)
                if e.get("id") in seen_ids:
                    probs.append(f"experiment id {e['id']!r} used twice")
                seen_ids.add(e.get("id"))
        if not probs:
            return ans["experiments"], ans["deferred"]
        log(f"specs try {attempt} rejected: {probs}")
        prompt = retry_prompt("answer (every experiment and the deferred claims)", attempt, probs, rules)
    return None


def _exp_brief(exp):
    """What builders, evaluators and auditors see of an experiment."""
    return {k: exp[k] for k in ("id", "arms", "procedure", "seeds", "seeds_reason", "metrics", "acceptance",
                                "open_choices") if k in exp}


def _record_jobs(eid, jobs):
    """Note the job ids of an experiment under the run root, so a resumed or restarted run can see what was already
    submitted instead of submitting it again (2026-11-06: the same experiment was resubmitted 5-7 times)."""
    root = _HOST["root"]
    if not root:
        return
    path = os.path.join(root, "jobs.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh) or {}
    except (OSError, ValueError):
        data = {}
    known = data.setdefault(eid, []) if isinstance(data, dict) else []
    for j in jobs:
        if j not in known:
            known.append(j)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
    except OSError:
        pass


async def watch_job(args, eid, job_uuid, n, wait_s):
    """One cheap turn that only waits: gpu_wait for `wait_s`, then report. It submits nothing, cancels nothing, reads
    no files."""
    per = int(wait_s)
    return await ask(
        f"Wait for GPU job `{job_uuid}` of experiment `{eid}` and report it. Call gpu_wait on that job with "
        f"max_wait_s={per}; if it returns still_running, call it again with the same argument "
        f"until it is done or about {per} s have passed. Then call gpu_output on the job "
        f"with tail_lines=25 and put that text in `tail`. Report `status` exactly as gpu_wait gives it (QUEUED, "
        f"RUNNING, COMPLETED, FAILED, CANCELLED), `exit_code` (null when there is none) and `done` true only when the "
        f"status is not QUEUED or RUNNING. Submit nothing, cancel nothing, read no files, run no commands.",
        label=_try(f"watch:{job_uuid}", n), schema=WATCH, options=_opts(args, "watch"))


async def wait_for_jobs(args, eid, jobs):
    """(finished, text) after waiting on the submitted jobs, with exponential backoff: `job_wait_start_s` (60 s),
    doubled up to `job_wait_s` (1800 s), inside `job_budget_s` (3 h) per job.

    Backoff is what gets both ends right: a job that dies after a minute is reported in a minute, a job queued for two
    hours is looked at a dozen times rather than fifty, and each look is one cheap watcher turn instead of a strong
    builder turn re-reading its context (2026-11-07)."""
    cap = float(args.get("job_wait_s") or 1800)
    start = min(float(args.get("job_wait_start_s") or 60), cap)
    budget = float(args.get("job_budget_s") or 10800)
    told = []
    for uuid in jobs:
        spent, wait, n, last = 0.0, start, 0, None
        while spent < budget:
            n += 1
            step = min(wait, cap, budget - spent)
            ans = await watch_job(args, eid, uuid, n, step)
            spent += step
            if not ans:
                log(f"{eid}: watch {uuid} gave no answer (waited {int(spent)}s of {int(budget)}s)")
                wait *= 2
                continue
            last = ans
            if ans.get("done"):
                tail = (ans.get("tail") or "").strip()[:2000]
                told.append(f"{uuid} finished: {ans.get('status')}"
                            + (f" (exit {ans['exit_code']})" if ans.get("exit_code") is not None else "")
                            + (f"\nLast output:\n{tail}" if tail else ""))
                log(f"{eid}: job {uuid} {ans.get('status')} after {int(spent)}s of waiting")
                break
            log(f"{eid}: job {uuid} still {ans.get('status')} (waited {int(spent)}s of {int(budget)}s, "
                f"next check in {int(min(wait * 2, cap))}s)")
            wait *= 2
        else:
            state = (last or {}).get("status") or "unknown"
            return False, (f"{uuid} is still {state} after {int(spent)} s of waiting ({n} checks, backing off to "
                           f"{int(cap)} s each). The queue is shared; do not resubmit it. If it is stuck, cancel it "
                           f"and report `failed` with the reason.")
    return True, "\n\n".join(told) or "the job reported no output"


_LEAD_LOCK = asyncio.Lock()  # one lead session, several experiments: a conference turn never interleaves


def apply_patch(exp, patch):
    """(experiment, applied keys, ignored reasons). Only the description fields can change, and an arm patch must
    keep the arm names: the claims, the specs' comparisons and every run file refer to them."""
    if not patch:
        return exp, [], []
    out, applied, ignored = dict(exp), [], []
    for key in ("acceptance", "procedure", "open_choices"):
        if patch.get(key):
            out[key] = patch[key]
            applied.append(key)
    arms = patch.get("arms")
    if arms:
        before, after = [a.get("name") for a in exp.get("arms") or []], [a.get("name") for a in arms]
        if after == before:
            out["arms"] = arms
            applied.append("arms")
        else:
            ignored.append(f"arms {after} are not the spec's {before}: arm names cannot change under the claims")
    return out, applied, ignored


async def confer(args, lead, exp, claims, items, n):
    """(experiment, answers, applied) after one lead turn on the open items. `items` are a builder's questions
    (`kind: question`) and a reviewer's spec-owned issues (`kind: issue`). An answer's `source` is `paper` or
    `reference` (with a citation) or `decision`; a decision is written into the experiment's open choices with
    `basis: "lead"`, so a setting the paper does not state stays visible in the report as a decision."""
    eid = exp["id"]
    brief = {**exp, "claims": [c for c in claims if c["id"] in {x["claim_id"] for x in exp["claims"]}]}
    item_md = "\n".join(
        (f"- [{i['key']}] QUESTION (the builder cannot go on): {i['text']}\n"
         f"  already checked: {i.get('detail') or '-'}\n  if wrong: {i.get('impact') or '-'}")
        if i["kind"] == "question" else
        (f"- [{i['key']}] SPEC ISSUE (the reviewer says the experiment description is wrong): {i['text']}\n"
         f"  evidence: {i.get('detail') or '-'}\n  proposed fix: {i.get('impact') or '-'}")
        for i in items)
    prompt = (
        "You wrote this experiment's spec; the build is paused on the items below. Answer every one, in one turn.\n"
        "Per answer: `answer` is the instruction the builder follows; `source` is where it comes from - `paper` or "
        "`reference` with the page, section or file in `citation`, or `decision` when the paper does not say and the "
        "call is yours (never present a decision as the paper's); `blocks` is true when it must be applied before the "
        "next run. If an item is a spec issue, return `spec_patch` with the corrected fields verbatim: `acceptance` "
        "(the whole list), `procedure`, `arms` (same names, corrected descriptions) or `open_choices`. Leave the "
        "claims, the metrics and the seeds alone. Check the paper yourself before answering - view_page renders a "
        "page, which settles a superscript that flat text loses - because a wrong number in the spec blocks the build "
        "for as many rounds as it takes.\n\nItems:\n" + item_md
        + f"\n\nExperiment as the builder has it:\n{_j(brief)}")
    async with _LEAD_LOCK:
        ans = await lead.send(prompt, schema=CONF, label=_try(f"lead:conf:{eid}", n))
    if not ans:
        return exp, [], []
    answers = [{**a, "blocks": bool(a.get("blocks", True))} for a in ans.get("answers") or []]
    exp2, applied, ignored = apply_patch(exp, ans.get("spec_patch"))
    for why in ignored:
        log(f"{eid}: spec patch ignored: {why}")
    if applied:
        log(f"{eid}: spec patched: {', '.join(applied)}")
    for a in answers:
        if a.get("source") == "decision":  # an unstated setting: keep it visible instead of absorbing it
            exp2 = {**exp2, "open_choices": list(exp2.get("open_choices") or []) + [{
                "choice": str(a.get("answer", ""))[:200], "paper_says": "nothing", "why_it_matters": "",
                "suggested": "as answered by the lead", "basis": "lead"}]}
        log(f"{eid}: conference {a.get('key')}: {a.get('source')} - {str(a.get('answer'))[:160]}")
    return exp2, answers, applied


def answer_text(answers, applied):
    """The builder's next prompt after a conference: what was answered, and whether the spec changed."""
    if not answers and not applied:
        return ("The lead gave no answer. Use your judgement, record the setting as a `guess` in `choices` with its "
                "measured effect, and go on.")
    head = "The lead answered:\n" + "\n".join(
        f"- [{a['key']}] {a['answer']} (source: {a['source']}" + (f"; {a['citation']}" if a.get("citation") else "")
        + ")" for a in answers) if answers else "The lead did not answer, but the spec changed."
    if applied:
        head += ("\nThe spec was patched (" + ", ".join(applied) + "); the experiment below supersedes the one you "
                 "have.")
    return head


async def build_experiment(args, exp, claims, lead):
    """Builder <-> reviewer rounds (pilot, then full runs) with one escalation channel to the lead.
    Returns the experiment's review state."""
    eid, rounds = exp["id"], int(args.get("build_rounds", 5))
    max_questions, max_fixes = int(args.get("max_questions", 2)), int(args.get("max_spec_fixes", 2))
    code_dir = f"repo/arms/{eid}"
    fmt = run_format("repo/arms")
    builder = _Turns(f"builder:{eid}", options=_opts(args, "builder"),
                     instructions=f"You reproduce experiment `{eid}` in {code_dir}/. {_compute_note(args)}")
    brief = {**exp, "claims": [c for c in claims if c["id"] in {x["claim_id"] for x in exp["claims"]}]}
    prompt = (f"Reproduce this experiment from paper.pdf (and scope.txt): implement it in {code_dir}/ as the paper "
              f"describes, run a short pilot first and report stage `pilot`; the full runs come after the pilot is "
              f"reviewed. {BUILD_RULES}\nIf you cannot go on without an answer the paper does not give, put the "
              f"question in `questions` with what you already checked instead of guessing: the lead answers in one "
              f"turn.\nAfter you submit a GPU job, report stage `submitted` with its `job_ids` and END YOUR TURN: the "
              f"flow waits for the job and wakes you with its output.\nWrite the run files in this format:"
              f"\n{fmt}\n\nExperiment:\n{_j(brief)}")
    stage, open_issues, record = None, ["never reviewed"], {}
    decisions, accepted, used, fixes, asked, reviews, turns = [], [], 0, 0, 0, 0, 0
    # `used` counts reviews only: a turn that asks or that triggers a spec patch is not a review round. `turns` is a
    # hard stop so an experiment whose lead never resolves anything still ends.
    while used < rounds and turns < 2 * rounds + max_questions + max_fixes + 3:
        turns += 1
        built = await builder.send(prompt, schema=BUILD)
        if not built or built["stage"] == "failed":
            prompt = "That did not work. Fix what failed and report again (`pilot` first if no pilot passed review)."
            open_issues = ["the builder reported failure" if built else "the builder gave no answer"]
            continue
        stage = built["stage"]
        record = {"choices": built.get("choices") or [], "limits": built.get("limits") or []}
        if stage == "submitted":
            submitted = [str(j) for j in (built.get("job_ids") or [])]
            if _local(args):
                prompt = ("There is no GPU and no gpurun here. Run the experiment in your shell and report the run "
                          "files.")
                continue
            if not submitted:
                prompt = "You reported `submitted` with no `job_ids`. Submit the run and report its job ids."
                continue
            _record_jobs(eid, submitted)
            finished, text = await wait_for_jobs(args, eid, submitted)
            prompt = (text + ("\nDownload the run files and report the stage you reached."
                              if finished else "\nReport what you have now.") + f"\nExperiment:\n{_j(brief)}")
            continue
        open_q = [q for q in (built.get("questions") or []) if q.get("blocking")]
        if open_q and asked >= max_questions:
            log(f"{eid}: {len(open_q)} question(s) left unanswered: the {max_questions}-question budget is spent")
        if open_q and asked < max_questions:
            items = [{"kind": "question", "key": q.get("key") or _slugify(str(q.get("question", "")))[:64],
                      "text": q.get("question", ""), "detail": q.get("what_i_checked", ""),
                      "impact": q.get("impact_if_wrong", "")} for q in open_q]
            asked += len(items)
            exp, answers, _ = await confer(args, lead, exp, claims, items, len(decisions) + 1)
            decisions += [{**a, "experiment": eid} for a in answers]
            brief = {**exp, "claims": [c for c in claims if c["id"] in {x["claim_id"] for x in exp["claims"]}]}
            prompt = (answer_text(answers, []) + f"\nGo on with the build.\nExperiment:\n{_j(brief)}\n"
                      f"Write the run files in this format:\n{fmt}")
            continue
        auto = build_problems(built, accepted) + build_shape_problems(built, exp)
        for i in auto:
            log(f"{eid}: build check: {i['issue']}")
        reviews += 1
        review = await ask(
            f"Review experiment `{eid}` at stage `{stage}`: read paper.pdf, the code in {code_dir}/ and the run "
            f"files, and judge whether this reproduces the paper's experiment faithfully and whether the results "
            f"look right. {REVIEW_RULES} Report each problem with evidence and the fix; mark it blocking if it "
            f"should be fixed before going on, give it a stable `key`, set `owner` to `spec` when the experiment "
            f"description rather than the code is wrong, and give `sensitivity_pct` (the claim's move, in percent) "
            f"when you measured how much a guess matters.\nRun format:\n{fmt}\nExperiment:\n{_j(brief)}\n"
            f"Run files: {built['run_files']}\nBuilder's choices, limits and changes:\n"
            f"{_j({k: built.get(k) or [] for k in ('choices', 'limits', 'changes')})}\n"
            f"Builder notes: {built['notes']}",
            label=_try(f"reviewer:{eid}", reviews), schema=REVIEW, options=_opts(args, "reviewer"))
        issues = auto + blocking(review)
        open_issues = [i["issue"] for i in issues]
        notes = [i for i in (review or {}).get("issues") or [] if not i.get("blocking") or measured_immaterial(i)]
        spec_issues = [i for i in issues if i.get("owner") == "spec"]
        if spec_issues and fixes < max_fixes:
            fixes += 1  # this round is void: the builder cannot fix an acceptance line only the lead can change
            items = [{"kind": "issue", "key": i.get("key") or "spec", "text": i["issue"],
                      "detail": i.get("evidence", ""), "impact": i.get("action", "")} for i in spec_issues]
            exp, answers, applied = await confer(args, lead, exp, claims, items, len(decisions) + 1)
            decisions += [{**a, "experiment": eid} for a in answers]
            accepted += [a["key"] for a in answers if a.get("source") == "decision"]  # ruled on: stop re-blocking
            brief = {**exp, "claims": [c for c in claims if c["id"] in {x["claim_id"] for x in exp["claims"]}]}
            prompt = (answer_text(answers, applied) + f"\nReport again at stage `{stage}`, against this experiment:"
                      f"\n{_j(brief)}")
            continue
        used += 1
        if issues:
            prompt = ("The reviewer found blocking problems. Fix them, re-run what they change, and report again at "
                      f"stage `{stage}`:\n" + format_issues(issues))
            continue
        if stage == "full":
            return {"experiment": eid, "ran": True, "reviewed": True, "rounds": used, "open": [],
                    "decisions": decisions, "spec": exp, **record}
        prompt = (f"The pilot passed review. Now run the full experiment, download every run file and report stage "
                  f"`full`." + (f"\nReviewer notes:\n{format_issues(notes)}" if notes else ""))
    return {"experiment": eid, "ran": stage == "full", "reviewed": False, "rounds": used, "open": open_issues,
            "decisions": decisions, "spec": exp, **record}


async def evaluate(args, exps, by_id, root):
    """{claim id: result}: the evaluator computes every claim from the run files under `root`."""
    want = [(by_id[c["claim_id"]], c["metric"]) for e in exps for c in e["claims"]]
    fields = _bullets(_result_fields(c, m) for c, m in want)
    prompt = (f"Compute every claim below from the raw run files with your own code in repo/eval/, as the paper "
              f"defines the metric. Fill the fields exactly as listed (a missing number is null) and list the run "
              f"files you used in `sources`.\nRun files:\n{run_format(root)}\n\nFields:\n{fields}\n\n"
              f"Experiments:\n{_j([_exp_brief(e) for e in exps])}\n\n{OUTPUT_RULES}")
    ev = _Turns("evaluator", options=_opts(args, "evaluator"))
    out = {}
    for attempt in (1, 2):
        ans = await ev.send(prompt, schema=RESULTS)
        out.update({r["claim_id"]: r for r in (ans or {}).get("results", [])})
        missing = [c["id"] for c, _ in want if c["id"] not in out]
        if not missing:
            break
        log(f"evaluator try {attempt}: no result for {', '.join(missing)}")
        prompt = f"Your results have no entry for: {', '.join(missing)}. Send results for every claim:\n{fields}"
    return out


async def audit_all(args, want, root, suffix=""):
    """{claim id: audit}: an independent recompute of each claim from the run files."""
    audits = await parallel([lambda c=c, m=m: ask(
        f"Independently recompute this claim from the raw run files with your own code (a python heredoc in your "
        f"read-only shell), not anyone else's. Fill the fields exactly as follows and list the run files you used:\n- {_result_fields(c, m)}\n"
        f"Run files:\n{run_format(root)}\nClaim:\n{_j(public_claim(c))}\n\n{OUTPUT_RULES}",
        label=f"audit:{c['id']}{suffix}", schema=AUDIT, options=_opts(args, "audit")) for c, m in want])
    return {c["id"]: a for (c, _), a in zip(want, audits)}


async def tune(args, exps, short, blind):
    """(best tuned result per claim, rounds used) after up to `tune_rounds` rounds towards the paper's numbers."""
    rounds, trials = int(args.get("tune_rounds", 4)), int(args.get("tune_trials", 4))
    metric = {c["claim_id"]: c["metric"] for e in exps for c in e["claims"]}
    fields = _bullets(_result_fields(c, metric[c["id"]]) for c in short)
    sess = _Turns("tuner", options=_opts(args, "tuner"))
    prompt = (
        f"The blind reproduction is finished. These claims did not match the paper. Read paper.pdf and scope.txt, and "
        f"tune the experiment's settings, as a researcher would, until each claim matches the paper's own number: "
        f"`paper_value` for a value claim and, for an ordering, `paper_a` for the method arm (`better`) and "
        f"`paper_b` for the baseline. The paper's method (a claim with `method` true, and an ordering's `better` "
        f"arm) may also do better than the paper's number. An ordering that holds while the method's score is far "
        f"below the paper's is not a match. "
        f"You may change any "
        f"setting, including one the paper states. Work in repo/tuned/<experiment>/ (copy the code from repo/arms/, "
        f"which you must not change), try at most {trials} settings per round, and write every setting you try and "
        f"its outcome to repo/tuned/<experiment>/trials.json. Write the best setting's runs, all seeds, as run files:\n"
        f"{run_format('repo/tuned')}\n{_compute_note(args)}\n\nAnswer for the best setting so far, fields exactly as "
        f"listed; `trials` counts every setting tried so far, `changes` lists every value that differs from the blind "
        f"run:\n{fields}\n\nClaims, with the paper's numbers:\n{_j(short)}\nBlind results:\n"
        f"{_j({c['id']: blind.get(c['id']) for c in short})}\nExperiments:\n{_j([_exp_brief(e) for e in exps])}"
        f"\n\n{OUTPUT_RULES}")
    best, used = {}, 0
    for rnd in range(1, rounds + 1):
        ans = await sess.send(prompt, schema=TUNED)
        used = rnd
        best.update({r["claim_id"]: r for r in (ans or {}).get("results", []) if r["claim_id"] in metric})
        left = [c for c in short if judge_tuned(c, best.get(c["id"])) != "reproduced"]
        log(f"tune round {rnd}: {len(short) - len(left)} of {len(short)} claims match")
        if not left:
            break
        prompt = ("Round results against the paper:\n"
                  + _bullets(f"{c['id']}: measured {_measured(c, best.get(c['id']))}, paper "
                             f"{c.get('paper_value') if c['kind'] != 'ordering' else (c.get('paper_a'), c.get('paper_b'))}"
                             for c in left)
                  + f"\nThese still do not match. Try new settings (at most {trials} this round), keep trials.json "
                    f"up to date, and answer again for the best setting so far:\n{fields}")
    return best, used


async def final_verdicts(args, claims, exps, rows, audits, caps, decisions=()):
    """Verdicts from the judge, with the lead's build-time decisions as evidence: a decision never turns a failed
    measurement into `reproduced`, but it is what justifies `could_not_test` when the protocol cannot identify the
    claim (2026-11-07: the lead ruled four level claims untestable for a paper-side inconsistency and the judge, which
    never saw that ruling, would have judged them on the raw numbers anyway)."""
    ids = [r["id"] for r in rows]
    decided = [d for d in decisions or [] if d.get("experiment") in {r["experiment"] for r in rows}]
    prompt = (
        "You are the final judge. Set each claim's final verdict: reproduced, partial, not_reproduced or "
        "could_not_test. Read paper.pdf and scope.txt, and check the run files in repo/ as far as you need. The rule "
        "verdicts and caps below come from fixed checks: they are your main evidence, but the decision is yours. If "
        "you depart from a rule verdict or a cap, say exactly why, citing a file or a number. Judge what was "
        "measured. A large gap in 'scale' means the setup may differ from the paper's; the paper's method doing "
        "better than the paper (a `method` claim, or an ordering's better arm) still reproduces it. A row with `tuned` was matched "
        "after a hyperparameter search, which the user allows: it may be `reproduced` if its audited numbers match. A "
        "lead's decision can justify `could_not_test` when the protocol cannot identify a claim, and it never turns a "
        "failed measurement into `reproduced`.\n\n"
        "Rule results (JSON):\n" + json.dumps(rows, sort_keys=True) + "\nEnd of rule results.\n\nCaps:\n" + _j(caps)
        + (("\n\nDecisions the lead took during the build (its call, not the paper's - say in `reason` which one you "
            "used):\n" + _j([{"experiment": d.get("experiment"), "key": d.get("key"), "answer": d.get("answer"),
                                "source": d.get("source"), "citation": d.get("citation")} for d in decided]))
           if decided else "")
        + "\n\nClaims:\n" + _j([c for c in claims if c["id"] in ids]) + "\nExperiments:\n" + _j(exps)
        + "\nAudits:\n" + _j({i: audits.get(i) for i in ids}) + "\n" + OUTPUT_RULES)
    sess = _Turns("judge", options=_opts(args, "judge"))
    for attempt in range(1, int(args.get("max_tries", 3)) + 1):
        ans = await sess.send(prompt, schema=JUDGE)
        named = [v["claim_id"] for v in (ans or {}).get("verdicts", [])]
        probs = (["empty reply"] if not ans else
                 [f"claim {c!r} has no verdict" for c in ids if c not in named]
                 + [f"claim {c!r} has more than one verdict" for c in sorted(set(named)) if named.count(c) > 1])
        if not probs:
            return {v["claim_id"]: (v["verdict"], v["reason"]) for v in ans["verdicts"]}
        log(f"judge try {attempt} rejected: {probs}")
        prompt = retry_prompt("set of verdicts", attempt, probs, ["every claim id appears exactly once: " + ", ".join(ids)])
    return None


# ======================================================================= run ==

def parse_args(args):
    """args as a dict: JiuwenSwarm passes the JSON string, the headless runner a dict."""
    if isinstance(args, str):
        return json.loads(args) if args.strip() else {}
    return dict(args or {})


async def run(args):
    args = parse_args(args)
    _HOST["root"] = (args.get("root") or "").rstrip("/")
    phase("Lead")
    args = {**args, "compute": await resolve_compute(args)}
    lead = _Turns("lead", options=_opts(args, "lead"))
    arms, claims, rejected = await read_claims(args, lead)
    if not claims:
        return _stop("the lead found no usable claims", rejected=rejected)
    # With a claim limit (max_claims, smoke) the lead selects claims first, so only the selected ones are read off
    # their figures: reading every claim's figures first cost a smoke run its whole budget (2026-11-05).
    limited = bool(args.get("max_claims") or args.get("smoke"))
    fig_records = []
    if not limited:
        claims, fig_records = await read_figures(args, claims)
    written = await write_specs(args, lead, arms, claims)
    if written is None:
        return _stop("the lead gave no valid experiments", rejected=rejected)
    exps, deferred = written
    if limited:
        chosen = {c["claim_id"] for e in exps for c in e["claims"]}
        read, fig_records = await read_figures(args, [c for c in claims if c["id"] in chosen])
        read = {c["id"]: c for c in read}
        claims = [read.get(c["id"], c) for c in claims]
    if args.get("spec_only"):  # test the lead's turns without paying for builds
        return {"verdict": "could_not_test", "reason": "spec_only: stopped after the specs", "claims": claims,
                "experiments": exps, "deferred": deferred, "figure_reads": fig_records, "rejected_claims": rejected}
    by_id = {c["id"]: c for c in claims}
    judged = [by_id[c["claim_id"]] for e in exps for c in e["claims"]]
    metric = {c["claim_id"]: c["metric"] for e in exps for c in e["claims"]}
    exp_of = {c["claim_id"]: e["id"] for e in exps for c in e["claims"]}

    phase("Build")
    states = await parallel([lambda e=e: build_experiment(args, e, claims, lead) for e in exps])
    for s in states:  # a spec the lead patched mid-build supersedes the one it first wrote, downstream too
        if s.get("spec"):
            exps = [s["spec"] if e["id"] == s["experiment"] else e for e in exps]
    caps = []
    for s in states:
        if not s["ran"]:
            caps.append({"experiment": s["experiment"], "ceiling": "could_not_test",
                         "reason": f"{s['experiment']}: the full runs never finished"})
        elif not s["reviewed"]:
            caps.append({"experiment": s["experiment"], "ceiling": "partial",
                         "reason": f"{s['experiment']}: review still open", "detail": s["open"]})
    if args.get("smoke"):
        caps.append({"experiment": None, "ceiling": "partial", "reason": "smoke test: reduced scope"})
    if not any(s["ran"] for s in states):
        return _stop("no experiment finished its runs", states=states)

    phase("Evaluate")
    results = await evaluate(args, exps, by_id, "repo/arms")
    audits = await audit_all(args, [(c, metric[c["id"]]) for c in judged], "repo/arms")

    phase("Tune")
    tuned, tune_rounds, tuned_audits = {}, 0, {}
    short = [c for c in judged if judge_tuned(c, results.get(c["id"])) != "reproduced"]
    if not (args.get("tune") and not args.get("smoke") and short):
        log("tuning skipped: " + ("not asked for" if not args.get("tune") else "smoke run" if args.get("smoke")
                                  else "every claim reproduced"))
    else:
        tuned, tune_rounds = await tune(args, exps, short, results)
        if tuned:
            tuned_audits = await audit_all(args, [(c, metric[c["id"]]) for c in short if c["id"] in tuned],
                                           "repo/tuned", ":tuned")

    phase("Judge")
    rows = []
    for c in judged:
        cid = c["id"]
        res, aud = results.get(cid), audits.get(cid)
        if res is not None:
            res = {**res, "diff": checked_diff(res, aud)}
        mine = [x for x in caps if x["experiment"] in (None, exp_of[cid])]
        ok, why = audit_agrees(c, res, aud)
        if not ok:
            mine = mine + [{"experiment": exp_of[cid], "ceiling": "partial", "reason": why}]
        rule = judge_item(c, res)
        tuned_row, match = None, False
        if cid in tuned:
            t = {**tuned[cid], "diff": checked_diff(tuned[cid], tuned_audits.get(cid))}
            t_ok, t_why = audit_agrees(c, t, tuned_audits.get(cid))
            tv = judge_tuned(c, t)
            match = t_ok and tv == "reproduced"
            tuned_row = {"measured": _measured(c, t), "verdict": tv, "audited": t_ok, "why": t_why,
                         "rounds": tune_rounds, "trials": t["trials"], "changes": t["changes"]}
        v = apply_caps("reproduced" if match else rule, mine)
        paper = (c.get("paper_value") if c["kind"] == "value" else c.get("target") if c["kind"] == "predicate"
                 else f"{c['better']} > {c['worse']} ({_fmt(c.get('paper_a'))} vs {_fmt(c.get('paper_b'))})")
        if c["kind"] != "predicate" and _tol(c) > REL_TOL:
            paper = f"{paper if isinstance(paper, str) else _fmt(paper)} (tol {_tol(c):.0%})"
        rows.append({"id": cid, "experiment": exp_of[cid], "paper": paper if isinstance(paper, str) else _fmt(paper),
                     "measured": _measured(c, res), "verdict": v, "caps": [x["reason"] for x in mine],
                     "scale": scale_note(c, res), **({"tuned": tuned_row, "tuned_match": match} if tuned_row else {})})
    verdicts = await final_verdicts(args, claims, exps, rows, audits, caps,
                                    [d for s in states for d in s.get("decisions") or []])
    for r in rows:
        r["rule_verdict"] = r["verdict"]
        if verdicts and r["id"] in verdicts:
            r["verdict"], r["judge_reason"] = verdicts[r["id"]]
    if not verdicts:
        caps = caps + [{"experiment": None, "ceiling": "reproduced",
                        "reason": "the judge gave no valid answer; rule verdicts stand"}]
    verdict = aggregate([r["verdict"] for r in rows])
    table = build_table(rows)
    caps_md = (caps_section(rows, caps) + tuning_section(rows) + figure_section(fig_records)
               + build_record_section(states))

    phase("Report")
    report = await ask(
        f"Write report.md in the workspace root. Paper verdict: {verdict}. Include this table unchanged, then for every "
        f"experiment what was run (code and run files are in repo/), and every assumption, deviation and cut. Do not "
        f"change any verdict, and do not give your own reasons for a cap: the caps section below is appended to your "
        f"report as written. Comment on 'Scale vs paper' where it is far from 1. Say that the builders could read "
        f"the paper, including its reported numbers (the results were computed independently from the run files).\n\n{table}\n\n{caps_md}\n"
        f"Claims the lead chose not to reproduce (list them with the reason):\n"
        f"{_bullets(d['claim_id'] + ': ' + d['reason'] for d in deferred)}",
        label="reporter", options=_opts(args, "reporter"))
    return {"verdict": verdict, "claims": rows, "table": table, "caps": caps, "caps_md": caps_md,
            "experiments": states, "audits": audits, "tuned_audits": tuned_audits, "deferred": deferred,
            "figure_reads": fig_records, "lead_decisions": [d for s in states for d in s.get("decisions") or []],
            "rejected_claims": rejected, "report": report}
