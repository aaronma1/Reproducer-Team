"""Paper reproducer (simplified, 2026-11-03).

lead (claims, then one spec per experiment) -> per experiment: builder <-> reviewer rounds (pilot, then full runs)
-> evaluator computes every claim from the run files -> audit recomputes each claim independently
-> tuner rounds on the claims that fell short (optional) -> rule verdicts -> judge -> reporter.

Every agent may read the paper (view_page renders a page as an image), so builds are not blind to the paper's numbers;
the results are still computed twice from the raw run files, by the evaluator and an independent auditor, and a
disagreement caps the claim. The rule verdicts are pure python; the judge agent sets the final verdicts with them, the
caps and the audits, and every departure from a rule verdict is shown. Prompts state goals, not procedures (2026-11-04).

args: {"paper": str, "compute": "remote" | "local", "gpu_hours": 8, "max_tries": 3, "build_rounds": 5,
       "tune": false, "tune_rounds": 4, "tune_trials": 4, "smoke": false, "max_claims": null,
       "models": {"cheap"?, "strong"?}}

Workers (lead, builder, evaluator, tuner) run on the `strong` tier and the agents that check them (reviewer, audit,
judge) on `cheap`, so a checker is never the same model as the worker it checks.

Imports only `swarmflow` and the standard library, so it also runs inside JiuwenSwarm.
"""
import json
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


def judge_value(measured, paper, rel_tol=REL_TOL):
    if measured is None or paper is None:
        return "could_not_test"
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
        return judge_value(item.get("value"), claim.get("paper_value"), _tol(claim))
    if claim["kind"] == "predicate":
        return judge_predicate(item.get("value"), claim.get("op"), claim.get("target"), _tol(claim))
    return judge_ordering(item.get("a"), item.get("b"), claim.get("higher_is_better", True), item.get("diff"))


def judge_tuned(claim, item):
    """A tuned result aims at the paper's own number for the method: an ordering with `paper_a` also needs the
    method arm's mean within tolerance of it, so beating a weak baseline is not a match."""
    v = judge_item(claim, item)
    if claim["kind"] == "ordering" and claim.get("paper_a") is not None:
        a = (item or {}).get("a") or {}
        v = cap(v, judge_value(a.get("mean"), claim["paper_a"], _tol(claim)))
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
    """measured / paper per compared arm of an ordering (not judged), so a setup gap that moves both arms shows."""
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


def combine_reads(claim, reads):
    """(claim with its paper numbers set from the figure readings, one record per field). Two readings are averaged;
    the uncertainty is the larger reported one, or half their gap if that is larger, and it widens the claim's
    tolerance (a number read off a plot is only as exact as the reading). A field nobody read keeps the lead's value."""
    out, records = dict(claim), []
    for field in figure_fields(claim):
        got = [(float(r["value"]), _unc(r)) for read in reads if read
               for r in read.get("readings") or [] if r.get("field") == field]
        rec = {"claim_id": claim["id"], "field": field, "lead": claim.get(field), "reads": got}
        if not got:
            records.append({**rec, "value": claim.get(field), "uncertainty": None, "agree": None})
            continue
        vals = [v for v, _ in got]
        value = sum(vals) / len(vals)
        gap = (max(vals) - min(vals)) / 2
        unc = max([u for _, u in got] + [gap])
        agree = len(got) < 2 or abs(vals[0] - vals[1]) <= got[0][1] + got[1][1] + 1e-12
        trial = {**out, field: value}
        if unit_problems(trial):  # a reading in the wrong units is not used
            records.append({**rec, "value": claim.get(field), "uncertainty": None, "agree": False,
                            "note": "readings not in the metric's units; kept the lead's value"})
            continue
        out = trial
        if value:
            out["rel_tol"] = max(float(out.get("rel_tol") or 0.0), unc / abs(value))
        records.append({**rec, "value": value, "uncertainty": unc, "agree": agree})
    return out, records


def figure_section(records):
    if not records:
        return ""
    out = ["## Paper numbers read off figures (written by the flow)", ""]
    for r in records:
        reads = ", ".join(f"{v:.4g} ± {u:.2g}" for v, u in r["reads"]) or "none"
        val = "kept the lead's value" if r["uncertainty"] is None else f"{r['value']:.4g} ± {r['uncertainty']:.2g}"
        out.append(f"- `{r['claim_id']}` {r['field']}: lead {r['lead']}, readings {reads} -> {val}"
                   + ("" if r["agree"] in (True, None) else " (the readings disagree; tolerance widened to cover both)"))
    return "\n".join(out) + "\n"


def retry_prompt(what, attempt, probs, rules):
    return (f"A deterministic check rejected that {what} (try {attempt}):\n{_bullets(probs)}\n"
            + (f"Every {what} must meet all of these rules:\n{_bullets(rules)}\n" if rules else "")
            + f"Send the whole corrected {what}.")


def blocking(review):
    """Blocking issues of a review. No usable review counts as one blocking issue."""
    if not review:
        return [{"issue": "the reviewer gave no usable answer", "evidence": "", "action": "review again", "blocking": True}]
    return [i for i in review.get("issues") or [] if i.get("blocking")]


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
          "source": {"enum": ["table", "text", "figure"]}, "figure": S}
CLAIMS = obj({"arms": arr(obj({"name": S, "description": S})),
              "claims": arr(obj(_CLAIM, [k for k in _CLAIM if k not in ("paper_a", "paper_b", "rel_tol", "source", "figure")]))})
FIELDS = ["paper_value", "paper_a", "paper_b"]
FIGREAD = obj({"claim_id": S, "page": {"type": "integer"},
               "readings": arr(obj({"field": {"enum": FIELDS}, "raw": N, "value": {"type": "number"},
                                    "uncertainty": {"type": "number"}, "how": S})), "notes": S})
EXPERIMENT = obj({
    "id": S, "arms": arr(obj({"name": S, "description": S})), "procedure": S,
    "seeds": {"type": "integer"}, "seeds_reason": S,
    "metrics": arr(obj({"name": S, "definition": S})),
    "claims": arr(obj({"claim_id": S, "metric": S})),
    "acceptance": arr(S)})
SPECS = obj({"experiments": arr(EXPERIMENT), "deferred": arr(obj({"claim_id": S, "reason": S}))})
REVIEW = obj({"issues": arr(obj({"issue": S, "evidence": S, "action": S, "blocking": B}))})
BUILD = obj({"stage": {"enum": ["pilot", "full", "failed"]}, "job_ids": arr(S), "run_files": arr(S), "notes": S})
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
         "higher_is_better": False, "better": None, "worse": None, "anchors": ["Table 1"], "source": "table"},
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
    "acceptance": ["20 run files per arm, seeds 0..19", "the update rules in the code match the arms' descriptions"]}],
    "deferred": []}


def _example(x):
    return "Example of the format, for a different, made-up paper (do not copy its content):\n" + _j(x)


# =================================================================== helpers ==

ROLE_TIER = {"lead": "strong", "figure": "strong", "figure-check": "cheap", "builder": "strong", "reviewer": "cheap", "evaluator": "strong", "audit": "cheap",
             "tuner": "strong", "judge": "cheap", "reporter": "cheap"}


def _opts(args, role):
    model = (args.get("models") or {}).get(ROLE_TIER[role])
    return {"model": model} if model else None


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
            'learning curve of each metric that changes during training}}. A run that did not finish has no file.')


def _compute_note(args):
    if _local(args):
        note = ("There is no GPU and no gpurun. Run jobs with python in your shell, each command finishing within "
                "about 8 minutes (split long work into chunks and keep partial results on disk).")
    else:
        note = ("Run experiments on the GPU with the gpurun tools (gpu_submit, gpu_wait, gpu_output, gpu_download). "
                "Your own shell has no network and no GPU, but its python has CPU torch and jax for tiny checks: debug "
                "there first. Put a pyproject.toml with pinned versions next to your code, submit the smallest "
                "directory a job needs, batch seeds into as few jobs as possible (jobs run one at a time on a queue "
                "shared with other teams), and gpu_download each job's result files into your own directory.")
    if args.get("smoke"):
        note += (" SMOKE TEST: cut every experiment to at most 3 seeds and at most 2% of the paper's steps; every shell "
                 "command must finish within 60 seconds. Record the cut.")
    return note


class _Turns:
    """A session whose turns are labelled base, base#2, ...; until one turn has answered, a retry carries the
    original request too (a skipped turn leaves nothing in the session)."""

    def __init__(self, base, **kw):
        self.base, self.n, self.sess = base, 0, agent_session(**kw)
        self.first, self.answered = None, False

    async def send(self, prompt, schema):
        self.n += 1
        if self.first is None:
            self.first = prompt
        elif not self.answered:
            prompt = f"{self.first}\n\n{prompt}"
        ans = await self.sess.send(prompt, schema=schema, options={"label": _try(self.base, self.n)})
        self.answered = self.answered or ans is not None
        return ans


def public_claim(c):
    """What blind agents may see of a claim: no paper numbers."""
    return {k: c.get(k) for k in ("id", "text", "kind", "metric", "higher_is_better", "better", "worse", "op")}


def _result_fields(c, metric):
    if c["kind"] == "ordering":
        return (f"{c['id']} (ordering, metric `{metric}`): `a` = mean over seeds and 95% CI (lo, hi) for arm "
                f"{c.get('better')}, `b` = the same for {c.get('worse')}, `diff` = mean and 95% CI of the paired "
                f"difference a - b over the seeds both ran (null if they share none); `value` null")
    return f"{c['id']} ({c['kind']}, metric `{metric}`): `value` = the mean over seeds; `a`, `b`, `diff` null"


def _stop(reason, **extra):
    return {"verdict": "could_not_test", "reason": reason, "claims": [], "table": "", "caps": [],
            "caps_md": f"Stopped: {reason}\n", **extra}


# ===================================================================== steps ==

READ_PROMPT = (
    "Read scope.txt, then paper.pdf. List the arms: every method or baseline the experiments in scope run, each with "
    "a lowercase slug name (e.g. `ppo`, `double-q`) and a one-line description. Then extract the claims the scope "
    "asks to reproduce. If the paper proposes a method or algorithm, its main claims are the benchmark numbers that "
    "method produces (its score, return, accuracy or error on each benchmark in scope): a `value` claim for each, "
    "then any comparison with baselines. Per claim: a lowercase slug id; kind `value` (a number), `ordering` (X "
    "beats Y; `better` and `worse` are arm names) or `predicate` (op `~=` or `>=` against a target, for a trend such "
    "as a loss that keeps falling); the metric; where it appears (anchors). For an ordering, also give `paper_a` and "
    "`paper_b`, the paper's value for `better` and for `worse` (from a table, or read off a plot; null if not "
    "reported). Every paper number is in the units of the claim's own metric: if the metric is normalized, "
    "normalize the paper's number the same way. Say where each claim's numbers come from in `source`: `table`, "
    "`text`, or `figure` when they are only in a plot; for a figure, `figure` names it exactly (figure and panel, "
    "page, which curves, and where to read them, e.g. at the end of training). Numbers only in a plot are read off "
    "it again afterwards by looking at the figure; you can look at a page yourself with view_page.\n\n"
    + _example(EXAMPLE_CLAIMS))
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

    def ask(c, role):
        fields = {"paper_value": "the number for the method in `text`",
                  "paper_a": f"the value of arm `{c.get('better')}`", "paper_b": f"the value of arm `{c.get('worse')}`"}
        return agent(
            f"Read the paper's own number(s) for this claim off its figure. Find the figure's page (read_file "
            f"paper.pdf or grep for the caption), then look at it with view_page, cropping to the panel. Read each "
            f"curve where the claim says (e.g. the end of training), using the axis ticks. For each field give the raw "
            f"reading in the plot's units, the value in the claim's metric units (apply the normalization the paper "
            f"states, if the metric is normalized), and an honest uncertainty (half-width, in the metric's units) from "
            f"the tick spacing, line width and any shaded band. Explain in `how` which curve and point you read.\n"
            f"Fields: {_j({k: fields[k] for k in figure_fields(c)})}\nWhere: {c.get('figure') or c.get('anchors')}\n"
            f"Claim:\n{_j(public_claim(c))}",
            label=f"{role}:{c['id']}", schema=FIGREAD, options=_opts(args, role))
    reads = await parallel([lambda c=c, role=role: ask(c, role) for c in todo for role in ("figure", "figure-check")])
    out, records = {c["id"]: c for c in claims}, []
    for i, c in enumerate(todo):
        out[c["id"]], recs = combine_reads(c, reads[2 * i:2 * i + 2])
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
        "in `acceptance`. The builders read the paper themselves.\n\nClaims:\n" + _j(claims)
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
    return {k: exp[k] for k in ("id", "arms", "procedure", "seeds", "seeds_reason", "metrics", "acceptance")}


async def build_experiment(args, exp, claims):
    """Builder <-> reviewer rounds: a pilot, then the full runs. Returns the experiment's review state."""
    eid, rounds = exp["id"], int(args.get("build_rounds", 5))
    code_dir = f"repo/arms/{eid}"
    fmt = run_format("repo/arms")
    builder = _Turns(f"builder:{eid}", options=_opts(args, "builder"),
                     instructions=f"You reproduce experiment `{eid}` in {code_dir}/. {_compute_note(args)}")
    brief = {**exp, "claims": [c for c in claims if c["id"] in {x["claim_id"] for x in exp["claims"]}]}
    prompt = (f"Reproduce this experiment from paper.pdf (and scope.txt): implement it in {code_dir}/ as the paper "
              f"describes, run a short pilot first and report stage `pilot`; the full runs come after the pilot is "
              f"reviewed. Write the run files in this format:\n{fmt}\n\nExperiment:\n{_j(brief)}")
    stage, open_issues = None, ["never reviewed"]
    for rnd in range(1, rounds + 1):
        built = await builder.send(prompt, schema=BUILD)
        if not built or built["stage"] == "failed":
            prompt = "That did not work. Fix what failed and report again (`pilot` first if no pilot passed review)."
            open_issues = ["the builder reported failure" if built else "the builder gave no answer"]
            continue
        stage = built["stage"]
        review = await agent(
            f"Review experiment `{eid}` at stage `{stage}`: read paper.pdf, the code in {code_dir}/ and the run "
            f"files, and judge whether this reproduces the paper's experiment faithfully and whether the results "
            f"look right. Report each problem with evidence and the fix; mark it blocking if it should be fixed "
            f"before going on.\nRun format:\n{fmt}\nExperiment:\n{_j(brief)}\nRun files: {built['run_files']}\n"
            f"Builder notes: {built['notes']}",
            label=_try(f"reviewer:{eid}", rnd), schema=REVIEW, options=_opts(args, "reviewer"))
        issues = blocking(review)
        open_issues = [i["issue"] for i in issues]
        notes = [i for i in (review or {}).get("issues") or [] if not i.get("blocking")]
        if issues:
            prompt = ("The reviewer found blocking problems. Fix them, re-run what they change, and report again at "
                      f"stage `{stage}`:\n" + format_issues(issues))
            continue
        if stage == "full":
            return {"experiment": eid, "ran": True, "reviewed": True, "rounds": rnd, "open": []}
        prompt = (f"The pilot passed review. Now run the full experiment, download every run file and report stage "
                  f"`full`." + (f"\nReviewer notes:\n{format_issues(notes)}" if notes else ""))
    return {"experiment": eid, "ran": stage == "full", "reviewed": False, "rounds": rounds, "open": open_issues}


async def evaluate(args, exps, by_id, root):
    """{claim id: result}: the evaluator computes every claim from the run files under `root`."""
    want = [(by_id[c["claim_id"]], c["metric"]) for e in exps for c in e["claims"]]
    fields = _bullets(_result_fields(c, m) for c, m in want)
    prompt = (f"Compute every claim below from the raw run files with your own code in repo/eval/, as the paper "
              f"defines the metric. Fill the fields exactly as listed (a missing number is null) and list the run "
              f"files you used in `sources`.\nRun files:\n{run_format(root)}\n\nFields:\n{fields}\n\n"
              f"Experiments:\n{_j([_exp_brief(e) for e in exps])}")
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
    audits = await parallel([lambda c=c, m=m: agent(
        f"Independently recompute this claim from the raw run files with your own code (a python heredoc in your "
        f"read-only shell), not anyone else's. Fill the fields exactly as follows and list the run files you used:\n- {_result_fields(c, m)}\n"
        f"Run files:\n{run_format(root)}\nClaim:\n{_j(public_claim(c))}",
        label=f"audit:{c['id']}{suffix}", schema=AUDIT, options=_opts(args, "audit")) for c, m in want])
    return {c["id"]: a for (c, _), a in zip(want, audits)}


async def tune(args, exps, short, blind):
    """(best tuned result per claim, rounds used) after up to `tune_rounds` rounds towards the paper's numbers."""
    phase("Tune")
    rounds, trials = int(args.get("tune_rounds", 4)), int(args.get("tune_trials", 4))
    metric = {c["claim_id"]: c["metric"] for e in exps for c in e["claims"]}
    fields = _bullets(_result_fields(c, metric[c["id"]]) for c in short)
    sess = _Turns("tuner", options=_opts(args, "tuner"))
    prompt = (
        f"The blind reproduction is finished. These claims did not match the paper. Read paper.pdf and scope.txt, and "
        f"tune the experiment's settings, as a researcher would, until each claim matches the paper's own number: "
        f"`paper_value` for a value claim and, for an ordering, `paper_a` for the method arm (`better`) and "
        f"`paper_b` for the baseline. An ordering that holds while the method's score is far off is not a match. "
        f"You may change any "
        f"setting, including one the paper states. Work in repo/tuned/<experiment>/ (copy the code from repo/arms/, "
        f"which you must not change), try at most {trials} settings per round, and write every setting you try and "
        f"its outcome to repo/tuned/<experiment>/trials.json. Write the best setting's runs, all seeds, as run files:\n"
        f"{run_format('repo/tuned')}\n{_compute_note(args)}\n\nAnswer for the best setting so far, fields exactly as "
        f"listed; `trials` counts every setting tried so far, `changes` lists every value that differs from the blind "
        f"run:\n{fields}\n\nClaims, with the paper's numbers:\n{_j(short)}\nBlind results:\n"
        f"{_j({c['id']: blind.get(c['id']) for c in short})}\nExperiments:\n{_j([_exp_brief(e) for e in exps])}")
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


async def final_verdicts(args, claims, exps, rows, audits, caps):
    ids = [r["id"] for r in rows]
    prompt = (
        "You are the final judge. Set each claim's final verdict: reproduced, partial, not_reproduced or "
        "could_not_test. Read paper.pdf and scope.txt, and check the run files in repo/ as far as you need. The rule "
        "verdicts and caps below come from fixed checks: they are your main evidence, but the decision is yours. If "
        "you depart from a rule verdict or a cap, say exactly why, citing a file or a number. Judge what was "
        "measured. A large gap in 'scale' means the setup may differ from the paper's. A row with `tuned` was matched "
        "after a hyperparameter search, which the user allows: it may be `reproduced` if its audited numbers match.\n\n"
        "Rule results (JSON):\n" + json.dumps(rows, sort_keys=True) + "\nEnd of rule results.\n\nCaps:\n" + _j(caps)
        + "\n\nClaims:\n" + _j([c for c in claims if c["id"] in ids]) + "\nExperiments:\n" + _j(exps)
        + "\nAudits:\n" + _j({i: audits.get(i) for i in ids}))
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

async def run(args):
    phase("Lead")
    lead = _Turns("lead", options=_opts(args, "lead"))
    arms, claims, rejected = await read_claims(args, lead)
    if not claims:
        return _stop("the lead found no usable claims", rejected=rejected)
    claims, fig_records = await read_figures(args, claims)
    written = await write_specs(args, lead, arms, claims)
    if written is None:
        return _stop("the lead gave no valid experiments", rejected=rejected)
    exps, deferred = written
    by_id = {c["id"]: c for c in claims}
    judged = [by_id[c["claim_id"]] for e in exps for c in e["claims"]]
    metric = {c["claim_id"]: c["metric"] for e in exps for c in e["claims"]}
    exp_of = {c["claim_id"]: e["id"] for e in exps for c in e["claims"]}

    phase("Build")
    states = await parallel([lambda e=e: build_experiment(args, e, claims) for e in exps])
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
        return _stop("no experiment finished its runs", experiments=states)

    phase("Evaluate")
    results = await evaluate(args, exps, by_id, "repo/arms")
    audits = await audit_all(args, [(c, metric[c["id"]]) for c in judged], "repo/arms")

    tuned, tune_rounds, tuned_audits = {}, 0, {}
    short = [c for c in judged if judge_tuned(c, results.get(c["id"])) != "reproduced"]
    if args.get("tune") and not args.get("smoke") and short:
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
    verdicts = await final_verdicts(args, claims, exps, rows, audits, caps)
    for r in rows:
        r["rule_verdict"] = r["verdict"]
        if verdicts and r["id"] in verdicts:
            r["verdict"], r["judge_reason"] = verdicts[r["id"]]
    if not verdicts:
        caps = caps + [{"experiment": None, "ceiling": "reproduced",
                        "reason": "the judge gave no valid answer; rule verdicts stand"}]
    verdict = aggregate([r["verdict"] for r in rows])
    table = build_table(rows)
    caps_md = caps_section(rows, caps) + tuning_section(rows) + figure_section(fig_records)

    phase("Report")
    report = await agent(
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
            "figure_reads": fig_records,
            "rejected_claims": rejected, "report": report}
