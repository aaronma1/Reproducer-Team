"""One pass over today's trace file: for every session, each builder's input tokens per call, in time order.
Prints the context at fixed call counts so runs can be compared at the same point in their work."""
import json, sys, collections
seq = collections.defaultdict(list)          # (session, member) -> [(t, tokens)]
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if "input_tokens" not in line or "builder" not in line:
        continue
    try:
        d = json.loads(line)
    except ValueError:
        continue
    for rs in d.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                at = {a["key"]: next(iter(a["value"].values()), None) for a in sp.get("attributes", [])}
                tok = at.get("gen_ai.usage.input_tokens")
                who = (at.get("openjiuwen.trajectory.subject_id") or at.get("gen_ai.agent.name") or "").split(":")[-1]
                if tok is None or "builder" not in who:
                    continue
                seq[(at.get("gen_ai.conversation.id"), who)].append((int(sp.get("startTimeUnixNano", 0)), int(tok)))
marks = (1, 10, 20, 30, 40, 50, 60, 80, 100)
print(f"{'session':30} {'builder':42} calls " + " ".join(f"@{m:<4}" for m in marks) + "  total_in")
for (sess, who), v in sorted(seq.items(), key=lambda x: min(t for t, _ in x[1])):
    v.sort()
    toks = [t for _, t in v]
    cells = " ".join(f"{toks[m - 1] / 1000:5.0f}k" if len(toks) >= m else "    - " for m in marks)
    print(f"{sess[:30]:30} {who[-42:]:42} {len(toks):5} {cells}  {sum(toks) / 1e6:6.2f}M")
