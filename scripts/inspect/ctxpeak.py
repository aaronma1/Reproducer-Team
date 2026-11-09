"""Per-member LLM input tokens (peak, last, calls, sum) for one session, from JiuwenSwarm's OTel trace file."""
import json, sys, collections
sess, path = sys.argv[1], sys.argv[2]
stats = collections.defaultdict(lambda: {"calls": 0, "peak": 0, "sum": 0, "last": 0, "first": 0})
keys = collections.Counter()
for line in open(path, encoding="utf-8", errors="replace"):
    if sess not in line or "input_tokens" not in line:
        continue
    try:
        d = json.loads(line)
    except ValueError:
        continue
    for rs in d.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                at = {a["key"]: next(iter(a["value"].values()), None) for a in sp.get("attributes", [])}
                if at.get("gen_ai.conversation.id") != sess:
                    continue
                tok = at.get("gen_ai.usage.input_tokens")
                if tok is None:
                    continue
                keys[sp.get("name")] += 1
                who = (at.get("openjiuwen.trajectory.subject_id") or at.get("gen_ai.agent.name") or "?").split(":")[-1]
                t = int(tok)
                s = stats[who]
                s["first"] = s["first"] or t; s["calls"] += 1; s["sum"] += t; s["peak"] = max(s["peak"], t); s["last"] = t
print(dict(keys))
for who, s in sorted(stats.items(), key=lambda x: -x[1]["sum"]):
    print(f"{who[:60]:60} calls={s['calls']:4} peak={s['peak']:7} last={s['last']:7} sum={s["sum"]:9} first={s["first"]:6}")
