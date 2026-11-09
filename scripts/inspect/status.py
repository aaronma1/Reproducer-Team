"""A session's events in order, without the per-call LLM noise: stage changes, members, tool calls and results.
Args: history.jsonl [first event index]."""
import json, sys, datetime, collections
p = sys.argv[1]
L = [json.loads(l) for l in open(p)]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0

def t(ts):
    if isinstance(ts, (int, float)):
        return datetime.datetime.fromtimestamp(ts if ts < 1e11 else ts / 1000).strftime("%H:%M:%S")
    return str(ts)[11:19]

print(len(L), "events")
cnt = collections.Counter((d.get("role"), d.get("event_type")) for d in L)
for k, v in sorted(cnt.items(), key=lambda x: -x[1])[:15]:
    print(v, k)
print()
for i in range(start, len(L)):
    d = L[i]
    et = d.get("event_type") or d.get("role")
    if et in ("chat.llm_call_start", "chat.llm_call_end", "context.usage"):
        continue
    body = d.get("tool_call") or d.get("result") or d.get("event") or d.get("content") or d.get("reasoning_content") or ""
    s = json.dumps(body, ensure_ascii=False)
    print(i, t(d.get("timestamp")), d.get("role"), d.get("member_id") or d.get("agent_name") or "", et, s[:260])
