"""Count teammate bash calls that wrapped themselves in `bwrap --args` (copied the sandbox wrapper). Arg: history.jsonl."""
import json, sys
L = [json.loads(l) for l in open(sys.argv[1])]
total = nested = 0
first = None
for i, d in enumerate(L):
    if d.get("event_type") != "chat.tool_call" or d.get("role") != "teammate":
        continue
    tc = d.get("tool_call") or {}
    if tc.get("name") != "bash":
        continue
    total += 1
    s = tc.get("description") or tc.get("arguments") or ""
    if "bwrap --args" in s:
        nested += 1
        if first is None:
            first = i
print("teammate bash calls:", total, "self-wrapped in bwrap:", nested, "first at event", first)
