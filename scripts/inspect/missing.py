"""Missing-path errors (cannot access / No such file / Permission denied) in a session's tool results, grouped by path,
with the command that caused each. Args: history.jsonl, the chat dir to abbreviate as <CHAT>."""
import json, re, sys, collections
L = [json.loads(l) for l in open(sys.argv[1])]
CHAT = sys.argv[2]
last_call = {}
hits = collections.defaultdict(list)
rx = re.compile(r"(?:cannot access '?|can't cd to |cd: |cannot open '?)(/[^\s:']+|~[^\s:']*|\.[^\s:']*)[^\n]*?(No such file|Permission denied|Read-only)")
for i, d in enumerate(L):
    m = d.get("member_name") or d.get("role")
    if d.get("event_type") == "chat.tool_call":
        last_call[m] = (i, json.dumps(d.get("tool_call")))
    if d.get("event_type") != "chat.tool_result":
        continue
    s = d.get("result") if isinstance(d.get("result"), str) else json.dumps(d.get("result"))
    for mm in rx.finditer(s):
        path = mm.group(1).replace(CHAT, "<CHAT>")
        hits[(path, mm.group(2))].append((i, m, last_call.get(m, (None, ""))[1]))
for (p, why), lst in sorted(hits.items(), key=lambda x: -len(x[1])):
    i, m, c = lst[0]
    print(len(lst), why, p, "| first", i, str(m)[-28:])
    print("     ", c.replace(CHAT, "<CHAT>")[:330])
