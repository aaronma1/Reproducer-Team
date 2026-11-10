"""Classify the builders' bash calls in one session by what they did, with the characters each kind put in history."""
import json, re, sys, collections
L = [json.loads(l) for l in open(sys.argv[1])]
KINDS = [
    ("look around (ls/find/tree/pwd/which/wc)", r"^(ls|find|tree|pwd|which|wc|du|stat|file|nproc)\b"),
    ("read a file (cat/head/tail/sed -n/grep)", r"^(cat|head|tail|sed -n|grep|rg|awk|less|jq)\b"),
    ("read the paper text (pdf/pypdf/paper.txt)", r"pypdf|PdfReader|pdftotext|paper\.txt|\.pgtxt|pgtxt|paper\.pdf"),
    ("env / install (uv venv/pip/import check)", r"uv (venv|pip|sync|add)|pip install|-c ['\"]import \w+['\"]|__version__|python3? -V"),
    ("write a file via shell (heredoc/echo >/cp/mkdir)", r"<<|^(echo|printf).*>|^cp |^mkdir|^touch|^rm "),
    ("run code (python script / -c)", r"python3?\b|\.venv/bin/python"),
]
calls = []
pending = {}
builders = {(d.get("event") or {}).get("member_id") for d in L
            if (d.get("event") or {}).get("agent_template") == "builder"}  # from team.member.spawned events
for d in L:
    m = d.get("member_name") or ""
    if m not in builders and "builder" not in m:
        continue
    if d.get("event_type") == "chat.tool_call":
        tc = d.get("tool_call") or {}
        if tc.get("name") != "bash":
            pending[m] = None
            continue
        try:
            cmd = json.loads(tc.get("description") or "{}").get("command", "")
        except ValueError:
            cmd = ""
        pending[m] = [m, cmd, len(tc.get("description") or ""), 0, ""]
        calls.append(pending[m])
    elif d.get("event_type") == "chat.tool_result" and pending.get(m):
        r = d.get("result")
        s = r if isinstance(r, str) else json.dumps(r)
        pending[m][3] = len(s)
        pending[m][4] = s
        pending[m] = None
kinds = collections.Counter(); size = collections.Counter(); ex = {}
for m, cmd, cs, rs, res in calls:
    body = re.sub(r"^cd \S+ (&&|;) *", "", cmd.strip())          # the leading cd to a directory
    k = next((name for name, rx in KINDS if re.search(rx, body)), "other")
    if k.startswith("run code") and re.search(KINDS[3][1], body):
        k = KINDS[3][0]
    kinds[k] += 1; size[k] += cs + rs
    ex.setdefault(k, body[:110].replace("\n", " "))
print(f"{len(calls)} builder bash calls, {sum(size.values())} chars (call + result)\n")
for k, n in kinds.most_common():
    print(f"{n:4}  {size[k]:7} chars  {k}\n        e.g. {ex[k]}")
cd = sum(1 for c in calls if re.match(r"^cd \S+", c[1].strip()))
print(f"\n{cd} of {len(calls)} start with `cd <absolute dir> &&` ({sum(len(re.match(r'^cd (\S+)', c[1].strip()).group(1)) for c in calls if re.match(r'^cd \S+', c[1].strip()))} chars of paths in calls alone)")
fails = sum(1 for c in calls if re.search(r"Exit Code: [1-9]", c[4]))
print(f"{fails} returned a non-zero exit code")
