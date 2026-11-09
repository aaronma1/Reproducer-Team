"""How do the two DeepSeek v4.1 flash endpoints treat reasoning sent back in history?

For each endpoint (from JiuwenSwarm's config, keys never printed), send the same short conversation:
  A. a past assistant turn with ~3k tokens of filler in `reasoning_content` (what openjiuwen sends for deepseek models)
  B. the same without it
  C. the same filler in `reasoning` (OpenRouter's own field name)
and compare the provider's prompt_tokens. Also report where the reply's own reasoning comes back.
"""
import json, os, re, sys, urllib.request
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
envfile = Path(sys.argv[1]).parent.parent / ".env"
env = dict(os.environ)
if envfile.exists():
    for line in envfile.read_text().splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
        if m:
            env.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))


def find_models(node):
    if isinstance(node, dict):
        mcc = node.get("model_client_config")
        if isinstance(mcc, dict) and "deepseek-v4.1-flash" in str(mcc.get("model_name", "")):
            yield node
        for v in node.values():
            yield from find_models(v)
    elif isinstance(node, list):
        for v in node:
            yield from find_models(v)


def resolve(v):
    m = re.fullmatch(r"\$\{([^}:]+)(?::-([^}]*))?\}", str(v or ""))
    return env.get(m.group(1), m.group(2) or "") if m else (v or "")


FILLER = " ".join(f"step{i}: consider the gradient of the loss with respect to head {i % 7}." for i in range(300))
seen = set()
for entry in find_models(cfg):
    mcc = entry["model_client_config"]
    key = (mcc["api_base"], mcc["model_name"])
    if key in seen:
        continue
    seen.add(key)
    api_key = resolve(mcc.get("api_key"))
    effort = (entry.get("model_config_obj") or {}).get("reasoning_level") or None
    print(f"\n=== {mcc['model_name']} @ {mcc['api_base']}  (reasoning_level={effort!r}, key {'set' if api_key else 'MISSING'})")

    def ask(extra, label):
        asst = {"role": "assistant", "content": "The answer is 4."}
        asst.update(extra)
        body = {"model": mcc["model_name"], "max_tokens": 400,
                "messages": [{"role": "user", "content": "What is 2+2?"}, asst,
                             {"role": "user", "content": "Reply with just: ok"}]}
        if effort:
            body["reasoning_effort"] = effort
        req = urllib.request.Request(mcc["api_base"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            print(f"  {label:34} HTTP {e.code}: {e.read()[:200]!r}")
            return
        msg = d["choices"][0]["message"]
        u = d.get("usage") or {}
        fields = {k: len(str(msg[k])) for k in ("reasoning_content", "reasoning", "thinking") if msg.get(k)}
        print(f"  {label:34} prompt_tokens={u.get('prompt_tokens')!s:6} completion={u.get('completion_tokens')!s:5} "
              f"reply={str(msg.get('content'))[:20]!r} reasoning fields in reply: {fields or 'none'}"
              f" reasoning_tokens={((u.get('completion_tokens_details') or {}).get('reasoning_tokens'))}")

    for n in range(int(sys.argv[2]) if len(sys.argv) > 2 else 1):
        ask({}, f"B{n}: no reasoning in history")
        ask({"reasoning_content": FILLER}, f"A{n}: reasoning_content (openjiuwen)")


print(f"\nfiller is {len(FILLER)} chars")
