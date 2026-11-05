"""Role agents: a tool-calling loop over an OpenAI-compatible endpoint (OpenRouter).

Chosen over openjiuwen's ReActAgent to keep full control of the `submit` tool, schema validation and
per-call accounting (see NOTES.md). Every file and shell action goes through a `tools.Scope`, so what
a role may touch is decided there, not by anything in a prompt.

Context policy (from the JiuwenSwarm cluster runs, where these models ran at 80-180k tokens per call
with ~96% provider prompt-cache hits): keep the conversation append-only so the cached prefix stays
valid, and only shrink old tool output when a call approaches the model's context window.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Callable

import httpx
from jsonschema import Draft202012Validator

from .config import Tier
from .tools import PAPER_FILES, AccessDenied, Scope


WEB_FETCH_CHARS = 60000  # text kept from one fetched page
_web_tools: dict = {}


async def _web(name: str, args: dict) -> str:
    """web_search / web_fetch through openjiuwen's free search and page fetcher (no API key needed)."""
    if not _web_tools:
        from openjiuwen.harness.tools.web.fetch_webpage import WebFetchWebpageTool
        from openjiuwen.harness.tools.web.free_search import WebFreeSearchTool
        _web_tools["web_search"] = WebFreeSearchTool(language="en", enabled_engines=("duckduckgo", "bing"))
        _web_tools["web_fetch"] = WebFetchWebpageTool(language="en")
    if name == "web_search":
        return str(await _web_tools[name].invoke({"query": str(args["query"]), "max_results": 8}))
    url = str(args["url"]).strip()
    try:  # plain text (source code, raw files) comes back as is: the HTML extractor would collapse its newlines
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
            r = await http.get(url, headers={"User-Agent": "Mozilla/5.0 (paper-reproducer)"})
        kind = r.headers.get("content-type", "")
        if r.status_code == 200 and "html" not in kind and ("text" in kind or "json" in kind or "xml" in kind):
            body = r.text
            cut = "\n[truncated]" if len(body) > WEB_FETCH_CHARS else ""
            return f"URL: {r.url}\nStatus: 200\nContent-Type: {kind}\n\n{body[:WEB_FETCH_CHARS]}{cut}"
    except httpx.HTTPError:
        pass  # fall through to the page fetcher, which reports the error
    return str(await _web_tools[name].invoke({"url": url, "max_chars": WEB_FETCH_CHARS}))


REASONING_LOG_CHARS = 20000  # reasoning and assistant text kept per model call in the event log


def _reasoning_text(msg: dict) -> str:
    """The model's reasoning for one call, as OpenRouter returns it (`reasoning`, or `reasoning_details` items)."""
    if isinstance(msg.get("reasoning"), str) and msg["reasoning"].strip():
        return msg["reasoning"].strip()
    parts = []
    for d in msg.get("reasoning_details") or []:
        if isinstance(d, dict):
            parts.append(d.get("text") or d.get("summary") or "")
    return "\n".join(p for p in parts if p).strip()


class ModelError(Exception):
    pass


class Client:
    def __init__(self, key: str, base_url: str, concurrency: int = 8, transport: httpx.AsyncBaseTransport | None = None):
        self._key = key
        self._http = httpx.AsyncClient(base_url=base_url, headers={"Authorization": f"Bearer {key}"},
                                       timeout=httpx.Timeout(600.0), transport=transport)
        self._sem = asyncio.Semaphore(concurrency)

    async def chat(self, body: dict) -> dict:
        last = ""
        for attempt in range(6):
            try:
                async with self._sem:
                    r = await self._http.post("/chat/completions", json=body)
                if r.status_code in (408, 429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}"
                elif r.status_code >= 400:
                    raise ModelError(f"HTTP {r.status_code}: {r.text[:300]}".replace(self._key, "***"))
                else:
                    data = r.json()
                    if data.get("choices"):
                        return data
                    last = str(data.get("error", "no choices"))[:200]
            except httpx.TransportError as e:
                last = type(e).__name__
            await asyncio.sleep(min(2 ** attempt, 30))
        raise ModelError(f"gave up after retries: {last}".replace(self._key, "***"))

    async def aclose(self) -> None:
        await self._http.aclose()


# ------------------------------------------------------------------ prompts --

TEAM = """\
The team reproduces one paper's results from scratch, the way a careful researcher would. In order:
- lead: reads the paper and scope.txt, lists the claims to reproduce and groups them into experiments.
- figure:<claim> / figure-check:<claim>: read paper numbers that only exist in plots, by looking at the figure.
- builder:<experiment>: implements and runs one experiment in repo/arms/<experiment>/ and writes one JSON file per run.
- reviewer:<experiment>: checks each builder round against the paper and sends back what must change.
- evaluator: computes every claim from the run files with its own code; audit:<claim> recomputes each one
  independently, and a disagreement caps the claim.
- tuner (optional): tunes settings in repo/tuned/ towards the paper's numbers.
- A fixed rule gives each claim a rule verdict; judge sets the final verdicts; reporter writes report.md.
Everyone may read paper.pdf (view_page shows a page as an image) and scope.txt."""

RULES = """\
Rules for everyone:
- Never state a number you did not read from a file, a command's output or a page you fetched; say where it came from.
- Follow scope.txt. Do not use the paper's own code or a third-party reproduction of it; write your own. If you
  have web tools, use them for the libraries, environments, datasets and reference implementations the paper uses
  or cites; do not open the paper's own code repository or other reproductions of it, and name the URL of every
  fact you take from the web.
- Report every deviation from the paper and every assumption in your answer.
- File tools take workspace-relative paths. The shell starts in /work, the workspace root.
- When done, call `submit` exactly once with the answer in the requested format."""

ROLE_GUIDE = {}


def _fn(name: str, desc: str, props: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": list(props)}}}


GPU_GUIDE = """\
Queue a job on the GPU box ({hw}): `path` is a workspace directory you may read; give the smallest one the job
needs (e.g. repo/arms/<arm>), never one holding datasets or old outputs: it is uploaded on every submit (no
.venv/.git/caches, max 500 MB), and a large upload times out. `commands` run in
order inside it, the first failure stopping the job. The GPU box HAS network, uv and a shared package cache;
your own shell does not. Put a pyproject.toml (pin torch/jax versions) in the uploaded dir and use
commands like ["uv sync", "uv run python train.py --out out"]; a bare `python` is the system interpreter
without packages. Download datasets inside the job from Hugging Face (fast; check the repo's real file names, a
404 makes code fall back to slow mirrors) into $XDG_CACHE_HOME, which persists across jobs, so each dataset is
downloaded once; never re-download it in every job. Print progress as the job runs. Jobs run one at a time
on a shared queue, so batch many runs/seeds into one job, keep each job well under the time you can afford,
and make sure it terminates. Write results to files (e.g. out/) and fetch them with gpu_download. `tag` names
the job for the logs, e.g. <claim_id>/<arm>."""


def gpu_tool_specs(mode: str, hardware: str) -> list[dict]:
    s, job = {"type": "string"}, {"job": {"type": "string"}}
    t = [_fn("gpu_status", "Status of one of this run's GPU jobs.", job),
         _fn("gpu_output", "stdout/stderr of a GPU job so far (works while it runs); tail_lines 0 = all.",
             {**job, "tail_lines": {"type": "integer"}}),
         _fn("gpu_list", "This run's GPU jobs and their status.", {})]
    if mode in ("run", "rw"):
        guide = GPU_GUIDE.format(hw=hardware or "GPU")
        if mode == "run":
            guide += ("\nYou are a reviewer: submit only a short check (minutes) when your own shell lacks a library "
                      "the code needs. You cannot download job files; read the output with gpu_output.")
        t += [_fn("gpu_submit", guide, {"path": s, "commands": {"type": "array", "items": s}, "tag": s}),
              _fn("gpu_wait", "Block until the job finishes or max_wait_s passes (at most 1800). It keeps running "
                              "either way; never resubmit because a wait timed out.",
                  {**job, "max_wait_s": {"type": "integer"}}),
              _fn("gpu_cancel", "Cancel a queued or running GPU job" + (" you submitted." if mode == "run" else "."), job)]
    if mode == "rw":
        t.append(_fn("gpu_download", "Copy files or directories from the job dir (paths relative to it) into `dest`, a "
                                     "workspace directory you may write, keeping their relative paths.",
                     {**job, "paths": {"type": "array", "items": s}, "dest": s}))
    return t


def _cpu_ml_libs() -> bool:
    """Whether the sandboxed shell's python can import torch and jax (true for the default .venv or a sandbox_venv)."""
    import importlib.util

    from . import tools as _tools
    if "sandbox-venv" in _tools.SANDBOX_PATH:
        return True
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "jax"))


def tool_specs(scope: Scope) -> list[dict]:
    pol, s = scope.policy, {"type": "string"}
    t = []
    if pol.read:
        t.append(_fn("read_file", "Read a whole file in the workspace (PDFs come back as text).", {"path": s}))
        t.append(_fn("list_dir", "List a directory in the workspace.", {"path": s}))
        t.append(_fn("grep", "Search files you may read for a regex. `path` is a file or directory, or '' for all.",
                     {"pattern": s, "path": s}))
        if any(r in ("paper.pdf", "") for r in pol.read):
            t.append(_fn("view_page", "Look at a page of paper.pdf as an image: read_file gives a PDF's text only, which "
                                      "keeps a plot's tick labels but not its curves, bars or markers. `page` is 1-based "
                                      "(the text marks pages as '--- page n ---'); `crop` is '' for the whole page or "
                                      "'x0,y0,x1,y1' as fractions of the page (0,0 top left) to zoom into one panel.",
                         {"page": {"type": "integer"}, "crop": s}))
    if pol.write:
        t.append(_fn("write_file", "Create or overwrite a file. Only some paths are writable for your role.",
                     {"path": s, "content": s}))
        t.append(_fn("edit_file", "Replace the one occurrence of `old` with `new` in a writable file.",
                     {"path": s, "old": s, "new": s}))
    if pol.shell:
        pkgs = "numpy, scipy, pandas and CPU-only torch and jax" if _cpu_ml_libs() else "numpy"
        t.append(_fn("shell", f"Run a bash command in /work (the workspace). Python with {pkgs} is available; "
                              "there is no network and no GPU." + (" Your shell is read-only." if pol.shell == "ro" else ""),
                     {"command": s}))
    if pol.gpu:
        t += gpu_tool_specs(pol.gpu, scope.ws.gpu.cfg.hardware if scope.ws.gpu else "")
    if pol.web:
        t.append(_fn("web_search", "Search the web (DuckDuckGo/Bing). Use it to check facts about the libraries, "
                                   "environments, datasets and reference implementations the paper uses or cites, "
                                   "e.g. a package's defaults in a given version. Returns titles, URLs and snippets.",
                     {"query": s}))
        t.append(_fn("web_fetch", "Fetch a web page or raw file as text (for GitHub source, prefer "
                                  "raw.githubusercontent.com URLs).", {"url": s}))
    t.append(_fn("submit", "Finish this turn. Call it exactly once, when done.", {"result": s}))
    return t


def submit_spec(schema_json: dict | None) -> dict:
    """`submit` for one turn. An object answer is declared as an object, so the model sends the JSON itself rather
    than hand-escaping a long document into a string (GLM broke 30 KB spec answers that way, 2026-11-02/03)."""
    if not schema_json or schema_json.get("type") != "object":
        return _fn("submit", "Finish this turn. Call it exactly once, when done.", {"result": {"type": "string"}})
    return _fn("submit", "Finish this turn. Call it exactly once, when done. `result` is your answer as a JSON "
                         "object (not a string) matching the schema in the request.", {"result": {"type": "object"}})


def system_prompt(role: str, instructions: str | None, scope: Scope) -> str:
    pol = scope.policy
    access = (f"read {', '.join(p or 'everything' for p in pol.read) or 'nothing'}; "
              f"write {', '.join(pol.write) or 'nothing'}; shell {pol.shell or 'none'}")
    parts = [f"You are `{role}` in a paper-reproduction team. Your access: {access}. Anything else is refused by the "
             "tools, not just discouraged.", TEAM, RULES]
    if role in ROLE_GUIDE:
        parts.append(f"Your job as {role}: {ROLE_GUIDE[role]}")
    if instructions:
        parts.append(instructions)
    return "\n\n".join(parts)


# --------------------------------------------------------------------- loop --

@dataclass
class Turn:
    text: str | None = None
    structured: object = None
    skipped: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    reason: str = ""  # why the turn was skipped


MAX_VIEWS = 6  # view_page calls per turn: figure readers zoomed into ever thinner slivers 60+ times (2026-11-03)
MAX_BAD = 5  # invalid submits (or bare replies) before a turn is given up
WRAP_UP_STEPS = 15  # model calls before max_steps at which the agent is told to submit
KEEP_RECENT = 8  # when compacting, tool results / written files this recent stay verbatim
COMPACT_AT = 0.85  # compact once a call uses this share of the context window
_STUB_AT = 600


def _shrink_args(raw: str) -> str:
    """Drop a long `content` argument (a file already on disk) from an old write_file call."""
    try:
        args = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(args, dict) and isinstance(args.get("content"), str) and len(args["content"]) > _STUB_AT:
        args["content"] = f"[{len(args['content'])} chars elided; the file is on disk]"
        return json.dumps(args)
    return raw


def _drop_extras(value, schema: dict):
    """A copy of `value` without object keys that `schema` forbids with additionalProperties: false."""
    if isinstance(value, dict) and "properties" in schema:
        props = schema["properties"]
        keep = props if schema.get("additionalProperties") is False else value
        return {k: _drop_extras(v, props.get(k, {})) for k, v in value.items() if k in keep}
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [_drop_extras(v, schema["items"]) for v in value]
    return value


def _close_braces(text: str):
    """`text` parsed after rebalancing its brackets, or None. GLM drops the brace that ends a list item before the
    next one (`...}, {"claim_id": ...`) and then pads the end with extra closers (reppo spec answers, 2026-11-03).
    A tracked bracket stack closes an object that is followed by `, {` (an object never holds a bare object),
    drops closers with nothing to close, closes the objects inside a list that `]` ends, and adds missing closers at
    the end. The caller still checks the result against the schema, so a wrong guess is rejected, not used."""
    out, stack, in_str, esc = [], [], False, False
    n = len(text)
    for i, ch in enumerate(text):
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            while out and out[-1].isspace():
                out.pop()
            if out and out[-1] == ",":  # a trailing comma before a closer
                out.pop()
            want = "{" if ch == "}" else "["
            if not stack or want not in stack:
                continue  # nothing to close: drop it
            while stack[-1] != want:  # close what the model left open inside
                out.append("}" if stack.pop() == "{" else "]")
            stack.pop()
            out.append(ch)
        elif ch == ",":
            k = i + 1
            while k < n and text[k].isspace():
                k += 1
            if stack and stack[-1] == "{" and k < n and text[k] == "{":
                stack.pop()
                out.append("}")
            out.append(ch)
        else:
            out.append(ch)
    out += ["}" if c == "{" else "]" for c in reversed(stack)]
    try:
        return json.loads("".join(out), strict=False)
    except ValueError:
        return None


def _parse(value, validator: Draft202012Validator | None):
    """(ok, parsed-or-error) for a submitted result."""
    if validator is None:
        text = value if isinstance(value, str) else json.dumps(value)
        return (bool(text.strip()), text.strip() or "the result is empty")
    if isinstance(value, str):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip())
        try:
            value = json.loads(text, strict=False)  # strict=False: raw newlines/tabs inside strings are fine
        except ValueError as e:
            value = _close_braces(text)
            if value is None or any(True for _ in validator.iter_errors(value)):
                return False, (f"result is not valid JSON ({e}). Send `result` as a JSON object, not as a string, "
                               "so nothing has to be escaped; close every object before starting the next list item")
    errs = sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))
    if errs:  # an extra field is no reason to lose a finished answer (a gate's whole review was lost to one)
        trimmed = _drop_extras(value, validator.schema)
        if not any(True for _ in validator.iter_errors(trimmed)):
            return True, trimmed
        shown = [f"at {'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message[:200]}" for e in errs[:6]]
        more = f" (and {len(errs) - 6} more)" if len(errs) > 6 else ""
        return False, f"does not match the schema; fix all of these: {'; '.join(shown)}{more}"
    return True, value


class Agent:
    """One conversation. A single-shot call uses a fresh Agent; a session keeps one across turns."""

    def __init__(self, *, label: str, role: str, scope: Scope, client: Client, tier: Tier, instructions: str | None,
                 timeout_s: int, out_limit: int, max_steps: int, read_limit: int = 1_000_000,
                 turn_token_limit: int = 0, on_usage: Callable[[int, float], None],
                 should_stop: Callable[[], bool], on_activity: Callable[[str, str, str], None]):
        self.label, self.scope, self.client, self.tier = label, scope, client, tier
        self.timeout_s, self.out_limit, self.max_steps = timeout_s, out_limit, max_steps
        self.read_limit, self.turn_token_limit = read_limit, turn_token_limit
        self.on_usage, self.should_stop, self.on_activity = on_usage, should_stop, on_activity
        self.messages: list[dict] = [{"role": "system", "content": system_prompt(role, instructions, scope)}]
        self.turns = 0
        self._read_papers: set[str] = set()
        self._keep_ids: set[str] = set()  # tool_call ids whose result must never shrink (paper reads)
        self._keep_whole = False
        self._last_prompt_tokens = 0
        self._images: list[tuple[str, bytes]] = []
        self._views = 0  # pages rendered by view_page, sent after this step's tool results

    async def _tool(self, name: str, args: dict) -> str:
        limit = self.out_limit
        try:
            if name == "read_file":
                path, limit = args["path"], self.read_limit
                if path in PAPER_FILES and path in self._read_papers:  # immutable: don't pay for it twice
                    return f"You already read {path} earlier in this conversation; its content is above and unchanged."
                out = self.scope.read_text(path, self.read_limit)
                if path in PAPER_FILES:
                    self._read_papers.add(path)
                    self._keep_whole = True
            elif name == "list_dir":
                out = "\n".join(self.scope.list_dir(args["path"] or ".")) or "(empty)"
            elif name == "grep":
                out = "\n".join(self.scope.grep(args["pattern"], args.get("path") or "")) or "(no matches)"
            elif name == "write_file":
                self.scope.write_file(args["path"], args["content"])
                out = f"wrote {len(args['content'])} chars to {args['path']}"
            elif name == "edit_file":
                self.scope.edit_file(args["path"], args["old"], args["new"])
                out = f"edited {args['path']}"
            elif name == "view_page":
                self._views += 1
                if self._views > MAX_VIEWS:
                    return (f"limit reached: you have viewed {MAX_VIEWS} images this turn. Answer now from what you saw; "
                            "if you are unsure, give your best reading with a wider uncertainty.")
                crop = str(args.get("crop") or "").strip()
                png = await asyncio.to_thread(self.scope.render_page, "paper.pdf", int(args["page"]), crop)
                what = f"paper.pdf page {int(args['page'])}" + (f", crop {crop}" if crop else "")
                self._images.append((what, png))
                out = f"rendered {what} ({len(png) // 1024} KB); the image follows in the next message"
            elif name in ("web_search", "web_fetch"):
                out = await _web(name, args)
            elif name.startswith("gpu_"):
                out = json.dumps(await asyncio.to_thread(self._gpu, name, args), indent=1)
            elif name == "shell":
                out = await asyncio.to_thread(self.scope.shell, args["command"], timeout=self.timeout_s, max_out=self.out_limit)
            else:
                out = f"error: unknown tool {name}"
        except AccessDenied as e:
            out = f"error: {e}"
        except (KeyError, TypeError) as e:
            out = f"error: bad arguments ({type(e).__name__}: {e})"
        except Exception as e:  # a tool failure goes back to the model; it must not crash the run
            out = f"error: {type(e).__name__}: {e}"
        return out if len(out) <= limit else out[:limit] + "\n[truncated]"

    def _gpu(self, name: str, args: dict):
        from .gpu import GpuError
        sc = self.scope
        try:
            if name == "gpu_submit":
                cmds = args["commands"]
                return sc.gpu_submit(self.label, args["path"], [cmds] if isinstance(cmds, str) else list(cmds),
                                     str(args.get("tag") or ""))
            if name == "gpu_download":
                paths = args["paths"]
                return sc.gpu_download(args["job"], [paths] if isinstance(paths, str) else list(paths), args["dest"])
            if name == "gpu_wait":
                return sc.gpu("run").wait(args["job"], int(args.get("max_wait_s") or 600))
            if name == "gpu_cancel":
                return sc.gpu_cancel(self.label, args["job"])
            if name == "gpu_status":
                return sc.gpu().status(args["job"])
            if name == "gpu_output":
                tl = args.get("tail_lines")
                return sc.gpu().output(args["job"], 200 if tl is None else int(tl))
            if name == "gpu_list":
                return sc.gpu().list()
            return {"error": f"unknown tool {name}"}
        except GpuError as e:
            return {"error": str(e)}

    def _compact(self) -> None:
        """Shrink tool output and written-file contents older than the last KEEP_RECENT steps.

        Only runs near the context window: it changes the conversation prefix, which costs the
        provider's prompt cache. Paper reads are kept whole; the agent cannot get them back cheaply.
        """
        if self._last_prompt_tokens < COMPACT_AT * self.tier.context:
            return
        tools = [m for m in self.messages if m["role"] == "tool"]
        for m in tools[:-KEEP_RECENT]:
            c = m["content"]
            if m["tool_call_id"] not in self._keep_ids and len(c) > _STUB_AT and not c.startswith("[elided"):
                m["content"] = f"[elided {len(c)} chars of old output; re-run or re-read if needed]\n" + c[:300]
        calls = [m for m in self.messages if m["role"] == "assistant" and m.get("tool_calls")]
        for m in calls[:-KEEP_RECENT]:
            for c in m["tool_calls"]:
                c["function"]["arguments"] = _shrink_args(c["function"].get("arguments") or "")

    async def turn(self, prompt: str, schema_json: dict | None) -> Turn:
        validator = Draft202012Validator(schema_json) if schema_json else None
        tail = ("When done, call `submit` with `result` set to your answer as a JSON object (not a string) matching "
                "this JSON Schema:\n" + json.dumps(schema_json) if schema_json else "When done, call `submit` with `result` set to your final answer.")
        self.messages.append({"role": "user", "content": f"{prompt}\n\n{tail}"})
        self._views = 0
        self.turns += 1
        specs, out, bad, idle = tool_specs(self.scope), Turn(), 0, 0
        specs = [submit_spec(schema_json) if t["function"]["name"] == "submit" else t for t in specs]
        out.reason = f"no answer after {self.max_steps} model calls"
        warn_at = max(1, self.max_steps - WRAP_UP_STEPS)
        for step in range(self.max_steps):
            if step == warn_at:  # without this, a thorough reviewer can spend every call and never answer
                self.messages.append({"role": "user", "content": (
                    f"You have {self.max_steps - step} model calls left in this turn. Stop investigating now and call "
                    "`submit` with your answer based on what you have found so far; say what you could not check.")})
            if self.should_stop():
                out.reason = "run budget exhausted"
                break
            if self.turn_token_limit and out.input_tokens + out.output_tokens >= self.turn_token_limit:
                out.reason = f"turn token cap ({self.turn_token_limit}) reached"
                break
            self._compact()
            last = step == self.max_steps - 1
            if last:  # the final call can only answer, so a long review still returns what it found
                self.messages.append({"role": "user", "content": "This is your last model call: call `submit` now."})
            body = {"model": self.tier.id, "messages": self.messages, "usage": {"include": True},
                    "tools": [t for t in specs if t["function"]["name"] == "submit"] if last else specs,
                    "tool_choice": "required" if last else "auto"}
            if self.tier.effort:
                body["reasoning"] = {"effort": self.tier.effort}
            if self.tier.max_tokens:
                body["max_tokens"] = self.tier.max_tokens
            try:
                data = await self.client.chat(body)
            except ModelError as e:
                out.reason = f"model error: {e}"
                break
            u = data.get("usage") or {}
            pin, pout = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
            cost = float(u.get("cost") or 0.0)
            self._last_prompt_tokens = pin
            out.input_tokens += pin
            out.output_tokens += pout
            out.cached_tokens += int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            out.cost += cost
            self.on_usage(pin + pout, cost)
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            thought = _reasoning_text(msg)
            if thought:
                self.on_activity(self.label, "thinking", thought[:REASONING_LOG_CHARS])
            if (msg.get("content") or "").strip() and calls:  # text next to tool calls; a bare reply is parsed below
                self.on_activity(self.label, "says", msg["content"][:REASONING_LOG_CHARS])
            self.messages.append({"role": "assistant", "content": msg.get("content") or "",
                                  **({"tool_calls": calls} if calls else {})})
            if not calls:  # a bare reply: accept it if it already is a valid answer
                ok, val = _parse((msg.get("content") or "").strip(), validator)
                if ok:
                    return self._done(out, val, schema_json)
                idle += 1
                if idle >= MAX_BAD:
                    out.reason = f"replied {MAX_BAD} times without calling submit"
                    break
                self.messages.append({"role": "user", "content": f"Call the `submit` tool with your final answer ({val})."})
                continue
            finished = False
            for c in calls:
                name = c["function"]["name"]
                if finished:
                    self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": "not run: turn already submitted"})
                    continue
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except ValueError:
                    args = {}
                if name == "submit":
                    ok, val = _parse(args.get("result"), validator)
                    self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": "ok" if ok else f"invalid: {val}"})
                    if ok:
                        finished, result = True, val
                    else:
                        bad += 1
                        got = args.get("result")
                        self._save_rejected(got if isinstance(got, str) else json.dumps(got, indent=1) if got is not None
                                            else c["function"].get("arguments") or "", val)
                else:
                    self.on_activity(self.label, name, json.dumps(args)[:2000])
                    self._keep_whole = False
                    res = await self._tool(name, args)
                    if self._keep_whole:
                        self._keep_ids.add(c["id"])
                    self.on_activity(self.label, name + " ->", res[:2000])
                    self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": res})
            if self._images:  # images cannot go in a tool result: they follow all of this step's results
                import base64
                content = []
                for what, png in self._images:
                    content += [{"type": "text", "text": what},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                                                          + base64.b64encode(png).decode()}}]
                self.messages.append({"role": "user", "content": content})
                self._images = []
            if finished:
                return self._done(out, result, schema_json)
            if bad >= MAX_BAD:
                out.reason = f"{MAX_BAD} invalid submits, last: {val}"
                break
        out.skipped = True
        self.on_activity(self.label, "gave up", out.reason)
        return out

    def _save_rejected(self, raw: str, why: str) -> None:
        """Keep a rejected submit next to the run (outside the agents' workspace) so its cause can be read later."""
        try:
            d = self.scope.ws.root.parent / "rejected"
            d.mkdir(exist_ok=True)
            stem = re.sub(r"[^A-Za-z0-9_.#-]+", "_", self.label)
            n = len(list(d.glob(f"{stem}.*.txt"))) + 1
            (d / f"{stem}.{n}.txt").write_text(f"{why}\n\n{raw}")
        except OSError:
            pass  # a diagnostic, never a reason to fail the turn

    @staticmethod
    def _done(out: Turn, val, schema_json) -> Turn:
        if schema_json:
            out.structured = val
        else:
            out.text = val
        return out
