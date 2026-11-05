"""team/: the role rules of RunPolicyRail, view_page, the package build, and the flow's `experts` option."""
import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEAM = ROOT / "team"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pr = _load(TEAM / "common" / "policy_rail.py", "policy_rail")
build = _load(TEAM / "build.py", "team_build")
ROLES = {k: {"role": k, **v["policy"]} for k, v in json.loads((TEAM / "roles.json").read_text()).items()
         if isinstance(v, dict) and "policy" in v}


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "work"
    (root / "repo" / "arms" / "exp1").mkdir(parents=True)
    (root / "repo" / "eval").mkdir()
    (root / "paper.pdf").write_bytes(b"%PDF")
    (root / "scope.txt").write_text("scope")
    return str(root)


def ok(role, root, name, args, experiment=None):
    pr.decide(ROLES[role], root, experiment, name, args, sandbox="bwrap")
    return args


def denied(role, root, name, args, experiment=None):
    with pytest.raises(pr.Denied):
        pr.decide(ROLES[role], root, experiment, name, args, sandbox="bwrap")


def test_reads_follow_the_role_table(run):
    ok("lead", run, "read_file", {"file_path": f"{run}/paper.pdf"})
    ok("lead", run, "read_file", {"file_path": "scope.txt"})            # relative = under the root
    denied("lead", run, "read_file", {"file_path": f"{run}/repo/arms/exp1/train.py"})
    ok("audit", run, "read_file", {"file_path": f"{run}/repo/arms/exp1/train.py"})
    ok("reporter", run, "read_file", {"file_path": f"{run}/repo/eval/x.py"})


def test_writes_follow_the_role_table_and_the_builders_experiment(run):
    ok("builder", run, "write_file", {"file_path": f"{run}/repo/arms/exp1/train.py"}, experiment="exp1")
    denied("builder", run, "write_file", {"file_path": f"{run}/repo/arms/exp2/train.py"}, experiment="exp1")
    denied("builder", run, "write_file", {"file_path": f"{run}/repo/arms/exp1/train.py"})   # experiment unknown
    ok("evaluator", run, "edit_file", {"file_path": f"{run}/repo/eval/eval.py"})
    denied("reviewer", run, "write_file", {"file_path": f"{run}/repo/arms/exp1/train.py"})
    ok("reporter", run, "write_file", {"file_path": f"{run}/report.md"})
    denied("reporter", run, "write_file", {"file_path": f"{run}/repo/eval/x.py"})


def test_paths_outside_the_root_and_escapes_are_refused(run):
    denied("reporter", run, "read_file", {"file_path": "/etc/passwd"})
    denied("builder", run, "write_file", {"file_path": f"{run}/repo/arms/exp1/../../eval/x"}, experiment="exp1")
    denied("lead", None, "read_file", {"file_path": "/x/paper.pdf"})     # root not known yet


def test_symlinks_in_the_run_tree_are_refused(run):
    Path(run, "repo", "arms", "exp1", "link").symlink_to("/etc")
    denied("builder", run, "read_file", {"file_path": f"{run}/repo/arms/exp1/link/passwd"}, experiment="exp1")


def test_search_tools_default_to_the_root(run):
    args = ok("audit", run, "grep", {"pattern": "seed"})
    assert args["path"] == run


def test_shell_is_wrapped_in_bubblewrap_with_only_the_roles_paths(run, monkeypatch):
    monkeypatch.setattr(pr.shutil, "which", lambda _: "/usr/bin/bwrap")
    args = ok("builder", run, "bash", {"command": "python train.py"}, experiment="exp1")
    argv = shlex.split(args["command"])
    assert argv[0] == "bwrap" and argv[-3:] == ["bash", "-c", "python train.py"]
    binds = [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a == "--bind"]
    assert binds == [(f"{run}/repo/arms/exp1", f"{run}/repo/arms/exp1")]
    assert argv.index("--share-net") > argv.index("--unshare-all")   # the builder has web access (order matters)
    mounted = [argv[i + 1] for i, a in enumerate(argv) if a in ("--bind", "--ro-bind")]
    assert not any(m.endswith("/.local/share/uv") or "credentials" in m for m in mounted)   # uv's stored credentials
    resolv = os.path.realpath("/etc/resolv.conf")
    if not resolv.startswith("/etc/") and os.path.isfile(resolv):
        assert resolv in mounted                                      # DNS works when resolv.conf is a symlink
    audit = shlex.split(ok("audit", run, "bash", {"command": "ls"})["command"])
    assert "--bind" not in audit and "--share-net" not in audit       # read-only, offline
    denied("lead", run, "bash", {"command": "ls"})
    denied("builder", run, "powershell", {"command": "ls"}, experiment="exp1")


def test_shell_without_bubblewrap_is_refused(run, monkeypatch):
    monkeypatch.setattr(pr.shutil, "which", lambda _: None)
    denied("builder", run, "bash", {"command": "ls"}, experiment="exp1")


def test_gpu_tools_by_level_and_path(run):
    denied("audit", run, "mcp_gpurun_gpu_list", {})
    ok("reviewer", run, "mcp_gpurun_gpu_submit", {"local_dir": f"{run}/repo/arms/exp1", "commands": ["ls"]})
    denied("reviewer", run, "mcp_gpurun_download_job_files", {"job_uuid": "j", "files": {"a": f"{run}/repo"}})
    ok("builder", run, "mcp_gpurun_download_job_files",
       {"job_uuid": "j", "files": {"out/m.json": f"{run}/repo/arms/exp1/out/m.json"}}, experiment="exp1")
    denied("builder", run, "mcp_gpurun_download_job_files",
           {"job_uuid": "j", "files": {"out/m.json": f"{run}/repo/eval/m.json"}}, experiment="exp1")


def test_web_tools_need_web_access(run):
    denied("judge", run, "mcp_free_search", {"query": "x"})
    ok("lead", run, "mcp_fetch_webpage", {"url": "https://example.org"})


def test_wrapped_tool_calls_are_checked_and_rewritten(run, monkeypatch):
    monkeypatch.setattr(pr.shutil, "which", lambda _: "/usr/bin/bwrap")
    denied("lead", run, "tool_call", {"name": "write_file", "args": {"file_path": f"{run}/x"}})
    args = ok("evaluator", run, "tool_call", {"name": "bash", "args": {"command": "ls"}})
    assert args["args"]["command"].startswith("bwrap ")
    ok("lead", run, "tool_search", {"query": "anything"})            # tools outside the table pass


def test_the_rail_reads_root_and_experiment_from_the_task(tmp_path):
    rail = pr.RunPolicyRail.__new__(pr.RunPolicyRail)
    rail.root = rail.experiment = None
    rail.observe("All files of this run are under /srv/run1. Every relative path in this task ...")
    rail.observe("You reproduce experiment `exp-2` in repo/arms/exp-2/.")
    assert (rail.root, rail.experiment) == ("/srv/run1", "exp-2")


def test_build_writes_a_package_that_passes_the_publish_rules(tmp_path):
    pkg = build.build(tmp_path)
    group = json.loads((pkg / "manifest.json").read_text())
    assert group["agents"][0] == "leader" and set(group["agents"][1:]) == set(ROLES)
    builder = json.loads((pkg / "agents" / "builder" / "manifest.json").read_text())
    assert builder["rails"] == [{"file": "rails/policy_rail.py", "class": "RunPolicyRail"}]
    assert json.loads((pkg / "agents" / "builder" / "rails" / "policy.json").read_text())["gpu"] == "rw"
    assert "Rules for everyone" in (pkg / "agents" / "audit" / "persona" / "audit.md").read_text()
    assert (pkg / "skills" / "paper-reproducer" / "scripts" / "workflow.py").read_text() == \
        (ROOT / "flows" / "reproducer_flow.py").read_text()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake JiuwenSwarm home with the allowed areas and the secrets next to them."""
    h = tmp_path / "jw"
    for d in ("agent/workspace/skills/s1", "agent/workspace/mcp", ".agent_teams/agent_groups/g1", "config",
              ".agent_teams/team1#builder", ".agent_teams/team2#other"):
        (h / d).mkdir(parents=True)
    (h / "agent/workspace/skills/s1/SKILL.md").write_text("skill")
    (h / "config/config.yaml").write_text("api_key: sk-secret")
    (h / "agent/workspace/skills/s1/.env").write_text("KEY=1")
    (h / "agent/workspace/skills/s1/cfg").symlink_to(h / "config")
    monkeypatch.setattr(pr, "OUTSIDE_READ", (h / "agent/workspace/skills", h / "agent/workspace/mcp",
                                             h / ".agent_teams/agent_groups"))
    return h


def test_outside_the_run_only_skills_mcp_packages_and_own_workspace_are_readable(run, home):
    own = str(home / ".agent_teams/team1#builder")
    read = lambda p: pr.decide(ROLES["lead"], run, None, "read_file", {"file_path": str(p)}, own_workspace=own)
    read(home / "agent/workspace/skills/s1/SKILL.md")
    read(home / "agent/workspace/mcp")
    read(home / ".agent_teams/agent_groups/g1")
    read(home / ".agent_teams/team1#builder")
    for p in (home / "config/config.yaml",                          # model API keys
              home / "agent/workspace/skills/s1/cfg/config.yaml",   # a symlink out of an allowed area
              home / "agent/workspace/skills/s1/.env",              # a secret-looking name inside one
              home / ".agent_teams/team2#other",                    # another worker's workspace
              home / "agent"):
        with pytest.raises(pr.Denied):
            read(p)


def test_view_page_only_renders_pdfs_under_the_run_root(run, home):
    ok("lead", run, "view_page", {"file_path": f"{run}/paper.pdf", "page": 1})
    (home / "agent/workspace/skills/s1/x.pdf").write_bytes(b"%PDF")
    denied("lead", run, "view_page", {"file_path": str(home / "agent/workspace/skills/s1/x.pdf"), "page": 1})


def test_the_rail_learns_its_own_workspace_in_init(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(pr, "load_policy", lambda: {"role": "lead", "read": [], "write": []})
    rail = pr.RunPolicyRail()
    rail.init(SimpleNamespace(deep_config=SimpleNamespace(workspace=SimpleNamespace(root_path="/w/team1#lead"))))
    assert rail.own_workspace == "/w/team1#lead"


def test_view_page_is_capped_per_turn(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    monkeypatch.setattr(pr, "load_policy", lambda: {"role": "lead", "read": ["paper.pdf"], "write": []})
    rail = pr.RunPolicyRail()
    rail.root = "/srv/run1"
    rejected = []
    monkeypatch.setattr(rail, "_reject", lambda ctx, msg: rejected.append(msg))

    def call(name="view_page", args=None):
        inputs = SimpleNamespace(tool_name=name, tool_args=args or {"file_path": "/srv/run1/paper.pdf", "page": 1})
        asyncio.run(rail.before_tool_call(SimpleNamespace(inputs=inputs, extra={})))

    for _ in range(pr.MAX_VIEWS):
        call()
    call("tool_call", {"name": "view_page", "args": {"file_path": "/srv/run1/paper.pdf", "page": 2}})
    assert len(rejected) == 1 and "limit reached (12" in rejected[0]
    asyncio.run(rail.before_invoke(SimpleNamespace(inputs=SimpleNamespace(query="next turn"))))
    call()
    assert len(rejected) == 1                                      # a new turn starts a new count
