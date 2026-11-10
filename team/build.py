"""Build the paper-reproducer-team expert team (AgentGroup) package from team/ sources.

    python team/build.py [--bwrap | --sandbox none|bwrap|auto] [--out team/dist] [--no-zip]

Writes <out>/paper-reproducer-team/ and <out>/paper-reproducer-team.zip (the file to import or share): the group
manifest, one expert per role in team/roles.json (persona, the RunPolicyRail with that role's policy.json, the
view_page tool), and the group-shared paper-reproducer skill with flows/reproducer_flow.py as its
scripts/workflow.py. Every member gets its own copy of the shared files because a package may not contain symlinks.
`--bwrap` makes every member's shell run in a bubblewrap sandbox with only its role's paths mounted; the host running
the team then needs `bwrap`, and a member's shell calls are refused without it. Without `--bwrap` shell commands run
unsandboxed and the role's paths bind the file tools only. `--sandbox auto` (the build to share) sandboxes shells
where the host can run bubblewrap and runs them unsandboxed elsewhere, with the unsandboxed persona rules in every
case. The choice is written into each member's policy.json.
The package README.md (the team's description in JiuwenSwarm) is team/README.md with the "Results and cost" table
from docs/swarmflow-notes.md inserted at the top.
Then it checks the result against the rules JiuwenSwarm's publish step applies and zips it, with the package
directory as the archive's single top-level entry.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NAME = "paper-reproducer-team"
SKILL = "paper-reproducer"
MEMBER_RX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
NOTES = REPO / "docs" / "swarmflow-notes.md"
RESULTS_HEADING = "### Results and cost"
SANDBOXES = ("bwrap", "none", "auto")  # how members' shells run; team/common/policy_rail.py reads it from policy.json


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def results_section() -> str:
    """The merged results-and-cost table from docs/swarmflow-notes.md, from its heading up to the next heading."""
    text = NOTES.read_text(encoding="utf-8")
    start = text.index(RESULTS_HEADING) + len(RESULTS_HEADING)
    end = text.find("\n#", start)
    body = text[start:end if end >= 0 else None].strip()
    assert "| Paper |" in body, f"{NOTES}: no table under {RESULTS_HEADING!r}"
    return body


SANDBOX_NOTE = {
    "bwrap": "**Shell sandbox: bubblewrap.** Every member's shell runs in a bubblewrap sandbox with only its role's "
             "paths mounted. The host needs `bwrap` installed; without it, shell calls are refused.",
    "none": "**Shell sandbox: none.** Shell commands run unsandboxed: the role's paths bind the file tools only, and a "
            "shell command can read or write anything the JiuwenSwarm user can. Build with `--bwrap` to confine them.",
    "auto": "**Shell sandbox: bubblewrap where available.** On a Linux host that can run bubblewrap (`bwrap`), every "
            "member's shell runs in a sandbox with only its role's paths mounted. Elsewhere (no `bwrap`, macOS, "
            "Windows, hosts that block user namespaces) shell commands run unsandboxed and can read or write anything "
            "the JiuwenSwarm user can; install bubblewrap to confine them.",
}


# Persona sentences that are true only for one sandbox mode, as {{name}} placeholders in team/personas/*.md.
PERSONA_TEXT = {
    "bwrap": {
        "tool_reach": "Your file, shell and GPU tools only reach the paths your role owns; a refused call says what "
                      "is allowed.",
        "shell_ro": "is read-only",
    },
    "none": {
        "tool_reach": "Your file and GPU tools only reach the paths your role owns, and a refused call says what is "
                      "allowed; your shell is not confined (see \"Your shell\" below).",
        "shell_ro": "must not change any file (nothing enforces it in this build; see \"Your shell\")",
    },
    "auto": {
        "tool_reach": "Your file and GPU tools only reach the paths your role owns, and a refused call says what is "
                      "allowed; your shell may not be confined (see \"Your shell\" below).",
        "shell_ro": "must not change any file (see \"Your shell\")",
    },
}
SHELL_INTRO = {
    "none": "This team was built without a shell sandbox: your shell commands run on the host exactly as you write "
            "them, and the role rules above bind only the file tools.",
    "auto": "Your shell runs in a sandbox only if this host supports one; otherwise your commands run on the host "
            "exactly as you write them, and the role rules above bind only the file tools. Follow these rules either "
            "way.",
}
PLACEHOLDER_RX = re.compile(r"\{\{(\w+)\}\}")


def render(text: str, sandbox: str) -> str:
    """Fill the sandbox-dependent {{placeholders}} of a persona."""
    return PLACEHOLDER_RX.sub(lambda m: PERSONA_TEXT[sandbox][m.group(1)], text)


def shell_note(policy: dict, sandbox: str = "none") -> str:
    """Persona section for a member whose shell runs unsandboxed: nothing but these instructions keeps it to its
    role's paths, so say exactly which ones (the rail starts each command in the directory named here)."""
    paths = lambda ps: ", ".join(f"`{p.replace('{experiment}', '<experiment>')}" + ("" if "." in p else "/") + "`"
                                 for p in ps) or "nothing"
    if policy["shell"] == "rw":
        home = f"your own directory, {paths(policy['write'][:1])} under the run directory"
        writes = f"Create, change and delete files only inside {paths(policy['write'])}."
    else:
        home = "the run directory"
        writes = "Do not create, change or delete any file with it: your shell is read-only by instruction."
    net = ("Use the network only for what your task needs (packages, the references the paper cites)."
           if policy.get("web") else "Do not use the network.")
    return (f"## Your shell\n\n{SHELL_INTRO[sandbox]} Each command starts in {home}; stay there. {writes} Read only {paths(policy['read'])} under the run directory, by absolute "
            f"path. Never read or touch anything outside the run directory: not your home directory, not "
            f"JiuwenSwarm's own files (its config holds API keys), not other runs or teams. {net}\n")


def readme(sandbox: str = "none") -> str:
    """team/README.md with the results table inserted right after its title, as the team's detail page shows it, and
    a line saying how this build runs shell commands."""
    title, _, rest = (HERE / "README.md").read_text(encoding="utf-8").partition("\n")
    return f"{title}\n\n{SANDBOX_NOTE[sandbox]}\n\n## Results so far\n\n{results_section()}\n\n{rest.lstrip()}"


def build(out: Path, sandbox: str = "none") -> Path:
    assert sandbox in SANDBOXES, sandbox
    roles = {k: v for k, v in json.loads((HERE / "roles.json").read_text(encoding="utf-8")).items()
             if not k.startswith("_")}
    pkg = out / NAME
    if pkg.exists():
        shutil.rmtree(pkg)
    team_md = (HERE / "personas" / "_team.md").read_text(encoding="utf-8")
    for role, spec in roles.items():
        d = pkg / "agents" / role
        persona = (HERE / "personas" / f"{role}.md").read_text(encoding="utf-8").rstrip() + "\n\n" + team_md
        if sandbox != "bwrap" and (spec.get("policy") or {}).get("shell"):
            persona = persona.rstrip() + "\n\n" + shell_note(spec["policy"], sandbox)
        persona = render(persona, sandbox)
        (d / "persona").mkdir(parents=True)
        (d / "persona" / f"{role}.md").write_text(persona, encoding="utf-8")
        manifest = {"package_type": "agent_template", "name": role, "display_name": spec["display_name"],
                    "description": spec["description"], "persona": {"dir": "persona"}}
        if "policy" in spec:  # the leader only launches the skill; it has no role policy
            (d / "rails").mkdir()
            shutil.copy(HERE / "common" / "policy_rail.py", d / "rails" / "policy_rail.py")
            _write_json(d / "rails" / "policy.json", {"role": role, **spec["policy"], "sandbox": sandbox})
            (d / "tools").mkdir()
            shutil.copy(HERE / "common" / "view_page.py", d / "tools" / "view_page.py")
            manifest["tools"] = [{"file": "tools/view_page.py", "class": "ViewPage"}]
            manifest["rails"] = [{"file": "rails/policy_rail.py", "class": "RunPolicyRail"}]
        if role == "leader":  # JiuwenSwarm reads the group leader's rules from AGENT.md
            (d / "AGENT.md").write_text(persona, encoding="utf-8")
        _write_json(d / "manifest.json", manifest)
    skill = pkg / "skills" / SKILL
    (skill / "scripts").mkdir(parents=True)
    for f in ("SKILL.md", "skill_display.md"):
        shutil.copy(HERE / "skill" / f, skill / f)
    shutil.copy(REPO / "flows" / "reproducer_flow.py", skill / "scripts" / "workflow.py")
    _write_json(pkg / "manifest.json", {
        "package_type": "agent_group", "name": NAME,
        "display_name": {"en": "Paper Reproduction Team", "zh": "论文复现专家团"},
        "description": "Reproduces an ML paper's claims from scratch: one expert per pipeline role, each limited to "
                       "the files, shell, GPU and web access its role needs.",
        "instruction": "Run the paper-reproducer skill with experts=true. The leader stages the paper and launches "
                       "the workflow; the script hands each step to the member of the same name.",
        "agents": list(roles), "skills": [SKILL]})
    (pkg / "README.md").write_text(readme(sandbox), encoding="utf-8")  # JiuwenSwarm shows it as the team's description
    check(pkg)
    return pkg


def make_zip(pkg: Path) -> Path:
    """Zip the package as <out>/<name>.zip, replacing any older archive."""
    archive = pkg.with_suffix(".zip")
    archive.unlink(missing_ok=True)
    return Path(shutil.make_archive(str(pkg), "zip", pkg.parent, pkg.name))


def check(pkg: Path) -> None:
    """The structural rules of JiuwenSwarm's agent_group publish adapter (asset_publish_adapters._agent_group)."""
    group = json.loads((pkg / "manifest.json").read_text(encoding="utf-8"))
    agents = group["agents"]
    assert group["package_type"] == "agent_group" and group.get("name") and group.get("description")
    assert agents and len(set(agents)) == len(agents) and "leader" in agents
    assert all(MEMBER_RX.fullmatch(a) for a in agents) and all(MEMBER_RX.fullmatch(s) for s in group["skills"])
    for a in agents:
        m = json.loads((pkg / "agents" / a / "manifest.json").read_text(encoding="utf-8"))
        assert m["package_type"] == "agent_template", a
        personas = list((pkg / "agents" / a / m["persona"]["dir"]).glob("*.md"))
        assert personas, f"{a}: persona needs Markdown"
        assert not any(PLACEHOLDER_RX.search(p.read_text(encoding="utf-8")) for p in personas), f"{a}: {{{{placeholder}}}} left"
        for entry in m.get("tools", []) + m.get("rails", []):
            assert (pkg / "agents" / a / entry["file"]).is_file(), f"{a}: missing {entry['file']}"
        policy = pkg / "agents" / a / "rails" / "policy.json"
        if policy.is_file():
            assert json.loads(policy.read_text(encoding="utf-8")).get("sandbox") in SANDBOXES, f"{a}: sandbox"
    for s in group["skills"]:
        assert (pkg / "skills" / s / "SKILL.md").is_file(), s
    assert not any(p.is_symlink() for p in pkg.rglob("*")), "a package may not contain symlinks"
    assert (pkg / "agents" / "leader" / "AGENT.md").is_file(), "the leader needs AGENT.md"
    assert not any((pkg / "agents" / a / "AGENT.md").exists() for a in agents if a != "leader"), \
        "only the leader may have AGENT.md"
    try:  # the loader JiuwenSwarm's import runs, when openjiuwen is installed
        from openjiuwen.agent_teams.experts import load_agent_group_package
    except ImportError:
        return
    load_agent_group_package(pkg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "dist")
    ap.add_argument("--sandbox", choices=SANDBOXES, default="none",
                    help="how members' shells run: none (default), bwrap (the host then needs bwrap), or auto "
                         "(bwrap where the host can run it, unsandboxed elsewhere: the build to share)")
    ap.add_argument("--bwrap", dest="sandbox", action="store_const", const="bwrap",
                    help="same as --sandbox bwrap")
    ap.add_argument("--no-zip", dest="zip", action="store_false",
                    help="skip writing <out>/paper-reproducer-team.zip")
    ap.add_argument("--zip", action="store_true", help=argparse.SUPPRESS)  # the default; kept for old commands
    ap.set_defaults(zip=True)
    a = ap.parse_args()
    pkg = build(a.out, a.sandbox)
    print(pkg)
    print({"bwrap": "shell sandbox: bubblewrap (the host running the team needs `bwrap`)",
           "auto": "shell sandbox: auto - bubblewrap where the host can run it, unsandboxed elsewhere",
           "none": "shell sandbox: none - shell commands run unsandboxed; rebuild with --bwrap to confine them",
           }[a.sandbox])
    if a.zip:
        print(make_zip(pkg))


if __name__ == "__main__":
    main()
