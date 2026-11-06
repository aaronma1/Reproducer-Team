"""Build the paper-reproducer-team expert team (AgentGroup) package from team/ sources.

    python team/build.py [--out team/dist] [--no-zip]

Writes <out>/paper-reproducer-team/ and <out>/paper-reproducer-team.zip (the file to import or share): the group
manifest, one expert per role in team/roles.json (persona, the RunPolicyRail with that role's policy.json, the
view_page tool), and the group-shared paper-reproducer skill with flows/reproducer_flow.py as its
scripts/workflow.py. Every member gets its own copy of the shared files because a package may not contain symlinks.
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


def readme() -> str:
    """team/README.md with the results table inserted right after its title, as the team's detail page shows it."""
    title, _, rest = (HERE / "README.md").read_text(encoding="utf-8").partition("\n")
    return f"{title}\n\n## Results so far\n\n{results_section()}\n\n{rest.lstrip()}"


def build(out: Path) -> Path:
    roles = {k: v for k, v in json.loads((HERE / "roles.json").read_text(encoding="utf-8")).items()
             if not k.startswith("_")}
    pkg = out / NAME
    if pkg.exists():
        shutil.rmtree(pkg)
    team_md = (HERE / "personas" / "_team.md").read_text(encoding="utf-8")
    for role, spec in roles.items():
        d = pkg / "agents" / role
        persona = (HERE / "personas" / f"{role}.md").read_text(encoding="utf-8").rstrip() + "\n\n" + team_md
        (d / "persona").mkdir(parents=True)
        (d / "persona" / f"{role}.md").write_text(persona, encoding="utf-8")
        manifest = {"package_type": "agent_template", "name": role, "display_name": spec["display_name"],
                    "description": spec["description"], "persona": {"dir": "persona"}}
        if "policy" in spec:  # the leader only launches the skill; it has no role policy
            (d / "rails").mkdir()
            shutil.copy(HERE / "common" / "policy_rail.py", d / "rails" / "policy_rail.py")
            _write_json(d / "rails" / "policy.json", {"role": role, **spec["policy"]})
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
    (pkg / "README.md").write_text(readme(), encoding="utf-8")  # JiuwenSwarm shows it as the team's description
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
        assert any((pkg / "agents" / a / m["persona"]["dir"]).glob("*.md")), f"{a}: persona needs Markdown"
        for entry in m.get("tools", []) + m.get("rails", []):
            assert (pkg / "agents" / a / entry["file"]).is_file(), f"{a}: missing {entry['file']}"
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
    ap.add_argument("--no-zip", dest="zip", action="store_false",
                    help="skip writing <out>/paper-reproducer-team.zip")
    ap.add_argument("--zip", action="store_true", help=argparse.SUPPRESS)  # the default; kept for old commands
    ap.set_defaults(zip=True)
    a = ap.parse_args()
    pkg = build(a.out)
    print(pkg)
    if a.zip:
        print(make_zip(pkg))


if __name__ == "__main__":
    main()
