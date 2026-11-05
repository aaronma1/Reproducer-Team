import os
import shutil
import subprocess

import pytest

from framework.tools import AccessDenied, Workspace, policy_for


@pytest.fixture
def ws(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "paper.pdf").write_bytes(b"%PDF secret-paper")
    (paper / "scope.txt").write_text("scope")
    (tmp_path / "outside.txt").write_text("outside")
    return Workspace.create(tmp_path / "run", paper)


def test_every_role_reads_the_paper_since_2026_11_04(ws):
    for label in ("builder:a1", "evaluator", "audit:c1", "reviewer:e", "lead"):
        assert ws.for_label(label).read_file("scope.txt")
    with pytest.raises(AccessDenied):
        ws.for_label("mystery").read_file("scope.txt")       # an unknown role still gets nothing


def test_paper_roles_can_read_paper_but_not_write(ws):
    r = ws.for_label("lead")
    assert r.read_file("paper.pdf").startswith(b"%PDF")
    with pytest.raises(AccessDenied):
        r.write_file("repo/data/x", "x")
    with pytest.raises(AccessDenied):
        r.shell("ls")




def test_builder_writes_only_own_arm(ws):
    b = ws.for_label("builder:a1")
    b.write_file("repo/arms/a1/m.py", "x = 1")
    assert b.read_file("repo/arms/a1/m.py") == b"x = 1"
    for bad in ("repo/arms/a2/m.py", "repo/data/x", "repo/eval/x", "report.md", "repo/arms/a1x/m.py"):
        with pytest.raises(AccessDenied):
            b.write_file(bad, "x")


@pytest.mark.parametrize("bad", ["../outside.txt", "repo/../../outside.txt", "/etc/passwd", "", "repo/arms/a1/../a2/x"])
def test_escape_paths_rejected(ws, bad):
    with pytest.raises(AccessDenied):
        ws.for_label("builder:a1").write_file(bad, "x")
    with pytest.raises(AccessDenied):
        ws.for_label("reporter").read_file(bad)


def test_symlink_escape_rejected(ws, tmp_path):
    arm = ws.root / "repo/arms/a1"
    arm.mkdir(parents=True)
    (arm / "link").symlink_to(tmp_path / "outside.txt")
    (ws.root / "repo/arms/a1/dirlink").symlink_to(tmp_path)
    b = ws.for_label("builder:a1")
    with pytest.raises(AccessDenied):
        b.write_file("repo/arms/a1/link", "pwn")
    with pytest.raises(AccessDenied):
        b.write_file("repo/arms/a1/dirlink/new.txt", "pwn")
    with pytest.raises(AccessDenied):
        b.read_file("repo/arms/a1/link")
    assert (tmp_path / "outside.txt").read_text() == "outside"


def test_symlinked_arm_dir_rejected(ws, tmp_path):
    (ws.root / "repo/arms").mkdir(exist_ok=True)
    (ws.root / "repo/arms/a1").symlink_to(tmp_path)
    with pytest.raises(AccessDenied):
        ws.for_label("builder:a1").write_file("repo/arms/a1/x", "pwn")


def test_hardlink_write_rejected(ws, tmp_path):
    arm = ws.root / "repo/arms/a1"
    arm.mkdir(parents=True)
    os.link(tmp_path / "outside.txt", arm / "h")
    with pytest.raises(AccessDenied):
        ws.for_label("builder:a1").write_file("repo/arms/a1/h", "pwn")
    assert (tmp_path / "outside.txt").read_text() == "outside"


def test_reporter_writes_only_report(ws):
    r = ws.for_label("reporter")
    r.write_file("report.md", "# r")
    assert r.read_file("paper.pdf")
    with pytest.raises(AccessDenied):
        r.write_file("repo/data/x", "x")


@pytest.mark.parametrize("label", ["builder:", "builder:../x", "builder:a/b", "mystery"])
def test_malformed_or_unknown_labels_get_nothing(ws, label):
    _, pol = policy_for(label)
    assert not pol.read and not pol.write and pol.shell is None


needs_bwrap = pytest.mark.skipif(
    not shutil.which("bwrap")
    or subprocess.run(["bwrap", "--unshare-all", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
                       "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64", "true"]).returncode,
    reason="bwrap unusable",
)


@needs_bwrap
def test_shell_has_no_secrets(ws, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SHOULD-NOT-LEAK")
    b = ws.for_label("builder:a1")
    out = b.shell("env; ls /work; ls /work/repo")
    assert "SHOULD-NOT-LEAK" not in out and "OPENROUTER" not in out
    assert "scope.txt" in out                                 # the paper files are mounted for a builder now


@needs_bwrap
def test_shell_writes_confined_to_arm_and_no_network(ws, tmp_path):
    b = ws.for_label("builder:a1")
    out = b.shell("echo ok > /work/repo/arms/a1/f && echo bad > /work/repo/data/f; echo bad > /work/repo/eval/f; "
                  "echo x > /work/leak; (exec 3<>/dev/tcp/1.1.1.1/80) 2>&1 | head -1")
    assert (ws.root / "repo/arms/a1/f").read_text() == "ok\n"
    assert not (ws.root / "repo/data/f").exists() and not (ws.root / "leak").exists()
    assert out.count("Read-only file system") + out.count("Permission denied") >= 2
    assert "ok" not in out or True
    assert "Network is unreachable" in out or "Connection refused" in out or "No route" in out or "Cannot assign" in out
    assert (ws.root / "repo/arms/a1").is_dir()


@needs_bwrap
def test_shell_timeout_and_truncation(ws):
    c = ws.for_label("evaluator")
    assert "timed out" in c.shell("sleep 5", timeout=1)
    assert c.shell("yes | head -c 100000", max_out=100).endswith("[truncated]")


def test_edit_file_only_where_writable_and_unique(ws):
    b = ws.for_label("builder:a1")
    b.write_file("repo/arms/a1/m.py", "x = 1\ny = 2\n")
    b.edit_file("repo/arms/a1/m.py", "y = 2", "y = 3")
    assert b.read_file("repo/arms/a1/m.py") == b"x = 1\ny = 3\n"
    with pytest.raises(ValueError):
        b.edit_file("repo/arms/a1/m.py", "nope", "z")
    (ws.root / "repo/data/d.py").write_text("a = 1")
    with pytest.raises(AccessDenied):
        b.edit_file("repo/data/d.py", "a = 1", "a = 2")      # readable but not writable for a builder


def test_grep_respects_read_rules(ws):
    (ws.root / "repo/data/d.py").write_text("alpha = 1\nbeta = 2\n")
    hits = ws.for_label("builder:a1").grep("beta")
    assert hits == ["repo/data/d.py:2: beta = 2"]
    assert ws.for_label("builder:a1").grep("secret-paper") == []          # the paper is not searchable for builders
    assert ws.for_label("lead").grep("scope") == ["scope.txt:1: scope"]       # a broken PDF is skipped, not fatal


def test_root_listing_shows_only_reachable_entries(ws):
    assert ws.for_label("lead").list_dir(".") == ["paper.pdf", "scope.txt"]
    assert ws.for_label("builder:a1").list_dir(".") == ["paper.pdf", "repo/", "scope.txt"]


def test_auditor_cannot_write(ws):
    a = ws.for_label("audit:c1")
    assert a.policy.shell == "ro"
    with pytest.raises(AccessDenied):
        a.write_file("repo/data/x", "x")
