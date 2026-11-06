#!/usr/bin/python3
"""Copy everything JiuwenSwarm stored for one session into ./<session_id>/ (or --out).

    python pull_session.py web_1a1798a0e92_3e624336a26a
    python pull_session.py <sid> --out ./pulled --data-dir /other/.jiuwenswarm

Collects:
  sessions/          ~/.jiuwenswarm/agent/sessions/<sid>/ (history.jsonl, metadata.json)
  traces/            .agent/traces and .code/traces dumps named after the session
  agent_teams/       team folders under .agent_teams/ that mention the session (team workspace, SwarmFlow runs)
  process_cli_logs/  Process CLI diagnostic files that mention the session
  otel/              OpenTelemetry span lines (.trace/traces-*.jsonl) that mention the session
  logs/              lines from app/framework logs that mention the session (and the team names found)
  other/             any other file under the data dir that mentions the session
  MANIFEST.txt       what was copied from where
"""
from __future__ import annotations

import argparse, os, shutil, sqlite3, sys
from pathlib import Path

SKIP_DIRS = {"node_modules", ".git", "__pycache__", "skills", "models", "cache"}
MAX_SCAN_BYTES = 200_000_000      # files larger than this are not grepped
MAX_WHOLE_DB = 20_000_000         # shared SQLite DBs larger than this get only the session's rows
SQLITE_SIDECARS = ("-wal", "-shm", "-journal")


def sqlite_main(path: Path) -> Path | None:
    """The main DB file if path is a SQLite DB or one of its sidecars, else None."""
    for suf in SQLITE_SIDECARS:
        if path.name.endswith(suf):
            path = path.with_name(path.name[: -len(suf)])
            break
    try:
        with open(path, "rb") as f:
            return path if f.read(16) == b"SQLite format 3\x00" else None
    except OSError:
        return None


def copy_sqlite(src: Path, dst: Path, sid: str, manifest: list[str]) -> None:
    """Snapshot a live DB with the backup API (folds in the WAL). Big shared DBs: only rows mentioning sid."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(dst)
    try:
        if src.stat().st_size <= MAX_WHOLE_DB:
            s.backup(d)
            manifest.append(f"{src} (sqlite snapshot) -> {dst}")
            return
        kept = []
        for name, sql in s.execute("select name, sql from sqlite_master where type='table' and sql is not null"):
            if name.startswith("sqlite_"):
                continue
            cols = [r[1] for r in s.execute(f'pragma table_info("{name}")')]
            where = " or ".join(f'instr(cast("{c}" as blob), cast(? as blob)) > 0' for c in cols)
            rows = s.execute(f'select * from "{name}" where {where}', [sid] * len(cols)).fetchall()
            if rows:
                d.execute(sql)
                d.executemany(f'insert into "{name}" values ({",".join("?" * len(cols))})', rows)
                kept.append(f"{name}:{len(rows)}")
        d.commit()
        manifest.append(f"{src} (sqlite rows {', '.join(kept) or 'none'}) -> {dst}")
    finally:
        s.close()
        d.close()


def contains(path: Path, needles: list[bytes]) -> bool:
    try:
        if path.stat().st_size > MAX_SCAN_BYTES:
            return False
        with open(path, "rb") as f:
            data = f.read()
        return any(n in data for n in needles)
    except OSError:
        return False


def grep_lines(path: Path, needles: list[str]) -> list[str]:
    out = []
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                if any(n in line for n in needles):
                    out.append(line)
    except OSError:
        pass
    return out


def copy(src: Path, dst: Path, manifest: list[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=True)
    else:
        shutil.copy2(src, dst)
    manifest.append(f"{src} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_id")
    ap.add_argument("--data-dir", default=os.environ.get("JIUWENSWARM_DATA_DIR", str(Path.home() / ".jiuwenswarm")))
    ap.add_argument("--out", default=None, help="output folder (default ./<session_id>)")
    ap.add_argument("--no-scan", action="store_true", help="skip the full-tree search for other mentions")
    a = ap.parse_args()

    sid, d = a.session_id, Path(a.data_dir).expanduser()
    out = Path(a.out or sid).resolve()
    if not d.is_dir():
        sys.exit(f"data dir not found: {d}")
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    done: set[Path] = set()

    # 1. the session folder itself
    sess = d / "agent" / "sessions" / sid
    if sess.is_dir():
        copy(sess, out / "sessions" / sid, manifest)
        done.add(sess.resolve())
    else:
        print(f"note: no session folder at {sess}")

    # 2. trace dumps named after the session
    for sub in (".agent/traces", ".code/traces"):
        for p in (d / sub).glob(f"*{sid}*") if (d / sub).is_dir() else []:
            copy(p, out / "traces" / p.name, manifest)
            done.add(p.resolve())

    # 3. team folders that mention the session (teams are named separately, so match by content)
    team_names: list[str] = []
    teams = d / ".agent_teams"
    if teams.is_dir():
        for team in teams.iterdir():
            if team.is_dir() and any(contains(f, [sid.encode()]) for f in team.rglob("*") if f.is_file()):
                copy(team, out / "agent_teams" / team.name, manifest)
                done.add(team.resolve())
                team_names.append(team.name)

    # 4. Process CLI diagnostic logs
    pcl = d / "agent" / "process_cli_logs"
    for p in pcl.rglob("*") if pcl.is_dir() else []:
        if p.is_file() and contains(p, [sid.encode()]):
            copy(p, out / "process_cli_logs" / p.relative_to(pcl), manifest)
            done.add(p.resolve())

    needles = [sid] + team_names

    # 5. OTel spans: keep only matching lines
    for p in sorted((d / ".trace").glob("traces-*.jsonl")) if (d / ".trace").is_dir() else []:
        lines = grep_lines(p, needles)
        if lines:
            (out / "otel").mkdir(parents=True, exist_ok=True)
            (out / "otel" / p.name).write_text("".join(lines))
            manifest.append(f"{p} ({len(lines)} lines) -> otel/{p.name}")
        done.add(p.resolve())

    # 6. app and framework logs: keep only matching lines
    for logdir in (d / "agent" / ".logs", d / "logs"):
        for p in logdir.rglob("*") if logdir.is_dir() else []:
            if p.is_file() and (p.suffix in (".log", ".txt", ".jsonl") or ".log." in p.name):
                lines = grep_lines(p, needles)
                if lines:
                    dst = out / "logs" / p.relative_to(d)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_text("".join(lines))
                    manifest.append(f"{p} ({len(lines)} lines) -> {dst.relative_to(out)}")
                done.add(p.resolve())

    # 7. anything else under the data dir that mentions the session
    if not a.no_scan:
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS and Path(root, x).resolve() not in done]
            for name in files:
                p = Path(root, name)
                if p.resolve() in done or "config" in p.relative_to(d).parts[:1]:
                    continue
                if not contains(p, [sid.encode()]):
                    continue
                db = sqlite_main(p)
                if db is None:
                    copy(p, out / "other" / p.relative_to(d), manifest)
                elif db.resolve() not in done:   # a hit in the -wal alone is useless without the main file
                    done.add(db.resolve())
                    copy_sqlite(db, out / "other" / db.relative_to(d), sid, manifest)

    (out / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print(f"copied {len(manifest)} items to {out}")
    if team_names:
        print(f"teams: {', '.join(team_names)}")
    if not manifest:
        print("nothing found; check the session id and --data-dir")


if __name__ == "__main__":
    main()