"""MockBackend fixtures from JSON: {label: value}. A file may name `"$extends": "<other>.json"` (relative to
itself) to start from another fixture file and override only the labels it lists."""
from __future__ import annotations

import json
from pathlib import Path


def load_fixtures(path: Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text())
    base = data.pop("$extends", None)
    out = load_fixtures(path.parent / base) if base else {}
    out.update(data)
    return out
