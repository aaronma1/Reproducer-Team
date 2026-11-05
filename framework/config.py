"""Run-time configuration: model tiers, paths, limits. Loaded from TOML."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .gpu import GpuConfig

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FLOW = ROOT / "flows" / "reproducer_flow.py"
RUNS_DIR = ROOT / "runs"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_env(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from .env into os.environ without overriding real env vars."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("\"'")
        if value:
            os.environ.setdefault(key.strip(), value)


def openrouter_credentials() -> tuple[str, str]:
    """(api_key, base_url) for the live backend; raises if no key is configured."""
    load_env()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set: put it in .env (see .env.example) or export it.")
    return key, os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)


@dataclass(frozen=True)
class Tier:
    id: str  # provider model id
    effort: str | None = None  # reasoning effort hint, e.g. "low"
    context: int = 1_048_576  # context window in tokens
    max_tokens: int | None = None  # output cap per call; None = the provider's default


@dataclass
class Config:
    tiers: dict[str, Tier] = field(default_factory=dict)
    default_tier: str = "cheap"
    papers_dir: Path = ROOT.parent / "papers"
    tool_timeout_s: int = 1800
    tool_output_limit: int = 100_000
    max_steps: int = 200
    role_max_steps: dict[str, int] = field(default_factory=dict)  # per-role override of max_steps
    read_limit: int = 1_000_000
    turn_token_limit: int = 0  # 0 = off
    gpu: GpuConfig = field(default_factory=GpuConfig)
    sandbox_venv: str = ""  # venv for the agents' shell (e.g. CPU torch/jax); "" = this project's venv

    @classmethod
    def load(cls, path: Path = ROOT / "config.toml") -> "Config":
        raw = tomllib.loads(path.read_text())
        tiers = {name: Tier(**t) for name, t in raw.pop("tiers", {}).items()}
        papers = Path(raw.pop("papers_dir", "papers"))
        raw["gpu"] = GpuConfig(**raw.pop("gpu", {}))
        return cls(tiers=tiers, papers_dir=(papers if papers.is_absolute() else (path.parent / papers).resolve()), **raw)

    def paper_dir(self, paper: str) -> Path:
        """`paper` is a directory path or a name under papers_dir."""
        direct = Path(paper)
        return direct if direct.is_dir() else self.papers_dir / paper
