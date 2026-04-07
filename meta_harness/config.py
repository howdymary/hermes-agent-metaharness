"""Configuration and Hermes repo discovery."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def get_hermes_agent_path() -> Path:
    """Discover the Hermes repo path."""
    env_path = os.getenv("HERMES_AGENT_REPO")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.exists():
            return candidate.resolve()

    sibling = Path(__file__).resolve().parent.parent.parent / "hermes-agent"
    if sibling.exists():
        return sibling.resolve()

    home_path = Path.home() / ".hermes" / "hermes-agent"
    if home_path.exists():
        return home_path.resolve()

    raise FileNotFoundError(
        "Could not locate hermes-agent. Set HERMES_AGENT_REPO or place a checkout at ../hermes-agent."
    )


@dataclass
class MetaHarnessConfig:
    """Base configuration for the standalone Meta-Harness repo."""

    hermes_agent_path: Path = field(default_factory=get_hermes_agent_path)
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    default_benchmark: str = "tblite"
    python_executable: str = field(default_factory=lambda: sys.executable)
