"""Configuration and Hermes repo discovery."""

from __future__ import annotations

import os
import shlex
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


def parse_command_prefix(value: str | None) -> tuple[str, ...]:
    """Parse a shell-style command prefix into argv tokens."""
    if not value:
        return ()
    return tuple(token for token in shlex.split(value) if token)


def _default_launch_prefix() -> tuple[str, ...]:
    """Default benchmark launcher prefix from the environment."""
    return parse_command_prefix(os.getenv("HERMES_LAUNCH_PREFIX"))


def _default_python_executable() -> str:
    """Choose the Python entrypoint used inside the benchmark launcher."""
    env_value = os.getenv("HERMES_PYTHON_EXECUTABLE")
    if env_value:
        return env_value
    return "python" if _default_launch_prefix() else sys.executable


@dataclass
class MetaHarnessConfig:
    """Base configuration for the standalone Meta-Harness repo."""

    hermes_agent_path: Path = field(default_factory=get_hermes_agent_path)
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    default_benchmark: str = "tblite"
    launch_prefix: tuple[str, ...] = field(default_factory=_default_launch_prefix)
    python_executable: str = field(default_factory=_default_python_executable)
