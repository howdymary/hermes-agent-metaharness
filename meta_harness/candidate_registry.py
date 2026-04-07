"""Resolve Meta-Harness candidates against a Hermes checkout."""

from __future__ import annotations

from pathlib import Path
from typing import List


def builtin_candidates_dir(hermes_agent_path: Path) -> Path:
    """Return the Hermes built-in candidate directory."""
    return hermes_agent_path / "environments" / "meta_harness" / "candidates"


def list_builtin_candidates(hermes_agent_path: Path) -> List[str]:
    """List built-in Hermes candidate names."""
    directory = builtin_candidates_dir(hermes_agent_path)
    if not directory.exists():
        return []
    return sorted(
        path.stem
        for path in directory.glob("*.py")
        if path.name != "__init__.py"
    )


def resolve_candidate_path(candidate: str, hermes_agent_path: Path) -> Path:
    """Resolve an explicit or built-in candidate path."""
    raw = Path(candidate).expanduser()
    if raw.is_absolute() and raw.exists():
        return raw.resolve()

    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists():
        return cwd_path

    if raw.suffix == ".py":
        hermes_relative = (hermes_agent_path / candidate).resolve()
        if hermes_relative.exists():
            return hermes_relative

    builtin = builtin_candidates_dir(hermes_agent_path) / f"{candidate}.py"
    if builtin.exists():
        return builtin.resolve()

    raise FileNotFoundError(
        f"Could not resolve candidate '{candidate}'. "
        "Checked absolute path, cwd-relative path, Hermes-relative path, and built-in candidates."
    )
