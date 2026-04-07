"""Resolve Meta-Harness candidates against a Hermes checkout."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence


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


def _is_within(path: Path, root: Path) -> bool:
    """Check whether *path* is inside *root* (both must be resolved)."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_candidate_containment(
    resolved: Path,
    hermes_agent_path: Path,
    extra_allowed_roots: Optional[Sequence[Path]] = None,
) -> None:
    """Raise if the resolved candidate path escapes all allowed roots.

    Allowed roots include the Hermes builtins dir, the Hermes repo,
    the current working directory, and any explicitly provided extra roots.
    """
    allowed = [
        builtin_candidates_dir(hermes_agent_path).resolve(),
        hermes_agent_path.resolve(),
        Path.cwd().resolve(),
    ]
    if extra_allowed_roots:
        allowed.extend(root.resolve() for root in extra_allowed_roots)

    if any(_is_within(resolved, root) for root in allowed):
        return

    raise ValueError(
        f"Candidate path '{resolved}' is outside all allowed directories. "
        f"Allowed roots: {[str(r) for r in allowed]}"
    )


def resolve_candidate_path(
    candidate: str,
    hermes_agent_path: Path,
    *,
    extra_allowed_roots: Optional[Sequence[Path]] = None,
) -> Path:
    """Resolve an explicit or built-in candidate path.

    The resolved path is validated to be within an allowed root directory
    (Hermes builtins, Hermes repo, cwd, or any extra roots passed in).
    """
    raw = Path(candidate).expanduser()
    if raw.is_absolute() and raw.exists():
        resolved = raw.resolve()
        _validate_candidate_containment(resolved, hermes_agent_path, extra_allowed_roots)
        return resolved

    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists():
        _validate_candidate_containment(cwd_path, hermes_agent_path, extra_allowed_roots)
        return cwd_path

    if raw.suffix == ".py":
        hermes_relative = (hermes_agent_path / candidate).resolve()
        if hermes_relative.exists():
            _validate_candidate_containment(hermes_relative, hermes_agent_path, extra_allowed_roots)
            return hermes_relative

    builtin = builtin_candidates_dir(hermes_agent_path) / f"{candidate}.py"
    if builtin.exists():
        return builtin.resolve()

    raise FileNotFoundError(
        f"Could not resolve candidate '{candidate}'. "
        "Checked absolute path, cwd-relative path, Hermes-relative path, and built-in candidates."
    )
