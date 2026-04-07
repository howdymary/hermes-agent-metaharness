"""Read Hermes Meta-Harness archives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from meta_harness.models import RunSummary


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def find_run_dirs(archive_root: Path) -> List[Path]:
    """Return run directories containing summary.json."""
    archive_root = archive_root.expanduser().resolve()
    if not archive_root.exists():
        return []
    return sorted(
        [path for path in archive_root.iterdir() if path.is_dir() and (path / "summary.json").exists()],
        key=lambda path: path.stat().st_mtime,
    )


def find_latest_run_dir(archive_root: Path) -> Path:
    """Return the newest run directory in an archive root."""
    run_dirs = find_run_dirs(archive_root)
    if not run_dirs:
        raise FileNotFoundError(f"No Meta-Harness run summaries found in {archive_root}")
    return run_dirs[-1]


def load_manifest(run_dir: Path) -> Dict:
    """Load manifest.json if present."""
    manifest_path = run_dir / "manifest.json"
    return _load_json(manifest_path) if manifest_path.exists() else {}


def load_task_records(run_dir: Path) -> List[Dict]:
    """Load all per-task JSON records for a run."""
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.exists():
        return []

    records = []
    for task_file in sorted(tasks_dir.glob("*.json")):
        try:
            records.append(_load_json(task_file))
        except json.JSONDecodeError:
            continue
    return records


def load_run_summary(run_dir: Path) -> RunSummary:
    """Load one run summary and its manifest."""
    run_dir = run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")

    payload = _load_json(summary_path)
    return RunSummary(
        benchmark_name=payload.get("benchmark_name", ""),
        candidate_name=payload.get("candidate_name", ""),
        candidate_path=payload.get("candidate_path", ""),
        run_dir=Path(payload.get("run_dir") or run_dir),
        eval_metrics=payload.get("eval_metrics", {}),
        task_results=payload.get("task_results", []),
        manifest=load_manifest(run_dir),
    )


def load_latest_run_summary(archive_root: Path) -> RunSummary:
    """Load the latest run summary from an archive root."""
    return load_run_summary(find_latest_run_dir(archive_root))
