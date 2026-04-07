"""Read Hermes Meta-Harness archives."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from meta_harness.models import RunSummary

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_object(path: Path, *, label: str) -> Dict:
    """Load a JSON object and raise a descriptive error when it is malformed."""
    try:
        payload = _load_json(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {label} in {path.parent}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"{label} in {path.parent} is not a JSON object (got {type(payload).__name__})"
        )

    return payload


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
    if not manifest_path.exists():
        return {}
    return _load_json_object(manifest_path, label="manifest.json")


def load_task_records(run_dir: Path) -> List[Dict]:
    """Load all per-task JSON records for a run."""
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.exists():
        return []

    records = []
    for task_file in sorted(tasks_dir.glob("*.json")):
        try:
            payload = _load_json(task_file)
            if not isinstance(payload, dict):
                logger.warning("Skipping non-object task file: %s", task_file)
                continue
            records.append(payload)
        except json.JSONDecodeError:
            logger.warning("Skipping corrupted task file: %s", task_file)
            continue
    return records


def load_run_summary(run_dir: Path) -> RunSummary:
    """Load one run summary and its manifest."""
    run_dir = run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")

    payload = _load_json_object(summary_path, label="summary.json")
    eval_metrics = payload.get("eval_metrics", {})
    task_results = payload.get("task_results", [])
    if not isinstance(eval_metrics, dict):
        raise ValueError(f"summary.json in {run_dir} has non-object eval_metrics")
    if not isinstance(task_results, list):
        raise ValueError(f"summary.json in {run_dir} has non-list task_results")

    return RunSummary(
        benchmark_name=payload.get("benchmark_name", ""),
        candidate_name=payload.get("candidate_name", ""),
        candidate_path=payload.get("candidate_path", ""),
        run_dir=run_dir,
        eval_metrics=eval_metrics,
        task_results=task_results,
        manifest=load_manifest(run_dir),
    )


def load_latest_run_summary(archive_root: Path) -> RunSummary:
    """Load the latest run summary from an archive root."""
    return load_run_summary(find_latest_run_dir(archive_root))
