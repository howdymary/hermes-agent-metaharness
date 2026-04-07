"""Helpers for run comparability across baseline/candidate evaluations."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional


def _normalize_csv(value: Optional[str]) -> List[str]:
    """Normalize comma-separated task selectors into a stable sorted list."""
    if not value:
        return []
    items = [item.strip() for item in value.split(",") if item.strip()]
    return sorted(dict.fromkeys(items))


def build_task_selection_metadata(
    *,
    task_filter: Optional[str],
    skip_tasks: Optional[str],
) -> Dict[str, object]:
    """Build stable metadata describing the requested task subset."""
    normalized = {
        "task_filter": _normalize_csv(task_filter),
        "skip_tasks": _normalize_csv(skip_tasks),
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return {
        **normalized,
        "selection_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def extract_task_selection_metadata(manifest: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Read the outer-loop task selection metadata from a run manifest."""
    outer_loop = manifest.get("outer_loop")
    if not isinstance(outer_loop, dict):
        return None
    benchmark_runner = outer_loop.get("benchmark_runner")
    if not isinstance(benchmark_runner, dict):
        return None
    task_selection = benchmark_runner.get("task_selection")
    if not isinstance(task_selection, dict):
        return None
    return task_selection
