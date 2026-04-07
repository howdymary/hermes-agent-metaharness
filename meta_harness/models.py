"""Core data models for Meta-Harness orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkRunSpec:
    """One benchmark execution request."""

    benchmark: str
    candidate: str
    archive_root: Path
    run_name: Optional[str] = None
    hermes_config_path: Optional[Path] = None
    task_filter: Optional[str] = None
    skip_tasks: Optional[str] = None
    python_executable: str = "python"


@dataclass
class RunSummary:
    """Parsed run summary from a Hermes Meta-Harness archive."""

    benchmark_name: str
    candidate_name: str
    candidate_path: str
    run_dir: Path
    eval_metrics: Dict[str, Any] = field(default_factory=dict)
    task_results: List[Dict[str, Any]] = field(default_factory=list)
    manifest: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskDelta:
    """Per-task comparison between baseline and candidate runs."""

    task_name: str
    baseline_passed: Optional[bool]
    candidate_passed: Optional[bool]
    baseline_reward: Optional[float]
    candidate_reward: Optional[float]
    status: str


@dataclass
class RunComparison:
    """Comparison between two benchmark runs."""

    baseline_run_dir: Path
    candidate_run_dir: Path
    benchmark_name: str
    baseline_candidate_name: str
    candidate_name: str
    metric_deltas: Dict[str, float] = field(default_factory=dict)
    task_deltas: List[TaskDelta] = field(default_factory=list)


@dataclass
class FrontierEntry:
    """Frontier record for one evaluated candidate."""

    candidate_name: str
    candidate_path: str
    benchmark_name: str
    run_dir: str
    pass_rate: float
    status: str = "evaluated"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
