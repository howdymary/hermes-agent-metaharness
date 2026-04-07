"""Core data models for Meta-Harness orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Maximum fraction of tasks that can regress before a candidate is rejected.
MAX_REGRESSION_RATIO = 0.10


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
class ComparisonReport:
    """Headline report for baseline-vs-candidate evaluation."""

    benchmark_name: str
    baseline_candidate_name: str
    candidate_name: str
    baseline_run_dir: Path
    candidate_run_dir: Path
    total_tasks: int
    overlapping_tasks: int
    improved_tasks: int
    regressed_tasks: int
    unchanged_tasks: int
    baseline_only_tasks: int
    candidate_only_tasks: int
    pass_rate_delta: float
    passed_tasks_delta: float
    evaluation_time_delta_seconds: Optional[float]
    metric_deltas: Dict[str, float] = field(default_factory=dict)
    improved_task_names: List[str] = field(default_factory=list)
    regressed_task_names: List[str] = field(default_factory=list)
    unchanged_task_names: List[str] = field(default_factory=list)

    @property
    def net_task_gain(self) -> int:
        return self.improved_tasks - self.regressed_tasks

    @property
    def candidate_better(self) -> bool:
        """A candidate is better only if it improves pass rate AND does not regress too many tasks."""
        if self.pass_rate_delta <= 0.0:
            return False
        if self.total_tasks > 0 and self.regressed_tasks / self.total_tasks > MAX_REGRESSION_RATIO:
            return False
        return True

    def ranking_key(self) -> Tuple[float, float, int, float]:
        return comparison_sort_key(self)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["baseline_run_dir"] = str(self.baseline_run_dir)
        payload["candidate_run_dir"] = str(self.candidate_run_dir)
        payload["candidate_better"] = self.candidate_better
        payload["net_task_gain"] = self.net_task_gain
        payload["ranking_key"] = list(self.ranking_key())
        return payload


@dataclass
class FrontierEntry:
    """Frontier record for one evaluated candidate."""

    candidate_name: str
    candidate_path: str
    benchmark_name: str
    run_dir: str
    pass_rate: float
    total_tasks: int = 0
    task_selection_hash: str = ""
    status: str = "evaluated"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchTrialResult:
    """One generated trial inside a structured search."""

    mutation_slug: str
    mutation_description: str
    candidate_name: str
    candidate_path: str
    run_dir: Optional[str] = None
    report: Optional[ComparisonReport] = None
    command: List[str] = field(default_factory=list)
    returncode: int = 0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.report is not None:
            payload["report"] = self.report.to_dict()
        return payload


@dataclass
class SearchSummary:
    """Top-level summary for one structured search session."""

    benchmark_name: str
    baseline_candidate: str
    baseline_source: str
    baseline_run_dir: Optional[str]
    baseline_reference: Optional[str]
    seed_candidate: str
    workspace_dir: str
    generated_candidates_dir: str
    best_mutation_slug: Optional[str] = None
    best_candidate_name: Optional[str] = None
    best_run_dir: Optional[str] = None
    trial_results: List[SearchTrialResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "baseline_candidate": self.baseline_candidate,
            "baseline_source": self.baseline_source,
            "baseline_run_dir": self.baseline_run_dir,
            "baseline_reference": self.baseline_reference,
            "seed_candidate": self.seed_candidate,
            "workspace_dir": self.workspace_dir,
            "generated_candidates_dir": self.generated_candidates_dir,
            "best_mutation_slug": self.best_mutation_slug,
            "best_candidate_name": self.best_candidate_name,
            "best_run_dir": self.best_run_dir,
            "trial_results": [trial.to_dict() for trial in self.trial_results],
        }


def comparison_sort_key(report: ComparisonReport) -> Tuple[float, float, int, float]:
    """Lexicographic comparison key for baseline-vs-candidate reports."""
    return comparison_sort_key_for_values(
        pass_rate_delta=report.pass_rate_delta,
        passed_tasks_delta=report.passed_tasks_delta,
        regressed_tasks=report.regressed_tasks,
        evaluation_time_delta_seconds=report.evaluation_time_delta_seconds,
    )


def comparison_sort_key_for_values(
    *,
    pass_rate_delta: float,
    passed_tasks_delta: float,
    regressed_tasks: int,
    evaluation_time_delta_seconds: Optional[float],
) -> Tuple[float, float, int, float]:
    """Shared ranking rule for deciding whether a candidate beats baseline.

    Regressions are weighted before pass_rate_delta so that a candidate
    with many regressions never outranks one with fewer regressions
    unless it also has a strictly higher pass rate delta.
    """
    eval_time = float(evaluation_time_delta_seconds or 0.0)
    return (
        -int(regressed_tasks),
        round(float(pass_rate_delta), 10),
        round(float(passed_tasks_delta), 10),
        round(-eval_time, 10),
    )
