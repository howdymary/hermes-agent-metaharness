"""Helpers for selecting a baseline candidate or run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from meta_harness.archive_reader import load_run_summary
from meta_harness.frontier import FrontierStore
from meta_harness.models import RunSummary


@dataclass
class BaselineSelection:
    """Resolved baseline choice for paired evaluation and search."""

    source: str
    candidate_name: str
    baseline_candidate: str
    summary: Optional[RunSummary] = None
    run_dir: Optional[Path] = None
    reference: Optional[str] = None

    @property
    def requires_fresh_run(self) -> bool:
        return self.summary is None

    @property
    def display_label(self) -> str:
        if self.source == "fresh_candidate":
            return self.baseline_candidate
        if self.summary is not None:
            return f"{self.summary.candidate_name} ({self.source})"
        return self.candidate_name


def resolve_baseline_selection(
    *,
    benchmark_name: str,
    baseline_candidate: str = "snapshot_baseline",
    baseline_run_dir: Optional[Path] = None,
    baseline_frontier_path: Optional[Path] = None,
) -> BaselineSelection:
    """Resolve whether baseline should come from a fresh candidate run or an existing artifact."""
    if baseline_run_dir is not None and baseline_frontier_path is not None:
        raise ValueError("Choose at most one baseline source: --baseline-run or --baseline-from-frontier.")

    if baseline_run_dir is not None:
        run_dir = baseline_run_dir.expanduser().resolve()
        summary = load_run_summary(run_dir)
        _validate_baseline_benchmark(summary, benchmark_name, run_dir)
        return BaselineSelection(
            source="existing_run",
            candidate_name=summary.candidate_name,
            baseline_candidate=summary.candidate_name,
            summary=summary,
            run_dir=run_dir,
            reference=str(run_dir),
        )

    if baseline_frontier_path is not None:
        frontier = FrontierStore(baseline_frontier_path.expanduser().resolve())
        entry = frontier.best_for_benchmark(benchmark_name)
        run_dir = Path(entry.run_dir).expanduser().resolve()
        summary = load_run_summary(run_dir)
        _validate_baseline_benchmark(summary, benchmark_name, run_dir)
        return BaselineSelection(
            source="frontier_best",
            candidate_name=entry.candidate_name,
            baseline_candidate=entry.candidate_name,
            summary=summary,
            run_dir=run_dir,
            reference=str(frontier.path),
        )

    return BaselineSelection(
        source="fresh_candidate",
        candidate_name=baseline_candidate,
        baseline_candidate=baseline_candidate,
    )


def _validate_baseline_benchmark(summary: RunSummary, benchmark_name: str, run_dir: Path) -> None:
    """Ensure an imported baseline run matches the requested benchmark when the summary declares one."""
    if summary.benchmark_name and summary.benchmark_name != benchmark_name:
        raise ValueError(
            f"Baseline run '{run_dir}' is for benchmark '{summary.benchmark_name}', "
            f"expected '{benchmark_name}'."
        )
