"""Helpers for selecting a baseline candidate or run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from meta_harness.archive_reader import load_run_summary
from meta_harness.comparability import build_task_selection_metadata, extract_task_selection_metadata
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
    task_filter: Optional[str] = None,
    skip_tasks: Optional[str] = None,
) -> BaselineSelection:
    """Resolve whether baseline should come from a fresh candidate run or an existing artifact."""
    if baseline_run_dir is not None and baseline_frontier_path is not None:
        raise ValueError("Choose at most one baseline source: --baseline-run or --baseline-from-frontier.")

    expected_selection = build_task_selection_metadata(
        task_filter=task_filter,
        skip_tasks=skip_tasks,
    )

    if baseline_run_dir is not None:
        run_dir = baseline_run_dir.expanduser().resolve()
        if not run_dir.is_dir():
            raise ValueError(
                f"Baseline run dir '{run_dir}' does not exist or is not a directory."
            )
        summary = load_run_summary(run_dir)
        _validate_baseline_benchmark(summary, benchmark_name, run_dir)
        _validate_task_selection(summary, run_dir, expected_selection)
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
        matching_entries = frontier.top_for_benchmark(
            benchmark_name,
            limit=1,
            task_selection_hash=str(expected_selection.get("selection_hash") or ""),
        )
        if not matching_entries:
            raise ValueError(
                "No frontier baseline matches the current task selection. "
                "Run a comparable baseline first or use --baseline-candidate/--baseline-run."
            )
        entry = matching_entries[0]
        run_dir = Path(entry.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise ValueError(
                f"Frontier entry run_dir '{run_dir}' does not exist or is not a directory."
            )
        summary = load_run_summary(run_dir)
        _validate_baseline_benchmark(summary, benchmark_name, run_dir)
        _validate_task_selection(summary, run_dir, expected_selection)
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


def _validate_task_selection(
    summary: RunSummary,
    run_dir: Path,
    expected_selection: dict,
) -> None:
    """Ensure a reused baseline was evaluated on the same task subset."""
    observed_selection = extract_task_selection_metadata(summary.manifest)
    if observed_selection is None:
        raise ValueError(
            f"Baseline run '{run_dir}' is missing outer-loop task selection metadata. "
            "Re-run the baseline with a current version of hermes-agent-metaharness."
        )

    expected_hash = str(expected_selection.get("selection_hash") or "")
    observed_hash = str(observed_selection.get("selection_hash") or "")
    if observed_hash != expected_hash:
        raise ValueError(
            f"Baseline run '{run_dir}' was evaluated on a different task selection. "
            f"expected_hash={expected_hash}, observed_hash={observed_hash}"
        )
