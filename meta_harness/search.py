"""Structured candidate search for Hermes Meta-Harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from meta_harness.benchmark_runner import run_benchmark
from meta_harness.candidate_registry import resolve_candidate_path
from meta_harness.comparison import build_comparison_report
from meta_harness.config import MetaHarnessConfig
from meta_harness.frontier import FrontierStore
from meta_harness.models import BenchmarkRunSpec, SearchSummary, SearchTrialResult
from meta_harness.mutation import MutationSpec, generate_variant_candidates, resolve_mutation_specs, safe_slug


@dataclass
class StructuredSearchRequest:
    """Configuration for one structured mutation search."""

    benchmark: str
    seed_candidate: str
    baseline_candidate: str = "snapshot_baseline"
    workspace_dir: Optional[Path] = None
    archive_root: Optional[Path] = None
    mutation_slugs: Sequence[str] = field(default_factory=tuple)
    hermes_config_path: Optional[Path] = None
    task_filter: Optional[str] = None
    skip_tasks: Optional[str] = None
    frontier_path: Optional[Path] = None
    python_executable: str = "python"


def default_workspace_dir(config: MetaHarnessConfig, benchmark: str, seed_candidate: str) -> Path:
    """Build a timestamped workspace path for one search run."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.output_dir / "searches" / benchmark / f"{stamp}__{safe_slug(seed_candidate)}"


def run_structured_search(
    config: MetaHarnessConfig,
    request: StructuredSearchRequest,
    *,
    dry_run: bool = False,
) -> SearchSummary:
    """Run a deterministic search over generated candidate wrappers."""
    workspace_dir = (
        request.workspace_dir.expanduser().resolve()
        if request.workspace_dir
        else default_workspace_dir(config, request.benchmark, request.seed_candidate)
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)

    archive_root = (
        request.archive_root.expanduser().resolve()
        if request.archive_root
        else workspace_dir / "archives"
    )
    generated_candidates_dir = workspace_dir / "generated_candidates"
    reports_dir = workspace_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    seed_candidate_path = resolve_candidate_path(request.seed_candidate, config.hermes_agent_path)
    seed_candidate_name = seed_candidate_path.stem
    mutations: List[MutationSpec] = resolve_mutation_specs(request.mutation_slugs)
    generated_candidates = generate_variant_candidates(
        seed_candidate_path=seed_candidate_path,
        seed_candidate_name=seed_candidate_name,
        output_dir=generated_candidates_dir,
        mutations=mutations,
    )

    baseline_spec = BenchmarkRunSpec(
        benchmark=request.benchmark,
        candidate=request.baseline_candidate,
        archive_root=archive_root,
        run_name=f"baseline__{safe_slug(request.baseline_candidate)}",
        hermes_config_path=request.hermes_config_path,
        task_filter=request.task_filter,
        skip_tasks=request.skip_tasks,
        python_executable=request.python_executable or config.python_executable,
    )
    baseline_result = run_benchmark(config, baseline_spec, dry_run=dry_run)

    summary = SearchSummary(
        benchmark_name=request.benchmark,
        baseline_candidate=request.baseline_candidate,
        baseline_run_dir=str(baseline_result.run_dir) if baseline_result.run_dir else None,
        seed_candidate=request.seed_candidate,
        workspace_dir=str(workspace_dir),
        generated_candidates_dir=str(generated_candidates_dir),
    )

    frontier = FrontierStore(request.frontier_path) if request.frontier_path else None

    for mutation, candidate_path in zip(mutations, generated_candidates):
        run_spec = BenchmarkRunSpec(
            benchmark=request.benchmark,
            candidate=str(candidate_path),
            archive_root=archive_root,
            run_name=f"trial__{mutation.slug}",
            hermes_config_path=request.hermes_config_path,
            task_filter=request.task_filter,
            skip_tasks=request.skip_tasks,
            python_executable=request.python_executable or config.python_executable,
        )
        trial_result = run_benchmark(config, run_spec, dry_run=dry_run)

        report = None
        if not dry_run:
            if baseline_result.summary is None or trial_result.summary is None:
                raise RuntimeError("Structured search expected benchmark summaries for baseline and trial.")
            report = build_comparison_report(baseline_result.summary, trial_result.summary)
            report_path = reports_dir / f"{mutation.slug}.json"
            report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            if frontier is not None:
                frontier.upsert_from_summary(
                    trial_result.summary,
                    status="candidate_beats_baseline" if report.candidate_better else "evaluated",
                    notes=(
                        f"baseline={request.baseline_candidate}; pass_rate_delta={report.pass_rate_delta:+.4f}; "
                        f"net_task_gain={report.net_task_gain}; regressions={report.regressed_tasks}"
                    ),
                )

        summary.trial_results.append(
            SearchTrialResult(
                mutation_slug=mutation.slug,
                mutation_description=mutation.description,
                candidate_name=candidate_path.stem,
                candidate_path=str(candidate_path),
                run_dir=str(trial_result.run_dir) if trial_result.run_dir else None,
                report=report,
                command=trial_result.command,
                returncode=trial_result.returncode,
            )
        )

    ranked = [trial for trial in summary.trial_results if trial.report is not None]
    if ranked:
        best = max(ranked, key=lambda trial: trial.report.ranking_key())
        summary.best_mutation_slug = best.mutation_slug
        summary.best_candidate_name = best.candidate_name
        summary.best_run_dir = best.run_dir

    if not dry_run:
        summary_path = workspace_dir / "search_summary.json"
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    return summary
