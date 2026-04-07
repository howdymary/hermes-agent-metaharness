"""Compare Hermes Meta-Harness benchmark runs."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from meta_harness.models import ComparisonReport, RunComparison, RunSummary, TaskDelta


def _numeric_metric_deltas(
    baseline_metrics: Dict,
    candidate_metrics: Dict,
) -> Dict[str, float]:
    deltas = {}
    keys = sorted(set(baseline_metrics) & set(candidate_metrics))
    for key in keys:
        baseline_value = baseline_metrics.get(key)
        candidate_value = candidate_metrics.get(key)
        if isinstance(baseline_value, (int, float)) and isinstance(candidate_value, (int, float)):
            deltas[key] = round(float(candidate_value) - float(baseline_value), 10)
    return deltas


def _task_map(task_results: List[Dict]) -> Dict[str, Dict]:
    return {
        str(task.get("task_name")): task
        for task in task_results
        if task.get("task_name")
    }


def _task_status(
    baseline_passed: Optional[bool],
    candidate_passed: Optional[bool],
) -> str:
    if baseline_passed is True and candidate_passed is False:
        return "regressed"
    if baseline_passed is False and candidate_passed is True:
        return "improved"
    return "unchanged"


def _metric_delta(metrics: Dict[str, float], key: str) -> Optional[float]:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def compare_runs(baseline: RunSummary, candidate: RunSummary) -> RunComparison:
    """Compare baseline and candidate run summaries."""
    baseline_tasks = _task_map(baseline.task_results)
    candidate_tasks = _task_map(candidate.task_results)

    task_deltas = []
    for task_name in sorted(set(baseline_tasks) | set(candidate_tasks)):
        base = baseline_tasks.get(task_name, {})
        cand = candidate_tasks.get(task_name, {})
        base_passed = base.get("passed")
        cand_passed = cand.get("passed")
        status = _task_status(base_passed, cand_passed)

        task_deltas.append(
            TaskDelta(
                task_name=task_name,
                baseline_passed=base_passed,
                candidate_passed=cand_passed,
                baseline_reward=base.get("reward"),
                candidate_reward=cand.get("reward"),
                status=status,
            )
        )

    return RunComparison(
        baseline_run_dir=baseline.run_dir,
        candidate_run_dir=candidate.run_dir,
        benchmark_name=candidate.benchmark_name or baseline.benchmark_name,
        baseline_candidate_name=baseline.candidate_name,
        candidate_name=candidate.candidate_name,
        metric_deltas=_numeric_metric_deltas(baseline.eval_metrics, candidate.eval_metrics),
        task_deltas=task_deltas,
    )


def build_comparison_report(baseline: RunSummary, candidate: RunSummary) -> ComparisonReport:
    """Build a ranking-friendly report for one baseline-vs-candidate comparison."""
    comparison = compare_runs(baseline, candidate)
    baseline_task_names: Set[str] = {delta.task_name for delta in comparison.task_deltas if delta.baseline_passed is not None}
    candidate_task_names: Set[str] = {delta.task_name for delta in comparison.task_deltas if delta.candidate_passed is not None}
    overlapping_task_names = baseline_task_names & candidate_task_names

    improved = [delta.task_name for delta in comparison.task_deltas if delta.status == "improved"]
    regressed = [delta.task_name for delta in comparison.task_deltas if delta.status == "regressed"]
    unchanged = [delta.task_name for delta in comparison.task_deltas if delta.status == "unchanged"]

    return ComparisonReport(
        benchmark_name=comparison.benchmark_name,
        baseline_candidate_name=comparison.baseline_candidate_name,
        candidate_name=comparison.candidate_name,
        baseline_run_dir=comparison.baseline_run_dir,
        candidate_run_dir=comparison.candidate_run_dir,
        total_tasks=len(comparison.task_deltas),
        overlapping_tasks=len(overlapping_task_names),
        improved_tasks=len(improved),
        regressed_tasks=len(regressed),
        unchanged_tasks=len(unchanged),
        baseline_only_tasks=len(baseline_task_names - candidate_task_names),
        candidate_only_tasks=len(candidate_task_names - baseline_task_names),
        pass_rate_delta=float(comparison.metric_deltas.get("eval/pass_rate", 0.0)),
        passed_tasks_delta=float(comparison.metric_deltas.get("eval/passed_tasks", 0.0)),
        evaluation_time_delta_seconds=_metric_delta(comparison.metric_deltas, "eval/evaluation_time_seconds"),
        metric_deltas=comparison.metric_deltas,
        improved_task_names=improved,
        regressed_task_names=regressed,
        unchanged_task_names=unchanged,
    )
