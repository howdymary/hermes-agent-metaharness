"""Compare Hermes Meta-Harness benchmark runs."""

from __future__ import annotations

from typing import Dict, List

from meta_harness.models import RunComparison, RunSummary, TaskDelta


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

        if base_passed is True and cand_passed is False:
            status = "regressed"
        elif base_passed is False and cand_passed is True:
            status = "improved"
        else:
            status = "unchanged"

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
