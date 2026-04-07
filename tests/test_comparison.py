from pathlib import Path

from meta_harness.comparison import build_comparison_report, compare_runs
from meta_harness.models import RunSummary


def test_compare_runs_detects_metric_and_task_deltas():
    baseline = RunSummary(
        benchmark_name="tblite",
        candidate_name="baseline",
        candidate_path="/tmp/base.py",
        run_dir=Path("/tmp/base"),
        eval_metrics={"eval/pass_rate": 0.2},
        task_results=[
            {"task_name": "a", "passed": False, "reward": 0.0},
            {"task_name": "b", "passed": True, "reward": 1.0},
        ],
    )
    candidate = RunSummary(
        benchmark_name="tblite",
        candidate_name="candidate",
        candidate_path="/tmp/candidate.py",
        run_dir=Path("/tmp/candidate"),
        eval_metrics={"eval/pass_rate": 0.3},
        task_results=[
            {"task_name": "a", "passed": True, "reward": 1.0},
            {"task_name": "b", "passed": True, "reward": 1.0},
        ],
    )

    comparison = compare_runs(baseline, candidate)
    assert comparison.metric_deltas["eval/pass_rate"] == 0.1
    statuses = {delta.task_name: delta.status for delta in comparison.task_deltas}
    assert statuses["a"] == "improved"
    assert statuses["b"] == "unchanged"


def test_build_comparison_report_summarizes_improvements():
    baseline = RunSummary(
        benchmark_name="tblite",
        candidate_name="baseline",
        candidate_path="/tmp/base.py",
        run_dir=Path("/tmp/base"),
        eval_metrics={
            "eval/pass_rate": 0.25,
            "eval/passed_tasks": 1,
            "eval/evaluation_time_seconds": 10.0,
        },
        task_results=[
            {"task_name": "a", "passed": False, "reward": 0.0},
            {"task_name": "b", "passed": True, "reward": 1.0},
        ],
    )
    candidate = RunSummary(
        benchmark_name="tblite",
        candidate_name="candidate",
        candidate_path="/tmp/candidate.py",
        run_dir=Path("/tmp/candidate"),
        eval_metrics={
            "eval/pass_rate": 0.50,
            "eval/passed_tasks": 2,
            "eval/evaluation_time_seconds": 8.0,
        },
        task_results=[
            {"task_name": "a", "passed": True, "reward": 1.0},
            {"task_name": "b", "passed": True, "reward": 1.0},
        ],
    )

    report = build_comparison_report(baseline, candidate)
    assert report.improved_tasks == 1
    assert report.regressed_tasks == 0
    assert report.net_task_gain == 1
    assert report.pass_rate_delta == 0.25
    assert report.passed_tasks_delta == 1.0
    assert report.evaluation_time_delta_seconds == -2.0
    assert report.improved_task_names == ["a"]
    assert report.candidate_better is True
