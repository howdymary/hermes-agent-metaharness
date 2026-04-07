import json
from pathlib import Path

import pytest

from meta_harness.baseline import resolve_baseline_selection
from meta_harness.frontier import FrontierStore
from meta_harness.models import RunSummary


def test_resolve_baseline_selection_defaults_to_fresh_candidate():
    selection = resolve_baseline_selection(benchmark_name="tblite", baseline_candidate="snapshot_baseline")
    assert selection.source == "fresh_candidate"
    assert selection.requires_fresh_run is True
    assert selection.baseline_candidate == "snapshot_baseline"


def test_resolve_baseline_selection_from_run_dir(tmp_path):
    run_dir = tmp_path / "baseline_run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "benchmark_name": "tblite",
        "candidate_name": "best_candidate",
        "candidate_path": "/tmp/best.py",
        "run_dir": str(run_dir),
        "eval_metrics": {"eval/pass_rate": 0.8},
        "task_results": [],
    }), encoding="utf-8")

    selection = resolve_baseline_selection(
        benchmark_name="tblite",
        baseline_run_dir=run_dir,
    )
    assert selection.source == "existing_run"
    assert selection.requires_fresh_run is False
    assert selection.summary is not None
    assert selection.summary.candidate_name == "best_candidate"


def test_resolve_baseline_selection_from_frontier(tmp_path):
    run_dir = tmp_path / "frontier_run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "benchmark_name": "tblite",
        "candidate_name": "frontier_best",
        "candidate_path": "/tmp/frontier_best.py",
        "run_dir": str(run_dir),
        "eval_metrics": {"eval/pass_rate": 0.9},
        "task_results": [],
    }), encoding="utf-8")

    frontier = FrontierStore(tmp_path / "frontier.json")
    frontier.upsert_from_summary(
        RunSummary(
            benchmark_name="tblite",
            candidate_name="frontier_best",
            candidate_path="/tmp/frontier_best.py",
            run_dir=run_dir,
            eval_metrics={"eval/pass_rate": 0.9},
        )
    )

    selection = resolve_baseline_selection(
        benchmark_name="tblite",
        baseline_frontier_path=frontier.path,
    )
    assert selection.source == "frontier_best"
    assert selection.summary is not None
    assert selection.summary.candidate_name == "frontier_best"


def test_resolve_baseline_selection_rejects_mixed_sources(tmp_path):
    with pytest.raises(ValueError, match="Choose at most one baseline source"):
        resolve_baseline_selection(
            benchmark_name="tblite",
            baseline_run_dir=tmp_path,
            baseline_frontier_path=tmp_path / "frontier.json",
        )


def test_resolve_baseline_selection_rejects_wrong_benchmark(tmp_path):
    run_dir = tmp_path / "wrong_benchmark"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "benchmark_name": "tb2",
        "candidate_name": "wrong_benchmark",
        "candidate_path": "/tmp/wrong.py",
        "run_dir": str(run_dir),
        "eval_metrics": {},
        "task_results": [],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 'tblite'"):
        resolve_baseline_selection(
            benchmark_name="tblite",
            baseline_run_dir=run_dir,
        )
