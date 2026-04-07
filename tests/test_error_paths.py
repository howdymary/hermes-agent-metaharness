"""Tests for error paths across the meta_harness package."""

import json

import pytest

from meta_harness.archive_reader import (
    find_run_dirs,
    load_manifest,
    load_run_summary,
    load_task_records,
)
from meta_harness.candidate_registry import resolve_candidate_path
from meta_harness.comparison import compare_runs
from meta_harness.frontier import FrontierStore
from meta_harness.models import RunSummary


# --- archive_reader error paths ---


def test_load_run_summary_malformed_json(tmp_path):
    run_dir = tmp_path / "bad_run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("not json at all")

    with pytest.raises(ValueError, match="Malformed summary.json"):
        load_run_summary(run_dir)


def test_load_run_summary_non_object_json(tmp_path):
    run_dir = tmp_path / "array_run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps([1, 2, 3]))

    with pytest.raises(ValueError, match="not a JSON object"):
        load_run_summary(run_dir)


def test_load_run_summary_missing_file(tmp_path):
    run_dir = tmp_path / "no_summary"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing summary.json"):
        load_run_summary(run_dir)


def test_find_run_dirs_missing_root(tmp_path):
    assert find_run_dirs(tmp_path / "nonexistent") == []


def test_load_task_records_skips_corrupted(tmp_path):
    run_dir = tmp_path / "run_with_bad_tasks"
    tasks_dir = run_dir / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "good.json").write_text(json.dumps({"task_name": "good"}))
    (tasks_dir / "bad.json").write_text("{{invalid")

    records = load_task_records(run_dir)
    assert len(records) == 1
    assert records[0]["task_name"] == "good"


def test_load_manifest_non_object_json(tmp_path):
    run_dir = tmp_path / "manifest_array"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(["not", "an", "object"]))

    with pytest.raises(ValueError, match="manifest.json.*not a JSON object"):
        load_manifest(run_dir)


# --- candidate_registry error paths ---


def test_resolve_candidate_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not resolve"):
        resolve_candidate_path("nonexistent_candidate", tmp_path)


def test_resolve_candidate_path_containment_rejects_escape(tmp_path):
    """A candidate outside all allowed roots should be rejected."""
    outside = tmp_path / "outside"
    outside.mkdir()
    evil = outside / "evil.py"
    evil.write_text("# evil")

    hermes = tmp_path / "hermes-agent"
    hermes.mkdir()

    # Passing an absolute path that exists but is outside hermes and cwd
    with pytest.raises(ValueError, match="outside all allowed directories"):
        resolve_candidate_path(str(evil), hermes)


# --- comparison edge cases ---


def test_compare_runs_one_sided_tasks():
    """Tasks present in only one run get 'unchanged' status but report tracks asymmetry."""
    baseline = RunSummary(
        benchmark_name="tblite",
        candidate_name="base",
        candidate_path="/tmp/base.py",
        run_dir=__import__("pathlib").Path("/tmp/base"),
        task_results=[
            {"task_name": "only_in_baseline", "passed": True, "reward": 1.0},
            {"task_name": "shared", "passed": False, "reward": 0.0},
        ],
    )
    candidate = RunSummary(
        benchmark_name="tblite",
        candidate_name="cand",
        candidate_path="/tmp/cand.py",
        run_dir=__import__("pathlib").Path("/tmp/cand"),
        task_results=[
            {"task_name": "only_in_candidate", "passed": True, "reward": 1.0},
            {"task_name": "shared", "passed": True, "reward": 1.0},
        ],
    )
    comparison = compare_runs(baseline, candidate)
    statuses = {d.task_name: d.status for d in comparison.task_deltas}
    assert statuses["shared"] == "improved"
    assert "only_in_baseline" in statuses
    assert "only_in_candidate" in statuses


# --- frontier error paths ---


def test_frontier_best_for_benchmark_empty(tmp_path):
    frontier = FrontierStore(tmp_path / "empty_frontier.json")
    with pytest.raises(FileNotFoundError, match="No frontier entries"):
        frontier.best_for_benchmark("tblite")


def test_frontier_upsert_updates_existing(tmp_path):
    """Upserting same identity should replace, not append."""
    frontier = FrontierStore(tmp_path / "frontier.json")
    summary_v1 = RunSummary(
        benchmark_name="tblite",
        candidate_name="test",
        candidate_path="/tmp/test.py",
        run_dir=__import__("pathlib").Path("/tmp/v1"),
        eval_metrics={"eval/pass_rate": 0.2},
    )
    summary_v2 = RunSummary(
        benchmark_name="tblite",
        candidate_name="test",
        candidate_path="/tmp/test.py",
        run_dir=__import__("pathlib").Path("/tmp/v2"),
        eval_metrics={"eval/pass_rate": 0.5},
    )
    frontier.upsert_from_summary(summary_v1)
    frontier.upsert_from_summary(summary_v2)

    entries = frontier.load()
    matching = [e for e in entries if e.candidate_name == "test"]
    assert len(matching) == 1
    assert matching[0].pass_rate == 0.5
