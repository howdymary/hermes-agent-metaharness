import json

from meta_harness.archive_reader import (
    find_latest_run_dir,
    load_manifest,
    load_run_summary,
    load_task_records,
)


def test_load_run_summary(tmp_path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "benchmark_name": "tblite",
        "candidate_name": "snapshot_baseline",
        "candidate_path": "/tmp/candidate.py",
        "run_dir": str(run_dir),
        "eval_metrics": {"eval/pass_rate": 0.4},
        "task_results": [{"task_name": "foo", "passed": True}],
    }))
    (run_dir / "manifest.json").write_text(json.dumps({"run_name": "smoke"}))

    summary = load_run_summary(run_dir)
    assert summary.benchmark_name == "tblite"
    assert summary.candidate_name == "snapshot_baseline"
    assert summary.manifest["run_name"] == "smoke"


def test_load_task_records(tmp_path):
    run_dir = tmp_path / "run_b"
    tasks_dir = run_dir / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "a.json").write_text(json.dumps({"task": {"task_name": "a"}}))
    (tasks_dir / "b.json").write_text(json.dumps({"task": {"task_name": "b"}}))

    records = load_task_records(run_dir)
    assert len(records) == 2


def test_find_latest_run_dir(tmp_path):
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    older.mkdir()
    newer.mkdir()
    (older / "summary.json").write_text("{}")
    (newer / "summary.json").write_text("{}")

    latest = find_latest_run_dir(tmp_path)
    assert latest == newer


def test_load_manifest_missing_is_empty(tmp_path):
    run_dir = tmp_path / "run_c"
    run_dir.mkdir()

    assert load_manifest(run_dir) == {}


def test_load_manifest_malformed_json_raises(tmp_path):
    run_dir = tmp_path / "run_d"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{not-valid", encoding="utf-8")

    try:
        load_manifest(run_dir)
    except ValueError as exc:
        assert "Malformed manifest.json" in str(exc)
    else:
        raise AssertionError("Expected malformed manifest.json to raise ValueError")


def test_load_run_summary_rejects_non_object_eval_metrics(tmp_path):
    run_dir = tmp_path / "run_e"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "benchmark_name": "tblite",
        "candidate_name": "snapshot_baseline",
        "candidate_path": "/tmp/candidate.py",
        "run_dir": str(run_dir),
        "eval_metrics": ["bad"],
        "task_results": [],
    }), encoding="utf-8")

    try:
        load_run_summary(run_dir)
    except ValueError as exc:
        assert "non-object eval_metrics" in str(exc)
    else:
        raise AssertionError("Expected invalid eval_metrics to raise ValueError")


def test_load_run_summary_rejects_non_list_task_results(tmp_path):
    run_dir = tmp_path / "run_f"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({
        "benchmark_name": "tblite",
        "candidate_name": "snapshot_baseline",
        "candidate_path": "/tmp/candidate.py",
        "run_dir": str(run_dir),
        "eval_metrics": {},
        "task_results": {"bad": True},
    }), encoding="utf-8")

    try:
        load_run_summary(run_dir)
    except ValueError as exc:
        assert "non-list task_results" in str(exc)
    else:
        raise AssertionError("Expected invalid task_results to raise ValueError")
