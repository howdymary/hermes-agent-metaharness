import subprocess
import json
from pathlib import Path

import pytest

from meta_harness.benchmark_runner import build_benchmark_command, resolve_benchmark_script, run_benchmark
from meta_harness.comparability import build_task_selection_metadata
from meta_harness.config import MetaHarnessConfig
from meta_harness.models import BenchmarkRunSpec, RunSummary


def test_resolve_benchmark_script(tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    script = hermes_repo / "environments" / "benchmarks" / "tblite" / "tblite_env.py"
    script.parent.mkdir(parents=True)
    script.write_text("# test")

    assert resolve_benchmark_script("tblite", hermes_repo) == script.resolve()


def test_build_benchmark_command(tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    benchmark_script = hermes_repo / "environments" / "benchmarks" / "tblite" / "tblite_env.py"
    benchmark_script.parent.mkdir(parents=True)
    benchmark_script.write_text("# test")

    candidates_dir = hermes_repo / "environments" / "meta_harness" / "candidates"
    candidates_dir.mkdir(parents=True)
    candidate_file = candidates_dir / "snapshot_baseline.py"
    candidate_file.write_text("# candidate")

    config = MetaHarnessConfig(
        hermes_agent_path=hermes_repo,
        output_dir=tmp_path / "output",
        python_executable="python3",
    )
    run_spec = BenchmarkRunSpec(
        benchmark="tblite",
        candidate="snapshot_baseline",
        archive_root=tmp_path / "archive",
        run_name="smoke",
        task_filter="foo,bar",
        python_executable=config.python_executable,
    )

    command = build_benchmark_command(config, run_spec)
    assert command[0] == "python3"
    assert command[1] == str(benchmark_script.resolve())
    assert "--env.meta_harness_candidate" in command
    assert str(candidate_file.resolve()) in command
    assert "--env.meta_harness_archive_dir" in command
    assert "--env.meta_harness_run_name" in command
    assert "smoke" in command
    assert "--env.task_filter" in command


def test_build_benchmark_command_with_launch_prefix(tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    benchmark_script = hermes_repo / "environments" / "benchmarks" / "tblite" / "tblite_env.py"
    benchmark_script.parent.mkdir(parents=True)
    benchmark_script.write_text("# test")

    candidates_dir = hermes_repo / "environments" / "meta_harness" / "candidates"
    candidates_dir.mkdir(parents=True)
    candidate_file = candidates_dir / "snapshot_baseline.py"
    candidate_file.write_text("# candidate")

    config = MetaHarnessConfig(
        hermes_agent_path=hermes_repo,
        output_dir=tmp_path / "output",
        launch_prefix=("uv", "run", "--python", "3.12", "--extra", "rl"),
        python_executable="python",
    )
    run_spec = BenchmarkRunSpec(
        benchmark="tblite",
        candidate="snapshot_baseline",
        archive_root=tmp_path / "archive",
        python_executable=config.python_executable,
    )

    command = build_benchmark_command(config, run_spec)
    assert command[:7] == ["uv", "run", "--python", "3.12", "--extra", "rl", "python"]
    assert command[7] == str(benchmark_script.resolve())
    assert str(candidate_file.resolve()) in command


def _make_runner_fixture(tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    benchmark_script = hermes_repo / "environments" / "benchmarks" / "tblite" / "tblite_env.py"
    benchmark_script.parent.mkdir(parents=True)
    benchmark_script.write_text("# test")

    candidates_dir = hermes_repo / "environments" / "meta_harness" / "candidates"
    candidates_dir.mkdir(parents=True)
    candidate_file = candidates_dir / "snapshot_baseline.py"
    candidate_file.write_text("# candidate")

    config = MetaHarnessConfig(
        hermes_agent_path=hermes_repo,
        output_dir=tmp_path / "output",
        python_executable="python3",
    )
    run_spec = BenchmarkRunSpec(
        benchmark="tblite",
        candidate="snapshot_baseline",
        archive_root=tmp_path / "archive",
        python_executable=config.python_executable,
    )
    return config, run_spec


def test_run_benchmark_timeout_raises_runtime_error(tmp_path, monkeypatch):
    config, run_spec = _make_runner_fixture(tmp_path)

    def fake_subprocess_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", args[0]), timeout=5)

    monkeypatch.setattr("meta_harness.benchmark_runner.subprocess.run", fake_subprocess_run)

    with pytest.raises(RuntimeError, match="Benchmark timed out"):
        run_benchmark(config, run_spec, timeout=5)


def test_run_benchmark_no_run_dir_raises_runtime_error(tmp_path, monkeypatch):
    config, run_spec = _make_runner_fixture(tmp_path)

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr("meta_harness.benchmark_runner.subprocess.run", lambda *args, **kwargs: Result())
    monkeypatch.setattr("meta_harness.benchmark_runner.find_run_dirs", lambda archive_root: [])

    with pytest.raises(RuntimeError, match="produced no run directory"):
        run_benchmark(config, run_spec)


def test_run_benchmark_failure_includes_stderr(tmp_path, monkeypatch):
    config, run_spec = _make_runner_fixture(tmp_path)

    class Result:
        returncode = 2
        stdout = ""
        stderr = "boom from benchmark"

    monkeypatch.setattr("meta_harness.benchmark_runner.subprocess.run", lambda *args, **kwargs: Result())

    with pytest.raises(RuntimeError, match="boom from benchmark"):
        run_benchmark(config, run_spec)


def test_run_benchmark_loads_latest_new_run_dir(tmp_path, monkeypatch):
    config, run_spec = _make_runner_fixture(tmp_path)
    archive_root = run_spec.archive_root
    before_run = archive_root / "before"
    after_run = archive_root / "after"
    before_run.mkdir(parents=True)
    after_run.mkdir(parents=True)

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    call_count = {"value": 0}

    def fake_find_run_dirs(path):
        call_count["value"] += 1
        return [before_run] if call_count["value"] == 1 else [before_run, after_run]

    monkeypatch.setattr("meta_harness.benchmark_runner.subprocess.run", lambda *args, **kwargs: Result())
    monkeypatch.setattr("meta_harness.benchmark_runner.find_run_dirs", fake_find_run_dirs)
    monkeypatch.setattr(
        "meta_harness.benchmark_runner.load_run_summary",
        lambda run_dir: RunSummary(
            benchmark_name="tblite",
            candidate_name="snapshot_baseline",
            candidate_path="/tmp/snapshot_baseline.py",
            run_dir=run_dir,
            eval_metrics={},
            task_results=[],
        ),
    )

    result = run_benchmark(config, run_spec)
    assert result.run_dir == after_run.resolve()


def test_run_benchmark_writes_outer_loop_manifest_metadata(tmp_path, monkeypatch):
    config, run_spec = _make_runner_fixture(tmp_path)
    run_spec.task_filter = "task_b, task_a"
    run_spec.skip_tasks = "task_x"
    run_dir = run_spec.archive_root / "run_a"
    run_dir.mkdir(parents=True)

    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    call_count = {"value": 0}

    def fake_find_run_dirs(path):
        call_count["value"] += 1
        return [] if call_count["value"] == 1 else [run_dir]

    monkeypatch.setattr("meta_harness.benchmark_runner.subprocess.run", lambda *args, **kwargs: Result())
    monkeypatch.setattr("meta_harness.benchmark_runner.find_run_dirs", fake_find_run_dirs)
    monkeypatch.setattr(
        "meta_harness.benchmark_runner.load_run_summary",
        lambda resolved_run_dir: RunSummary(
            benchmark_name="tblite",
            candidate_name="snapshot_baseline",
            candidate_path="/tmp/snapshot_baseline.py",
            run_dir=resolved_run_dir,
            eval_metrics={"eval/pass_rate": 0.5, "eval/total_tasks": 12},
            task_results=[{"task_name": "a", "passed": True}],
            manifest={},
        ),
    )

    run_benchmark(config, run_spec)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    runner_manifest = manifest["outer_loop"]["benchmark_runner"]
    assert runner_manifest["benchmark"] == "tblite"
    assert runner_manifest["candidate"] == "snapshot_baseline"
    assert runner_manifest["total_tasks"] == 12
    assert runner_manifest["task_selection"] == build_task_selection_metadata(
        task_filter="task_b, task_a",
        skip_tasks="task_x",
    )
