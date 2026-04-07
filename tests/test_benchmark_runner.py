from pathlib import Path

from meta_harness.benchmark_runner import build_benchmark_command, resolve_benchmark_script
from meta_harness.config import MetaHarnessConfig
from meta_harness.models import BenchmarkRunSpec


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
