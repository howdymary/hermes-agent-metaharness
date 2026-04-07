from pathlib import Path

from click.testing import CliRunner

from meta_harness.benchmark_runner import BenchmarkRunResult
from meta_harness.cli import main
from meta_harness.frontier import FrontierStore
from meta_harness.models import RunSummary, SearchSummary, SearchTrialResult


def test_show_frontier_cli_lists_ranked_entries(tmp_path):
    frontier = FrontierStore(tmp_path / "frontier.json")
    frontier.upsert_from_summary(
        RunSummary(
            benchmark_name="tblite",
            candidate_name="strong_candidate",
            candidate_path="/tmp/strong.py",
            run_dir=Path("/tmp/strong"),
            eval_metrics={"eval/pass_rate": 0.8, "eval/total_tasks": 20},
        )
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "show-frontier",
            "--frontier-path",
            str(frontier.path),
            "--benchmark",
            "tblite",
        ],
    )

    assert result.exit_code == 0
    assert "strong_candidate" in result.output
    assert "20" in result.output


def test_evaluate_vs_baseline_cli_dry_run(monkeypatch, tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    hermes_repo.mkdir()
    calls = []

    def fake_run_benchmark(config, run_spec, *, dry_run=False, timeout=None):
        calls.append(run_spec.candidate)
        return BenchmarkRunResult(
            command=["python3", str(run_spec.candidate), "--dry-run"],
            archive_root=run_spec.archive_root,
            returncode=0,
        )

    monkeypatch.setattr("meta_harness.cli.run_benchmark", fake_run_benchmark)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-vs-baseline",
            "--candidate",
            "candidates/template_candidate.py",
            "--benchmark",
            "tblite",
            "--hermes-repo",
            str(hermes_repo),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Baseline vs Candidate Evaluation" in result.output
    assert "fresh_candidate" in result.output
    assert len(calls) == 2


def test_search_candidates_cli_dry_run(monkeypatch, tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    hermes_repo.mkdir()

    def fake_run_structured_search(config, request, *, dry_run=False):
        assert dry_run is True
        return SearchSummary(
            benchmark_name=request.benchmark,
            baseline_candidate=request.baseline_candidate,
            baseline_source="fresh_candidate",
            baseline_run_dir=None,
            baseline_reference=None,
            seed_candidate=request.seed_candidate,
            workspace_dir=str(tmp_path / "workspace"),
            generated_candidates_dir=str(tmp_path / "workspace" / "generated_candidates"),
            trial_results=[
                SearchTrialResult(
                    mutation_slug="plan_briefly",
                    mutation_description="Prepend a brief planning reminder.",
                    candidate_name="seed__plan_briefly",
                    candidate_path=str(tmp_path / "workspace" / "generated_candidates" / "seed__plan_briefly.py"),
                )
            ],
        )

    monkeypatch.setattr("meta_harness.cli.run_structured_search", fake_run_structured_search)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "search-candidates",
            "--seed-candidate",
            "candidates/template_candidate.py",
            "--benchmark",
            "tblite",
            "--hermes-repo",
            str(hermes_repo),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Structured Search" in result.output
    assert "fresh_candidate" in result.output
    assert "seed__plan_briefly" in result.output


def test_evaluate_candidate_cli_supports_launcher_prefix(monkeypatch, tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    hermes_repo.mkdir()
    captured = {}

    def fake_run_benchmark(config, run_spec, *, dry_run=False, timeout=None):
        captured["launch_prefix"] = config.launch_prefix
        captured["python_executable"] = config.python_executable
        return BenchmarkRunResult(
            command=["uv", "run", "--python", "3.12", "--extra", "rl", "python", "bench.py"],
            archive_root=run_spec.archive_root,
            returncode=0,
        )

    monkeypatch.setattr("meta_harness.cli.run_benchmark", fake_run_benchmark)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "evaluate-candidate",
            "--candidate",
            "snapshot_baseline",
            "--benchmark",
            "tblite",
            "--hermes-repo",
            str(hermes_repo),
            "--launcher-prefix",
            "uv run --python 3.12 --extra rl",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert captured["launch_prefix"] == ("uv", "run", "--python", "3.12", "--extra", "rl")
    assert captured["python_executable"] == "python"
