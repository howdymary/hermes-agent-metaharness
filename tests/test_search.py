from pathlib import Path

from meta_harness.benchmark_runner import BenchmarkRunResult
from meta_harness.config import MetaHarnessConfig
from meta_harness.frontier import FrontierStore
from meta_harness.models import RunSummary
from meta_harness.search import StructuredSearchRequest, run_structured_search


def _make_config(tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    hermes_repo.mkdir()
    return MetaHarnessConfig(
        hermes_agent_path=hermes_repo,
        output_dir=tmp_path / "output",
        python_executable="python3",
    )


def test_run_structured_search_dry_run_generates_candidates(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    seed_candidate = tmp_path / "seed.py"
    seed_candidate.write_text("# seed\n", encoding="utf-8")

    calls = []

    def fake_run_benchmark(config_arg, run_spec, *, dry_run=False, capture_output=False):
        assert dry_run is True
        calls.append(run_spec.candidate)
        return BenchmarkRunResult(
            command=["python3", run_spec.candidate],
            archive_root=run_spec.archive_root,
            returncode=0,
        )

    monkeypatch.setattr("meta_harness.search.run_benchmark", fake_run_benchmark)

    summary = run_structured_search(
        config,
        StructuredSearchRequest(
            benchmark="tblite",
            seed_candidate=str(seed_candidate),
            baseline_candidate="snapshot_baseline",
            workspace_dir=tmp_path / "workspace",
            mutation_slugs=("plan_briefly", "shorter_loop"),
            python_executable="python3",
        ),
        dry_run=True,
    )

    generated_dir = Path(summary.generated_candidates_dir)
    assert generated_dir.exists()
    assert len(list(generated_dir.glob("*.py"))) == 2
    assert len(summary.trial_results) == 2
    assert summary.best_mutation_slug is None
    assert calls[0] == "snapshot_baseline"


def test_run_structured_search_writes_reports_and_updates_frontier(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    seed_candidate = tmp_path / "seed.py"
    seed_candidate.write_text("# seed\n", encoding="utf-8")
    frontier_path = tmp_path / "frontier.json"

    def fake_run_benchmark(config_arg, run_spec, *, dry_run=False, capture_output=False):
        run_dir = run_spec.archive_root / (run_spec.run_name or Path(run_spec.candidate).stem)
        run_dir.mkdir(parents=True, exist_ok=True)

        if str(run_spec.candidate) == "snapshot_baseline":
            summary = RunSummary(
                benchmark_name=run_spec.benchmark,
                candidate_name="snapshot_baseline",
                candidate_path="/tmp/snapshot_baseline.py",
                run_dir=run_dir,
                eval_metrics={
                    "eval/pass_rate": 0.30,
                    "eval/passed_tasks": 3,
                    "eval/evaluation_time_seconds": 12.0,
                },
                task_results=[
                    {"task_name": "a", "passed": True, "reward": 1.0},
                    {"task_name": "b", "passed": False, "reward": 0.0},
                ],
            )
        elif "plan_briefly" in str(run_spec.candidate):
            summary = RunSummary(
                benchmark_name=run_spec.benchmark,
                candidate_name=Path(run_spec.candidate).stem,
                candidate_path=str(run_spec.candidate),
                run_dir=run_dir,
                eval_metrics={
                    "eval/pass_rate": 0.50,
                    "eval/passed_tasks": 4,
                    "eval/evaluation_time_seconds": 11.0,
                },
                task_results=[
                    {"task_name": "a", "passed": True, "reward": 1.0},
                    {"task_name": "b", "passed": True, "reward": 1.0},
                ],
            )
        else:
            summary = RunSummary(
                benchmark_name=run_spec.benchmark,
                candidate_name=Path(run_spec.candidate).stem,
                candidate_path=str(run_spec.candidate),
                run_dir=run_dir,
                eval_metrics={
                    "eval/pass_rate": 0.20,
                    "eval/passed_tasks": 2,
                    "eval/evaluation_time_seconds": 10.0,
                },
                task_results=[
                    {"task_name": "a", "passed": False, "reward": 0.0},
                    {"task_name": "b", "passed": False, "reward": 0.0},
                ],
            )

        return BenchmarkRunResult(
            command=["python3", str(run_spec.candidate)],
            archive_root=run_spec.archive_root,
            returncode=0,
            run_dir=run_dir,
            summary=summary,
        )

    monkeypatch.setattr("meta_harness.search.run_benchmark", fake_run_benchmark)

    summary = run_structured_search(
        config,
        StructuredSearchRequest(
            benchmark="tblite",
            seed_candidate=str(seed_candidate),
            baseline_candidate="snapshot_baseline",
            workspace_dir=tmp_path / "workspace",
            mutation_slugs=("plan_briefly", "shorter_loop"),
            frontier_path=frontier_path,
            python_executable="python3",
        ),
        dry_run=False,
    )

    assert summary.best_mutation_slug == "plan_briefly"
    assert (tmp_path / "workspace" / "search_summary.json").exists()
    assert (tmp_path / "workspace" / "reports" / "plan_briefly.json").exists()
    assert (tmp_path / "workspace" / "reports" / "shorter_loop.json").exists()

    frontier = FrontierStore(frontier_path)
    best = frontier.best_for_benchmark("tblite")
    assert "plan_briefly" in best.candidate_name
    assert best.status == "candidate_beats_baseline"
