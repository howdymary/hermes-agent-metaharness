from pathlib import Path

from meta_harness.frontier import FrontierStore
from meta_harness.models import RunSummary


def test_frontier_upsert_and_best(tmp_path):
    frontier = FrontierStore(tmp_path / "frontier.json")

    baseline = RunSummary(
        benchmark_name="tblite",
        candidate_name="baseline",
        candidate_path="/tmp/baseline.py",
        run_dir=Path("/tmp/base"),
        eval_metrics={"eval/pass_rate": 0.2},
    )
    improved = RunSummary(
        benchmark_name="tblite",
        candidate_name="candidate",
        candidate_path="/tmp/candidate.py",
        run_dir=Path("/tmp/candidate"),
        eval_metrics={"eval/pass_rate": 0.4},
    )

    frontier.upsert_from_summary(baseline)
    frontier.upsert_from_summary(improved)

    best = frontier.best_for_benchmark("tblite")
    assert best.candidate_name == "candidate"
    assert best.pass_rate == 0.4
