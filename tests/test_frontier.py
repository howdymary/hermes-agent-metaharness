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
        eval_metrics={"eval/pass_rate": 0.2, "eval/total_tasks": 10},
        manifest={"outer_loop": {"benchmark_runner": {"task_selection": {"selection_hash": "hash_a"}}}},
    )
    improved = RunSummary(
        benchmark_name="tblite",
        candidate_name="candidate",
        candidate_path="/tmp/candidate.py",
        run_dir=Path("/tmp/candidate"),
        eval_metrics={"eval/pass_rate": 0.4, "eval/total_tasks": 10},
        manifest={"outer_loop": {"benchmark_runner": {"task_selection": {"selection_hash": "hash_a"}}}},
    )

    frontier.upsert_from_summary(baseline)
    frontier.upsert_from_summary(improved)

    best = frontier.best_for_benchmark("tblite")
    assert best.candidate_name == "candidate"
    assert best.pass_rate == 0.4


def test_frontier_top_for_benchmark_orders_entries(tmp_path):
    frontier = FrontierStore(tmp_path / "frontier.json")

    for name, pass_rate, total_tasks in [("alpha", 0.1, 50), ("gamma", 0.7, 20), ("beta", 0.7, 50)]:
        frontier.upsert_from_summary(
            RunSummary(
                benchmark_name="tblite",
                candidate_name=name,
                candidate_path=f"/tmp/{name}.py",
                run_dir=Path(f"/tmp/{name}"),
                eval_metrics={"eval/pass_rate": pass_rate, "eval/total_tasks": total_tasks},
                manifest={"outer_loop": {"benchmark_runner": {"task_selection": {"selection_hash": f"hash_{name}"}}}},
            )
        )

    ranked = frontier.top_for_benchmark("tblite", limit=3)
    assert [entry.candidate_name for entry in ranked] == ["beta", "alpha", "gamma"]
    assert [entry.total_tasks for entry in ranked] == [50, 50, 20]


def test_frontier_top_for_benchmark_filters_by_task_selection_hash(tmp_path):
    frontier = FrontierStore(tmp_path / "frontier.json")

    for name, selection_hash in [("alpha", "hash_a"), ("beta", "hash_b")]:
        frontier.upsert_from_summary(
            RunSummary(
                benchmark_name="tblite",
                candidate_name=name,
                candidate_path=f"/tmp/{name}.py",
                run_dir=Path(f"/tmp/{name}"),
                eval_metrics={"eval/pass_rate": 0.5, "eval/total_tasks": 10},
                manifest={"outer_loop": {"benchmark_runner": {"task_selection": {"selection_hash": selection_hash}}}},
            )
        )

    ranked = frontier.top_for_benchmark("tblite", task_selection_hash="hash_b")
    assert [entry.candidate_name for entry in ranked] == ["beta"]
