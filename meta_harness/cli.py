"""CLI for standalone Hermes Meta-Harness orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from meta_harness.archive_reader import load_run_summary
from meta_harness.benchmark_runner import run_benchmark
from meta_harness.candidate_registry import list_builtin_candidates
from meta_harness.comparison import compare_runs
from meta_harness.config import MetaHarnessConfig
from meta_harness.frontier import FrontierStore
from meta_harness.models import BenchmarkRunSpec

console = Console()


@click.group()
def main() -> None:
    """Standalone outer-loop Meta-Harness tooling for Hermes."""


@main.command("list-builtins")
@click.option("--hermes-repo", help="Path to the hermes-agent checkout.")
def list_builtins_cmd(hermes_repo: str = None) -> None:
    """List Hermes built-in Meta-Harness candidates."""
    config = MetaHarnessConfig()
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo).expanduser().resolve()

    for name in list_builtin_candidates(config.hermes_agent_path):
        console.print(name)


@main.command("evaluate-candidate")
@click.option("--candidate", required=True, help="Candidate file path or built-in Hermes candidate name.")
@click.option("--benchmark", default="tblite", type=click.Choice(["tblite", "tb2"]), show_default=True)
@click.option("--hermes-repo", help="Path to the hermes-agent checkout.")
@click.option("--archive-dir", help="Archive root for Hermes Meta-Harness run outputs.")
@click.option("--run-name", help="Optional friendly run name.")
@click.option("--hermes-config-path", help="Optional Hermes benchmark YAML config path.")
@click.option("--task-filter", help="Optional comma-separated task filter.")
@click.option("--skip-tasks", help="Optional comma-separated task skip list.")
@click.option("--frontier-path", help="Optional path to a frontier JSON file to update after evaluation.")
@click.option("--dry-run", is_flag=True, help="Print the benchmark command without executing it.")
def evaluate_candidate_cmd(
    candidate: str,
    benchmark: str,
    hermes_repo: str = None,
    archive_dir: str = None,
    run_name: str = None,
    hermes_config_path: str = None,
    task_filter: str = None,
    skip_tasks: str = None,
    frontier_path: str = None,
    dry_run: bool = False,
) -> None:
    """Evaluate one Meta-Harness candidate against Hermes."""
    config = MetaHarnessConfig()
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo).expanduser().resolve()

    archive_root = Path(archive_dir).expanduser().resolve() if archive_dir else (
        config.output_dir / "archives" / benchmark
    )

    run_spec = BenchmarkRunSpec(
        benchmark=benchmark,
        candidate=candidate,
        archive_root=archive_root,
        run_name=run_name,
        hermes_config_path=Path(hermes_config_path).expanduser().resolve() if hermes_config_path else None,
        task_filter=task_filter,
        skip_tasks=skip_tasks,
        python_executable=config.python_executable,
    )

    console.print(f"\n[bold cyan]Meta-Harness Candidate Evaluation[/bold cyan]")
    console.print(f"  Hermes repo: {config.hermes_agent_path}")
    console.print(f"  Benchmark: {benchmark}")
    console.print(f"  Candidate: {candidate}")
    console.print(f"  Archive root: {archive_root}")

    result = run_benchmark(config, run_spec, dry_run=dry_run)

    console.print("\n[bold]Command[/bold]")
    console.print("  " + " ".join(result.command))

    if dry_run:
        console.print("\n[yellow]Dry run only — benchmark not executed.[/yellow]")
        return

    if result.summary is None:
        console.print("\n[red]No summary was produced.[/red]")
        return

    summary = result.summary
    table = Table(title="Benchmark Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    pass_rate = summary.eval_metrics.get("eval/pass_rate")
    total_tasks = summary.eval_metrics.get("eval/total_tasks")
    passed_tasks = summary.eval_metrics.get("eval/passed_tasks")
    eval_time = summary.eval_metrics.get("eval/evaluation_time_seconds")

    table.add_row("Candidate", summary.candidate_name)
    table.add_row("Benchmark", summary.benchmark_name)
    table.add_row("Pass Rate", f"{pass_rate:.4f}" if isinstance(pass_rate, (int, float)) else str(pass_rate))
    table.add_row("Passed / Total", f"{passed_tasks}/{total_tasks}")
    table.add_row("Eval Time (s)", f"{eval_time:.1f}" if isinstance(eval_time, (int, float)) else str(eval_time))
    table.add_row("Run Dir", str(summary.run_dir))
    console.print()
    console.print(table)

    if frontier_path:
        frontier = FrontierStore(Path(frontier_path))
        entry = frontier.upsert_from_summary(summary)
        console.print(f"\n[green]Frontier updated:[/green] {entry.candidate_name} @ {entry.pass_rate:.4f}")


@main.command("compare-runs")
@click.option("--baseline-run", required=True, help="Path to the baseline Hermes Meta-Harness run dir.")
@click.option("--candidate-run", required=True, help="Path to the candidate Hermes Meta-Harness run dir.")
def compare_runs_cmd(baseline_run: str, candidate_run: str) -> None:
    """Compare two Hermes Meta-Harness run directories."""
    baseline = load_run_summary(Path(baseline_run))
    candidate = load_run_summary(Path(candidate_run))
    comparison = compare_runs(baseline, candidate)

    metrics_table = Table(title="Metric Deltas")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Delta", justify="right")
    for key, delta in sorted(comparison.metric_deltas.items()):
        metrics_table.add_row(key, f"{delta:+.4f}")
    console.print()
    console.print(metrics_table)

    task_table = Table(title="Task-Level Deltas")
    task_table.add_column("Task", style="bold")
    task_table.add_column("Status")
    task_table.add_column("Baseline")
    task_table.add_column("Candidate")
    for delta in comparison.task_deltas:
        task_table.add_row(
            delta.task_name,
            delta.status,
            str(delta.baseline_passed),
            str(delta.candidate_passed),
        )
    console.print()
    console.print(task_table)


@main.command("show-run")
@click.option("--run-dir", required=True, help="Path to a Hermes Meta-Harness run dir.")
def show_run_cmd(run_dir: str) -> None:
    """Show the parsed summary for one run."""
    summary = load_run_summary(Path(run_dir))
    console.print(json.dumps(
        {
            "benchmark_name": summary.benchmark_name,
            "candidate_name": summary.candidate_name,
            "candidate_path": summary.candidate_path,
            "run_dir": str(summary.run_dir),
            "eval_metrics": summary.eval_metrics,
        },
        indent=2,
        sort_keys=True,
    ))
