"""CLI for standalone Hermes Meta-Harness orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import click
from rich.console import Console
from rich.table import Table

from meta_harness.archive_reader import load_run_summary
from meta_harness.baseline import resolve_baseline_selection
from meta_harness.benchmark_runner import BenchmarkRunResult, run_benchmark
from meta_harness.candidate_registry import list_builtin_candidates
from meta_harness.comparison import build_comparison_report
from meta_harness.config import MetaHarnessConfig
from meta_harness.frontier import FrontierStore
from meta_harness.models import BenchmarkRunSpec
from meta_harness.mutation import builtin_mutations, safe_slug
from meta_harness.search import StructuredSearchRequest, run_structured_search

console = Console()


@click.group()
def main() -> None:
    """Standalone outer-loop Meta-Harness tooling for Hermes."""


def _format_optional_delta(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def _emit_report(report, *, show_task_names: bool = True) -> None:
    summary_table = Table(title="Baseline vs Candidate")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Benchmark", report.benchmark_name)
    summary_table.add_row("Baseline", report.baseline_candidate_name)
    summary_table.add_row("Candidate", report.candidate_name)
    summary_table.add_row("Candidate Better", "yes" if report.candidate_better else "no")
    summary_table.add_row("Pass Rate Delta", _format_optional_delta(report.pass_rate_delta))
    summary_table.add_row("Passed Tasks Delta", _format_optional_delta(report.passed_tasks_delta))
    summary_table.add_row(
        "Eval Time Delta (s)",
        _format_optional_delta(report.evaluation_time_delta_seconds),
    )
    summary_table.add_row("Improved Tasks", str(report.improved_tasks))
    summary_table.add_row("Regressed Tasks", str(report.regressed_tasks))
    summary_table.add_row("Net Task Gain", str(report.net_task_gain))
    summary_table.add_row("Overlapping Tasks", str(report.overlapping_tasks))
    summary_table.add_row("Total Compared Tasks", str(report.total_tasks))
    summary_table.add_row("Baseline Run", str(report.baseline_run_dir))
    summary_table.add_row("Candidate Run", str(report.candidate_run_dir))
    console.print()
    console.print(summary_table)

    if report.metric_deltas:
        metrics_table = Table(title="Metric Deltas")
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Delta", justify="right")
        for key, delta in sorted(report.metric_deltas.items()):
            metrics_table.add_row(key, f"{delta:+.4f}")
        console.print()
        console.print(metrics_table)

    if not show_task_names:
        return

    if report.regressed_task_names:
        regressed_table = Table(title="Regressed Tasks")
        regressed_table.add_column("Task", style="bold red")
        for task_name in report.regressed_task_names:
            regressed_table.add_row(task_name)
        console.print()
        console.print(regressed_table)

    if report.improved_task_names:
        improved_table = Table(title="Improved Tasks")
        improved_table.add_column("Task", style="bold green")
        for task_name in report.improved_task_names:
            improved_table.add_row(task_name)
        console.print()
        console.print(improved_table)


def _build_config(hermes_repo: Optional[str]) -> MetaHarnessConfig:
    config = MetaHarnessConfig()
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo).expanduser().resolve()
    return config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_command(command: Iterable[str], *, fallback: str) -> str:
    rendered = " ".join(command).strip()
    return rendered if rendered else fallback


@main.command("list-builtins")
@click.option("--hermes-repo", type=click.Path(exists=True, file_okay=False), help="Path to the hermes-agent checkout.")
def list_builtins_cmd(hermes_repo: Optional[str] = None) -> None:
    """List Hermes built-in Meta-Harness candidates."""
    config = _build_config(hermes_repo)

    for name in list_builtin_candidates(config.hermes_agent_path):
        console.print(name)


@main.command("list-mutations")
def list_mutations_cmd() -> None:
    """List built-in structured search mutations."""
    table = Table(title="Built-in Mutations")
    table.add_column("Slug", style="bold")
    table.add_column("Description")
    for slug, mutation in builtin_mutations().items():
        table.add_row(slug, mutation.description)
    console.print(table)


@main.command("show-frontier")
@click.option("--frontier-path", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to a frontier JSON file.")
@click.option("--benchmark", required=True, help="Benchmark name to inspect.")
@click.option("--limit", default=10, show_default=True, type=int, help="Maximum entries to show.")
def show_frontier_cmd(frontier_path: str, benchmark: str, limit: int) -> None:
    """Show ranked frontier entries for one benchmark."""
    frontier = FrontierStore(Path(frontier_path).expanduser().resolve())
    entries = frontier.top_for_benchmark(benchmark, limit=limit)
    if not entries:
        raise click.ClickException(f"No frontier entries found for benchmark '{benchmark}'.")

    table = Table(title=f"Frontier: {benchmark}")
    table.add_column("Rank", justify="right")
    table.add_column("Candidate", style="bold")
    table.add_column("Total Tasks", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Status")
    table.add_column("Run Dir")
    for index, entry in enumerate(entries, start=1):
        table.add_row(
            str(index),
            entry.candidate_name,
            str(entry.total_tasks),
            f"{entry.pass_rate:.4f}",
            entry.status,
            entry.run_dir,
        )
    console.print(table)


@main.command("evaluate-candidate")
@click.option("--candidate", required=True, help="Candidate file path or built-in Hermes candidate name.")
@click.option("--benchmark", default="tblite", type=click.Choice(["tblite", "tb2"]), show_default=True)
@click.option("--hermes-repo", type=click.Path(exists=True, file_okay=False), help="Path to the hermes-agent checkout.")
@click.option("--archive-dir", help="Archive root for Hermes Meta-Harness run outputs.")
@click.option("--run-name", help="Optional friendly run name.")
@click.option("--hermes-config-path", type=click.Path(exists=True, dir_okay=False), help="Optional Hermes benchmark YAML config path.")
@click.option("--task-filter", help="Optional comma-separated task filter.")
@click.option("--skip-tasks", help="Optional comma-separated task skip list.")
@click.option("--frontier-path", help="Optional path to a frontier JSON file to update after evaluation.")
@click.option("--dry-run", is_flag=True, help="Print the benchmark command without executing it.")
def evaluate_candidate_cmd(
    candidate: str,
    benchmark: str,
    hermes_repo: Optional[str] = None,
    archive_dir: Optional[str] = None,
    run_name: Optional[str] = None,
    hermes_config_path: Optional[str] = None,
    task_filter: Optional[str] = None,
    skip_tasks: Optional[str] = None,
    frontier_path: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Evaluate one Meta-Harness candidate against Hermes."""
    config = _build_config(hermes_repo)

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
@click.option("--json-output", help="Optional path to write the comparison report as JSON.")
@click.option("--hide-task-names", is_flag=True, help="Hide improved/regressed task name tables.")
def compare_runs_cmd(
    baseline_run: str,
    candidate_run: str,
    json_output: Optional[str] = None,
    hide_task_names: bool = False,
) -> None:
    """Compare two Hermes Meta-Harness run directories."""
    baseline = load_run_summary(Path(baseline_run))
    candidate = load_run_summary(Path(candidate_run))
    report = build_comparison_report(baseline, candidate)
    _emit_report(report, show_task_names=not hide_task_names)

    if json_output:
        _write_json(Path(json_output).expanduser().resolve(), report.to_dict())


@main.command("evaluate-vs-baseline")
@click.option("--candidate", required=True, help="Candidate file path or built-in Hermes candidate name.")
@click.option("--baseline-candidate", default="snapshot_baseline", show_default=True)
@click.option("--baseline-run", type=click.Path(exists=True, file_okay=False), help="Reuse an existing baseline run directory instead of running one.")
@click.option("--baseline-from-frontier", type=click.Path(exists=True, dir_okay=False), help="Reuse the current best frontier entry for this benchmark as baseline.")
@click.option("--benchmark", default="tblite", type=click.Choice(["tblite", "tb2"]), show_default=True)
@click.option("--hermes-repo", type=click.Path(exists=True, file_okay=False), help="Path to the hermes-agent checkout.")
@click.option("--archive-dir", help="Archive root for the paired evaluation.")
@click.option("--candidate-run-name", help="Optional candidate run name.")
@click.option("--baseline-run-name", help="Optional baseline run name.")
@click.option("--hermes-config-path", type=click.Path(exists=True, dir_okay=False), help="Optional Hermes benchmark YAML config path.")
@click.option("--task-filter", help="Optional comma-separated task filter.")
@click.option("--skip-tasks", help="Optional comma-separated task skip list.")
@click.option("--frontier-path", help="Optional frontier JSON path to update with the candidate run.")
@click.option("--json-output", help="Optional path to write the comparison report as JSON.")
@click.option("--dry-run", is_flag=True, help="Print benchmark commands without executing them.")
def evaluate_vs_baseline_cmd(
    candidate: str,
    benchmark: str,
    baseline_candidate: str,
    baseline_run: Optional[str] = None,
    baseline_from_frontier: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    archive_dir: Optional[str] = None,
    candidate_run_name: Optional[str] = None,
    baseline_run_name: Optional[str] = None,
    hermes_config_path: Optional[str] = None,
    task_filter: Optional[str] = None,
    skip_tasks: Optional[str] = None,
    frontier_path: Optional[str] = None,
    json_output: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Run a baseline and candidate, then emit a richer comparison report."""
    config = _build_config(hermes_repo)
    archive_root = Path(archive_dir).expanduser().resolve() if archive_dir else (
        config.output_dir / "comparisons" / benchmark
    )
    try:
        baseline_selection = resolve_baseline_selection(
            benchmark_name=benchmark,
            baseline_candidate=baseline_candidate,
            baseline_run_dir=Path(baseline_run).expanduser().resolve() if baseline_run else None,
            baseline_frontier_path=Path(baseline_from_frontier).expanduser().resolve() if baseline_from_frontier else None,
            task_filter=task_filter,
            skip_tasks=skip_tasks,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    common_kwargs = {
        "benchmark": benchmark,
        "archive_root": archive_root,
        "hermes_config_path": Path(hermes_config_path).expanduser().resolve() if hermes_config_path else None,
        "task_filter": task_filter,
        "skip_tasks": skip_tasks,
        "python_executable": config.python_executable,
    }
    candidate_spec = BenchmarkRunSpec(
        candidate=candidate,
        run_name=candidate_run_name or f"candidate__{safe_slug(candidate)}",
        **common_kwargs,
    )

    console.print(f"\n[bold cyan]Baseline vs Candidate Evaluation[/bold cyan]")
    console.print(f"  Hermes repo: {config.hermes_agent_path}")
    console.print(f"  Benchmark: {benchmark}")
    console.print(f"  Baseline: {baseline_selection.display_label}")
    console.print(f"  Baseline source: {baseline_selection.source}")
    console.print(f"  Candidate: {candidate}")
    console.print(f"  Archive root: {archive_root}")

    if baseline_selection.requires_fresh_run:
        baseline_spec = BenchmarkRunSpec(
            candidate=baseline_selection.baseline_candidate,
            run_name=baseline_run_name or f"baseline__{safe_slug(baseline_selection.baseline_candidate)}",
            **common_kwargs,
        )
        baseline_result = run_benchmark(config, baseline_spec, dry_run=dry_run)
    else:
        baseline_result = BenchmarkRunResult(
            command=[],
            archive_root=archive_root,
            returncode=0,
            run_dir=baseline_selection.run_dir,
            summary=baseline_selection.summary,
        )
    candidate_result = run_benchmark(config, candidate_spec, dry_run=dry_run)

    command_table = Table(title="Commands")
    command_table.add_column("Run", style="bold")
    command_table.add_column("Command")
    command_table.add_row(
        "Baseline",
        _format_command(
            baseline_result.command,
            fallback=f"(reused {baseline_selection.source}: {baseline_selection.reference or baseline_selection.display_label})",
        ),
    )
    command_table.add_row("Candidate", _format_command(candidate_result.command, fallback="(no command)"))
    console.print()
    console.print(command_table)

    if dry_run:
        console.print("\n[yellow]Dry run only — benchmark not executed.[/yellow]")
        return

    if baseline_result.summary is None or candidate_result.summary is None:
        raise click.ClickException("Expected both baseline and candidate summaries.")

    report = build_comparison_report(baseline_result.summary, candidate_result.summary)
    _emit_report(report)

    if json_output:
        _write_json(Path(json_output).expanduser().resolve(), report.to_dict())

    if frontier_path:
        frontier = FrontierStore(Path(frontier_path))
        entry = frontier.upsert_from_summary(
            candidate_result.summary,
            status="candidate_beats_baseline" if report.candidate_better else "evaluated",
            notes=(
                f"baseline={baseline_selection.baseline_candidate}; pass_rate_delta={report.pass_rate_delta:+.4f}; "
                f"net_task_gain={report.net_task_gain}; regressions={report.regressed_tasks}"
            ),
        )
        console.print(f"\n[green]Frontier updated:[/green] {entry.candidate_name} @ {entry.pass_rate:.4f}")


@main.command("search-candidates")
@click.option("--seed-candidate", required=True, help="Seed candidate file path or built-in Hermes candidate name.")
@click.option("--baseline-candidate", default="snapshot_baseline", show_default=True)
@click.option("--baseline-run", type=click.Path(exists=True, file_okay=False), help="Reuse an existing baseline run directory instead of running one.")
@click.option("--baseline-from-frontier", type=click.Path(exists=True, dir_okay=False), help="Reuse the current best frontier entry for this benchmark as baseline.")
@click.option("--benchmark", default="tblite", type=click.Choice(["tblite", "tb2"]), show_default=True)
@click.option("--hermes-repo", type=click.Path(exists=True, file_okay=False), help="Path to the hermes-agent checkout.")
@click.option("--workspace-dir", help="Workspace directory for generated candidates, reports, and search summary.")
@click.option("--archive-dir", help="Archive root for evaluations inside this search.")
@click.option("--mutation", "mutations", multiple=True, help="Mutation slug to include. Repeat to add more.")
@click.option("--hermes-config-path", type=click.Path(exists=True, dir_okay=False), help="Optional Hermes benchmark YAML config path.")
@click.option("--task-filter", help="Optional comma-separated task filter.")
@click.option("--skip-tasks", help="Optional comma-separated task skip list.")
@click.option("--frontier-path", help="Optional frontier JSON path to update for each evaluated trial.")
@click.option("--dry-run", is_flag=True, help="Generate candidate files and print commands without executing benchmarks.")
def search_candidates_cmd(
    seed_candidate: str,
    benchmark: str,
    baseline_candidate: str,
    baseline_run: Optional[str] = None,
    baseline_from_frontier: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    archive_dir: Optional[str] = None,
    mutations: Iterable[str] = (),
    hermes_config_path: Optional[str] = None,
    task_filter: Optional[str] = None,
    skip_tasks: Optional[str] = None,
    frontier_path: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Run a small deterministic search over generated wrapper candidates."""
    config = _build_config(hermes_repo)
    request = StructuredSearchRequest(
        benchmark=benchmark,
        seed_candidate=seed_candidate,
        baseline_candidate=baseline_candidate,
        baseline_run_dir=Path(baseline_run).expanduser().resolve() if baseline_run else None,
        baseline_frontier_path=Path(baseline_from_frontier).expanduser().resolve() if baseline_from_frontier else None,
        workspace_dir=Path(workspace_dir).expanduser().resolve() if workspace_dir else None,
        archive_root=Path(archive_dir).expanduser().resolve() if archive_dir else None,
        mutation_slugs=tuple(mutations),
        hermes_config_path=Path(hermes_config_path).expanduser().resolve() if hermes_config_path else None,
        task_filter=task_filter,
        skip_tasks=skip_tasks,
        frontier_path=Path(frontier_path).expanduser().resolve() if frontier_path else None,
        python_executable=config.python_executable,
    )

    try:
        summary = run_structured_search(config, request, dry_run=dry_run)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    overview = Table(title="Structured Search")
    overview.add_column("Metric", style="bold")
    overview.add_column("Value")
    overview.add_row("Benchmark", summary.benchmark_name)
    overview.add_row("Baseline Candidate", summary.baseline_candidate)
    overview.add_row("Baseline Source", summary.baseline_source)
    overview.add_row("Baseline Reference", summary.baseline_reference or "-")
    overview.add_row("Seed Candidate", summary.seed_candidate)
    overview.add_row("Workspace", summary.workspace_dir)
    overview.add_row("Generated Candidates", summary.generated_candidates_dir)
    overview.add_row("Trials", str(len(summary.trial_results)))
    overview.add_row("Best Mutation", summary.best_mutation_slug or "-")
    overview.add_row("Best Candidate", summary.best_candidate_name or "-")
    console.print()
    console.print(overview)

    trials_table = Table(title="Trials")
    trials_table.add_column("Mutation", style="bold")
    trials_table.add_column("Candidate")
    trials_table.add_column("Pass Rate Delta", justify="right")
    trials_table.add_column("Net Gain", justify="right")
    trials_table.add_column("Better?", justify="right")
    for trial in summary.trial_results:
        pass_rate_delta = _format_optional_delta(trial.report.pass_rate_delta) if trial.report else "-"
        net_gain = str(trial.report.net_task_gain) if trial.report else "-"
        better = "yes" if trial.report and trial.report.candidate_better else "no"
        if trial.report is None:
            better = "-"
        trials_table.add_row(
            trial.mutation_slug,
            trial.candidate_name,
            pass_rate_delta,
            net_gain,
            better,
        )
    console.print()
    console.print(trials_table)

    if dry_run:
        console.print("\n[yellow]Dry run only — generated candidates and commands were prepared.[/yellow]")


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
            "task_count": len(summary.task_results),
        },
        indent=2,
        sort_keys=True,
    ))
