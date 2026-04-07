"""Launch Hermes benchmarks with Meta-Harness candidates."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

from meta_harness.archive_reader import find_run_dirs, load_run_summary
from meta_harness.candidate_registry import resolve_candidate_path
from meta_harness.config import MetaHarnessConfig
from meta_harness.models import BenchmarkRunSpec, RunSummary

logger = logging.getLogger(__name__)

# Default timeout for benchmark subprocess (30 minutes).
DEFAULT_BENCHMARK_TIMEOUT_S = 1800

BENCHMARK_SCRIPTS = {
    "tblite": "environments/benchmarks/tblite/tblite_env.py",
    "tb2": "environments/benchmarks/terminalbench_2/terminalbench2_env.py",
}


@dataclass
class BenchmarkRunResult:
    """Outcome of one benchmark command execution."""

    command: list
    archive_root: Path
    returncode: int
    run_dir: Optional[Path] = None
    summary: Optional[RunSummary] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


def resolve_benchmark_script(benchmark: str, hermes_agent_path: Path) -> Path:
    """Resolve a benchmark script inside Hermes."""
    if benchmark not in BENCHMARK_SCRIPTS:
        raise ValueError(f"Unsupported benchmark '{benchmark}'. Expected one of {sorted(BENCHMARK_SCRIPTS)}")
    script = (hermes_agent_path / BENCHMARK_SCRIPTS[benchmark]).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Benchmark script not found: {script}")
    return script


def build_benchmark_command(config: MetaHarnessConfig, run_spec: BenchmarkRunSpec) -> list:
    """Build a Hermes benchmark command for one candidate run."""
    benchmark_script = resolve_benchmark_script(run_spec.benchmark, config.hermes_agent_path)
    candidate_path = resolve_candidate_path(run_spec.candidate, config.hermes_agent_path)

    command = [
        run_spec.python_executable or config.python_executable,
        str(benchmark_script),
        "evaluate",
        "--env.meta_harness_candidate",
        str(candidate_path),
        "--env.meta_harness_archive_dir",
        str(run_spec.archive_root),
    ]

    if run_spec.run_name:
        command.extend(["--env.meta_harness_run_name", run_spec.run_name])
    if run_spec.hermes_config_path:
        command.extend(["--config", str(run_spec.hermes_config_path)])
    if run_spec.task_filter:
        command.extend(["--env.task_filter", run_spec.task_filter])
    if run_spec.skip_tasks:
        command.extend(["--env.skip_tasks", run_spec.skip_tasks])

    return command


def _existing_run_dirs(archive_root: Path) -> Set[Path]:
    return {path.resolve() for path in find_run_dirs(archive_root)}


def run_benchmark(
    config: MetaHarnessConfig,
    run_spec: BenchmarkRunSpec,
    *,
    dry_run: bool = False,
    timeout: Optional[int] = None,
) -> BenchmarkRunResult:
    """Run one Hermes benchmark and load the resulting archive summary."""
    archive_root = run_spec.archive_root.expanduser().resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    command = build_benchmark_command(config, run_spec)

    if dry_run:
        return BenchmarkRunResult(
            command=command,
            archive_root=archive_root,
            returncode=0,
        )

    effective_timeout = timeout if timeout is not None else DEFAULT_BENCHMARK_TIMEOUT_S

    before = _existing_run_dirs(archive_root)
    try:
        result = subprocess.run(
            command,
            cwd=str(config.hermes_agent_path),
            text=True,
            capture_output=True,
            check=False,
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Benchmark timed out after {effective_timeout}s. "
            f"Command: {' '.join(command)}"
        ) from exc

    if result.returncode != 0:
        logger.error("Benchmark stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"Benchmark command failed with exit code {result.returncode}.\n"
            f"stderr: {(result.stderr or '').strip()[:2000]}"
        )

    after = _existing_run_dirs(archive_root)
    new_run_dirs = sorted(after - before, key=lambda path: path.stat().st_mtime)

    if new_run_dirs:
        run_dir = new_run_dirs[-1]
    else:
        fallback = find_run_dirs(archive_root)
        if not fallback:
            raise RuntimeError(
                f"Benchmark succeeded but produced no run directory under {archive_root}"
            )
        run_dir = fallback[-1]

    summary = load_run_summary(run_dir)

    return BenchmarkRunResult(
        command=command,
        archive_root=archive_root,
        returncode=result.returncode,
        run_dir=run_dir,
        summary=summary,
        stdout=result.stdout,
        stderr=result.stderr,
    )
