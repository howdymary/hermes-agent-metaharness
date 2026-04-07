"""Persist a simple frontier of evaluated candidates."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from filelock import FileLock

from meta_harness.comparability import extract_task_selection_metadata
from meta_harness.models import FrontierEntry, RunSummary


class FrontierStore:
    """Simple JSON-backed frontier store with cross-platform file locking."""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_suffix(".lock")
        self._lock = FileLock(str(self._lock_path))

    def load(self) -> List[FrontierEntry]:
        """Load all frontier entries."""
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [
            FrontierEntry(**{k: v for k, v in entry.items() if k in FrontierEntry.__dataclass_fields__})
            for entry in payload
        ]

    def save(self, entries: List[FrontierEntry]) -> None:
        """Save the frontier atomically via temp file + rename."""
        with self._lock:
            self._save_unlocked(entries)

    def _save_unlocked(self, entries: List[FrontierEntry]) -> None:
        """Save the frontier assuming the caller already holds the file lock."""
        content = json.dumps(
            [entry.to_dict() for entry in entries], indent=2, sort_keys=True
        )
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".tmp", prefix=".frontier_"
        )
        closed = False
        try:
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            closed = True
            os.replace(tmp_path, self.path)
        except BaseException:
            if not closed:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def upsert_from_summary(
        self,
        summary: RunSummary,
        *,
        status: str = "evaluated",
        notes: str = "",
    ) -> FrontierEntry:
        """Insert or update a frontier record from a run summary.

        Uses an exclusive file lock to prevent concurrent lost updates.
        """
        pass_rate = float(summary.eval_metrics.get("eval/pass_rate", 0.0) or 0.0)
        new_entry = FrontierEntry(
            candidate_name=summary.candidate_name,
            candidate_path=summary.candidate_path,
            benchmark_name=summary.benchmark_name,
            run_dir=str(summary.run_dir),
            pass_rate=pass_rate,
            total_tasks=int(summary.eval_metrics.get("eval/total_tasks") or len(summary.task_results) or 0),
            task_selection_hash=str(
                (extract_task_selection_metadata(summary.manifest) or {}).get("selection_hash") or ""
            ),
            status=status,
            notes=notes,
        )

        with self._lock:
            entries = self.load()
            updated = False
            for index, entry in enumerate(entries):
                same_identity = (
                    entry.candidate_name == new_entry.candidate_name
                    and entry.candidate_path == new_entry.candidate_path
                    and entry.benchmark_name == new_entry.benchmark_name
                )
                if same_identity:
                    entries[index] = new_entry
                    updated = True
                    break

            if not updated:
                entries.append(new_entry)

            self._save_unlocked(entries)

        return new_entry

    def best_for_benchmark(self, benchmark_name: str) -> FrontierEntry:
        """Return the best entry for a benchmark by pass rate."""
        entries = self.top_for_benchmark(benchmark_name, limit=1)
        if not entries:
            raise FileNotFoundError(f"No frontier entries for benchmark '{benchmark_name}'")
        return entries[0]

    def top_for_benchmark(
        self,
        benchmark_name: str,
        *,
        limit: Optional[int] = None,
        statuses: Optional[List[str]] = None,
        task_selection_hash: Optional[str] = None,
    ) -> List[FrontierEntry]:
        """Return the top frontier entries for a benchmark."""
        entries = [entry for entry in self.load() if entry.benchmark_name == benchmark_name]
        if statuses is not None:
            allowed = set(statuses)
            entries = [entry for entry in entries if entry.status in allowed]
        if task_selection_hash is not None:
            entries = [entry for entry in entries if entry.task_selection_hash == task_selection_hash]
        ranked = sorted(
            entries,
            key=lambda entry: (-entry.total_tasks, -entry.pass_rate, entry.candidate_name, entry.run_dir),
        )
        if limit is None:
            return ranked
        return ranked[:limit]
