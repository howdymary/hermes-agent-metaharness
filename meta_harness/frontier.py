"""Persist a simple frontier of evaluated candidates."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List

from meta_harness.models import FrontierEntry, RunSummary


class FrontierStore:
    """Simple JSON-backed frontier store."""

    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[FrontierEntry]:
        """Load all frontier entries."""
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [FrontierEntry(**entry) for entry in payload]

    def save(self, entries: List[FrontierEntry]) -> None:
        """Save the frontier atomically via temp file + rename."""
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
                os.close(fd)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def upsert_from_summary(
        self,
        summary: RunSummary,
        *,
        status: str = "evaluated",
        notes: str = "",
    ) -> FrontierEntry:
        """Insert or update a frontier record from a run summary."""
        pass_rate = float(summary.eval_metrics.get("eval/pass_rate", 0.0) or 0.0)
        new_entry = FrontierEntry(
            candidate_name=summary.candidate_name,
            candidate_path=summary.candidate_path,
            benchmark_name=summary.benchmark_name,
            run_dir=str(summary.run_dir),
            pass_rate=pass_rate,
            status=status,
            notes=notes,
        )

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

        self.save(entries)
        return new_entry

    def best_for_benchmark(self, benchmark_name: str) -> FrontierEntry:
        """Return the best entry for a benchmark by pass rate."""
        entries = [entry for entry in self.load() if entry.benchmark_name == benchmark_name]
        if not entries:
            raise FileNotFoundError(f"No frontier entries for benchmark '{benchmark_name}'")
        return max(entries, key=lambda entry: entry.pass_rate)
