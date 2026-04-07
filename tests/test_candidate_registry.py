from pathlib import Path

import pytest

from meta_harness.candidate_registry import list_builtin_candidates, resolve_candidate_path


def test_resolve_builtin_candidate(tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    candidates_dir = hermes_repo / "environments" / "meta_harness" / "candidates"
    candidates_dir.mkdir(parents=True)
    candidate_file = candidates_dir / "snapshot_baseline.py"
    candidate_file.write_text("# test candidate")

    resolved = resolve_candidate_path("snapshot_baseline", hermes_repo)
    assert resolved == candidate_file.resolve()


def test_list_builtin_candidates(tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    candidates_dir = hermes_repo / "environments" / "meta_harness" / "candidates"
    candidates_dir.mkdir(parents=True)
    (candidates_dir / "alpha.py").write_text("")
    (candidates_dir / "beta.py").write_text("")
    (candidates_dir / "__init__.py").write_text("")

    assert list_builtin_candidates(hermes_repo) == ["alpha", "beta"]


def test_resolve_explicit_candidate_path(tmp_path):
    candidate_file = tmp_path / "custom_candidate.py"
    candidate_file.write_text("# custom")

    resolved = resolve_candidate_path(str(candidate_file), tmp_path)
    assert resolved == candidate_file.resolve()


def test_resolve_candidate_accepts_extra_allowed_root(tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    hermes_repo.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    candidate_file = external_root / "candidate.py"
    candidate_file.write_text("# external", encoding="utf-8")

    resolved = resolve_candidate_path(
        str(candidate_file),
        hermes_repo,
        extra_allowed_roots=[external_root],
    )
    assert resolved == candidate_file.resolve()


def test_resolve_candidate_rejects_outside_allowed_roots(tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    hermes_repo.mkdir()
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    candidate_file = outside_root / "candidate.py"
    candidate_file.write_text("# outside", encoding="utf-8")

    with pytest.raises(ValueError, match="outside all allowed directories"):
        resolve_candidate_path(str(candidate_file), hermes_repo)
