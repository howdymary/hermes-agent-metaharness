from pathlib import Path

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
