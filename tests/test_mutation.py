from meta_harness.mutation import generate_variant_candidates, resolve_mutation_specs, variant_candidate_name


def test_generate_variant_candidates_writes_wrapper_files(tmp_path):
    seed_candidate = tmp_path / "seed_candidate.py"
    seed_candidate.write_text("class HarnessCandidate:\n    pass\n", encoding="utf-8")

    mutation = resolve_mutation_specs(["plan_briefly"])[0]
    generated = generate_variant_candidates(
        seed_candidate_path=seed_candidate,
        seed_candidate_name="Seed Candidate",
        output_dir=tmp_path / "generated",
        mutations=[mutation],
    )

    assert len(generated) == 1
    generated_path = generated[0]
    assert generated_path.name == f"{variant_candidate_name('Seed Candidate', mutation)}.py"

    content = generated_path.read_text(encoding="utf-8")
    assert str(seed_candidate.resolve()) in content
    assert "plan_briefly" in content
    compile(content, str(generated_path), "exec")
