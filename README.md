# Hermes Agent Meta-Harness

`hermes-agent-metaharness` is the standalone outer-loop Meta-Harness repo for Hermes.

It treats `hermes-agent` as the execution backend for benchmark harness candidates and focuses on:

- candidate resolution
- benchmark evaluation orchestration
- archive reading
- run comparison
- richer baseline-vs-candidate reporting
- frontier tracking
- structured candidate mutation and search

## Origin

This project is directly inspired by the paper [Meta-Harness: End-to-End Optimization of Model Harnesses](https://arxiv.org/abs/2603.28052) and the companion [project page](https://yoonholee.com/meta-harness/).

The paper’s core argument is that LLM system quality depends not only on model weights, but also on the harness: the surrounding code that decides what context to collect, store, retrieve, and show to the model. Instead of hand-tuning that harness, Meta-Harness proposes an outer-loop optimizer that searches over harness code. Its proposer has access to the source code, scores, and execution traces of prior candidates through a filesystem, which gives it much richer diagnostic context than methods that only optimize from scores or short summaries. The paper reports gains on online text classification, retrieval-augmented math reasoning, and agentic coding, including improved TerminalBench-2 harnesses.

## How Hermes Adapts Meta-Harness

Hermes uses the same high-level idea, but adapts it to a research-safe benchmark workflow:

- `hermes-agent` owns the inner runtime: candidate protocol, benchmark integration, loop hooks, and archive writing.
- `hermes-agent-metaharness` owns the outer loop: candidate evaluation, archive analysis, baseline reuse, frontier tracking, and search.
- The current target is verifiable coding benchmarks such as TBLite and TB2, not general production chat behavior.
- Candidate search is intentionally conservative today: this repo generates deterministic wrapper candidates around a seed candidate instead of rewriting Hermes core.

In other words, the project applies Meta-Harness to Hermes by optimizing how Hermes is run on benchmarks, not by changing model weights and not by letting the production runtime self-modify.

## Boundary

`hermes-agent` owns the inner Meta-Harness runtime:

- candidate protocol
- TB2/TBLite integration
- optional loop hooks
- per-task archive writing

`hermes-agent-metaharness` owns the outer loop:

- candidate evaluation and comparison
- archive analysis
- baseline helpers
- frontier management
- mutation and search

## Current Scope

The current release provides:

- candidate resolution by explicit path or Hermes built-in candidate name
- TBLite and TB2 benchmark orchestration through Hermes
- archive parsing for `manifest.json`, `summary.json`, and `tasks/*.json`
- paired baseline-vs-candidate evaluation and reporting
- baseline reuse from an existing run or the current frontier-best entry
- task-selection comparability metadata for reused baselines
- a simple JSON-backed frontier with cross-platform locking
- deterministic wrapper-mutation search over generated candidate variants

## Quick Start

```bash
git clone https://github.com/howdymary/hermes-agent-metaharness.git
cd hermes-agent-metaharness
pip install -e ".[dev]"
```

Point it at Hermes with either:

- `HERMES_AGENT_REPO=/path/to/hermes-agent`
- a sibling checkout at `../hermes-agent`
- or `~/.hermes/hermes-agent`

Dry-run a built-in Hermes candidate on TBLite:

```bash
python -m meta_harness evaluate-candidate \
  --candidate snapshot_baseline \
  --benchmark tblite \
  --hermes-repo /path/to/hermes-agent \
  --dry-run
```

Compare two Hermes Meta-Harness run directories:

```bash
python -m meta_harness compare-runs \
  --baseline-run /path/to/baseline-run \
  --candidate-run /path/to/candidate-run
```

Run a candidate directly against a baseline and emit a richer report:

```bash
python -m meta_harness evaluate-vs-baseline \
  --candidate candidates/template_candidate.py \
  --baseline-candidate snapshot_baseline \
  --benchmark tblite \
  --hermes-repo /path/to/hermes-agent
```

Reuse an existing baseline run instead of rerunning baseline:

```bash
python -m meta_harness evaluate-vs-baseline \
  --candidate candidates/template_candidate.py \
  --baseline-run /path/to/baseline-run \
  --benchmark tblite \
  --hermes-repo /path/to/hermes-agent
```

Run a small deterministic search over generated wrapper candidates:

```bash
python -m meta_harness search-candidates \
  --seed-candidate candidates/template_candidate.py \
  --baseline-candidate snapshot_baseline \
  --benchmark tblite \
  --hermes-repo /path/to/hermes-agent
```

Inspect the current frontier for a benchmark:

```bash
python -m meta_harness show-frontier \
  --frontier-path output/frontier.json \
  --benchmark tblite
```

## Repo Layout

```text
meta_harness/
├── archive_reader.py
├── baseline.py
├── benchmark_runner.py
├── candidate_registry.py
├── cli.py
├── comparison.py
├── config.py
├── frontier.py
├── models.py
├── mutation.py
├── search.py
└── __main__.py
```

Candidate files can live in `candidates/`, with an example in `candidates/template_candidate.py`.

## Release Notes

This repo is intentionally research-oriented:

- it optimizes harness procedure, not model weights
- it is designed around verifiable benchmark feedback
- it keeps Hermes core stable by treating Hermes as the execution backend
- reused baselines are validated against the same task-selection hash before comparison

## Near-Term Roadmap

1. Better ranking/reporting and frontier-backed baseline policies
2. More expressive mutation spaces and composition
3. Trace-driven reflective candidate improvement
4. Frontier-aware search strategies
5. Stronger benchmark-aware candidate generation
