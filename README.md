# Hermes Agent Meta-Harness

`hermes-agent-metaharness` is the standalone outer-loop Meta-Harness repo for Hermes.

It does not reimplement Hermes runtime behavior. Instead, it treats `hermes-agent` as the execution backend for benchmark harness candidates and focuses on:

- candidate resolution
- benchmark evaluation orchestration
- archive reading
- run comparison
- frontier tracking
- later, candidate mutation and search

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
- later, mutation and search

## Current Scope

V1 provides:

- resolve a candidate by explicit path or Hermes built-in candidate name
- launch TBLite or TB2 with the Hermes Meta-Harness flags
- read Hermes-produced archives (`manifest.json`, `summary.json`, `tasks/*.json`)
- compare two runs
- persist a simple frontier

## Quick Start

```bash
git clone https://github.com/howdymary/hermes-agent-metaharness.git
cd hermes-agent-metaharness
pip install -e ".[dev]"
```

Point it at Hermes with either:

- `HERMES_AGENT_REPO=/path/to/hermes-agent`
- or a sibling checkout at `../hermes-agent`
- or `~/.hermes/hermes-agent`

Dry-run a built-in Hermes candidate on TBLite:

```bash
python -m meta_harness evaluate-candidate \
  --candidate snapshot_baseline \
  --benchmark tblite \
  --hermes-repo /Users/maryliu/Projects/hermes-agent \
  --dry-run
```

Compare two Hermes Meta-Harness run directories:

```bash
python -m meta_harness compare-runs \
  --baseline-run /path/to/baseline-run \
  --candidate-run /path/to/candidate-run
```

## Repo Layout

```text
meta_harness/
├── config.py
├── models.py
├── candidate_registry.py
├── benchmark_runner.py
├── archive_reader.py
├── comparison.py
├── frontier.py
├── cli.py
└── __main__.py
```

Local candidate files can live in [`candidates/README.md`](/Users/maryliu/Projects/hermes-agent-metaharness/candidates/README.md), with an example in [`template_candidate.py`](/Users/maryliu/Projects/hermes-agent-metaharness/candidates/template_candidate.py).

## Near-Term Roadmap

1. Stable candidate evaluation and comparison
2. Baseline helpers and richer reports
3. Candidate templating and mutation
4. Search/frontier management
5. Trace-driven reflective candidate improvement
