# Hermes Agent Meta-Harness Product Spec

## Purpose

`hermes-agent-metaharness` is the standalone outer-loop system for Hermes Meta-Harness candidate evolution.

It operates on a checked-out `hermes-agent` repo and uses Hermes benchmarks as the execution engine for one candidate at a time.

It does **not** own:

- the candidate protocol
- Hermes benchmark runtime behavior
- the inner agent loop

Those belong in `hermes-agent`.

## Core Distinction

| Layer | Lives In | Owns |
|------|----------|------|
| Inner runtime | `hermes-agent` | candidate protocol, benchmark integration, loop hooks, archive writing |
| Outer loop | `hermes-agent-metaharness` | candidate orchestration, archive analysis, comparison, frontier, later mutation/search |

## Objective

Meta-Harness optimizes the benchmark operating procedure, not model weights.

Primary target:

- TBLite first
- TB2 second

Primary levers:

- prompt additions
- pre-run bootstrap
- tool filtering or ordering
- stop/completion heuristics
- lightweight context policy

Primary objective:

- improve benchmark pass rate without destabilizing Hermes core

## V1 Scope

V1 is the minimum outer-loop foundation:

1. Evaluate an explicit candidate file through Hermes.
2. Resolve built-in Hermes candidates by name.
3. Read Hermes-produced archives.
4. Compare baseline vs candidate runs.
5. Persist a simple frontier of evaluated candidates.

## Current Milestone

The current milestone extends V1 in the most direct way:

1. Richer baseline-vs-candidate reporting
2. Paired evaluation flow for baseline plus candidate
3. Deterministic wrapper mutations around a seed candidate
4. Structured search over generated variants
5. Frontier updates based on baseline-relative outcomes

## V1 Functional Requirements

### Candidate Resolution

Support:

- explicit candidate file paths
- cwd-relative paths
- Hermes built-in candidates by name

### Benchmark Evaluation

Support:

- TBLite and TB2
- explicit archive root
- explicit run name
- optional task filters and skip lists
- safe dry-run mode

### Archive Reading

Read:

- `manifest.json`
- `summary.json`
- `tasks/*.json`

Treat `summary.json` as the canonical outer-loop run summary.

### Run Comparison

Compute:

- numeric metric deltas
- task-level improvements/regressions
- pass-rate deltas
- net task gain
- candidate-better judgement using a stable ranking rule

### Frontier Persistence

Persist:

- candidate identity
- benchmark
- pass rate
- run dir
- notes/status

### Baseline Reporting

Support:

- evaluating a candidate directly against a baseline candidate
- emitting a richer report with improvement/regression counts
- writing report JSON for later inspection

### Structured Search

Support:

- generating wrapper candidates from a seed candidate
- deterministic prompt/tool/loop mutations
- evaluating each generated candidate against the same baseline
- recording per-trial reports and a search summary

## Non-Goals for V1

- changing `run_agent.py`
- reproducing Hermes runtime logic in this repo
- autonomous merge/deploy
- weight training or finetuning
- open-ended non-verifiable optimization

## Next Stage

After the current milestone works:

1. Better mutation spaces and composition
2. Smarter baseline selection and report ranking
3. Trace-driven reflective edits
4. Frontier-aware search policies
5. Benchmark-aware candidate generation

## Success Criteria

V1 succeeds when a user can:

- point this repo at a Hermes checkout
- evaluate a candidate on TBLite
- parse the resulting archive
- compare it against another run
- record it in a frontier for future search
