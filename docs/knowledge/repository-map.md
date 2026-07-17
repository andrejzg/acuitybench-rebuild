---
type: Architecture
title: Repository Map and Source-of-Truth Guide
description: Where implementation, configuration, data, results, schemas and documentation live.
tags: [architecture, repository, navigation]
timestamp: 2026-07-17T00:00:00+01:00
status: active
---

# High-level map

```text
acuitybench/                 Python package and CLI
  providers/                 Provider abstraction and OpenAI adapter
  static_student.py          Static-first plan/data/evaluation contracts
  synthetic.py               Fictional-pilot generation, labels and leakage checks
  interactive/               Interactive seed, schemas, simulator and costing logic
configs/                     Model, rubric, static and interactive versioned configuration
data/cache/sources/          Pinned source snapshots (Git LFS)
data/processed/              Reconstructed benchmark and build report
data/interactive/            Evaluation-only interactive seed and manifest
data/static/                 Separate static-pilot requests, outputs and manifests
prompts/                     Versioned synthetic generation and labelling prompts
schemas/                     Strict JSON Schemas for static and interactive contracts
results/                     SQLite state, reports, exports and frontier plots
tests/                       Unit and integration-style deterministic tests
docs/                        Design documentation
docs/knowledge/              This OKF v0.1 handover bundle
AGENTS.md                    Repository-specific operating constraints for AI agents
HANDOVER.md                  Dated concise continuation brief
sources.lock.json            Source URLs, revisions, sizes, hashes and licence notes
ARTIFACTS.sha256             Checksums for committed data/result artifacts
```

# Python modules

| Module | Responsibility |
| --- | --- |
| `acuitybench/cli.py` | Command-line entrypoint and subcommand wiring. |
| `acuitybench/sources.py` | Pinned-source fetching and checksum enforcement. |
| `acuitybench/pipeline.py` | Benchmark reconstruction and output generation. |
| `acuitybench/validation.py` | Core benchmark invariants. |
| `acuitybench/models.py` | Model/judge config parsing and fingerprints. |
| `acuitybench/providers/` | Provider interface and OpenAI streaming implementation. |
| `acuitybench/store.py` | Resumable SQLite schema and migrations. |
| `acuitybench/evaluation.py` | Generation/judging orchestration and timing. |
| `acuitybench/distributional.py` | Physician-distribution metrics. |
| `acuitybench/reporting.py` | Tables, exports, costs, latency and manifests. |
| `acuitybench/plotting.py` | Deterministic frontier SVG generation. |
| `acuitybench/static_student.py` | Static plan inspection, separate-data validation and evaluation-contract construction. |
| `acuitybench/synthetic.py` | Deterministic fictional slots, resumable generation/double labelling, acceptance and lexical leakage checks. |
| `acuitybench/interactive/seed.py` | Seed selection, transformation, provenance and validation. |
| `acuitybench/interactive/simulator.py` | Closed action validation, deterministic replay and scoring. |
| `acuitybench/interactive/costing.py` | Versioned Tinker/human-work cost formulas. |
| `acuitybench/interactive/schema_validation.py` | Dependency-light strict JSON Schema validation. |

# Configuration contracts

- [`../../configs/models.yaml`](../../configs/models.yaml): target models,
  prices, reasoning settings, GPT-4.1 judge profile and OpenAI-compatible
  endpoint metadata.
- [`../../configs/static_student.v1.yaml`](../../configs/static_student.v1.yaml):
  accepted static-first objective, metrics, training boundary and progression
  gate.
- [`../../configs/static/synthetic_pilot.v0.yaml`](../../configs/static/synthetic_pilot.v0.yaml):
  balanced 20-case fictional-pilot slots, prompts, schemas, acceptance gates
  and output contract.
- [`../../schemas/static-acuity-example-v1.schema.json`](../../schemas/static-acuity-example-v1.schema.json):
  separately sourced static training/example contract.
- [`../../configs/rubric.yaml`](../../configs/rubric.yaml): A-D disposition
  criteria used by the judge.
- [`../../configs/interactive/seed_set.v1.yaml`](../../configs/interactive/seed_set.v1.yaml):
  exact seed selection, exclusions, quotas and paths.
- [`../../configs/interactive/question_catalog.v1.yaml`](../../configs/interactive/question_catalog.v1.yaml):
  deterministic fact slots.
- [`../../configs/interactive/action_catalog.v1.yaml`](../../configs/interactive/action_catalog.v1.yaml):
  policy-visible actions and acuity meanings.
- [`../../configs/interactive/cost_assumptions.v1.yaml`](../../configs/interactive/cost_assumptions.v1.yaml):
  workload and price assumptions.

# Generated evidence

Generated outputs are not disposable build clutter. They are part of the audit
trail:

- build reports prove reconstructed counts and source hashes;
- run manifests preserve the inference contract and returned model snapshots;
- raw/request-attempt exports preserve model outputs, retries, usage and timing;
- comparison CSVs are the source for frontier plots; and
- interactive manifests bind the seed to schemas, configs, sources and hashes.
- fictional-pilot manifests bind deterministic requests to prompts, schemas,
  benchmark-screening input, provider-call counts and training blockers.

# Source-of-truth hierarchy

Use this order when facts conflict:

1. executable schemas and validation code;
2. versioned configuration and source lock;
3. generated manifests/checksums;
4. committed result tables and sample-level exports;
5. narrative documentation and this bundle.

Repair stale lower-priority documentation in the same change.

# Extension points

- Add an OpenAI-hosted model by copying a model entry in `configs/models.yaml`.
  For a local or hosted compatible student, use `provider: openai_compatible`,
  a `base_url_env` and stable `deployment`; always use a fresh run ID.
- Add a provider through `acuitybench/providers/` and its registry entry.
- Add a new interactive version by creating new schema/config/artifact versions;
  do not silently mutate a frozen reviewed version.
- Add new curated knowledge as a focused concept and link it from
  [`index.md`](index.md).
