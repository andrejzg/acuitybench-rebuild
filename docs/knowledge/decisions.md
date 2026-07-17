---
type: Decision Record
title: Project Decision Record
description: Accepted repository, data, safety, simulation, latency and knowledge-management decisions.
tags: [decisions, governance, safety, reproducibility, okf]
timestamp: 2026-07-17T00:00:00+01:00
status: active
---

# How to use this record

These are active decisions, not immutable facts. A later change should add a
new dated entry that supersedes the old one, update affected contracts and
preserve the historical rationale.

## D-001 — Keep the repository public

- **Date:** 2026-07-17
- **Status:** accepted by the repository owner
- **Decision:** Keep the GitHub repository public. Do not change visibility
  without a new explicit owner instruction.
- **Licence caveat:** Public visibility does not grant permission to use or
  redistribute third-party data. PMR is CC BY-NC 4.0, Semigran has layered
  terms, structured-triage data terms need verification, and the AcuityBench
  physician-release licence is still pending upstream.
- **Consequence:** Review source terms before adding or redistributing derived
  data and before publication or commercial use. Do not present visibility as
  licence clearance.

## D-002 — Treat released labels as evaluation references

- **Date:** 2026-07-17
- **Status:** accepted
- **Decision:** A/B/C/D and boundary labels are released
  physician/reference targets, not objective clinical truth.
- **Consequence:** Report agreement, disagreement, under-triage relative to the
  reference and label provenance. Do not equate exact match with medical
  correctness or deployment safety.

## D-003 — Use the released physician CSV as canonical

- **Date:** 2026-07-17
- **Status:** accepted
- **Decision:** The checksum-pinned released
  `normalized_survey_labels.csv`, stored as
  `data/cache/sources/physician_labels.csv`, is the canonical annotation
  artifact. Recompute its aggregates for verification; do not regenerate and
  substitute labels using a conflicting upstream CLI default.
- **Rationale:** This preserves the published artifact and makes deviations
  auditable. Its SHA-256 is fixed in `sources.lock.json`.

## D-004 — Apply the strict ambiguity threshold

- **Date:** 2026-07-17
- **Status:** accepted
- **Decision:** Consensus is `avg_sd <= 0.75`; ambiguity is strictly
  `avg_sd > 0.75`.
- **Rationale:** This reproduces the released 450 consensus / 217 ambiguous
  split. The alternative upstream `1.0` threshold would move 27 cases and
  produce 477 / 190.

## D-005 — Never train on an interactive evaluation seed

- **Date:** 2026-07-17
- **Status:** accepted and enforced
- **Decision:** `seed_v1` and future reviewed descendants are held-out
  evaluation artifacts. Their cards, rewrites, paraphrases and seed-derived
  teacher traces may not be used for training.
- **Enforcement:** Cards and manifests declare `training_allowed: false`;
  validation checks the guard. Training needs a separately sourced, grouped
  and deduplicated pool.

## D-006 — Keep the v1 simulator deterministic

- **Date:** 2026-07-17
- **Status:** accepted
- **Decision:** A question is resolved by exact catalogued `question_id`.
  Known answers contain only routed source evidence; unknown answers are fixed;
  repeated questions return the same response. The simulator makes no model
  call and invents no patient fact.
- **Rationale:** This isolates policy behaviour and enables byte-reproducible
  tests.
- **Consequence:** Natural-conversation realism requires a separate future
  experiment and must not be inferred from this simulator.

## D-007 — Define unsafe disposition by exact ordinal under-triage

- **Date:** 2026-07-17
- **Status:** accepted and schema-enforced
- **Decision:** Every disposition below the reference acuity is unsafe for the
  evaluator:

| Reference | Unsafe dispositions |
| --- | --- |
| A | none |
| B | A |
| C | A, B |
| D | A, B, C |

- **Consequence:** Over-triage, exact match and signed ordinal error remain
  separately reportable. `HANDOFF` is also separate; a model cannot improve
  autonomous accuracy merely by handing off every case.

## D-008 — Make latency a primary serving objective

- **Date:** 2026-07-17
- **Status:** accepted
- **Decision:** The primary serving comparison is client-observed p95 service
  latency, reported alongside accuracy and under-triage. Per-turn TTFT is a
  second mandatory latency view, and full-consultation latency is required for
  interactive policies.
- **Consequence:** Queue wait, request wall time, retry sleep, TTFT, stream
  tail, total logical duration and provider processing stay distinct. Cost
  remains a separate frontier; no provider header substitutes for a missing
  client clock.

## D-009 — Use OKF v0.1 as the durable knowledge layer

- **Date:** 2026-07-17
- **Status:** accepted
- **Decision:** Maintain focused Markdown concepts with YAML frontmatter,
  ordinary cross-links, `index.md` progressive disclosure and `log.md` history
  under `docs/knowledge/`.
- **Rationale:** The bundle is readable by humans and agents, diffable in Git
  and portable across AI tools.
- **Consequence:** OKF does not replace root `AGENTS.md`, the human README,
  executable schemas, manifests, source locks or tests. Those higher-priority
  artifacts remain authoritative.

## D-010 — Version reviewed and generated artifacts immutably

- **Date:** 2026-07-17
- **Status:** accepted
- **Decision:** Preserve source revisions and hashes. Freeze reviewed output as
  a new version rather than editing v1 in place; run configurations and model
  transport changes receive new run IDs.
- **Consequence:** Historical reports retain their embedded builder/config
  version even when newer code exists.

## D-011 — Train a static student before an interactive policy

- **Date:** 2026-07-17
- **Status:** accepted by the repository owner
- **Decision:** First train a low-latency single-shot student that maps a
  complete case to A/B/C/D. Treat QA exact match as primary and the paper's
  one-shot conversational response plus GPT-4.1 judge as a secondary paired
  comparison. Defer multi-turn `ASK`/`DISPOSE`/`HANDOFF` training.
- **Rationale:** This preserves the closest AcuityBench comparison, gives a
  simpler and cheaper proof of distillation, and isolates model serving
  latency before adding consultation policy and patient-simulation variance.
- **Boundary:** Training cases must be separately sourced and family-grouped;
  none of the 914 AcuityBench cases or interactive seed may enter training.
- **Progression gate:** Begin interactive training only after the static
  student has a reproducible accuracy/under-triage/latency result and a human
  decision records that the extra interaction is worth the complexity.

# Evidence and related concepts

- [Data and labels](data-and-labels.md)
- [Known limitations](known-limitations.md)
- [Next steps](next-steps.md)
- [Static-first student plan](static-first.md)
- [`../../AUDIT.md`](../../AUDIT.md)
- [`../../AGENTS.md`](../../AGENTS.md)
- [`../../schemas/interactive-case-card-v1.schema.json`](../../schemas/interactive-case-card-v1.schema.json)
