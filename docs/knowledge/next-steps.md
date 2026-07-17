---
type: Roadmap
title: Prioritised Next Steps and Acceptance Gates
description: Static-first student roadmap followed by a deliberately deferred interactive-triage phase.
tags: [roadmap, static-first, training, evaluation, interactive-triage]
timestamp: 2026-07-17T00:00:00+01:00
status: active
---

# Current sequencing decision

Build and measure a **static AcuityBench-style student first**. Interactive
`ASK`/`DISPOSE`/`HANDOFF` is phase two.

Do not train on any reconstructed AcuityBench case or on
`data/interactive/seed_v1/`. Do not run paid provider inference without
explicit spend authorisation. Preserve versioned evaluation artifacts and use
new model IDs, deployment descriptions and run IDs when provenance changes.

# G0 — clear source terms for training

The repository remains public by explicit owner decision. Public visibility is
not permission to train on, redistribute or commercialise third-party data.

Before building a training pool, record for every candidate source:

1. permitted research, training, publication, redistribution and commercial
   uses;
2. attribution, notice, share-alike or non-commercial duties;
3. privacy and platform-content constraints;
4. whether teacher-labelled/generated derivatives remain restricted;
5. unresolved questions requiring the source owner or qualified legal review;
   and
6. the source page, revision and date checked.

**Gate G0:** every intended training input and derivative has a written terms
disposition or explicit blocking status. No right is inferred from GitHub
visibility.

# Static phase S1 — freeze the experiment contract

This code gate is now implemented in
[`../../configs/static_student.v1.yaml`](../../configs/static_student.v1.yaml)
and [`static-first.md`](static-first.md).

The contract declares:

- QA as the primary single-shot task;
- one-shot conversational response plus GPT-4.1 judging as the secondary
  apples-to-apples paper task;
- 914 held-out cases, with the 527 clear primary cases forming the main score;
- five samples and severe tie-breaking for the full contract;
- exact agreement, under/over-triage, ordinal error, cost, p95 service latency
  and TTFT; and
- interactive work as a later gated phase.

**Gate S1:** `acuitybench static-plan` passes, the contract is versioned, and
the benchmark is explicitly evaluation-only. **Status: complete locally.**

# Static phase S2 — build a separate training pool

Start with 500–1,000 unique cases as a pipeline and learning-curve pilot. This
is a planning heuristic, not a sample-size guarantee.

The [fictional pilot](synthetic-pilot.md) has now verified the paid mechanics at
200-case scale: all generations and blinded labels completed, with 131 current
machine passes and 69 rejections. Before the larger build, add semantic
screening, manually inspect all 200 cases, and record whether ambiguity flags
should remain an automatic rejection or be separated from substantive acuity
disagreement. The completed API run alone does not satisfy Gate S2.

For every case:

- preserve source revision, source identity, source-text SHA-256, label basis
  and transformation;
- assign a stable case-family ID before splitting;
- keep families within exactly one of train/development/evaluation;
- deduplicate exact and near-duplicate narratives and generated variants;
- exclude every AcuityBench source identity, descendant and paraphrase; and
- retain ambiguous/rejected cases and their reasons rather than silently
  selecting only easy examples.

Write records against
[`../../schemas/static-acuity-example-v1.schema.json`](../../schemas/static-acuity-example-v1.schema.json)
and run `acuitybench static-data-validate`.

**Gate S2:** terms are cleared, schema validation passes, family split
crossings and AcuityBench contamination are zero, and deduplication evidence is
retained.

# Static phase S3 — create labels and train a small student

Use a versioned strong teacher—hosted or open-weight—to produce A-D targets and
concise rationales for the separate pool. Record exact model/deployment,
prompt, sampling, output and configuration hashes. Define filtering and review
rules before generation; retain failed, ambiguous and rejected targets.

Clinician review is most valuable for:

- a blinded sample across labels and presentation strata;
- teacher/student disagreements;
- severe under-triage candidates; and
- cases intended for a small independent development/evaluation slice.

Train the simplest small model that can emit `ACUITY: <A|B|C|D>`. Use nested
case subsets—such as 200, 500 and 1,000—to measure the learning curve before a
larger run. Tinker is optional; no Tinker client or training loop exists yet.

**Gate S3:** the dataset, teacher outputs, review sampling, checkpoint and
training configuration are versioned and reproducible; the student has never
seen AcuityBench evaluation cases.

# Static phase S4 — run the held-out benchmark

Evaluate the student through the existing resumable instrumentation:

```bash
uv run python -m acuitybench static-evaluate \
  --model <student-profile> \
  --qa-only \
  --samples 1 \
  --limit 10 \
  --run-id <new-smoke-id>
```

After the smoke run, execute five-sample QA. If complete and credible, run the
default paired QA/conversation contract so the model can enter the same
accuracy-cost and accuracy-latency comparison as the paper baselines.

Report:

- QA exact, under-triage, over-triage and ordinal distance;
- conversation exact and judge cost for the paired run;
- per-acuity/source results and severe failures;
- p50/p95 service latency and TTFT under a pinned deployment profile;
- cost per 1,000 target calls using a real provider rate or documented
  amortized self-hosted cost; and
- training-data and benchmark hashes.

**Gate S4:** at least one complete, reproducible student run exists with no
contamination, compatible latency instrumentation, non-placeholder cost and an
explicit human decision that its accuracy/safety/latency trade-off merits the
interactive phase. No numerical promotion threshold is silently assumed.

# Interactive phase I1 — review calibration

This phase is deferred until S4. Then run the proposed blinded 10-card
calibration across A-D, both seed sources and varied presentation groups. Two
GPs rehearse source-grounding, fact routing, red flags, required questions,
independent A-D labels and accept/rewrite/reject decisions.

**Gate I1:** both reviewers complete calibration independently; the guide,
review-record schema and adjudication method are frozen.

# Interactive phase I2 — review and freeze seed v2

Apply the calibrated protocol to all 100 v1 candidates. Preserve independent
votes and inherited references, adjudicate conflicts, reject unsupported
cases, and review replacements under the same workflow. Never mutate v1.

Freeze accepted cards as evaluation-only `seed_v2` with source/config/schema,
review and output digests. Decide explicitly whether the primary scoring target
is the adjudicated GP label or inherited AcuityBench reference; preserve and
report both either way.

**Gate I2:** every scoring card is source-grounded and independently labelled,
all conflicts are adjudicated, manifests reconcile, deterministic validation
passes, and no reviewed seed case can enter training.

# Interactive phase I3 — implement the policy loop

Build a provider-neutral resumable runner around `ASK`, `DISPOSE` and
`HANDOFF`. For every attempt and turn, retain queue entry, dispatch, TTFT,
terminal latency, retries, token usage, cost, model/deployment metadata, action
validity, simulator state digest and terminal safety metrics.

Keep per-turn service latency, TTFT and full-consultation latency separate.
Report autonomous accuracy alongside handoff rate.

**Gate I3:** fake-provider contract tests, deterministic replay and resume tests
pass; every turn has timing/usage evidence or an explicit unavailable reason.

# Interactive phase I4 — collect traces and adapt the student

Generate teacher trajectories only from a separate grouped training pool.
Define validity, evidence-grounding, safety, question-efficiency and review
rules in advance. Compare the static student, strong teachers and interactive
student on reviewed seed v2. Only measured models enter interactive frontiers.

**Gate I4:** accepted traces and training provenance are auditable; no seed
contamination exists; under-triage, handoff, latency and cost are all reported.

# Gate summary

| Gate | Required outcome |
| --- | --- |
| G0 Terms | Every intended training use is cleared or explicitly blocked. |
| S1 Contract | Versioned static plan and held-out cohort validate. **Complete locally.** |
| S2 Data | Separate, grouped, deduplicated, provenance-linked pool with zero benchmark contamination. |
| S3 Student | Reproducible labels/training and a small static checkpoint. |
| S4 Static result | Complete accuracy/safety/latency/cost run and explicit decision to proceed. |
| I1 Calibration | Ten blinded interactive cards and reviewer process calibrated. |
| I2 Seed v2 | Clinician-reviewed, immutable, evaluation-only interactive seed. |
| I3 Runner | Resumable per-turn provider-neutral policy instrumentation. |
| I4 Interactive student | Reviewed traces, reproducible adaptation and full interactive frontier. |

# Open human-owned decisions

- Which source licences permit the intended static training and downstream use?
- Which teacher model, student family, hardware and spend ceiling are in scope?
- What review fraction is required for teacher-labelled training cases?
- What exact/near-duplicate method defines a case family?
- What static under-triage, agreement, p95 latency and cost trade-off merits
  proceeding to interactivity?
- Who reviews and adjudicates the interactive seed, and what label becomes the
  primary v2 scoring target?
- What interactive safety/handoff and full-consultation latency thresholds are
  acceptable?

Record resolved choices in [Decision record](decisions.md); do not silently
choose them in code.

# Related concepts

- [Static-first contract](static-first.md)
- [Fictional static pilot](synthetic-pilot.md)
- [Training strategy](training-strategy.md)
- [Model evaluation](model-evaluation.md)
- [Clinical-review protocol](clinical-review-protocol.md)
- [Interactive triage](interactive-triage.md)
- [Known limitations](known-limitations.md)
