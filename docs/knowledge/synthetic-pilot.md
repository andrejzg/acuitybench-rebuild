---
type: Experiment Strategy
title: Fictional Static Pilot v0
description: Zero-call scaffold for a 20-case fictional pipeline check before building the larger static training pool.
tags: [synthetic-data, static-first, teacher-labels, leakage, pilot]
timestamp: 2026-07-17T00:00:00+01:00
status: scaffold-ready
last_verified: 2026-07-17
---

# Purpose and current status

The first implementation step is a deliberately tiny, wholly fictional
pipeline check—not a useful training set and not a clinical dataset. Version 0
plans 20 vignettes: five presentation groups crossed with A/B/C/D, one case per
cell. Sixteen slots are assigned to training and four to development, with one
development case per acuity label.

The scaffold has been initialized with **zero provider calls**. No vignette or
teacher label has been generated, and `training_ready` remains false. The
machine-readable status is
[`../../data/static/synthetic_pilot_v0/manifest.json`](../../data/static/synthetic_pilot_v0/manifest.json).

# Frozen contract

The versioned plan is
[`../../configs/static/synthetic_pilot.v0.yaml`](../../configs/static/synthetic_pilot.v0.yaml).
It binds the slot design, seed, prompts, output schemas, acceptance rules,
leakage thresholds and artifact paths. Generation and blinded labelling use
separate prompts:

- [`../../prompts/synthetic_case_generator.v0.md`](../../prompts/synthetic_case_generator.v0.md)
  creates fictional complete-case vignettes without benchmark or real-patient
  content;
- [`../../prompts/synthetic_labeler.v0.md`](../../prompts/synthetic_labeler.v0.md)
  receives only the vignette and independently predicts A/B/C/D without seeing
  the generator's intended label.

Each case receives two independent label calls. The planned paid workload is
therefore 20 generation calls plus 40 label calls: **60 calls total**. This is
call arithmetic, not a cost estimate; model choices, current prices and a
spend ceiling must be recorded first.

# Free inspection commands

These commands do not call a provider:

```bash
uv run python -m acuitybench synthetic-plan
uv run python -m acuitybench synthetic-init
uv run python -m acuitybench synthetic-validate --allow-incomplete
```

Initialization is deterministic and resumable. It writes the 20 requests and
a manifest containing configuration, prompt, benchmark and request hashes.

# Paid phases

Generation and labelling are separate so outputs can be inspected between
phases. Both commands require explicit spend and provider-terms confirmations:

```bash
uv run python -m acuitybench synthetic-generate \
  --model <generator-profile> \
  --confirm-spend \
  --terms-reviewed

uv run python -m acuitybench synthetic-label \
  --model <independent-labeler-profile> \
  --confirm-spend \
  --terms-reviewed
```

No paid command should run until the exact profiles, deployment metadata,
provider terms and expected maximum cost are reviewed. Raw attempt logs are
append-only and retain prompt/model configuration, timing, usage, provider
metadata, errors and retry identity. Re-running resumes completed samples.

# Acceptance and leakage boundary

A machine-accepted candidate must have:

- valid strict-schema generation and label outputs;
- two unanimous blinded labels;
- agreement between both labels and the generator's intended A-D target;
- no ambiguity flag, acuity-label leakage or internal generated duplicate; and
- no exact or configured lexical near-match to held-out AcuityBench text.

The held-out benchmark is never included in a generation prompt. It is loaded
only after generation for contamination screening. This separation is a hard
training boundary, not evidence that lexical screening proves independence.

Semantic/paraphrase screening is explicitly **not implemented** and is
required before scaling. All 20 candidates also require manual review before
any training use. Accordingly, even a candidate written to `examples.jsonl`
is only machine-accepted: its dedicated schema requires
`training_allowed: false`, and the overall manifest must remain
`training_ready: false` until the manual and semantic gates are satisfied.
Promotion to the static training-example schema must be a later, explicit
reviewed step.

# Artifact contract

The durable directory is
[`../../data/static/synthetic_pilot_v0/`](../../data/static/synthetic_pilot_v0/).
It starts with deterministic requests and a zero-call manifest. A completed
run additionally writes raw generation and label attempts, accepted
candidates, rejected cases, a contamination report and refreshed hashes and
counts. These artifacts must remain versioned; change the pilot version rather
than silently rewriting a reviewed contract.

# Next decision

Choose one generator profile and one independent labeler profile, confirm
their terms for generated training data, calculate a conservative 60-call
cost ceiling, and request explicit spend authorization. After generation,
manually inspect every case before treating the pilot as evidence that the
larger 500–1,000-case static-data workflow is ready.

# Related concepts

- [Static-first contract](static-first.md)
- [Training strategy](training-strategy.md)
- [Prioritised next steps](next-steps.md)
- [Known limitations](known-limitations.md)
