---
type: Experiment Strategy
title: Static-First Student Contract
description: Decision, executable contract, data boundary and commands for training and evaluating a single-shot acuity student before interactive triage.
tags: [static-first, student, distillation, evaluation, latency]
timestamp: 2026-07-17T00:00:00+01:00
status: active
last_verified: 2026-07-17
---

# Decision

The first specialised model should solve the same **static, single-shot acuity
task** as AcuityBench before the project adds adaptive questioning. Given the
complete stored presentation, it predicts A, B, C or D. This isolates acuity
recognition, distillation quality, serving latency and cost from the additional
failure modes introduced by patient simulation and question selection.

“Static” includes both AcuityBench paper formats:

- `qa` is the primary capability test and requires `ACUITY: <A|B|C|D>`;
- `conv` is the secondary, still-one-shot patient-facing response scored by
  the GPT-4.1 paper judge.

Neither format lets the model ask a new question and receive a new patient
answer. `ASK`/`DISPOSE`/`HANDOFF` remains the later interactive phase.

# Held-out evaluation contract

The versioned contract is
[`../../configs/static_student.v1.yaml`](../../configs/static_student.v1.yaml).
It binds:

- all 914 reconstructed cases as the auditable evaluation population;
- the 527 clear primary A-D cases as the main exact-agreement score;
- five samples per case and severe tie-breaking, matching the paper contract;
- QA as primary and one-shot conversation as secondary;
- exact agreement, under/over-triage, ordinal distance, p95 service latency,
  TTFT and target cost; and
- the gate that interactive work follows only after a measured static result.

The reconstructed AcuityBench data is **evaluation-only for student work**.
Do not train on its 914 cases, the 100-card interactive seed, rewrites of those
cases, or teacher outputs derived from them.

# Current code readiness

| Capability | State |
| --- | --- |
| Validate and inspect the static plan without API calls | Implemented: `static-plan`. |
| Run a resumable QA-only pilot | Implemented: `static-evaluate --qa-only`. |
| Run the paired paper-style static contract | Implemented: `static-evaluate` (default). |
| Preserve the static experiment contract in run identity/SQLite/manifests | Implemented. |
| Reuse cost, TTFT, service-latency, under-triage and report exports | Implemented through the existing evaluator. |
| Evaluate an OpenAI-hosted model | Implemented. |
| Evaluate a local/self-hosted OpenAI-compatible student | Implemented through `provider: openai_compatible`. |
| Validate provenance/grouping and AcuityBench contamination for a separate JSONL pool | Implemented: `static-data-validate`. |
| Initialize and validate a balanced 20-case fictional pipeline check | Implemented with zero provider calls: `synthetic-plan`, `synthetic-init`, `synthetic-validate`. |
| Resumable fictional generation and double-blinded labelling | Implemented and fake-provider tested; no paid run has occurred. |
| Separate training pool | **Not built.** |
| Accepted, manually reviewed teacher targets | **Not built.** |
| SFT/Tinker training loop or trained student | **Not implemented.** |

# Commands

The readiness check is local and free:

```bash
uv run python -m acuitybench static-plan
```

It currently reports 914 evaluation cases, 527 main-score cases, 4,570 QA
target calls for a five-sample run, and 9,140 target plus 4,570 judge calls for
the paired contract.

Validate a future separate training/development pool:

```bash
uv run python -m acuitybench static-data-validate \
  --input data/static/<version>/examples.jsonl
```

The strict schema is
[`../../schemas/static-acuity-example-v1.schema.json`](../../schemas/static-acuity-example-v1.schema.json).
The validator rejects schema violations, duplicate example IDs, case families
crossing splits, and any train/development example whose source identity occurs
in AcuityBench.

Paid evaluation requires explicit spend authorisation and a new run ID. A
small QA-only smoke run is:

```bash
uv run python -m acuitybench static-evaluate \
  --model <student-profile> \
  --qa-only \
  --samples 1 \
  --limit 10 \
  --run-id <new-static-smoke-id>
```

The default paired run matches the paper's two static formats:

```bash
uv run python -m acuitybench static-evaluate \
  --model <student-profile> \
  --run-id <new-static-paper-contract-id> \
  --concurrency 20 \
  --judge-concurrency 20
```

# Plugging in a self-hosted student

Expose the checkpoint through an OpenAI-compatible chat-completions endpoint,
then add a profile to `configs/models.yaml` following this shape:

```yaml
student-checkpoint-v1:
  display_name: Student checkpoint v1
  provider: openai_compatible
  api_model: <served-model-name>
  endpoint: chat_completions
  api_key_env: STUDENT_API_KEY
  base_url_env: STUDENT_BASE_URL
  deployment: <checkpoint-hardware-server-description>
  temperature: 0.0
  send_temperature: true
  max_output_tokens: 32
  token_parameter: max_tokens
  input_cost_per_million: <measured-or-amortized-rate>
  cached_input_cost_per_million: <measured-or-amortized-rate>
  output_cost_per_million: <measured-or-amortized-rate>
```

Put endpoint and credential values in `.env`, never in the YAML. The stable
deployment description becomes part of model fingerprinting, so a checkpoint,
server stack or hardware change should receive a new model ID/run ID. Do not
enter zero token prices merely because a model is self-hosted; calculate a
documented amortized serving cost before placing it on the cost frontier.

# First pilot

First run the 20-case fictional pipeline check described in
[`synthetic-pilot.md`](synthetic-pilot.md). It is intended to expose prompt,
schema, resumption, provenance, rejection and review failures cheaply. It is
not a training corpus and does not replace the 500–1,000-case learning-curve
pilot.

The smallest credible learning experiment is:

1. clear source terms for the intended training use;
2. build 500–1,000 **separate** unique static cases as a pipeline/learning-curve
   pilot, grouped before splitting;
3. create teacher A-D labels and rationales under a versioned prompt/model
   contract, retaining rejected/ambiguous cases and sampled clinical review;
4. train a small student to emit the short QA contract;
5. evaluate it first with `--qa-only`, then run the paired contract for an
   apples-to-apples paper/frontier point; and
6. decide whether the measured accuracy, under-triage, cost and latency justify
   proceeding to interactive triage.

The 500–1,000 range is a pilot heuristic, not a power calculation. Use nested
training subsets and a learning curve rather than assuming a fixed sufficient
sample size.

# Progression to interactive work

There is no silently assumed accuracy threshold. Before the interactive phase,
the project must have at least one fully reproducible static student run with:

- complete QA metrics and under-triage reporting;
- a compatible streaming/concurrency/deployment latency profile;
- measured target cost rather than a zero-cost placeholder;
- documented training provenance and contamination checks; and
- an explicit human decision about whether the trade-off is useful enough to
  add adaptive questioning.

Only then should the project invest in reviewed interactive seed v2,
model-driven `ASK`/`DISPOSE`/`HANDOFF`, per-turn latency and interactive teacher
traces.

# Related concepts

- [Training strategy](training-strategy.md)
- [Fictional static pilot](synthetic-pilot.md)
- [Prioritised next steps](next-steps.md)
- [Model evaluation](model-evaluation.md)
- [Interactive triage](interactive-triage.md)
- [Data and labels](data-and-labels.md)
