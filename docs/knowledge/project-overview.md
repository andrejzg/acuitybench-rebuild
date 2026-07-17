---
type: Project Overview
title: AcuityBench Rebuild Project Overview
description: Mission, success criteria, research boundary and current workstreams.
tags: [acuitybench, clinical-acuity, research, handover]
timestamp: 2026-07-17T00:00:00+01:00
status: active
source_of_truth: ../../README.md
---

# Mission

This repository reconstructs the 914-case benchmark introduced in
[AcuityBench: Evaluating Clinical Acuity Identification and Uncertainty Alignment](https://arxiv.org/abs/2605.11398),
runs reproducible model evaluations, and establishes a staged path from a
low-latency static acuity student to a future interactive acuity model.

The long-term product hypothesis is a specialised model that can ask a small
number of useful questions and select one of four care-urgency dispositions
with high agreement, low under-triage and substantially lower latency than a
frontier general-purpose model.

# Success criteria

The project optimises a constrained frontier rather than accuracy alone:

1. **Accuracy**: retain or improve agreement with released
   physician/reference labels.
2. **Safety**: minimise under-triage and report handoff behaviour separately.
3. **Latency**: materially reduce p95 client-observed service latency; TTFT is a
   second important serving metric.
4. **Cost**: reduce cost per consultation without hiding review, retry or judge
   costs.
5. **Reproducibility**: pin inputs, configs, schemas, model metadata and
   digests; preserve sample-level evidence.

The green `Our trained model?` point in the frontier figures is aspirational.
It is not a measured result.

The two-plot presentation—accuracy versus cost and accuracy versus latency—is
inspired by Thinking Machines Lab's
[expert-judgment replication study](https://thinkingmachines.ai/news/learning-to-replicate-expert-judgment-in-financial-tasks/).
Here the clinically relevant additions are explicit under-triage reporting,
held-out physician/reference labels, and end-to-end multi-turn consultation
latency once an interactive policy runner exists.

# Workstreams

## Benchmark reconstruction

The source pipeline downloads checksum-pinned snapshots, reconstructs the
authors' released cases and labels, independently audits physician aggregation,
and emits CSV/Parquet artifacts plus a build report.

## Frontier-model evaluation

The evaluation system runs target models in both QA and conversational prompt
formats, uses GPT-4.1 as the paper-style rubric judge for conversational
outputs, stores resumable sample-level state, and exports accuracy,
distributional, cost and latency reports.

## Interactive triage

The interactive workstream defines a closed question/action protocol and a
deterministic patient simulator. Its current 100-case seed is a review and
evaluation artifact, not training data.

## Distilled low-latency model

No specialised student has been trained yet. The accepted first stage is
static: complete case to one A/B/C/D label. The intended sequence is:

1. build a separately sourced, grouped static training pool;
2. collect and review teacher targets, then train a small student;
3. evaluate QA accuracy/under-triage and latency, then the paired one-shot
   conversational paper comparison;
4. record whether the result justifies moving to interaction;
5. only then review/freeze the interactive seed and train a multi-turn policy.

This order proves the simplest distillation and serving hypothesis before
adding question selection, patient simulation and consultation-level latency.

# Explicit non-goals

- This is not a medical device or patient-facing clinical service.
- Reference-label agreement is not proof of clinical correctness.
- The project does not diagnose conditions; it studies acuity/disposition.
- The deterministic simulator does not attempt to generate free-form patient
  behaviour.
- The current 100-case seed is not a training corpus.

# Related concepts

- [Current state](current-state.md)
- [Data and labels](data-and-labels.md)
- [Model evaluation](model-evaluation.md)
- [Static-first student plan](static-first.md)
- [Interactive triage](interactive-triage.md)
- [Conversation-data landscape](conversation-data-landscape.md)
- [Training strategy](training-strategy.md)
- [Next steps](next-steps.md)

# Citations

[1] [AcuityBench paper](https://arxiv.org/abs/2605.11398)
