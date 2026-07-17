---
type: Handover
title: Current Repository and Research State
description: Dated snapshot of completed work, local changes, external repository state and immediate blockers.
tags: [handover, status, risk]
timestamp: 2026-07-17T00:00:00+01:00
status: active
last_verified: 2026-07-17
---

# Snapshot warning

This document is a dated handover snapshot. Before acting, run `git status`,
inspect the current branch and compare generated manifests with the values
below.

# Repository state

| Field | Verified value |
| --- | --- |
| Remote | `https://github.com/andrejzg/acuitybench-rebuild.git` |
| Branch | `main` |
| Baseline HEAD | `78e5706` (`origin/main` before the current fictional-pilot work) |
| GitHub visibility | **PUBLIC** |
| Package version | `0.4.0` |
| Test baseline | 195 passing, including synthetic-pilot, static-contract and handover/OKF integrity tests |

The fictional-pilot implementation and documentation are local uncommitted
work at the time of this snapshot. Git LFS 3.7.1 is installed in the verified
environment; ordinary `git status` is still required to distinguish current
work from unrelated local changes.

# Public-repository policy and licensing risk

The owner explicitly confirmed on 2026-07-17 that the GitHub repository should
remain **public**. No visibility change is required. Public visibility does not
resolve the underlying licence constraints:

- the AcuityBench physician-label release still says its licence is pending;
- PMR-Reddit/PMR-Synth are CC BY-NC 4.0; and
- the repository includes source-derived clinical text and result artifacts.

Before adding or redistributing source-derived data, review the applicable
upstream terms and preserve required attribution. Do not infer a commercial-use
right from GitHub visibility, and do not change repository visibility without
a new explicit owner instruction.

# Completed and verified locally

## Core benchmark

- 914 reconstructed rows and stable reference IDs.
- 697 primary cases and 217 ambiguous cases.
- 527 clear A/B/C/D primary cases used by the paper-style main score.
- 170 primary boundary-label cases retained for separate reporting.
- Source snapshots, reconstructed data and artifact checksums are committed or
  represented through Git LFS.

## Model evaluation

- Resumable generation, judging and reporting backed by SQLite.
- Attempt-level usage and latency instrumentation, including streaming TTFT.
- Complete paired GPT-5-mini and GPT-5.4 runs with five samples per case and
  prompt format.
- Deterministic accuracy-vs-cost and accuracy-vs-latency SVGs.

## Static-first student path

- `configs/static_student.v1.yaml` freezes the first-stage objective: complete
  case to one A/B/C/D label, with QA exact match as the primary score.
- `schemas/static-acuity-example-v1.schema.json` defines separately sourced
  train/development examples with family grouping and source provenance.
- `static-data-validate` rejects schema failures, duplicate example IDs,
  family crossings between splits, and exact source-identity reuse from the
  AcuityBench benchmark.
- `static-evaluate` can run the held-out QA task alone or the paired QA plus
  one-shot conversational task used for paper-style reporting. It persists a
  versioned experiment contract in each run.
- `provider: openai_compatible` supports a separately served student through
  an OpenAI-compatible endpoint while recording a stable deployment ID.

These are evaluation and integration capabilities, not a trained model. No
separate training pool, teacher-labelled training set, training loop, or
student checkpoint exists yet.

## Fictional static pilot

- `configs/static/synthetic_pilot.v0.yaml` freezes 20 fictional slots balanced
  across A/B/C/D and five presentation groups, with 16 train and four
  development assignments.
- Separate strict prompts and schemas support fictional generation and two
  blinded independent labels per case.
- `synthetic-plan`, `synthetic-init` and incomplete validation are free;
  generation and labelling require explicit spend and terms confirmations.
- Raw attempts are append-only and resumable, with model/prompt provenance,
  timing, usage, provider metadata, errors and retry identity.
- Exact and lexical leakage checks run only after generation; AcuityBench is
  never supplied to the generator.
- The default scaffold contains 20 requests, zero provider calls and no
  generated cases. Semantic screening is not implemented and every candidate
  requires manual review, so candidate records require
  `training_allowed: false` and the manifest is `training_ready: false`.

## Interactive triage

- Versioned case-card and action schemas.
- Closed 33-question catalog with `ASK`, `DISPOSE` and `HANDOFF`.
- Deterministic simulator, trajectory scoring and aggregate safety metrics.
- 100-case balanced seed with opaque IDs and byte-reproducible build.
- Seed JSONL SHA-256:
  `882037b01df43f9155a21969c1a29b55c4458b96dd1ff6053791bce4adfd3f65`.
- Independent data/code audit found no structural blockers.

# What is not complete

- No clinician has reviewed the transformed interactive card content.
- No new independent GP labels have been collected for the interactive cards.
- No natural free-form patient simulator has been validated.
- No teacher-policy interactive benchmark has been run.
- No separate static or interactive training pool has been built.
- The fictional pilot has not made provider calls and has no accepted teacher
  targets; it is pipeline infrastructure, not training data.
- No student model has been trained.
- No measured result occupies the aspirational green frontier point.

# Immediate handoff

Follow the [fictional-pilot contract](synthetic-pilot.md). The immediate
decision is generator/labeler selection, provider-terms review and a maximum
cost for 60 calls before requesting spend authorization. The following build
is the separately sourced 500–1,000-case static learning pilot; clinician
review remains a later prerequisite for freezing a defensible interactive
evaluation set.
The continuing governance gate is source-by-source licence review before new
publication, redistribution, or commercial use; public visibility itself is a
settled owner decision.

# Evidence

- [`../../data/processed/build_report.json`](../../data/processed/build_report.json)
- [`../../data/interactive/seed_v1/manifest.json`](../../data/interactive/seed_v1/manifest.json)
- [`../../data/static/synthetic_pilot_v0/manifest.json`](../../data/static/synthetic_pilot_v0/manifest.json)
- [`../../results/README.md`](../../results/README.md)
- [`../../NOTICE.md`](../../NOTICE.md)
- [`../../sources.lock.json`](../../sources.lock.json)
