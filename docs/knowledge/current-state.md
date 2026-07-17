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
| Baseline HEAD | `4b22d5e` (`origin/main` at verification time) |
| GitHub visibility | **PUBLIC** |
| Package version | `0.4.0` in the current local worktree |
| Test baseline | 190 passing, including static-contract and handover/OKF integrity tests |

The interactive-triage implementation and this handover bundle are local
uncommitted work at the time of this snapshot. Git LFS 3.7.1 is installed in
the verified environment and `git lfs status` showed no LFS objects pending;
ordinary `git status` is still required to distinguish the current work from
unrelated local changes.

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
- No student model has been trained.
- No measured result occupies the aspirational green frontier point.

# Immediate handoff

Follow [Next steps](next-steps.md). The immediate build is the separately
sourced static-student pilot; clinician review remains a later prerequisite
for freezing a defensible interactive evaluation set.
The continuing governance gate is source-by-source licence review before new
publication, redistribution, or commercial use; public visibility itself is a
settled owner decision.

# Evidence

- [`../../data/processed/build_report.json`](../../data/processed/build_report.json)
- [`../../data/interactive/seed_v1/manifest.json`](../../data/interactive/seed_v1/manifest.json)
- [`../../results/README.md`](../../results/README.md)
- [`../../NOTICE.md`](../../NOTICE.md)
- [`../../sources.lock.json`](../../sources.lock.json)
