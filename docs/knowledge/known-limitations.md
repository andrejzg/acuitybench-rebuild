---
type: Risk Register
title: Known Limitations and Unsupported Claims
description: Clinical, data, evaluation, latency, tooling and repository-state limits that constrain interpretation.
tags: [limitations, risk, clinical-review, evaluation, handover]
timestamp: 2026-07-17T00:00:00+01:00
status: active
---

# Clinical validation boundary

The interactive seed is reproducible research plumbing, not a validated
clinical benchmark or patient-facing system.

- `clinician_content_reviewed_cases` is **0 of 100**.
- No new independent GP labels have been collected for the transformed cards.
- The opening/fact split, red flags and required questions were generated
  mechanically.
- Released reference-label agreement is not proof of clinical correctness.

Until the review and adjudication gate is complete, do not call seed results
clinical performance, clinical safety or validated consultation efficiency.

# Mechanical fact routing

Source fragments are routed to a fixed 33-question catalog by deterministic
rules. A fragment may populate multiple question IDs. Direct inspection of the
current JSONL finds at least two known facts with the same answer in **86 of
100 cards**, across 145 within-card duplicate-answer groups.

This can reward or punish question selection for artifacts of routing rather
than clinical information gain. Clinicians must split, rewrite or reject these
facts without adding information unsupported by the source. Version 1 remains
immutable; reviewed content belongs in a new version.

# Sparse safety annotations

The current lexical rules emit 27 red-flag records across 22 cards; 78 cards
have none. Only 15 required-question records occur across 13 cards; 87 cards
have none. A required question is currently a detected red flag whose evidence
is not already present in the opening utterance.

These counts show that required-question recall and red-flag coverage are
sparse development signals, not a complete clinical questioning standard.
Absence of a marker does not establish absence of risk. The rules and every
emitted marker require clinician review.

# Simulator realism

The patient simulator performs exact `question_id` lookup. It never infers a
fact, paraphrases spontaneously, changes its story, models misunderstanding or
generates natural conversational behaviour. Unknown answers are always
`I'm not sure.`, and repeats are byte-identical.

Determinism is valuable for contract tests and reproducibility, but it cannot
support claims about natural-patient interaction quality, robustness to
language variation or production consultation latency.

# Missing experiment and training components

The repository has a static-student plan/schema/validator/evaluation wrapper,
an OpenAI-compatible serving adapter, deterministic scripted simulation,
trajectory scoring and a cost calculator. It does **not** yet have:

- a separately sourced static training pool or accepted teacher targets;
- a student-training pipeline or trained static checkpoint;
- a provider-neutral autonomous multi-turn policy runner;
- a completed teacher-policy benchmark on the interactive seed;
- accepted teacher traces from a separate training pool;
- a grouped/deduplicated train/evaluation split;
- an interactive student-training pipeline; or
- any trained specialised model.

The green `Our trained model?` point in frontier figures is aspirational, not a
measurement.

# Sample size and selection

The interactive seed has `n=100`, deliberately balanced at 25 per A-D label
and selected for information richness under fixed source quotas. It is not
prevalence representative, and its effective uncertainty can be larger within
source, presentation or safety subgroups. Current reports do not establish a
power analysis, clinically justified non-inferiority margin or uncertainty
gate for model promotion.

Report numerators, denominators and uncertainty intervals where appropriate;
do not generalise a point estimate beyond this selected set.

# Static result presentation

The accuracy-cost and accuracy-latency charts are deterministic static SVGs
generated from committed result tables. They are not live dashboards and do
not update when a database row, model alias, price or worktree file changes.
Always tie a claim to a run ID, manifest, source CSV and generation date.

# Latency comparability

The legacy non-streaming reproduction's historical `latency_ms` includes local
semaphore waiting, provider attempts and retry backoff. It is not API request
latency, and TTFT cannot be recovered from it. Schema-v2 streaming runs record
queue wait, request wall time, retry sleep, TTFT, stream tail, total duration
and provider processing separately.

Do not compare these clocks as if they were interchangeable. The paired July
2026 runs also intentionally differ from the paper's unspecified transport by
using streaming, concurrency 20 and the default service tier.

# Historical artifact versions

`data/processed/build_report.json` records core benchmark builder version
`0.2.0`. The current local package and interactive seed manifest identify
version `0.4.0`. This is provenance, not evidence that the core benchmark was
rebuilt by the newer interactive code. Read the version embedded in each
artifact and do not silently rewrite historical reports.

# Licensing and public visibility

The repository is public by explicit owner decision dated 2026-07-17. That
decision does not resolve mixed third-party terms: PMR inputs are CC BY-NC
4.0, Semigran has layered terms, structured-triage data terms need checking,
and the physician-release licence remains pending upstream.

No inference about redistribution or commercial rights may be drawn from
public GitHub visibility. Complete a source-by-source terms review before new
publication, redistribution or commercial use.

# Dirty local worktree

At this handover the worktree contains substantial modified and untracked work,
including the interactive implementation and knowledge bundle. Materialised
Git LFS result files can also look modified when local Git stores pointers.
Treat `git status` as volatile, preserve unrelated changes, and never stage or
rewrite files merely to make the tree look clean.

# Claims this repository does not support

- that any evaluated model is medically correct or deployment-ready;
- that the seed is clinician reviewed or representative of real prevalence;
- that required-question/red-flag coverage is clinically exhaustive;
- that static one-shot latency predicts a multi-turn production consultation;
- that exact source-identity checks alone rule out semantic benchmark leakage;
- that a specialised student has been trained or measured; or
- that public visibility authorises third-party data reuse or commercial use.

# Related concepts

- [Data and labels](data-and-labels.md)
- [Current state](current-state.md)
- [Next steps](next-steps.md)
- [`../../AUDIT.md`](../../AUDIT.md)
