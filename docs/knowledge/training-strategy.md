---
type: Training Strategy
title: Proposed Training and Distillation Strategy
description: Static-first, then interactive, plan for building a low-latency acuity model without contaminating held-out evaluation.
tags: [training, distillation, simulation, sample-size, tinker]
timestamp: 2026-07-17T00:00:00+01:00
status: proposed
last_verified: 2026-07-17
---

# Non-negotiable evaluation boundary

Never train on any reconstructed AcuityBench benchmark case, prompt, label,
paraphrase, or teacher trace derived from those cases. This includes
`data/interactive/seed_v1/`, whose 100 cards are held-out evaluation assets and
declare `training_allowed: false`.

# Accepted sequence: static first

The first student is a single-shot classifier: complete case in, one A/B/C/D
acuity label out. QA exact agreement is primary. The paper's one-shot
conversational response plus GPT-4.1 rubric judge is secondary and preserves
an apples-to-apples comparison; it is not a multi-turn consultation.

Build the first pilot from roughly 500–1,000 separately sourced, family-grouped
cases. Generate and review teacher targets, train a small student, and compare
accuracy, under-triage, p95 service latency, TTFT and cost against the measured
frontier. The repository implements the plan, schema, validation, evaluation
contract and OpenAI-compatible serving adapter, but not the training data,
teacher-generation workflow or training loop. See
[`static-first.md`](static-first.md).

Interactive `ASK`/`DISPOSE`/`HANDOFF` is a later phase. Before it becomes a
defensible benchmark, the 100 transformed cards still need a 10-case
calibration, two blinded GP reviews, adjudication and an immutable reviewed
version. See [`interactive-triage.md`](interactive-triage.md) and
[`clinical-review-protocol.md`](clinical-review-protocol.md).

# Separate training pool

Training cases must come from a separate pool with provenance and usage rights.
Before generating any trajectories:

1. define a stable case-family or patient-family grouping key;
2. deduplicate exact and near-duplicate narratives;
3. split groups—not individual rows—between train, development and evaluation;
4. stratify deliberately across acuity, presentation family, source and
   clinically relevant demographics; and
5. freeze the split manifest before teacher generation.

This prevents related variants or multiple rollouts from the same underlying
case leaking across boundaries.

# What an interactive training trace means

One trace is one complete synthetic consultation trajectory:

- immutable case ID and provenance;
- patient-visible opening;
- ordered GP `ASK` actions and exact patient answers;
- terminal `DISPOSE` or `HANDOFF` action;
- source/reference label and review status;
- per-action validity, turn index, token usage, cost, request start, TTFT and
  terminal service latency; and
- trajectory outcomes such as exact acuity, signed ordinal error,
  under-triage, handoff and required/red-flag coverage.

“GP decisions” or “action targets” count all GP actions in traces, not just the
final labels. The planning arithmetic below assumes roughly six GP actions per
consultation. That is a workload assumption to be measured and revised.

# Patient and teacher options

The design should remain provider-neutral.

## Structured deterministic patient

Use a reviewed case state and return a fixed answer for each question ID. This
is cheap, reproducible and easy to audit, but language variation and natural
repair behaviour are limited. DDXPlus-style structured patient rows could
eventually seed this kind of environment after a clinically reviewed acuity
mapping.

## Model-simulated patient

Give a model a hidden case card and require it to answer only from that state.
This produces linguistic variety but risks adding facts, leaking the label or
changing the case across turns. Use fact-consistency checks, deterministic
state, adversarial tests and sampled clinician review.

## Teacher GP policy

The teacher can be a strong hosted model or a large open-source/open-weight
model. Hosted models make high-quality pilots easy but introduce API cost,
alias drift and external data handling. Open models offer weight control and
potentially easier large-scale generation but require serving infrastructure
and careful capability validation. Record the exact model snapshot,
configuration, prompt, sampling parameters and returned metadata in either
case.

A useful hybrid is a deterministic patient state plus a strong teacher GP:
the clinical facts stay fixed while the teacher explores different question
orders and dispositions.

# Suggested later interactive stages

These numbers are **planning heuristics**, not evidence-based sample-size
guarantees. They were derived from three practical considerations: cover many
different cases, sample more than one trajectory per case, and budget about six
GP decisions per consultation.

| Stage | Unique cases | Traces per case | Consultations | Approximate GP action targets |
| --- | ---: | ---: | ---: | ---: |
| Pipeline check | 200 | 2 | 400 | ~2,400 |
| First real pilot | 1,000 | 2 | 2,000 | ~12,000 |
| Credible v1 | 2,000–5,000 | 3–5 | 6,000–25,000 | ~40,000–150,000 |
| Strong research run | 10,000+ | 3–5 | 30,000+ | ~180,000+ |

Why distinguish cases from traces:

- more **unique cases** broaden presentation and safety coverage;
- more **traces per case** expose stochastic question order, wording and
  teacher-policy variation; and
- more **actions per trace** create token-level supervision but do not replace
  case diversity.

The lower credible-v1 action estimate rounds `6,000 x 6 = 36,000` to roughly
40,000. All totals should be recomputed from observed trajectory length and
acceptance rate before budgeting a larger run.

# Measure a learning curve instead of assuming N

Train comparable checkpoints on nested subsets—for example 200, 500, 1,000,
2,000 and 5,000 unique grouped cases—while holding the reviewed evaluation set
and inference contract fixed. Plot:

- overall and per-stratum accuracy;
- under-triage and severe ordinal error;
- handoff rate and autonomous accuracy;
- required/red-flag question recall;
- questions per consultation;
- p50/p95 per-action TTFT and service latency;
- p50/p95 end-to-end consultation latency; and
- cost per completed consultation.

Scale only if performance is still improving, important strata remain weak,
or uncertainty is too large. A plateau or systematic safety failure calls for
better cases, labels, rewards or architecture—not automatically more traces.

# Terminology

## Stratum

A stratum is a predefined slice of examples sharing attributes used for
sampling or reporting. The current case-card schema records acuity,
presentation group and source dataset in `stratum`; future analyses may add
reviewed demographic or difficulty slices. “Twenty examples per stratum” means
twenty examples in each selected slice, not twenty total.

Strata can overlap conceptually, so the split manifest must say whether a
sampling design uses independent fields, cross-product cells or post-hoc
reporting slices.

## Sweep

A sweep is a controlled set of training configurations used to choose
hyperparameters. It is not another dataset and does not multiply unique cases.
The local illustrative cost model uses three learning rates and two LoRA ranks,
for six configurations, each trained for one epoch on 25% of the accepted
teacher rollouts. A real sweep should be small, predeclared, evaluated on a
development set, and followed by one final training configuration.

## On-policy prompt group

“On-policy” means the rollouts are generated by the **current student policy**
or its current checkpoint, rather than replayed only from a fixed teacher
dataset. The student can then receive a task reward, teacher-distribution
signal, or both before the next update.

In group-based RL terminology, one prompt/problem forms a group and the current
policy produces multiple independent trajectories for it. The group is the
unit used to compare or centre rewards; it is not necessarily a unique case.
The [Tinker EnvGroupBuilder documentation](https://tinker-docs.thinkingmachines.ai/tutorials/cookbook-abstractions/env-and-envgroupbuilder/)
describes each environment in a group as an independent episode of the same
problem, and the [Tinker RL quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)
shows the cycle of current-policy sampling, scoring and updating.

Therefore, a report of **102,000 on-policy prompt groups** would mean 102,000
group instances were sampled during training. It would not by itself mean
102,000 unique cases or 102,000 conversations: prompts may repeat, and each
group may contain several trajectories. The group size, resampling policy and
unique prompt count are all needed to interpret that number.

# Proposed experiment sequence

1. **Freeze the static contract.** Use `static-plan` and preserve AcuityBench
   exclusively as held-out evaluation.
2. **Build static training cases.** Acquire, deduplicate, group and split a
   separate 500–1,000-case pilot with provenance and licences; validate it
   with `static-data-validate`.
3. **Train and evaluate the static student.** Generate reviewed targets, train
   a small model, measure QA first, then the paired paper comparison, and
   record a human progression decision.
4. **Review and freeze interactive evaluation.** Complete the 100-card
   clinician gate only when progressing to multi-turn work.
5. **Build interactive training cases.** Acquire and split another appropriate
   separate pool; never derive it from the held-out static or interactive
   benchmarks.
6. **Pipeline check.** Generate and inspect 400 consultations from 200 cases;
   prove schema validity, provenance, replay, safety metrics and timing.
7. **First interactive pilot.** Generate 2,000 accepted consultations, train a small
   baseline and start the learning curve.
8. **Targeted data improvement.** Add cases for weak strata and under-triage
   failures rather than scaling uniformly.
9. **Credible interactive v1.** Compare simple SFT, a small hyperparameter sweep and, only
   if useful, one on-policy correction round.
10. **Scale conditionally.** Attempt a strong research run only after the
   smaller stages show reproducible gains and the review process can support
   the volume.

The financial-task project that motivated the desired accuracy/cost frontier
used expert-labelled held-out evaluation and iterative training, but its
sample size and recipe are not evidence for this clinical task. Treat it as
inspiration, not a power calculation. See [Thinking Machines Lab and
Bridgewater's write-up](https://thinkingmachines.ai/news/learning-to-replicate-expert-judgment-in-financial-tasks/).

# Tinker status

Tinker is one optional managed implementation path. Its current documentation
includes [off-policy and on-policy distillation recipes](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/),
including multi-turn environments. This repository currently contains only a
versioned **cost model** for a possible Tinker pilot. It has no Tinker SDK
dependency, client, training loop, environment adapter or checkpoint.

Do not describe the $62.2905 compute estimate as a completed run. Before using
Tinker or any alternative, add an explicit implementation plan, credential and
data-handling review, smoke test, measured token counts and a new cost scenario
based on the chosen models.

# Governance

The repository is intentionally public, but every source licence and privacy
constraint still applies. Synthetic generation does not erase restrictions in
the source cases used to condition it. Preserve source provenance, prohibit
seed contamination, document teacher terms, and route clinically consequential
label or simulator changes through qualified review.
