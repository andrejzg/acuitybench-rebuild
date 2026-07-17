---
type: Data Landscape
title: Conversation Data Landscape for Interactive Acuity
description: What the current benchmark, HealthBench and DDXPlus do and do not provide for a live patient-GP triage task.
tags: [data, conversations, healthbench, ddxplus, simulation]
timestamp: 2026-07-17T00:00:00+01:00
status: active
last_verified: 2026-07-17
---

# The interaction the project ultimately wants

The target research task resembles a live primary-care telephone or chat
triage encounter:

1. a patient or caregiver gives an initial concern;
2. the GP policy chooses a question based on everything revealed so far;
3. the patient answers with information that was not necessarily in the
   opening;
4. the GP repeats this conditional process; and
5. the GP ends with an acuity disposition or a safe human handoff.

A natural corpus for that task would contain authentic sequential turns, the
information available at each turn, and a defensible final disposition. Audio
would add timing, interruption, hesitation and speech phenomena. No such
natural patient-GP call corpus is currently pinned or committed in this
repository. This is a statement about the repository, not a claim that no
suitable dataset exists anywhere.

# What AcuityBench contains

The reconstructed [AcuityBench paper benchmark](https://arxiv.org/abs/2605.11398)
contains 914 static examples assembled from five sources. The evaluator renders
each example in two formats:

- `qa`: the model reasons and returns an A/B/C/D label directly; and
- `conv`: the target model writes a response to the final stored user turn,
  then GPT-4.1 applies the paper rubric to map that response to acuity.

The word “conversational” therefore describes a prompt and scoring format. It
does not mean that a model runs a new patient interview turn by turn. For
HealthBench rows, `build_conversational_prompt()` passes the stored message
history through as context; other sources are wrapped as a single user turn.
The target model still produces only the next response.

The exact source revisions and checksums are in
[`../../sources.lock.json`](../../sources.lock.json), and the reconstructed
counts are in
[`../../data/processed/build_report.json`](../../data/processed/build_report.json).

# What HealthBench contains

[OpenAI's HealthBench description](https://openai.com/index/healthbench/)
calls its 5,000 examples realistic health “conversations.” They simulate
interactions between an AI assistant and individual users or clinicians, were
created through synthetic generation and human adversarial testing, and can be
multi-turn and multilingual. The benchmark task is to produce the best
response to the user's **last** message.

Consequently, some HealthBench inputs already contain user/assistant history.
That history is useful context and can resemble a dialogue, but it is not a
corpus of naturally observed, longitudinal patient-GP telephone calls. The
assistant turns are prior model messages, not verified GP questions, and the
example does not expose an environment that answers arbitrary new questions.

The interactive v1 seed deliberately takes a different route. For selected
HealthBench cases, its builder keeps only user-authored messages, removes the
stored assistant messages, and deterministically routes source fragments into
33 queryable fact slots. This creates a reproducible simulator interface, but
not natural dialogue. The transformation and its limitations are documented
in [`interactive-triage.md`](interactive-triage.md) and
[`../../data/interactive/README.md`](../../data/interactive/README.md).

# What DDXPlus contains

[DDXPlus](https://github.com/mila-iqia/ddxplus) is a large **synthetic,
structured patient** dataset for automatic symptom detection and diagnosis.
The [DDXPlus paper](https://arxiv.org/abs/2205.09148) describes roughly 1.3
million generated patients covering 49 pathologies in a cough, sore-throat and
breathing-related chief-complaint family, with 110 symptoms and 113
antecedents. It was generated from a proprietary medical knowledge base and a
commercial rule-based diagnosis system.

The often-repeated number **1,025,602** has a narrower meaning: it is the
number of patient records in the **training CSV**, not the number of natural
conversations and not the total across all splits. Direct line counts of the
official [English Figshare v2 release](https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585)
give:

| Split | Physical CSV lines | Header | Synthetic patient records |
| --- | ---: | ---: | ---: |
| Train | 1,025,603 | 1 | **1,025,602** |
| Validation | 132,449 | 1 | 132,448 |
| Test | 134,530 | 1 | 134,529 |
| **Total** | 1,292,582 | 3 | **1,292,579** |

Each row describes one synthesized patient with fields including age, sex,
ground-truth pathology, a set of binary/categorical/multi-choice evidences, an
initial evidence and a differential diagnosis. It is a case state, not a
transcript. The project README also links a conversion script that can render
structured fields as text; generated text still should not be described as an
observed conversation.

# How DDXPlus could support simulated questioning

The structure is suitable for building a controlled environment:

1. expose age, sex and `INITIAL_EVIDENCE` as the opening state;
2. let a policy ask one catalogued evidence question;
3. answer from the row's evidence set and the evidence metadata's value or
   default;
4. record the conditional question trajectory; and
5. stop with a prediction or disposition.

This is close to the interaction protocol used in DDXPlus experiments: a
model starts from an initial evidence and iteratively asks about symptoms or
antecedents. It can support repeated trajectories from the same underlying
case without pretending those trajectories were spoken by people.

Several adaptations would be required for AcuityBench-style triage:

- DDXPlus pathology severity is not an A/B/C/D care-disposition label;
- any acuity mapping would require explicit clinical design and review;
- its pathology scope is narrower than general primary-care triage;
- synthetic demographic and evidence assumptions require subgroup auditing;
- generated natural-language patient answers need consistency constraints;
  and
- DDXPlus cases must remain separate from the held-out AcuityBench seed.

# Repository status and acquisition rule

DDXPlus is **not** listed in `sources.lock.json`, `ARTIFACTS.sha256`, the source
cache or the processed data inventory. No DDXPlus row is currently part of the
benchmark, interactive seed or training data.

If it is added later, pin a specific release version and file digest, document
the transformation and split policy, and preserve the upstream CC BY 4.0
attribution. This repository is intentionally public, but public visibility
does not override any source's licence, privacy constraints or benchmark
anti-contamination guidance.

# Practical conclusion

The current assets provide three different things:

| Asset | What it provides | What it does not provide |
| --- | --- | --- |
| AcuityBench | Static acuity examples and physician/reference labels | A live environment that answers new GP questions |
| HealthBench histories | Synthetic/adversarial user-assistant context, sometimes multi-turn | Naturally observed patient-GP calls or arbitrary counterfactual answers |
| DDXPlus | Large structured synthetic patient states that can drive questioning | Natural transcripts or ready-made A/B/C/D labels |

The near-term defensible route is therefore a reviewed, deterministic or
carefully constrained simulator, with all generated dialogue labelled as
synthetic. A natural-call dataset, if later identified and lawfully usable,
would be a separate acquisition and governance workstream.
