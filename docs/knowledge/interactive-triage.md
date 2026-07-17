---
type: Architecture
title: Interactive Triage Protocol and Current Boundary
description: Executable v1 case-card and action contracts, deterministic simulator, seed design, limitations, and the boundary between current and proposed work.
tags: [interactive-triage, case-cards, simulator, evaluation, handover]
timestamp: 2026-07-17T00:00:00+01:00
status: active
last_verified: 2026-07-17
---

# Status in one sentence

The repository currently has deterministic interactive-evaluation plumbing and
an unreviewed 100-card seed. It does **not** yet have a clinically reviewed
interactive benchmark, a model-driven interactive policy runner, teacher
traces, a separate training pool, or a trained student.

# Current implementation versus proposed work

| Capability | Current state |
| --- | --- |
| Strict case-card and action schemas | Implemented as `interactive-case-card/v1` and `interactive-action/v1`. |
| Closed action vocabulary | Implemented: `ASK`, `DISPOSE`, and `HANDOFF`. |
| Closed question vocabulary | Implemented: 33 stable question IDs. |
| Patient simulator | Implemented as deterministic `question_id` lookup; it does not generate facts. |
| Trajectory scoring | Implemented for disposition error, under/over-triage, handoff, question counts, and question coverage. |
| Evaluation seed | Implemented: 100 automatically routed, evaluation-only cards. |
| Clinician review of interactive content | **Not done:** the manifest records 0 reviewed cards and all 100 require review. |
| Independent GP re-labelling | **Not done.** The current labels are inherited references. |
| Interactive model/policy runner | **Not implemented.** `interactive-simulate` replays a supplied JSON/JSONL action list; it does not call a model. |
| Per-turn model TTFT/cost collection | **Not implemented** for this interactive protocol. |
| Separate training data and teacher traces | **Not built.** |
| Student training or Tinker integration | **Not implemented.** The committed Tinker figures are a planning estimate only. |

# Care-urgency labels

Severity is ordinal from A to D. These are dispositions, not diagnoses.

| Label | Meaning | Current rubric boundary |
| --- | --- | --- |
| A | Monitor at home | Self-care or home monitoring; in-person assessment is not the main recommendation. |
| B | See a doctor within weeks | Routine, non-urgent outpatient assessment. |
| C | See a doctor within 24–48 hours | Time-sensitive outpatient assessment without default emergency-department referral. |
| D | Go to the emergency department now | Immediate emergency evaluation or an equivalent emergency response. |

The labels in the v1 seed are released physician/reference labels. They are not
objective clinical ground truth. For scoring, any disposition below the
reference label is currently considered under-triage and unsafe; over-triage is
reported separately.

# Policy contract

A policy begins with the clinical content in `opening_utterance`. Public
simulator state also includes the opaque case ID, transcript, question count,
remaining question budget, and terminal status. Hidden case-card fields must
not be included in a policy prompt.

The policy submits exactly one strict action at a time:

- `ASK(question_id, wording)` is non-terminal. `question_id` must be one of the
  33 catalogued IDs, while `wording` may be natural patient-facing language.
- `DISPOSE(acuity, rationale)` terminates the consultation with A, B, C, or D.
- `HANDOFF(reason, target="human_clinician")` terminates autonomous control
  when safe progression is not possible.

Unknown action fields, branch-specific fields on the wrong action, unknown
question IDs, invalid labels, and actions after termination fail closed. Each
card currently permits at most 12 `ASK` actions. Repeating a question returns
the same answer and still consumes a turn.

The simulator responds by exact ID lookup:

- a known slot returns only text routed from the source;
- an absent fact returns the fixed response `I'm not sure.`;
- it does not infer a negative finding from missing source text;
- it does not diagnose, improvise, paraphrase, or call a model.

# Visible and hidden state

| Policy-visible over time | Evaluator-only |
| --- | --- |
| Opening utterance | Full `queryable_facts` table before questions are asked |
| The policy's previous actions | `gold_acuity` |
| Answers to question IDs actually asked | `unsafe_dispositions` |
| Question budget and transcript | `red_flags` and `required_question_ids` |
| Terminal state | Provenance details not required by the policy |

Evidence already present verbatim in the opening counts as observed for
required-question and red-flag coverage. This is why a card can legitimately
have no required questions even when its opening contains a red flag. See the
[case-card example](case-card-example.md).

# Two question catalogs: do not version one alone

The same 33 IDs intentionally appear in two YAML catalogs with different
responsibilities:

1. [`question_catalog.v1.yaml`](../../configs/interactive/question_catalog.v1.yaml)
   defines deterministic source-routing slots, routing-oriented prompts and
   descriptions, plus the fixed unknown response.
2. [`action_catalog.v1.yaml`](../../configs/interactive/action_catalog.v1.yaml)
   defines the policy action contract, patient-facing canonical wording,
   safety-critical markers, and A–D meanings.

The IDs are also duplicated as closed enums in both JSON Schemas. The seed
builder and tests require identical ordered ID sets across both catalogs and
both schemas. A question-contract change must therefore create and validate a
coherent new version of all four artifacts. Do not silently edit one v1 file
or mutate a frozen reviewed seed.

# Current 100-card seed

The committed `seed_v1` is selected from clear primary AcuityBench cases and is
balanced for macro comparison:

| Reference label | HealthBench | PMR-Reddit | Total |
| --- | ---: | ---: | ---: |
| A | 5 | 20 | 25 |
| B | 10 | 15 | 25 |
| C | 10 | 15 | 25 |
| D | 15 | 10 | 25 |
| **Total** | **40** | **60** | **100** |

Reference-label basis:

- 87 cards inherit the released median of five physician labels;
- 13 cards inherit HealthBench's physician-agreed emergency category; all 13
  are D cards in this seed;
- boundary and ambiguous labels are excluded;
- every card is `evaluation_only` with `training_allowed: false`.

The seed spans 12 mechanically derived presentation groups. Source ID,
benchmark case ID, source-text SHA-256, mapping method, schemas, catalogs,
configuration and output digest are retained in the card or manifest.

# Quality boundary of v1

The transformation is reproducible, but the dialogue split is not clinically
validated. Counts below were recomputed from the committed JSONL on
2026-07-17:

- `clinician_content_reviewed_cases` is 0; all 100 cards say that clinician
  review is required;
- every card reuses at least one evidence fragment across question slots;
- 86 cards contain an identical known answer under more than one question ID;
- 87 cards have an empty `required_question_ids` list;
- 78 cards have no lexical `red_flags` entry;
- red flags are produced by `unreviewed_lexical_rule_v1`, not by a clinician;
- absent source information becomes unknown, so the simulator can be much less
  informative than a real patient;
- original source text can be incomplete, awkward, or instruction-like.

The current required-question list is derived from positive lexical red flags
whose evidence is not already visible in the opening. It is not a comprehensive
clinical minimum-question checklist. Coverage recall is defined as 1.0 when a
card has no required items, so a macro mean can look high largely because of
empty sets. Do not use current required/red-flag recall as a clinical safety
claim.

Whole-fragment reuse also makes question efficiency, consultation length, and
interactive latency provisional: one question can reveal text that properly
belongs under several concepts, while a poorly routed question can reveal
irrelevant text.

# Current scoring semantics

A terminal `DISPOSE` records:

- exact A–D agreement;
- signed and absolute ordinal error;
- under-triage, over-triage, and unsafe disposition;
- question, repetition, required-question, and red-flag coverage counts.

A `HANDOFF` has no predicted acuity and is not counted as an unsafe
disposition. Aggregate output therefore reports autonomous exact accuracy and
handoff rate separately. Overall correct-disposition rate treats a handoff as
not exactly correct, preventing an always-handoff policy from appearing
accurate.

These metrics are executable research definitions. They are not proof of
clinical safety, and no model has yet been evaluated end to end through this
interactive protocol.

# Next gates

1. Apply the proposed [blinded clinical-review protocol](clinical-review-protocol.md).
2. Freeze accepted, adjudicated output as a new immutable seed version while
   retaining inherited labels in provenance.
3. Implement a model-driven policy runner with strict action parsing,
   per-action request/TTFT/terminal timing, token usage, cost and trajectory
   exports.
4. Benchmark strong teacher and candidate student policies on the reviewed
   seed.
5. Build a separate, deduplicated training pool. Never generate training
   traces from `seed_v1` or its reviewed descendants.
6. Only then generate/review teacher traces and train a student.

With 100 cases, worst-case binomial 95% precision is roughly ±10 percentage
points overall and roughly ±20 points within a 25-case label class. Paired
model comparisons can be more informative, but this remains a pilot set for
clear differences and failure analysis, not clinical validation or fine
subgroup claims.

# Repository visibility and data rights

Repository visibility is intentionally **public** by owner decision. Public
visibility does not grant a licence to AcuityBench annotations, PMR-derived
text, HealthBench material, or generated derivatives. Reuse and redistribution
remain governed by each upstream source's terms; the physician-annotation
licence is still recorded as pending. Consult [`NOTICE.md`](../../NOTICE.md)
and [`sources.lock.json`](../../sources.lock.json) before copying data or
publishing derivatives.

# Evidence and implementation

- [`../interactive-triage-v1.md`](../interactive-triage-v1.md)
- [`../../data/interactive/README.md`](../../data/interactive/README.md)
- [`../../data/interactive/seed_v1/manifest.json`](../../data/interactive/seed_v1/manifest.json)
- [`../../schemas/interactive-case-card-v1.schema.json`](../../schemas/interactive-case-card-v1.schema.json)
- [`../../schemas/interactive-action-v1.schema.json`](../../schemas/interactive-action-v1.schema.json)
- [`../../acuitybench/interactive/seed.py`](../../acuitybench/interactive/seed.py)
- [`../../acuitybench/interactive/simulator.py`](../../acuitybench/interactive/simulator.py)
