# Interactive triage v1

This package is the first executable step toward a low-latency GP-style acuity
model. It fixes the interface and evaluation contract before any teacher trace
generation or student training.

It is not a clinical system. The simulator is deterministic research
infrastructure, and its automatically transformed case content still requires
clinician review.

## Contract

The case-card schema is
`schemas/interactive-case-card-v1.schema.json`. Hidden evaluator fields include
the queryable facts, gold acuity, red flags, required questions, and unsafe
dispositions. A policy initially sees only `opening_utterance`, then its own
actions and the corresponding patient answers.

The action schema is `schemas/interactive-action-v1.schema.json`:

- `ASK(question_id, wording)` requests one of 33 versioned facts;
- `DISPOSE(acuity, rationale)` ends with A, B, C, or D; and
- `HANDOFF(reason, target="human_clinician")` safely ends autonomous control.

The closed catalog in `configs/interactive/action_catalog.v1.yaml` keeps
question identity stable while allowing a model to phrase the question
naturally. Unknown fields and uncatalogued questions fail closed.

## Determinism

`PatientSimulator` performs an exact `question_id` lookup. It does not call a
model, infer a diagnosis, or generate a new fact. A known answer is assembled
only from routed source fragments; an absent fact always returns the card's
fixed unknown response. Repeated questions return the same answer, and the
case-specific question limit is enforced.

One complete trace can be replayed with:

```bash
python -m acuitybench interactive-simulate \
  --case-id acuity-interactive-v1-038 \
  --actions examples/interactive_actions.json
```

The terminal result reports exact disposition, signed and absolute ordinal
error, under-triage, over-triage, unsafe disposition, handoff, question count,
and coverage of required/red-flag questions. Aggregate helpers keep autonomous
accuracy and handoff rate separate so a model cannot inflate apparent safety
by handing off every case.

## Seed design

The committed v1 seed contains exactly 100 clear primary benchmark cases:

| Reference acuity | HealthBench | PMR-Reddit | Total |
|---|---:|---:|---:|
| A | 5 | 20 | 25 |
| B | 10 | 15 | 25 |
| C | 10 | 15 | 25 |
| D | 15 | 10 | 25 |
| **Total** | **40** | **60** | **100** |

This balance supports macro comparisons across acuity classes. The final set
spans 12 derived presentation groups. Eighty-seven labels are five-physician
panel medians and 13 are direct HealthBench physician-agreed emergency labels.

The case cards are evaluation-only (`training_allowed: false`). Training on
them would contaminate the evaluation because they derive from AcuityBench.
Teacher traces and student SFT data must be generated from a separate case
pool, grouped so related cases cannot cross train/evaluation boundaries.

## Current quality boundary

The build is reproducible, but it is not yet a clinically validated
interactive benchmark:

- source narratives were mechanically routed into question slots;
- one whole source fragment can be reused under multiple question IDs, so
  turn-efficiency and latency conclusions are provisional until clinicians
  split and rewrite the facts;
- a fact not stated in the source remains unknown rather than being invented;
- red-flag markers come from labelled, unreviewed lexical rules;
- the original source can itself be incomplete or awkward;
- no clinician has reviewed the transformed opening/fact split; and
- PMR data is non-commercial and carries privacy/redistribution constraints.

Accordingly, `manifest.json` reports zero clinician-content-reviewed cases.
No result on this seed should be described as clinical performance until the
review gate below is complete.

## The next gate

The next step is a blinded clinician pass over all 100 cards—not model
training. For each case, reviewers should:

1. confirm that every revealed answer is supported by the source;
2. correct the opening/fact routing without exposing the gold label;
3. confirm or replace required questions and red-flag markers;
4. independently re-label acuity, with adjudication for disagreement; and
5. reject cases that cannot support a realistic GP interaction.

Freeze accepted cards as a new immutable seed version; do not silently edit
v1. Only then use the interface to benchmark teacher policies, measure per-turn
TTFT and end-to-end consultation latency, and design a separate training pool.

## Cost

The deterministic build itself makes no provider calls. The standalone
100-case clinical preparation estimate is **$6,300** under the versioned
assumptions: 10 minutes of content review per case, two five-minute independent
labels per case, 20% five-minute adjudication, and a fully loaded clinician
rate of $180/hour. Engineering time is excluded.

The larger illustrative training pilot assumes 500 unique training cases,
four accepted teacher rollouts per case, six GP actions per rollout, SFT, a
six-configuration learning-rate/rank sweep, evaluation, and one on-policy
correction round. Its estimate is:

| Component | Estimate |
|---|---:|
| Tinker token compute | $62.2905 |
| Clinician review | $36,000.0000 |
| **Total** | **$36,062.2905** |

These are planning assumptions, not observed billing or a quote. Human review,
not model compute, dominates. Prices and every token/workload assumption are
versioned in `configs/interactive/cost_assumptions.v1.yaml`; the deterministic
JSON and Markdown reports live in `results/interactive-pilot-v1/`.

Regenerate them with:

```bash
python -m acuitybench interactive-cost
```

## Recommended post-review experiment

Once the reviewed seed is frozen, run the same action protocol with a strong
teacher and a small candidate student. Record, per action and per full
consultation, request start, time to first token, terminal latency, token usage,
cost, action validity, and simulator turn. Compare models on three fronts:

- acuity accuracy and under-triage safety;
- cost per 1,000 completed consultations; and
- p50/p95 TTFT plus p50/p95 end-to-end consultation latency.

That produces the two intended frontiers—accuracy versus cost and accuracy
versus latency—without conflating a static one-shot benchmark with a real
multi-turn triage policy.
