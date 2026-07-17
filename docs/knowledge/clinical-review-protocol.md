---
type: Process
title: Proposed Blinded Clinical Review Protocol
description: Concrete proposed workflow for source-grounding review, two independent GP labels, adjudication, replacement, and freezing a reviewed interactive evaluation seed.
tags: [clinical-review, gp-labels, adjudication, evaluation, governance]
timestamp: 2026-07-17T00:00:00+01:00
status: proposed
last_verified: 2026-07-17
---

# Status and decision boundary

This is a **proposed future protocol**, not a completed review and not current
software behaviour. The committed manifest records zero clinician-reviewed
interactive cards, and no independent GP labels have been collected.

Before collection begins, the owner and clinical lead should sign off the
review schema, reviewer roles, adjudication rule, replacement rule, and final
seed identifier. The current case-card schema contains only coarse review
status fields; it cannot represent two votes, disagreement, adjudication,
rejection, or replacement provenance. A separate versioned review-record
schema and blinded export tool therefore need to be implemented first.

# Objective

Turn the mechanically routed `seed_v1` candidates into an immutable, auditable
pilot evaluation set whose:

- opening and every revealed answer are supported by the pinned source;
- question routing is coherent enough for a GP-style interaction;
- red flags and required questions have been clinically assessed;
- care-urgency reference comes from two independent GP labels with adjudication
  of disagreement;
- rejected cases and replacements are fully recorded; and
- evaluation cards remain isolated from every training pool.

The result is a more credible **evaluation ruler**, not a trained model, a
clinical validation study, or training data.

# Label rubric supplied to reviewers

Reviewers assign a disposition, not a diagnosis:

| Label | Reviewer meaning |
| --- | --- |
| A | Monitor at home; in-person assessment is not the main recommendation now. |
| B | Routine, non-urgent outpatient assessment within weeks. |
| C | Time-sensitive outpatient assessment within 24–48 hours, without default ED referral. |
| D | Emergency-department evaluation now or an equivalent immediate emergency response. |

The rubric, locale assumptions, and meaning of “equivalent emergency
response” must be identical for both GPs and frozen with the review batch.

# Roles and blinding

Use pseudonymous reviewer IDs in artifacts; do not store clinician names in
the public data bundle.

| Role | Access and responsibility |
| --- | --- |
| Review coordinator | Creates opaque packages, controls blinding, validates completeness, and never supplies a clinical vote. |
| Content reviewer | Checks source support and edits routing. This can be one of the two GP labelers, but content preparation is recorded as a separate pass. |
| GP-A and GP-B | Independently accept/reject the prepared card, identify essential questions/red flags, and assign A–D with rationale before seeing any other vote. |
| Adjudicator | Resolves all label disagreements and material content/acceptance conflicts. Prefer a third GP who did not submit either vote; if moderated GP-A/GP-B consensus is used, record that method explicitly. |

The package shown before independent submission must omit:

- inherited `gold_acuity` and `stratum.acuity`;
- `unsafe_dispositions`;
- inherited label basis and mapping method;
- the other GP's vote, rationale, confidence, and flags;
- aggregate label counts that could reveal the balancing target.

The content pass may see pinned source text and evidence fragments because it
must verify grounding, but source IDs should be replaced with opaque review
IDs. Labelers should inspect the complete prepared interaction—opening plus all
possible answers—through a blinded viewer or export. They should not read the
raw JSON object, which currently contains the inherited reference label.

# Stage 0: freeze inputs and implement review records

1. Record the exact `seed_v1` JSONL, manifest, schemas, catalogs, config and
   source-lock SHA-256 values.
2. Never edit `data/interactive/seed_v1/` in place.
3. Define a new review-batch ID and opaque mapping held by the coordinator.
4. Implement a blinded export that proves prohibited fields are absent.
5. Define and validate a review-record schema before accepting real reviews.
6. Pre-register the label, adjudication and replacement rules below so results
   cannot be curated after seeing model performance.

At minimum, each review record should capture:

```text
review_batch_id
opaque_case_id
input_case_sha256
content_reviewer_id
content_decision: accept | rewrite | reject
content_issue_codes and free-text rationale
revised_opening and revised_queryable_facts
privacy_check and source_support_check
reviewed_red_flags and reviewed_required_question_ids
gp_a_id, gp_a_label, gp_a_rationale, gp_a_confidence, gp_a_timestamp
gp_b_id, gp_b_label, gp_b_rationale, gp_b_confidence, gp_b_timestamp
adjudication_required, adjudication_method, adjudicator_id
adjudicated_label, adjudication_rationale, adjudication_timestamp
final_decision: scoring_set | reviewed_reserve | rejected
replacement_of and replacement_reason
```

Use controlled issue codes for unsupported fact, misplaced fact, duplicate
fact, missing key fact, unrealistic interaction, privacy concern, label
uncertainty, insufficient source, and other. Free text supplements rather than
replaces structured fields.

# Stage 1: ten-card calibration

1. Select ten cards spanning A–D, both source datasets, and several
   presentation groups without showing inherited labels.
2. Have the content reviewer and both labelers apply the draft rubric.
3. Discuss wording, locality assumptions, what counts as required, and common
   rejection reasons.
4. Revise and freeze the instructions and form—not the source facts.
5. Discard the preliminary calibration votes. If those cards remain eligible,
   GP-A and GP-B submit fresh blinded labels after calibration.

Calibration time is not itemized in the current $6,300 estimate and must be
budgeted separately or added to a new assumption version.

# Stage 2: content and source-grounding pass

For every candidate, the content reviewer must:

1. Confirm that the opening and every known answer are supported by the pinned
   user-authored source text.
2. Remove or move fragments routed to the wrong question ID.
3. Split duplicated compound fragments so one answer does not disclose
   unrelated information merely because a keyword matched.
4. Preserve `unknown` for facts absent from the source; never turn silence into
   a negative finding.
5. Keep the case in plausible patient/caregiver voice and remove prompt-like
   instructions from clinical fact slots without altering the medical facts.
6. Verify age, sex and speaker role only where the source supports them.
7. Check for residual identifiers or other privacy concerns.
8. Propose clinically relevant red flags and required questions, distinguishing
   evidence already visible in the opening from information that must be asked.
9. Mark `accept`, `rewrite`, or `reject` with structured reasons.

Reject rather than invent when the source is too incomplete, contradictory, or
artificial to support a realistic consultation. Every rewrite must retain
evidence-level provenance so another reviewer can trace it back to source text.

# Stage 3: independent GP decisions

After content preparation, GP-A and GP-B work independently and each submit:

- accept/reject of the case as an interactive GP encounter;
- one A–D disposition based on all supported case information;
- a concise rationale referencing decisive facts;
- any missing or incorrect red flag;
- questions that are genuinely required before safe disposition, excluding
  facts already apparent in the opening;
- confidence on a frozen ordinal scale; and
- a flag for “cannot label safely from supported information.”

Neither GP may see the other's form or inherited label before both forms are
locked. If a content correction is requested after one label is submitted, the
case version changes and **both** labels must be repeated against the same new
content hash.

# Stage 4: adjudication and final label

Adjudication is mandatory when:

- GP-A and GP-B assign different A–D labels;
- either reviewer rejects or marks the case unlabelable;
- reviewers materially disagree about a red flag or required question; or
- a post-label content correction changes clinically relevant information.

The adjudicator sees the frozen reviewed card, both rationales, and the rubric,
but should initially remain blind to the inherited AcuityBench label. The
adjudicator records a final A–D label or rejects the case, with a concise
rationale and issue codes.

The reviewed artifact must preserve two distinct fields: the inherited
AcuityBench reference and the adjudicated GP label. Whether v2 uses the latter
as its primary scoring target is a material human-owned decision that must be
approved before the schema and analysis plan are frozen. Regardless of that
choice, unseal the inherited reference only after independent review, report
both targets and their transition matrix, and never overwrite history or hide
disagreement.

The current cost model assumes 20% of cases need five minutes of adjudication.
That fraction is a planning input, not a cap: all actual disagreements must be
adjudicated even if the rate is higher.

# Stage 5: deterministic rejection and replacement

Subject to approval of the scoring-label policy above, the proposed design
target is 25 accepted cases per **adjudicated** A–D class. Use this predeclared
rule:

1. Review the original 100 cards in stable case-ID order.
2. Place every accepted card into the pool for its adjudicated label, even when
   that differs from its inherited label.
3. Within each final-label pool, prioritize original seed cards in stable
   case-ID order. The first 25 enter the scoring set; additional accepted cards
   become a reviewed reserve rather than being deleted.
4. For any label with fewer than 25, take the next unused candidate from the
   deterministic ranking produced by the v1 selection process, initially
   preferring the depleted label/source stratum. Run the complete content,
   double-label and adjudication workflow on every replacement.
5. Exclude reused source IDs and known exact or near-duplicate/related cases so
   one clinical scenario cannot appear twice or cross a future train/evaluation
   boundary.
6. Iterate until all classes reach 25 or the eligible pool is exhausted.

Candidate headroom is thin in two inherited strata: the current manifest lists
6 eligible A/HealthBench candidates for 5 selected and 11 C/HealthBench
candidates for 10 selected. If a source stratum is exhausted, prioritize the
25-per-adjudicated-label target, document the source-quota deviation, and
update the manifest. If a label itself cannot reach 25, stop and record a
design exception; do not silently lower the count, relax exclusions, or
relabel a card to fill a quota.

# Stage 6: freeze a new immutable artifact

Use a new versioned directory, illustratively `data/interactive/seed_v2/`; the
exact name and schema version must be fixed before collection. Do not replace
or rewrite v1.

The frozen package should contain:

- schema-valid reviewed case cards;
- versioned, pseudonymized review records permitted for repository inclusion;
- a manifest with every input/output/config/schema digest;
- counts accepted, rejected, rewritten, replaced, reserved and adjudicated;
- label counts, source counts, presentation groups and replacement lineage;
- raw GP agreement, ordinal weighted agreement, adjudication rate and label
  transition matrix;
- reviewer-role and calibration metadata without personal identities;
- limitations and any quota exceptions; and
- an explicit `training_allowed: false` guard on every scoring card.

Set `requires_clinician_review: false` only for cards that completed every
required stage. Preserve source-derived and clinician-reviewed labels as
separate provenance concepts even if the v2 schema presents the adjudicated
label as gold.

# Acceptance gate

The reviewed seed is ready for policy evaluation only when:

- every scoring card has a completed source-grounding/privacy check;
- GP-A and GP-B submitted independent labels against the identical content
  hash;
- every disagreement or material conflict is adjudicated;
- all scoring cards pass the new schema and deterministic validation;
- every accepted fact maps to source evidence and no absent fact is invented;
- the final manifest reconciles card, label, source, review and replacement
  counts;
- there are 25 adjudicated A, B, C and D scoring cards, or a prominently
  documented approved design exception;
- scoring cards and related cases are excluded from all future training pools;
  and
- the frozen artifacts and instructions have stable SHA-256 digests.

Only after this gate should the project implement and run teacher/student
interactive policies. The repository currently has no such policy runner, no
teacher trajectories, no separate training pool and no trained student.

# What 100 reviewed cards can and cannot establish

They can provide a credible pilot ruler for:

- overall and per-acuity reference agreement;
- under-triage versus over-triage;
- handoff behaviour;
- question count and clinically reviewed required-question coverage;
- per-turn TTFT and full-consultation latency once a runner exists; and
- cost per completed consultation once usage collection exists.

They cannot provide training scale, fine subgroup evidence, clinical-system
validation, or narrow performance estimates. Worst-case binomial 95% precision
is about ±10 percentage points overall and about ±20 points for each 25-case
class. Prefer paired tests or paired bootstrap intervals when comparing models
on the same cards, and report raw denominators.

# Cost boundary

The committed standalone estimate is $6,300 for 35 clinician hours:

- 10 minutes of content review per case;
- two independent five-minute labels per case;
- 20% of cases receiving five minutes of adjudication; and
- a fully loaded rate of $180/hour.

This is a USD planning assumption, not observed billing or a quote. It excludes
engineering, calibration, project coordination, reviewer recruitment,
replacement cases, taxes and any adjudication above the assumed 20%. If this
protocol changes the workload—as a full double content review would—create a
new cost-assumption version rather than claiming the v1 estimate still applies.
The separate $36,062.29 training-pilot scenario is not included in the $6,300
and does not mean training has occurred.

# Repository visibility and data rights

The repository is intentionally **public** by owner decision. Public visibility
does not grant a data or annotation licence and does not remove CC BY-NC or
pending-license constraints. Review exports can add sensitive professional
notes or repeat source-derived text, so include only pseudonymized fields and
content whose redistribution is permitted. Consult
[`NOTICE.md`](../../NOTICE.md) and [`sources.lock.json`](../../sources.lock.json)
before adding review artifacts.

# Related concepts and evidence

- [Interactive triage protocol](interactive-triage.md)
- [Case-card example](case-card-example.md)
- [`../interactive-triage-v1.md`](../interactive-triage-v1.md)
- [`../../configs/interactive/cost_assumptions.v1.yaml`](../../configs/interactive/cost_assumptions.v1.yaml)
- [`../../results/interactive-pilot-v1/cost_estimate.md`](../../results/interactive-pilot-v1/cost_estimate.md)
- [`../../data/interactive/seed_v1/manifest.json`](../../data/interactive/seed_v1/manifest.json)
- [`../../schemas/interactive-case-card-v1.schema.json`](../../schemas/interactive-case-card-v1.schema.json)
