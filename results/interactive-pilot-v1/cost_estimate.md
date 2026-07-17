# Interactive pilot cost estimate

> **Planning estimate:** generated from versioned assumptions; this is not observed billing or a provider quote.

## Assumptions and provenance

- Assumption set: `interactive-pilot-tinker-2026-07-17-v1`
- Description: 500-case feasibility pilot with four accepted teacher rollouts per case
- Assumptions schema: v1
- Formula version: v1
- Assumptions fingerprint: `e401a4c797f94bbb07eb33cc202cedb28779924ae6862264d8905e4c2f25c407`
- Pricing source: [https://tinker-docs.thinkingmachines.ai/tinker/models/](https://tinker-docs.thinkingmachines.ai/tinker/models/)
- Pricing effective: 2026-07-17
- Pricing accessed: 2026-07-17

## Scenario

| Quantity | Value |
| --- | ---: |
| Unique clinical cases | 500 |
| Accepted teacher consultations | 2,000 |
| Mean GP actions per consultation | 6 |
| Supervised action targets | 12,000 |
| Evaluation cases | 500 |
| On-policy consultations | 2,000 |

## Standalone seed evaluation preparation

> **Not included in the 500-case pilot totals below.** This separately estimates preparation of the actual 100-case seed evaluation set.

- Estimate version: v1
- Seed cases: **100**
- Content review: 10 minutes per case
- Independent labels: 2 per case × 5 minutes
- Adjudication: 0.2 of cases × 5 minutes
- Clinician time: 35 hours at $180.0000/hour
- Clinician review: **$6,300.0000**
- Deterministic local build and provider compute: **$0.0000**
- Standalone seed preparation total: **$6,300.0000**
- Engineering cost: **excluded**

Formula: `cases * (content_review_minutes_per_case + independent_labels_per_case * minutes_per_label + adjudication_fraction * minutes_per_adjudication) / 60 * clinician_hourly_rate`

## Models and unit prices

Prices are USD per million tokens.

| Role | Model | Prefill | Cached prefill | Sample | Train |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher | `Qwen/Qwen3.5-397B-A17B` | $3.000 | $0.600 | $7.500 | — |
| Student | `Qwen/Qwen3.5-4B` | $0.330 | $0.066 | $1.005 | $0.737 |

## Cost summary

| Component | Category | Estimated cost |
| --- | --- | ---: |
| Teacher generation | Tinker compute | $36.0000 |
| SFT | Tinker compute | $6.6330 |
| Sweep | Tinker compute | $3.3165 |
| Evaluation | Tinker compute | $1.0260 |
| On-policy | Tinker compute | $15.3150 |
| Clinician review | Human review | $36,000.0000 |

- Tinker compute: **$62.2905**
- Clinician review: **$36,000.0000**
- Overall estimate: **$36,062.2905**

## Formula details

### Teacher generation

Formula: `(uncached_prefill_tokens * teacher_prefill_rate + cached_prefill_tokens * teacher_cached_prefill_rate + sampled_tokens * teacher_sample_rate) / 1_000_000`

| Quantity | Value |
| --- | ---: |
| `cached_prefill_cost_usd` | $0.0000 |
| `cached_prefill_tokens` | 0 |
| `rollouts` | 2,000 |
| `sample_cost_usd` | $6.0000 |
| `sampled_tokens` | 800,000 |
| `uncached_prefill_cost_usd` | $30.0000 |
| `uncached_prefill_tokens` | 10,000,000 |

### SFT

Formula: `teacher_rollouts * sequence_tokens_per_rollout * epochs * student_train_rate / 1_000_000`

| Quantity | Value |
| --- | ---: |
| `epochs` | 3 |
| `sequence_tokens_per_rollout` | 1,500 |
| `teacher_rollouts` | 2,000 |
| `train_tokens` | 9,000,000 |

### Sweep

Formula: `teacher_rollouts * rollout_fraction * learning_rate_count * lora_rank_count * sequence_tokens_per_rollout * sweep_epochs * student_train_rate / 1_000_000`

| Quantity | Value |
| --- | ---: |
| `configurations` | 6 |
| `epochs_per_configuration` | 1 |
| `learning_rate_count` | 3 |
| `lora_rank_count` | 2 |
| `rollouts_per_configuration` | 500 |
| `train_tokens` | 4,500,000 |

### Evaluation

Formula: `(uncached_prefill_tokens * student_prefill_rate + cached_prefill_tokens * student_cached_prefill_rate + sampled_tokens * student_sample_rate) / 1_000_000`

| Quantity | Value |
| --- | ---: |
| `cached_prefill_cost_usd` | $0.0000 |
| `cached_prefill_tokens` | 0 |
| `rollouts` | 500 |
| `sample_cost_usd` | $0.2010 |
| `sampled_tokens` | 200,000 |
| `uncached_prefill_cost_usd` | $0.8250 |
| `uncached_prefill_tokens` | 2,500,000 |

### On-policy

Formula: `student_generation_cost + teacher_scored_tokens * teacher_prefill_rate / 1_000_000 + student_train_tokens * student_train_rate / 1_000_000`

| Quantity | Value |
| --- | ---: |
| `cached_prefill_cost_usd` | $0.0000 |
| `cached_prefill_tokens` | 0 |
| `rollouts` | 2,000 |
| `rounds` | 1 |
| `sample_cost_usd` | $0.8040 |
| `sampled_tokens` | 800,000 |
| `student_train_tokens` | 3,000,000 |
| `student_training_cost_usd` | $2.2110 |
| `teacher_scored_tokens` | 3,000,000 |
| `teacher_scoring_cost_usd` | $9.0000 |
| `uncached_prefill_cost_usd` | $3.3000 |
| `uncached_prefill_tokens` | 10,000,000 |

### Clinician review

Formula: `(training_case_review_hours + rollout_review_hours + evaluation_label_hours + adjudication_hours) * clinician_hourly_rate`

| Quantity | Value |
| --- | ---: |
| `adjudication_hours` | 8.333333 |
| `evaluation_label_hours` | 83.333333 |
| `hourly_rate_usd` | $180.0000 |
| `rollout_review_hours` | 25 |
| `total_hours` | 200 |
| `training_case_review_hours` | 83.333333 |

## Scope note

The estimate includes only the itemized Tinker token operations and clinician-review workload. It excludes unmodeled engineering, data acquisition, taxes, and production serving costs.
