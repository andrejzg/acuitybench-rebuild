---
type: Cost Model
title: Interactive Review and Training-Pilot Cost Model
description: Reproducible arithmetic, assumptions and exclusions for the standalone seed review and illustrative Tinker pilot.
tags: [cost, clinician-review, tinker, training, planning]
timestamp: 2026-07-17T00:00:00+01:00
status: active
last_verified: 2026-07-17
---

# Scope

There are two separate planning estimates. The **$6,300** standalone review of
the actual 100-case evaluation seed is not included in the **$36,062.2905**
illustrative 500-case training-pilot total.

Both figures come from
[`../../configs/interactive/cost_assumptions.v1.yaml`](../../configs/interactive/cost_assumptions.v1.yaml)
and deterministic formulas in `acuitybench/interactive/costing.py`. They are
planning assumptions dated 2026-07-17, not invoices, observed billing or
provider quotes.

# Standalone 100-case review gate

Per case, the estimate allows:

- 10 minutes to check and rewrite the mechanically routed content;
- two independent acuity labels at 5 minutes each; and
- adjudication for 20% of cases at 5 minutes each.

The exact arithmetic is:

```text
100 x (10 + 2 x 5 + 0.20 x 5) minutes / 60
= 35 clinician hours

35 hours x $180/hour = $6,300
```

The deterministic seed build makes no provider calls, so local build and
provider compute are recorded as $0. Engineering is excluded. This review is
the next evaluation-quality gate; it does not create enough independent data
to train a model.

# Illustrative 500-case training pilot

The separate pilot assumes:

- 500 unique training cases;
- four accepted teacher consultations per case, or 2,000 rollouts;
- six mean GP actions per rollout, or 12,000 supervised action targets;
- three SFT epochs;
- a six-configuration sweep from 3 learning rates x 2 LoRA ranks;
- 500 evaluation rollouts; and
- one on-policy round with 2,000 student consultations.

## Human review: $36,000

| Work item | Arithmetic | Hours |
| --- | --- | ---: |
| Training-case review | `500 x 10 minutes` | 83.333333 |
| Teacher-rollout review | `2,000 x 25% x 3 minutes` | 25.000000 |
| Independent evaluation labels | `500 x 2 x 5 minutes` | 83.333333 |
| Evaluation adjudication | `500 x 20% x 5 minutes` | 8.333333 |
| **Total** |  | **200.000000** |

```text
200 clinician hours x $180/hour = $36,000
```

## Tinker token compute: $62.2905

The versioned price assumptions are USD per million tokens:

| Role | Model | Prefill | Cached prefill | Sample | Train |
| --- | --- | ---: | ---: | ---: | ---: |
| Teacher | `Qwen/Qwen3.5-397B-A17B` | $3.000 | $0.600 | $7.500 | n/a |
| Student | `Qwen/Qwen3.5-4B` | $0.330 | $0.066 | $1.005 | $0.737 |

The base estimate conservatively assumes zero cached-prefill share.

| Component | Formula summary | Cost |
| --- | --- | ---: |
| Teacher generation | `10M x $3/M + 0.8M x $7.5/M` | $36.0000 |
| SFT | `2,000 x 1,500 x 3 = 9M train tokens`; `9M x $0.737/M` | $6.6330 |
| Sweep | `6 x 500 x 1,500 = 4.5M train tokens`; `4.5M x $0.737/M` | $3.3165 |
| Evaluation | `2.5M x $0.33/M + 0.2M x $1.005/M` | $1.0260 |
| On-policy | Student generation $4.104 + teacher scoring $9.000 + student training $2.211 | $15.3150 |
| **Tinker compute** |  | **$62.2905** |

On-policy detail:

```text
student generation = 10M prefill x $0.33/M
                   + 0.8M sample x $1.005/M = $4.104
teacher scoring    = 3M tokens x $3.00/M       = $9.000
student training   = 3M tokens x $0.737/M      = $2.211
total                                                $15.315
```

The overall pilot estimate is therefore:

```text
$36,000 human review + $62.2905 token compute = $36,062.2905
```

# Exclusions and sensitivity

The estimate excludes:

- engineering and research labour;
- data acquisition, licensing and privacy review;
- clinician recruitment overhead beyond the fully loaded hourly rate;
- taxes and payment fees;
- failed or rejected teacher rollouts beyond the stated accepted workload;
- production deployment, monitoring and serving; and
- future price or model changes.

Human review dominates the estimate. Changing the review protocol, acceptance
rate or hourly rate can matter much more than modest token-price changes. Do
not combine the standalone seed-review cost with the pilot total unless a new
scenario explicitly intends to fund both.

# Regeneration and evidence

Regenerate the committed planning artifacts without network or provider calls:

```bash
uv run python -m acuitybench interactive-cost
```

Inspect:

- [`../../results/interactive-pilot-v1/cost_estimate.json`](../../results/interactive-pilot-v1/cost_estimate.json) for machine-readable formulas and quantities;
- [`../../results/interactive-pilot-v1/cost_estimate.md`](../../results/interactive-pilot-v1/cost_estimate.md) for the rendered report; and
- the assumptions fingerprint in both artifacts before comparing estimates.

If any workload, formula or unit price changes, create a new immutable
assumption version rather than rewriting the historical v1 scenario in place.
