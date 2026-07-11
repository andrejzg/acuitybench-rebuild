# AcuityBench report: gpt-5-mini-paper-stream-medium-20260711

Scope: full 914-case benchmark. Requested model: `gpt-5-mini`.
Returned target snapshot(s): gpt-5-mini-2025-08-07.

## Inference contract

Reasoning effort: `medium` (paper_unreported_resolved_to_documented_provider_default). The paper did not report a reasoning effort or separate reasoning-token budget.
Completion cap: 4,096 tokens; retry cap: 4,096. This combined cap includes hidden reasoning and visible output.
Configured temperature: 1.0; parameter sent: False. Service tier: `default` (paper unreported).
Execution streaming value(s): [True]; concurrency value(s): [20].

## Paper-style main table

| Model      | Scope                   | QA N | QA Exact | QA Over | QA Under | Conv N | Conv Exact | Conv Over | Conv Under |
| ---------- | ----------------------- | ---- | -------- | ------- | -------- | ------ | ---------- | --------- | ---------- |
| gpt-5-mini | full 914-case benchmark | 527  | 0.782    | 0.049   | 0.169    | 527    | 0.693      | 0.023     | 0.285      |

The main table scores only primary cases with clear A/B/C/D gold labels; valid sample labels are aggregated by mode with severe tie-breaking.

## Usage and estimated cost

| phase  | configured_model | reasoning_effort | reasoning_effort_basis                                   | service_tier | max_output_tokens | max_retry_output_tokens | returned_models       | calls | attempts_tracked | legacy_parent_records | usage_records_expected | usage_records_complete | missing_usage_records | missing_cache_breakdown_records | usage_coverage | cache_breakdown_coverage | reasoning_token_coverage | cost_completeness | billing_basis                                   | input_tokens | cached_input_tokens | output_tokens | reasoning_tokens | reasoning_tokens_observed | reasoning_tokens_per_successful_call | estimated_cost_usd | input_cost_per_million | cached_input_cost_per_million | output_cost_per_million |
| ------ | ---------------- | ---------------- | -------------------------------------------------------- | ------------ | ----------------- | ----------------------- | --------------------- | ----- | ---------------- | --------------------- | ---------------------- | ---------------------- | --------------------- | ------------------------------- | -------------- | ------------------------ | ------------------------ | ----------------- | ----------------------------------------------- | ------------ | ------------------- | ------------- | ---------------- | ------------------------- | ------------------------------------ | ------------------ | ---------------------- | ----------------------------- | ----------------------- |
| target | gpt-5-mini       | medium           | paper_unreported_resolved_to_documented_provider_default | default      | 4096              | 4096                    | gpt-5-mini-2025-08-07 | 9140  | 9143             | 0                     | 9143                   | 9143                   | 0                     | 0                               | 1.0            | 1.0                      | 1.0                      | complete          | all_tracked_attempts_plus_untracked_parent_rows | 3108922      | 15744               | 9176494       | 5429450          | 5429450                   | 594.0317286652079                    | 19.1267            | 0.25                   | 0.025                         | 2.0                     |
| judge  | gpt-4.1          |                  |                                                          |              | 1024              | 8192                    | gpt-4.1-2025-04-14    | 4570  | 4570             | 0                     | 4570                   | 4570                   | 0                     | 0                               | 1.0            | 1.0                      | 1.0                      | complete          | all_tracked_attempts_plus_untracked_parent_rows | 5964365      | 0                   | 697604        | 0                | 0                         | 0.0                                  | 17.5096            | 2.0                    | 0.5                           | 8.0                     |

Estimated total: $36.6362 USD.

## Latency

| phase  | task_type | metric               | n_measured | coverage | p50_ms  | p95_ms  |
| ------ | --------- | -------------------- | ---------- | -------- | ------- | ------- |
| target | conv      | service_latency_ms   | 4570       | 100.0%   | 14099.2 | 23885.6 |
| target | conv      | ttft_ms              | 4570       | 100.0%   | 8081.9  | 13578.4 |
| target | conv      | server_processing_ms | 4570       | 100.0%   | 7871.5  | 13393.0 |
| target | qa        | service_latency_ms   | 4570       | 100.0%   | 6427.4  | 10582.1 |
| target | qa        | ttft_ms              | 4570       | 100.0%   | 4815.8  | 8239.4  |
| target | qa        | server_processing_ms | 4570       | 100.0%   | 4613.5  | 8026.0  |
| judge  | conv      | service_latency_ms   | 4570       | 100.0%   | 1652.0  | 2635.0  |
| judge  | conv      | ttft_ms              | 4570       | 100.0%   | 557.9   | 765.8   |
| judge  | conv      | server_processing_ms | 4570       | 100.0%   | 333.0   | 484.0   |

The primary serving metric is p95 `service_latency_ms`; TTFT is the first non-empty visible text delta. Provider processing is reported separately and is not pure model compute.

## Published comparison

| model_id   | task_type | metric | published | fresh_run | delta  |
| ---------- | --------- | ------ | --------- | --------- | ------ |
| gpt-5-mini | qa        | exact  | 0.780     | 0.782     | 0.002  |
| gpt-5-mini | qa        | over   | 0.055     | 0.049     | -0.006 |
| gpt-5-mini | qa        | under  | 0.165     | 0.169     | 0.004  |
| gpt-5-mini | conv      | exact  | 0.677     | 0.693     | 0.016  |
| gpt-5-mini | conv      | over   | 0.036     | 0.023     | -0.013 |
| gpt-5-mini | conv      | under  | 0.286     | 0.285     | -0.001 |

Baseline: [AcuityBench Table 2](https://arxiv.org/pdf/2605.11398), published to three decimals. Delta is fresh run minus published.

This is a fresh stochastic run; the published aliases were not immutable experiment artifacts.

## Physician-panel ambiguous cases

| task_type | n   | n_evaluable | jsd_mean | wasserstein_mean | rater_probability_mean | consensus_loo_change_rate_mean | consensus_loo_mean_delta_mean | baseline_human_alpha | mean_loo_alpha | alpha_delta |
| --------- | --- | ----------- | -------- | ---------------- | ---------------------- | ------------------------------ | ----------------------------- | -------------------- | -------------- | ----------- |
| qa        | 217 | 217         | 0.2502   | 0.8651           | 0.3845                 | 0.3409                         | 0.2904                        | 0.0409               | 0.1408         | 0.0999      |
| conv      | 217 | 217         | 0.2821   | 1.045            | 0.3612                 | 0.3742                         | 0.3474                        | 0.0409               | 0.0745         | 0.0336      |

JSD uses natural logarithms for compatibility with the released analysis (its maximum is ln(2), not 1).
