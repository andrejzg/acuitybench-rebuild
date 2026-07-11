# AcuityBench report: gpt-5.4-paper-stream-none-20260711

Scope: full 914-case benchmark. Requested model: `gpt-5.4`.
Returned target snapshot(s): gpt-5.4-2026-03-05.

## Inference contract

Reasoning effort: `none` (paper_unreported_resolved_to_documented_provider_default). The paper did not report a reasoning effort or separate reasoning-token budget.
Completion cap: 4,096 tokens; retry cap: 4,096. This combined cap includes hidden reasoning and visible output.
Configured temperature: 1.0; parameter sent: True. Service tier: `default` (paper unreported).
Execution streaming value(s): [True]; concurrency value(s): [20].

## Paper-style main table

| Model   | Scope                   | QA N | QA Exact | QA Over | QA Under | Conv N | Conv Exact | Conv Over | Conv Under |
| ------- | ----------------------- | ---- | -------- | ------- | -------- | ------ | ---------- | --------- | ---------- |
| gpt-5.4 | full 914-case benchmark | 527  | 0.767    | 0.152   | 0.082    | 527    | 0.780      | 0.036     | 0.184      |

The main table scores only primary cases with clear A/B/C/D gold labels; valid sample labels are aggregated by mode with severe tie-breaking.

## Usage and estimated cost

| phase  | configured_model | reasoning_effort | reasoning_effort_basis                                   | service_tier | max_output_tokens | max_retry_output_tokens | returned_models    | calls | attempts_tracked | legacy_parent_records | usage_records_expected | usage_records_complete | missing_usage_records | missing_cache_breakdown_records | usage_coverage | cache_breakdown_coverage | reasoning_token_coverage | cost_completeness | billing_basis                                   | input_tokens | cached_input_tokens | output_tokens | reasoning_tokens | reasoning_tokens_observed | reasoning_tokens_per_successful_call | estimated_cost_usd | input_cost_per_million | cached_input_cost_per_million | output_cost_per_million |
| ------ | ---------------- | ---------------- | -------------------------------------------------------- | ------------ | ----------------- | ----------------------- | ------------------ | ----- | ---------------- | --------------------- | ---------------------- | ---------------------- | --------------------- | ------------------------------- | -------------- | ------------------------ | ------------------------ | ----------------- | ----------------------------------------------- | ------------ | ------------------- | ------------- | ---------------- | ------------------------- | ------------------------------------ | ------------------ | ---------------------- | ----------------------------- | ----------------------- |
| target | gpt-5.4          | none             | paper_unreported_resolved_to_documented_provider_default | default      | 4096              | 4096                    | gpt-5.4-2026-03-05 | 9140  | 9140             | 0                     | 9140                   | 9140                   | 0                     | 0                               | 1.0            | 1.0                      | 1.0                      | complete          | all_tracked_attempts_plus_untracked_parent_rows | 3107920      | 0                   | 2580501       | 0                | 0                         | 0.0                                  | 46.4773            | 2.5                    | 0.25                          | 15.0                    |
| judge  | gpt-4.1          |                  |                                                          |              | 1024              | 8192                    | gpt-4.1-2025-04-14 | 4570  | 4570             | 0                     | 4570                   | 4570                   | 0                     | 0                               | 1.0            | 1.0                      | 1.0                      | complete          | all_tracked_attempts_plus_untracked_parent_rows | 4987734      | 0                   | 672517        | 0                | 0                         | 0.0                                  | 15.3556            | 2.0                    | 0.5                           | 8.0                     |

Estimated total: $61.8329 USD.

## Latency

| phase  | task_type | metric               | n_measured | coverage | p50_ms | p95_ms  |
| ------ | --------- | -------------------- | ---------- | -------- | ------ | ------- |
| target | conv      | service_latency_ms   | 4570       | 100.0%   | 4861.9 | 10564.4 |
| target | conv      | ttft_ms              | 4570       | 100.0%   | 610.1  | 1096.5  |
| target | conv      | server_processing_ms | 4570       | 100.0%   | 384.0  | 861.6   |
| target | qa        | service_latency_ms   | 4570       | 100.0%   | 2175.6 | 3401.3  |
| target | qa        | ttft_ms              | 4570       | 100.0%   | 591.1  | 1102.5  |
| target | qa        | server_processing_ms | 4570       | 100.0%   | 386.0  | 876.0   |
| judge  | conv      | service_latency_ms   | 4570       | 100.0%   | 1739.6 | 4324.7  |
| judge  | conv      | ttft_ms              | 4570       | 100.0%   | 593.1  | 1062.1  |
| judge  | conv      | server_processing_ms | 4570       | 100.0%   | 336.0  | 615.8   |

The primary serving metric is p95 `service_latency_ms`; TTFT is the first non-empty visible text delta. Provider processing is reported separately and is not pure model compute.

## Published comparison

| model_id | task_type | metric | published | fresh_run | delta  |
| -------- | --------- | ------ | --------- | --------- | ------ |
| gpt-5.4  | qa        | exact  | 0.772     | 0.767     | -0.005 |
| gpt-5.4  | qa        | over   | 0.142     | 0.152     | 0.010  |
| gpt-5.4  | qa        | under  | 0.085     | 0.082     | -0.003 |
| gpt-5.4  | conv      | exact  | 0.772     | 0.780     | 0.008  |
| gpt-5.4  | conv      | over   | 0.049     | 0.036     | -0.013 |
| gpt-5.4  | conv      | under  | 0.178     | 0.184     | 0.006  |

Baseline: [AcuityBench Table 2](https://arxiv.org/pdf/2605.11398), published to three decimals. Delta is fresh run minus published.

This is a fresh stochastic run; the published aliases were not immutable experiment artifacts.

## Physician-panel ambiguous cases

| task_type | n   | n_evaluable | jsd_mean | wasserstein_mean | rater_probability_mean | consensus_loo_change_rate_mean | consensus_loo_mean_delta_mean | baseline_human_alpha | mean_loo_alpha | alpha_delta |
| --------- | --- | ----------- | -------- | ---------------- | ---------------------- | ------------------------------ | ----------------------------- | -------------------- | -------------- | ----------- |
| qa        | 217 | 217         | 0.2799   | 0.8681           | 0.3758                 | 0.3216                         | 0.2792                        | 0.0409               | 0.126          | 0.0851      |
| conv      | 217 | 217         | 0.2591   | 0.8883           | 0.381                  | 0.3651                         | 0.3163                        | 0.0409               | 0.1306         | 0.0896      |

JSD uses natural logarithms for compatibility with the released analysis (its maximum is ln(2), not 1).
