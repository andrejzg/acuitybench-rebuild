# AcuityBench report: smoke-gpt-5.4-none-stream-20260711

Scope: selected 1-case run. Requested model: `gpt-5.4`.
Returned target snapshot(s): gpt-5.4-2026-03-05.

## Inference contract

Reasoning effort: `none` (paper_unreported_resolved_to_documented_provider_default). The paper did not report a reasoning effort or separate reasoning-token budget.
Completion cap: 4,096 tokens; retry cap: 4,096. This combined cap includes hidden reasoning and visible output.
Configured temperature: 1.0; parameter sent: True. Service tier: `default` (paper unreported).
Execution streaming value(s): [True]; concurrency value(s): [1].

## Paper-style main table

| Model   | Scope               | QA N | QA Exact | QA Over | QA Under | Conv N | Conv Exact | Conv Over | Conv Under |
| ------- | ------------------- | ---- | -------- | ------- | -------- | ------ | ---------- | --------- | ---------- |
| gpt-5.4 | selected 1-case run | 1    | 1.000    | 0.000   | 0.000    | 1      | 1.000      | 0.000     | 0.000      |

The main table scores only primary cases with clear A/B/C/D gold labels; valid sample labels are aggregated by mode with severe tie-breaking.

## Usage and estimated cost

| phase  | configured_model | reasoning_effort | reasoning_effort_basis                                   | service_tier | max_output_tokens | max_retry_output_tokens | returned_models    | calls | attempts_tracked | legacy_parent_records | usage_records_expected | usage_records_complete | missing_usage_records | missing_cache_breakdown_records | usage_coverage | cache_breakdown_coverage | reasoning_token_coverage | cost_completeness | billing_basis                                   | input_tokens | cached_input_tokens | output_tokens | reasoning_tokens | reasoning_tokens_observed | reasoning_tokens_per_successful_call | estimated_cost_usd | input_cost_per_million | cached_input_cost_per_million | output_cost_per_million |
| ------ | ---------------- | ---------------- | -------------------------------------------------------- | ------------ | ----------------- | ----------------------- | ------------------ | ----- | ---------------- | --------------------- | ---------------------- | ---------------------- | --------------------- | ------------------------------- | -------------- | ------------------------ | ------------------------ | ----------------- | ----------------------------------------------- | ------------ | ------------------- | ------------- | ---------------- | ------------------------- | ------------------------------------ | ------------------ | ---------------------- | ----------------------------- | ----------------------- |
| target | gpt-5.4          | none             | paper_unreported_resolved_to_documented_provider_default | default      | 4096              | 4096                    | gpt-5.4-2026-03-05 | 2     | 2                | 0                     | 2                      | 2                      | 0                     | 0                               | 1.0            | 1.0                      | 1.0                      | complete          | all_tracked_attempts_plus_untracked_parent_rows | 345          | 0                   | 388           | 0                | 0                         | 0.0                                  | 0.0067             | 2.5                    | 0.25                          | 15.0                    |
| judge  | gpt-4.1          |                  |                                                          |              | 1024              | 8192                    | gpt-4.1-2025-04-14 | 1     | 1                | 0                     | 1                      | 1                      | 0                     | 0                               | 1.0            | 1.0                      | 1.0                      | complete          | all_tracked_attempts_plus_untracked_parent_rows | 814          | 0                   | 130           | 0                | 0                         | 0.0                                  | 0.0027             | 2.0                    | 0.5                           | 8.0                     |

Estimated total: $0.0094 USD.

## Latency

| phase  | task_type | metric               | n_measured | coverage | p50_ms | p95_ms |
| ------ | --------- | -------------------- | ---------- | -------- | ------ | ------ |
| target | conv      | service_latency_ms   | 1          | 100.0%   | 4675.5 | 4675.5 |
| target | conv      | ttft_ms              | 1          | 100.0%   | 623.2  | 623.2  |
| target | conv      | server_processing_ms | 1          | 100.0%   | 421.0  | 421.0  |
| target | qa        | service_latency_ms   | 1          | 100.0%   | 1558.7 | 1558.7 |
| target | qa        | ttft_ms              | 1          | 100.0%   | 817.2  | 817.2  |
| target | qa        | server_processing_ms | 1          | 100.0%   | 355.0  | 355.0  |
| judge  | conv      | service_latency_ms   | 1          | 100.0%   | 2001.4 | 2001.4 |
| judge  | conv      | ttft_ms              | 1          | 100.0%   | 475.5  | 475.5  |
| judge  | conv      | server_processing_ms | 1          | 100.0%   | 274.0  | 274.0  |

The primary serving metric is p95 `service_latency_ms`; TTFT is the first non-empty visible text delta. Provider processing is reported separately and is not pure model compute.
