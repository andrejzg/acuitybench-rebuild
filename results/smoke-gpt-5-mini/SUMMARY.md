# AcuityBench report: smoke-gpt-5-mini

Scope: selected 2-case run. Requested model: `gpt-5-mini`.
Returned target snapshot(s): gpt-5-mini-2025-08-07.

## Paper-style main table

| Model      | Scope               | QA N | QA Exact | QA Over | QA Under | Conv N | Conv Exact | Conv Over | Conv Under |
| ---------- | ------------------- | ---- | -------- | ------- | -------- | ------ | ---------- | --------- | ---------- |
| gpt-5-mini | selected 2-case run | 2    | 1.000    | 0.000   | 0.000    | 2      | 1.000      | 0.000     | 0.000      |

The main table scores only primary cases with clear A/B/C/D gold labels; valid sample labels are aggregated by mode with severe tie-breaking.

## Usage and estimated cost

| phase  | configured_model | returned_models       | calls | attempts_tracked | legacy_parent_records | usage_records_expected | usage_records_complete | missing_usage_records | missing_cache_breakdown_records | usage_coverage | cache_breakdown_coverage | cost_completeness | billing_basis                                   | input_tokens | cached_input_tokens | output_tokens | reasoning_tokens | estimated_cost_usd | input_cost_per_million | cached_input_cost_per_million | output_cost_per_million |
| ------ | ---------------- | --------------------- | ----- | ---------------- | --------------------- | ---------------------- | ---------------------- | --------------------- | ------------------------------- | -------------- | ------------------------ | ----------------- | ----------------------------------------------- | ------------ | ------------------- | ------------- | ---------------- | ------------------ | ---------------------- | ----------------------------- | ----------------------- |
| target | gpt-5-mini       | gpt-5-mini-2025-08-07 | 4     | 0                | 4                     | 4                      | 4                      | 0                     | 0                               | 1.0            | 1.0                      | complete          | all_tracked_attempts_plus_untracked_parent_rows | 690          | 0                   | 2596          | 1664             | 0.0054             | 0.25                   | 0.025                         | 2.0                     |
| judge  | gpt-4.1          | gpt-4.1-2025-04-14    | 2     | 0                | 2                     | 2                      | 2                      | 0                     | 0                               | 1.0            | 1.0                      | complete          | all_tracked_attempts_plus_untracked_parent_rows | 1680         | 0                   | 342           | 0                | 0.0061             | 2.0                    | 0.5                           | 8.0                     |

Estimated total: $0.0115 USD.

## Latency

| phase  | task_type | metric               | n_measured | coverage | p50_ms  | p95_ms  |
| ------ | --------- | -------------------- | ---------- | -------- | ------- | ------- |
| target | conv      | server_processing_ms | 2          | 100.0%   | 11390.0 | 13621.1 |
| target | qa        | server_processing_ms | 2          | 100.0%   | 5473.0  | 6459.4  |
| judge  | conv      | server_processing_ms | 2          | 100.0%   | 2173.5  | 2518.6  |

The primary serving metric is p95 `service_latency_ms`; TTFT is the first non-empty visible text delta. Provider processing is reported separately and is not pure model compute.

## Published comparison

| model_id   | task_type | metric | published | fresh_run | delta  |
| ---------- | --------- | ------ | --------- | --------- | ------ |
| gpt-5-mini | qa        | exact  | 0.780     | 1.000     | 0.220  |
| gpt-5-mini | qa        | over   | 0.055     | 0.000     | -0.055 |
| gpt-5-mini | qa        | under  | 0.165     | 0.000     | -0.165 |
| gpt-5-mini | conv      | exact  | 0.677     | 1.000     | 0.323  |
| gpt-5-mini | conv      | over   | 0.036     | 0.000     | -0.036 |
| gpt-5-mini | conv      | under  | 0.286     | 0.000     | -0.286 |

This is a fresh stochastic run; the published aliases were not immutable experiment artifacts.
