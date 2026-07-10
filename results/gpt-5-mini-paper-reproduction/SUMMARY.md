# AcuityBench report: gpt-5-mini-paper-reproduction

Scope: full 914-case benchmark. Requested model: `gpt-5-mini`.
Returned target snapshot(s): gpt-5-mini-2025-08-07.

## Paper-style main table

| Model      | Scope                   | QA N | QA Exact | QA Over | QA Under | Conv N | Conv Exact | Conv Over | Conv Under |
| ---------- | ----------------------- | ---- | -------- | ------- | -------- | ------ | ---------- | --------- | ---------- |
| gpt-5-mini | full 914-case benchmark | 527  | 0.768    | 0.065   | 0.167    | 527    | 0.681      | 0.027     | 0.292      |

The main table scores only primary cases with clear A/B/C/D gold labels; valid sample labels are aggregated by mode with severe tie-breaking.

## Usage and estimated cost

| phase  | configured_model | returned_models       | calls | attempts_tracked | legacy_parent_records | usage_records_expected | usage_records_complete | missing_usage_records | missing_cache_breakdown_records | usage_coverage | cache_breakdown_coverage | cost_completeness | billing_basis                                   | input_tokens | cached_input_tokens | output_tokens | reasoning_tokens | estimated_cost_usd | input_cost_per_million | cached_input_cost_per_million | output_cost_per_million |
| ------ | ---------------- | --------------------- | ----- | ---------------- | --------------------- | ---------------------- | ---------------------- | --------------------- | ------------------------------- | -------------- | ------------------------ | ----------------- | ----------------------------------------------- | ------------ | ------------------- | ------------- | ---------------- | ------------------ | ---------------------- | ----------------------------- | ----------------------- |
| target | gpt-5-mini       | gpt-5-mini-2025-08-07 | 9140  | 0                | 9140                  | 9140                   | 9140                   | 0                     | 0                               | 1.0            | 1.0                      | complete          | all_tracked_attempts_plus_untracked_parent_rows | 3107920      | 7296                | 9091918       | 5379840          | 18.9592            | 0.25                   | 0.025                         | 2.0                     |
| judge  | gpt-4.1          | gpt-4.1-2025-04-14    | 4570  | 0                | 4570                  | 4570                   | 4570                   | 0                     | 0                               | 1.0            | 1.0                      | complete          | all_tracked_attempts_plus_untracked_parent_rows | 5945034      | 0                   | 696370        | 0                | 17.461             | 2.0                    | 0.5                           | 8.0                     |

Estimated total: $36.4202 USD.

## Latency

| phase  | task_type | metric               | n_measured | coverage | p50_ms  | p95_ms  |
| ------ | --------- | -------------------- | ---------- | -------- | ------- | ------- |
| target | conv      | server_processing_ms | 4570       | 100.0%   | 17456.0 | 30684.0 |
| target | qa        | server_processing_ms | 4570       | 100.0%   | 7820.5  | 13032.2 |
| judge  | conv      | server_processing_ms | 4570       | 100.0%   | 1788.0  | 3670.7  |

The primary serving metric is p95 `service_latency_ms`; TTFT is the first non-empty visible text delta. Provider processing is reported separately and is not pure model compute.

## Published comparison

| model_id   | task_type | metric | published | fresh_run | delta  |
| ---------- | --------- | ------ | --------- | --------- | ------ |
| gpt-5-mini | qa        | exact  | 0.780     | 0.769     | -0.011 |
| gpt-5-mini | qa        | over   | 0.055     | 0.065     | 0.010  |
| gpt-5-mini | qa        | under  | 0.165     | 0.167     | 0.002  |
| gpt-5-mini | conv      | exact  | 0.677     | 0.681     | 0.004  |
| gpt-5-mini | conv      | over   | 0.036     | 0.027     | -0.009 |
| gpt-5-mini | conv      | under  | 0.286     | 0.292     | 0.006  |

This is a fresh stochastic run; the published aliases were not immutable experiment artifacts.

## Physician-panel ambiguous cases

| task_type | n   | n_evaluable | jsd_mean | wasserstein_mean | rater_probability_mean | consensus_loo_change_rate_mean | consensus_loo_mean_delta_mean | baseline_human_alpha | mean_loo_alpha | alpha_delta |
| --------- | --- | ----------- | -------- | ---------------- | ---------------------- | ------------------------------ | ----------------------------- | -------------------- | -------------- | ----------- |
| qa        | 217 | 217         | 0.2546   | 0.8678           | 0.3868                 | 0.336                          | 0.2961                        | 0.0409               | 0.1383         | 0.0974      |
| conv      | 217 | 217         | 0.2912   | 1.0576           | 0.3423                 | 0.3843                         | 0.3578                        | 0.0409               | 0.0736         | 0.0326      |

JSD uses natural logarithms for compatibility with the released analysis (its maximum is ln(2), not 1).
