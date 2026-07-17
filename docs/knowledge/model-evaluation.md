---
type: Evaluation
title: Model Evaluation, Cost and Latency
description: Verified static-benchmark evaluation contract, completed model results, timing semantics, charts and model-extension path.
tags: [evaluation, models, latency, cost, openai]
timestamp: 2026-07-17T00:00:00+01:00
status: active
last_verified: 2026-07-17
---

# Evaluation boundary

The completed model runs evaluate the reconstructed, static 914-case
AcuityBench benchmark. They do **not** exercise the newer interactive patient
simulator and are not end-to-end consultation measurements.

All 914 cases are retained for audit and secondary analyses. The paper-style
main score uses only the 527 primary cases with an unambiguous A/B/C/D
reference label. For each case and prompt format, the evaluator takes the mode
of five valid sample labels and resolves ties toward the more severe label.
Boundary and ambiguous cases are reported separately.

# Paper model roster

The paper evaluated 12 target models. This repository has so far reproduced
only GPT-5-mini and GPT-5.4:

- OpenAI: GPT-5.4, GPT-5-mini and GPT-4.1;
- Anthropic: Claude Opus 4.7, Claude Sonnet 4.6 and Claude Haiku 4.5;
- Google: Gemini 2.5 Pro and Gemini 2.5 Flash;
- open-weight: DeepSeek V3.1, Qwen 2.5 72B, Qwen 2.5 7B and Llama 3.3 70B.

This list is the paper's May 2026 experiment roster, not a recommendation of
currently available model aliases. See the
[AcuityBench paper, section 4.1.1](https://arxiv.org/pdf/2605.11398).

# Target generation and judging

Target generation and rubric judging are different API workloads:

1. The target model receives every case in both `qa` and `conv` prompt formats,
   with five samples per format:

   ```text
   914 cases x 2 formats x 5 samples = 9,140 target generations
   ```

2. QA responses contain a directly parsed acuity label. Conversational target
   responses are scored by the configured GPT-4.1 paper-rubric judge:

   ```text
   914 conversational cases x 5 samples = 4,570 judge calls
   ```

The judge is not the target model and its tokens, latency and cost are reported
as a separate phase. The current judge profile is `paper-gpt-4.1`, backed by
returned snapshot `gpt-4.1-2025-04-14`, temperature 0, and a 1,024-token output
limit.

# Completed paired runs

The paired runs streamed at concurrency 20 on the OpenAI `default` service
tier. Both use temperature 1 and a hard 4,096-token combined
reasoning-plus-visible-output cap. The paper did not state reasoning effort;
the manifests record the explicit reconstruction choices rather than treating
them as paper-reported facts.

| Model run | Reasoning | QA exact | Conv exact | Average exact | Delta from paper average | Target cost / 1K successful calls | Complete whole-run cost including judge | p95 service latency, macro | p95 TTFT, macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt-5-mini-paper-stream-medium-20260711` | `medium`; 5,429,450 observed reasoning tokens | 78.178% | 69.260% | **73.719%** | +0.869 pp | $2.0926 | $36.6362 | 17.234 s | 10.909 s |
| `gpt-5.4-paper-stream-none-20260711` | `none`; zero reasoning tokens with complete coverage | 76.660% | 77.989% | **77.324%** | +0.124 pp | $5.0850 | $61.8329 | 6.983 s | 1.100 s |

Task-specific p95 client service latency was 10.582 seconds QA and 23.886
seconds conversational for GPT-5-mini, versus 3.401 seconds QA and 10.564
seconds conversational for GPT-5.4. The macro columns above are arithmetic
means of the two task-specific percentiles; they are comparison summaries, not
pooled request percentiles.

Each final run has 9,140 successful logical target calls and 4,570 successful
judge calls. GPT-5-mini has 9,143 billed target attempts because three
superseded failed attempts are retained in the attempt audit and cost total.

Sources of truth:

- [`../../results/model-comparison/frontier.csv`](../../results/model-comparison/frontier.csv)
- [`../../results/model-comparison/table2.csv`](../../results/model-comparison/table2.csv)
- [`../../results/model-comparison/latency_summary.csv`](../../results/model-comparison/latency_summary.csv)
- [`../../results/model-comparison/usage_and_cost.csv`](../../results/model-comparison/usage_and_cost.csv)
- the two run manifests and summaries under `results/<run-id>/`

# Latency vocabulary

Keep these quantities separate:

| Metric | Meaning |
| --- | --- |
| `service_latency_ms` | All provider request time plus retry backoff for one logical call; excludes local semaphore queueing. |
| `request_wall_ms` | Terminal provider request from dispatch through stream EOF. |
| `request_wall_total_ms` | Cumulative provider-request wall time across all attempts. |
| `ttft_ms` | Terminal request dispatch to the first non-empty visible text delta. Available only for instrumented streaming calls. |
| `time_after_first_token_ms` | First visible text delta through stream EOF. |
| `server_processing_ms` | Provider-header duration. It is provider-defined and is not pure model-compute time. |
| `queue_wait_ms` | Local time waiting for the concurrency semaphore. |
| `retry_backoff_ms` | Actual client retry sleep. |
| `total_duration_ms` | Logical task residence time, including local queueing, provider calls and backoff. |

The primary frontier latency is p95 client `service_latency_ms`. Compare it
only across runs with compatible streaming, concurrency and service-tier
profiles. The older `gpt-5-mini-paper-reproduction` run is non-streaming and
has only a legacy provider-processing proxy; it cannot supply true TTFT or
client request-wall latency.

# Frontier figures

The comparison report emits:

- [`../../results/model-comparison/accuracy-vs-cost.svg`](../../results/model-comparison/accuracy-vs-cost.svg)
- [`../../results/model-comparison/accuracy-vs-latency.svg`](../../results/model-comparison/accuracy-vs-latency.svg)

Gray circles are measured static-benchmark runs. The green
`Our trained model?` point is deliberately aspirational: it is not a model
result and does not appear in `frontier.csv`.

Future interactive frontiers must be produced from a separate interactive
policy runner. They should use accuracy and under-triage from reviewed cards,
cost per completed consultation, per-action TTFT/service latency, and full
consultation latency. Static one-shot latency must not be relabelled as
multi-turn consultation latency.

# Running or extending evaluation

Inspect configured profiles without making provider calls:

```bash
uv run python -m acuitybench models
uv run python -m acuitybench runs
```

For the accepted static-student path, inspect the contract without provider
calls:

```bash
uv run python -m acuitybench static-plan
uv run python -m acuitybench static-data-validate --input <separate-training.jsonl>
```

`static-evaluate` is the paid wrapper that persists the
`static-student-evaluation/v1` contract. It defaults to paired QA plus one-shot
conversation so the result can enter the paper-style average and both charts:

```bash
uv run python -m acuitybench static-evaluate \
  --model <student-model-id> \
  --run-id <new-descriptive-run-id>
```

Use `--qa-only` for the primary classification check without conversational
generation or GPT-4.1 judge calls. A compatible self-hosted student is declared
in `configs/models.yaml` with `provider: openai_compatible`, the environment
variable holding its base URL, and a stable deployment/checkpoint identifier.
Its price fields must contain a measured or documented amortised serving cost,
not an assumed zero.

A full paired-style target run is expensive and requires explicit API-spend
authorisation:

```bash
uv run python -m acuitybench evaluate \
  --model gpt-5-mini \
  --samples 5 \
  --run-id <new-descriptive-run-id> \
  --concurrency 20 \
  --judge-concurrency 20
```

Use a new run ID whenever transport or intended provenance changes. Streaming
is recorded per invocation but is not part of the output-cache identity, so
reusing an old run ID can silently reuse cached generations.

To add another OpenAI model, copy a target entry in
[`../../configs/models.yaml`](../../configs/models.yaml) and set its API model,
reasoning contract, token parameter, service tier and prices. Configuration is
fingerprinted, preventing incompatible response-cache reuse. To support a new
provider, implement the provider protocol in `acuitybench/providers/` and add
its registry entry in `acuitybench.providers.get_provider()`.

After compatible reports exist, regenerate the comparison and charts with:

```bash
uv run python -m acuitybench compare \
  --run-ids <run-a> <run-b> \
  --output results/model-comparison
```
