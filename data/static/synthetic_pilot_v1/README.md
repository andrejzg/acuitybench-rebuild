# Fictional static pilot v1 (200 cases)

This directory contains the completed, explicitly authorised standard-API run
of the immutable 200-case fictional static contract. It contains no real
patient/source record and was not conditioned on AcuityBench text.

## Fixed models and workload

- Generator: Claude Fable 5, adaptive thinking `medium`, Anthropic Messages
  API with returned service tier `standard`: 200 successful calls.
- Blinded label sample 0: GPT-5.6 Terra, reasoning `low`: 200 successful calls.
- Blinded label sample 1: GPT-5.4, reasoning `none`: 200 successful calls.

The 200 slots are balanced at 50 per A/B/C/D target and 40 per presentation
group, with 160 intended training candidates and 40 development candidates.

## Machine-screen outcome

- 131 candidates passed the strict current machine gates.
- 69 cases were rejected: 68 included a teacher ambiguity flag, seven had at
  least one teacher disagree with generator intent, and five had teacher-label
  disagreement (reasons can overlap).
- Zero cases crossed the configured exact/fuzzy lexical-contamination
  thresholds against the 914 held-out AcuityBench cases.

This is **not** clinical acceptance. Every candidate has
`training_allowed: false`; semantic similarity screening and manual clinician
review of the complete 200-case pool are still required. The generator's
requested label is an experimental construction target, not a physician gold
label.

## Successful-call usage, cost and latency

| Model | Successful calls | Input tokens | Output tokens | Reasoning tokens | Estimated cost | E2E p50 | E2E p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude Fable 5 | 200 | 260,240 | 152,773 | 5,879 | $10.2411 | 14.040 s | 17.975 s |
| GPT-5.6 Terra | 200 | 93,958 | 21,339 | 991 | $0.5550 | 1.797 s | 4.005 s |
| GPT-5.4 | 200 | 94,758 | 24,382 | 0 | $0.6026 | 2.469 s | 3.841 s |
| **Total** | **600** | **448,956** | **198,494** | **6,870** | **$11.3987** | — | — |

Latency is client-observed end-to-end duration per successful non-streaming
request. TTFT is unavailable and is recorded as such; it is not inferred from
total duration. Cost uses recorded successful-call token usage and the frozen
model-registry prices. Provider-rejected attempts reported no usage and are
not included in the cost total.

## Attempt history

The append-only raw logs also retain two corrected setup failures:

- 386 Fable attempts received HTTP 429 while probing an overly high
  connection count; resumption at concurrency one completed all cases.
- the first 400 OpenAI label attempts were rejected before generation because
  the provider's strict-schema dialect required `type` alongside `const`;
  a provider-only schema normalization fixed this, after which all 400 labels
  completed.

These failed attempts remain in the LFS-backed raw traces for auditability and
are excluded from successful-call latency summaries.

## Files

- `generation_requests.jsonl`: deterministic balanced slots and intended split;
- `generated_raw.jsonl`: append-only Fable attempts, responses and metadata;
- `labels_raw.jsonl`: append-only blinded Terra/GPT-5.4 attempts;
- `examples.jsonl`: 131 machine-screened, training-blocked candidates;
- `rejected.jsonl`: 69 cases with complete rejection reasons and teacher outputs;
- `contamination_report.json`: held-out lexical and internal-duplicate screen;
- `manifest.json`: frozen input hashes, artifact hashes, counts, usage, cost and
  latency aggregates.

Validate the complete artifact set with:

```bash
uv run python -m acuitybench synthetic-validate \
  --config configs/static/synthetic_pilot.v1.yaml
```
