# Evaluation artifacts

This directory contains the complete local evaluation state and exports.

- `evaluations.sqlite3`: durable sample-level target generations and GPT-4.1
  judgments, including prompt/config hashes, raw responses, parsed labels,
  token usage, request metadata, retries, and latency fields.
  Schema v2 stores execution-level concurrency plus one row per provider
  attempt, separating TTFT, request wall time, retry backoff, local queueing,
  total duration, and provider-reported processing time. Cost reports sum all
  tracked attempts and flag missing usage instead of silently pricing it at
  zero.
- `gpt-5-mini-paper-reproduction/`: the full 914-case, five-sample GPT-5-mini
  legacy non-streaming run.
- `gpt-5-mini-paper-stream-medium-20260711/` and
  `gpt-5.4-paper-stream-none-20260711/`: complete 914-case, five-sample paired
  runs with streamed TTFT and client service-latency instrumentation, exact
  request/response provenance, attempt-aware cost, and self-contained
  manifests.
- `smoke-gpt-5-mini/` and the two `smoke-*-20260711/` directories: end-to-end
  validation runs.
- `model-comparison/` and `smoke-comparison/`: combined table exports.
  Each comparison includes ready-to-view accuracy-vs-cost and
  accuracy-vs-latency SVGs generated from `frontier.csv`.
- `interactive-pilot-v1/`: deterministic planning estimate for the standalone
  100-case clinician-review gate and an illustrative 500-case Tinker training
  pilot. It is assumptions-based, not observed billing.

Each new full run has 9,140 successful target generations and 4,570 successful
GPT-4.1 judgments. GPT-5-mini returned `gpt-5-mini-2025-08-07`; GPT-5.4
returned `gpt-5.4-2026-03-05`; the judge returned `gpt-4.1-2025-04-14`.
Requested and returned target service tier is `default` with 100% coverage.
Both use streaming and concurrency 20, and every target request is capped at
4,096 completion tokens.

GPT-5-mini (`medium`) used 5,429,450 observed reasoning tokens and cost $36.64
including judging. GPT-5.4 (`none`) reported zero reasoning tokens with 100%
telemetry coverage and cost $61.83 including judging. Their average exact
accuracy is 73.719% and 77.324%; their p95 client service-latency macro-average
is 17.234s and 6.983s respectively. See `model-comparison/frontier.csv` for the
full cost, accuracy, latency, TTFT, and profile provenance.

The older `gpt-5-mini-paper-reproduction` run predates schema-v2 client
instrumentation. Its report exposes the retained provider-processing header
but correctly leaves TTFT and request-wall latency unmeasured.

These outputs may reproduce or derive from clinical text with mixed licensing.
They contain no API keys, but the repository should remain private; see
`NOTICE.md` and `sources.lock.json`.
