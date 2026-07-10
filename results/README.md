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
  run, including CSV/Parquet exports, main and appendix tables, usage/cost data,
  and a self-contained `run_manifest.json`.
- `smoke-gpt-5-mini/`: the two-case end-to-end validation run.
- `model-comparison/` and `smoke-comparison/`: combined table exports.

The full run has 9,140 successful target generations and 4,570 successful
GPT-4.1 judgments. Its manifest records the requested aliases, returned model
snapshots, benchmark SHA-256, scoring contract, and completeness checks.
Because that run predates schema-v2 client instrumentation, its report exposes
the retained provider-processing header but correctly leaves TTFT and request
wall latency unmeasured.

These outputs may reproduce or derive from clinical text with mixed licensing.
They contain no API keys, but the repository should remain private; see
`NOTICE.md` and `sources.lock.json`.
