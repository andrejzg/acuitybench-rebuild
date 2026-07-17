# AcuityBench agent instructions

This repository is medical-AI research infrastructure. It is not a clinical
system and must not be described or deployed as one.

## Required reading order

Before making a material change:

1. Read [`HANDOVER.md`](HANDOVER.md).
2. Read the OKF knowledge index at [`docs/knowledge/index.md`](docs/knowledge/index.md).
3. Read [`docs/knowledge/current-state.md`](docs/knowledge/current-state.md) and
   [`docs/knowledge/next-steps.md`](docs/knowledge/next-steps.md).
4. Open only the linked concept documents relevant to the task.
5. Verify volatile claims against code, manifests, and `git status` before
   acting. The knowledge bundle is curated context, not a substitute for the
   executable source of truth.

## Non-negotiable constraints

- Treat all benchmark and seed A/B/C/D reference fields as released
  physician/source-derived labels, not objective clinical truth.
- Never train on any reconstructed AcuityBench benchmark case, label, prompt,
  paraphrase, or benchmark-derived teacher trace. This includes
  `data/interactive/seed_v1/`; every card there records
  `training_allowed: false`. AcuityBench is held-out evaluation only.
- Follow the accepted static-first sequence in
  [`docs/knowledge/static-first.md`](docs/knowledge/static-first.md): first
  train and evaluate a single-shot A/B/C/D student on separately sourced
  cases, then begin interactive `ASK`/`DISPOSE`/`HANDOFF` work only after the
  documented progression gate is met.
- Do not claim that the interactive seed is clinically validated. It has zero
  clinician-content-reviewed cards and mechanically routed facts.
- Preserve source identifiers, revisions, SHA-256 digests, manifests, and
  immutable versioning. Create a new version instead of silently rewriting a
  frozen reviewed artifact.
- The owner explicitly chose to keep this repository public on 2026-07-17.
  Do not change its visibility without a new explicit instruction. Public
  visibility does **not** grant a licence to third-party data: review
  `NOTICE.md` and `sources.lock.json` before adding, redistributing, or making
  commercial-use claims about source-derived artifacts.
- Never print, commit, or copy `.env` contents or API keys. Provider credentials
  are read from environment variables such as `OPENAI_API_KEY`.
- Do not conflate TTFT, provider processing time, service latency, queue wait,
  or total task duration. Their definitions are documented in the knowledge
  bundle and implemented separately.
- Preserve unrelated user changes. In particular, materialized Git LFS result
  files can appear modified when Git LFS is unavailable; do not stage or
  rewrite them unless the task explicitly requires it.

## Source-of-truth order

When documents disagree, use this precedence and then repair the stale
documentation:

1. Versioned schemas and executable validation code.
2. Versioned configuration and source locks.
3. Generated manifests and checksum inventories.
4. Committed result tables and run manifests.
5. The OKF knowledge bundle and narrative documentation.

## Minimum verification

For ordinary code or documentation changes, run:

```bash
uv run --extra dev pytest
uv run python -m acuitybench validate
uv run python -m acuitybench interactive-validate
uv run python -m acuitybench static-plan
```

For data or result changes, also verify:

```bash
shasum -a 256 -c ARTIFACTS.sha256
```

Full model evaluations incur API cost. Do not launch `infer`, `judge`, or
`evaluate` without explicit spend authorisation, and never launch one merely
as a generic verification step. Smoke runs and full runs must use new,
descriptive run IDs when the transport or configuration changes.

## Documentation maintenance

When a material decision, result, data contract, cost assumption, or next step
changes:

- update the relevant concept under `docs/knowledge/`;
- update `docs/knowledge/current-state.md` if the handover snapshot changed;
- add a newest-first entry to `docs/knowledge/log.md`; and
- keep `README.md` concise by linking to the knowledge bundle instead of
  duplicating its detail.
