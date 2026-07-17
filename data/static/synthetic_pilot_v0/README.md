# Fictional static pilot v0

This directory begins as a free, deterministic scaffold for 20 entirely
fictional cases: five presentation groups crossed with A/B/C/D. It is not a
clinical dataset and contains no real patient/source case.

Committed scaffold artifacts:

- `generation_requests.jsonl`: 20 non-clinical generation slots containing
  only case IDs, fictional seeds, presentation group, intended label and split;
- `manifest.json`: config, prompt, benchmark-screening and artifact hashes,
  with zero paid calls and `training_ready: false`.

Artifacts created only after an explicitly authorised paid run:

- `generated_raw.jsonl`: append-only provider generation attempts;
- `labels_raw.jsonl`: append-only blinded label attempts;
- `examples.jsonl`: candidates passing the machine gates, each still carrying
  `training_allowed: false` under the dedicated candidate schema;
- `rejected.jsonl`: every rejected or incomplete case with reasons; and
- `contamination_report.json`: exact/fuzzy lexical comparisons against all 914
  held-out AcuityBench cases and within the generated pool.

Machine acceptance is not training approval. All 20 candidates require manual
review, and semantic/embedding similarity remains an explicit missing gate
before scaling. The repository never sends AcuityBench text to the generator.

Free commands:

```bash
uv run python -m acuitybench synthetic-plan
uv run python -m acuitybench synthetic-init
uv run python -m acuitybench synthetic-validate --allow-incomplete
```

`synthetic-generate` and `synthetic-label` can spend provider credits and each
requires both `--confirm-spend` and `--terms-reviewed`.
