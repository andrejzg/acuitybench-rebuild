# Interactive evaluation data

`seed_v1/` is a deterministic, **evaluation-only** interactive-acuity seed.
It contains 100 cases, balanced at 25 each for reference acuity A, B, C, and
D. It must not be added to a training corpus.

## Files

- `seed_v1/case_cards.jsonl`: one strict
  `interactive-case-card/v1` object per line.
- `seed_v1/manifest.json`: counts, artifact digests, source mappings,
  limitations, and the exact build/config provenance.

The source rows remain in `data/cache/sources/` and the reconstructed
benchmark remains in `data/processed/acuitybench.parquet`. `source_id`,
`benchmark_case_id`, the source-text SHA-256, and `sources.lock.json` preserve
the trace from each case card to the pinned upstream snapshot.

## What the gold label means

- 87 cases inherit the released median of a five-physician panel.
- 13 cases inherit HealthBench's physician-agreed emergent category.
- All 100 use a clear primary A/B/C/D reference label; no boundary or
  ambiguous label is included.

Those are reference acuity labels. The newly split opening utterances,
queryable facts, and lexical red-flag markers are automatic source-grounded
transformations and have **not** been clinician reviewed. Every card records
that distinction in `review_status`.

## Construction

The seed uses 40 HealthBench and 60 PMR-Reddit cases. HealthBench assistant
messages are removed; only user-authored messages enter the simulator. Source
text is routed to a fixed 33-question catalog without a model call. A whole
source fragment can currently be reused under multiple question IDs, so this
v1 artifact is suitable for pipeline and review work, not yet for defensible
question-efficiency or consultation-latency claims. When the
source does not state an answer, the patient always replies `I'm not sure.`
Repeated questions return byte-identical answers.

Selection is fixed by `configs/interactive/seed_set.v1.yaml`, including source
quotas and heuristic exclusions for URLs, email addresses, Reddit handles,
and instruction-like role-play prompts. The manifest includes hashes of the
config, question catalog, benchmark, source lock, and emitted JSONL.

Rebuild and validate with:

```bash
python -m acuitybench interactive-build
python -m acuitybench interactive-validate
```

See `docs/interactive-triage-v1.md` for the action protocol, simulator,
evaluation metrics, limitations, costing, and the next review gate.

## Use and redistribution

This is research plumbing, not a medical device or clinical decision aid.
PMR-Reddit is pinned as CC-BY-NC-4.0, and the upstream physician-label license
is still marked pending in `sources.lock.json`. The repository is intentionally
public, but that does not grant additional rights; verify source terms and
privacy obligations before reuse, redistribution, or commercial use.
