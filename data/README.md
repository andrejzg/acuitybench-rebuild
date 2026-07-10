# Data inventory and provenance

This private repository intentionally includes the complete data needed to
inspect and reproduce the current benchmark.

## `cache/sources/`

These are the immutable upstream snapshots used by the builder. Every file is
listed in the repository-root `sources.lock.json` with its original URL,
revision, SHA-256 digest, byte count, homepage, and known licence. Running
`python -m acuitybench fetch --offline` verifies the committed snapshots without
network access.

The directory includes the authors' released anonymised physician-label CSV.
It contains label and anonymised-rater fields, not physician identities or the
private Qualtrics exports used during recruitment.

## `processed/`

- `acuitybench.csv`: normalized benchmark in the authors' released schema.
- `acuitybench_transformed.csv`: benchmark plus split and evaluation prompts.
- `acuitybench.parquet`: typed copy with stable case IDs.
- `build_report.json`: source and output digests, build version, annotation
  audit, and validation counts.

The processed benchmark has 914 rows: 697 primary and 217 ambiguous. Of these,
667 cases have five released anonymised physician-panel labels.

## Integrity and terms

`ARTIFACTS.sha256` inventories all committed data and result artifacts.
Licensing is mixed and the physician-annotation licence remains pending
upstream. Keep this repository private and consult `NOTICE.md` before any
redistribution or commercial use.
