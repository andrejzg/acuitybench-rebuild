# Data inventory and provenance

This public repository intentionally includes the complete data needed to
inspect and reproduce the current benchmark. Public availability does not
grant additional rights to third-party material.

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

## `interactive/`

`interactive/seed_v1/` contains the deterministic, evaluation-only 100-case
interactive seed and its manifest. The set is balanced at 25 clear A/B/C/D
reference labels. Its manifest records exact source/config/schema digests and
marks all transformed case content as requiring clinician review. See
`interactive/README.md`; do not use these held-out cases for training.

## `static/`

`static/synthetic_pilot_v0/` contains a deterministic, zero-call scaffold for
the first 20-case fictional static-data experiment. It records only generation
slots and provenance hashes at this stage; no vignette, teacher label or
training-ready example exists. See `static/README.md` and the pilot README.

## Integrity and terms

`ARTIFACTS.sha256` inventories all committed data and result artifacts, while
the interactive manifest additionally locks the exact inputs and schema files
used to derive that seed.
Licensing is mixed and the physician-annotation licence remains pending
upstream. The repository is intentionally public; consult `NOTICE.md` and each
upstream licence before reuse, redistribution, or commercial use.
