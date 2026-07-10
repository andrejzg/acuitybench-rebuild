# AcuityBench rebuild

This project reconstructs the 914-case AcuityBench benchmark from pinned public
source snapshots and the authors' released anonymised physician annotations.
Every download is checked against a SHA-256 digest before it is used.

This private repository includes the pinned source snapshots, reconstructed
benchmark and labels, and completed evaluation artifacts so a checkout is
self-contained. See [Data, provenance, and licensing](#data-provenance-and-licensing)
before changing repository visibility or redistributing files.

Large data and result files use Git LFS. After cloning, install Git LFS and
materialize the tracked objects:

```bash
git lfs install
git lfs pull
```

## Build

Python 3.10 or newer is required. With `uv`:

```bash
uv run --python 3.13 python -m acuitybench build
```

Or install the package and use its command:

```bash
python -m pip install -e .
acuitybench build
```

The first build downloads roughly 45 MB into `data/cache/`. Later builds reuse
the verified cache. Use `--refresh` to fetch all source files again.

Because this repository includes the verified cache, an offline rebuild works:

```bash
uv run python -m acuitybench build --offline
```

## Outputs

- `data/processed/acuitybench.csv`: normalized benchmark in the authors'
  released schema.
- `data/processed/acuitybench_transformed.csv`: normalized rows plus primary or
  ambiguous split and both evaluation prompt formats.
- `data/processed/acuitybench.parquet`: convenient typed version with a stable
  `case_id` column.
- `data/processed/build_report.json`: source digests and validation counts.

Validate an existing build with:

```bash
python -m acuitybench validate
```

Run unit tests with:

```bash
uv run --extra dev pytest
```

## Run models

Model evaluation is configuration-driven and resumable. Put provider secrets in
`.env` (never in `configs/models.yaml`), then inspect the available profiles:

```bash
uv run python -m acuitybench models
```

A cheap end-to-end smoke run exercises both prompt formats, the conversational
rubric judge, token accounting, and report generation:

```bash
uv run python -m acuitybench evaluate \
  --model gpt-5-mini \
  --samples 1 \
  --limit 2 \
  --run-id smoke-gpt-5-mini
```

The paper-compatible full run is:

```bash
uv run python -m acuitybench evaluate \
  --model gpt-5-mini \
  --samples 5 \
  --run-id gpt-5-mini-paper-reproduction \
  --concurrency 100 \
  --judge-concurrency 100
```

This makes 9,140 target-model calls (914 cases × two formats × five samples)
and 4,570 GPT-4.1 rubric-judge calls. Every result is committed immediately to
`results/evaluations.sqlite3`; rerunning the identical command skips completed
work. Use `acuitybench infer`, `acuitybench judge`, and `acuitybench report` to
run those stages separately, `acuitybench runs` to inspect cached runs, or
`acuitybench compare --run-ids <run-a> <run-b>` to build a cross-model table.

Reports are written to `results/<run-id>/`:

- `tables/table2.csv` and `.md`: the paper-style main row.
- `tables/metrics_long.csv`: overall, per-dataset, and per-acuity metrics.
- `tables/boundary_metrics.csv`: results on the 170 boundary-label cases.
- `tables/distributional_metrics.csv`: physician-panel JSD, ordinal
  Wasserstein, consensus leave-one-out, and reported custom alpha metrics for
  the 450 panel-consensus and 217 ambiguous cases.
- `tables/confusion_*.csv`: QA and conversational confusion matrices.
- `tables/usage_and_cost.csv`: token usage and price-based cost estimate.
- `exports/raw_samples.*`, `judged_samples.*`, and `case_predictions.*`:
  auditable sample- and case-level results.
- `run_manifest.json`: exact configuration, data digest, returned model
  snapshots, completeness, and aggregation contract.
- `SUMMARY.md`: compact main results, cost, paper comparison, and ambiguous-case
  physician-panel summary.

The main table follows the released analysis: only the 527 primary cases with
clear A/B/C/D gold labels are scored. It takes the mode of valid labels across
five samples and resolves ties toward the more severe label. Boundary and
ambiguous cases remain in the raw exports for separate analyses.

To add another OpenAI model, copy an entry in `configs/models.yaml` and change
its ID and API settings. A configuration change creates a different run
fingerprint, preventing stale cached responses from being reused. A new model
provider needs a provider adapter in `acuitybench/providers/` plus one registry
entry in `get_provider()`.

Model aliases are not immutable. The runner records both the requested alias
and the exact model string returned by the API; a new run should be described
as a fresh replication rather than assumed byte-for-byte identical to the
authors' April 2026 run.

## Data, provenance, and licensing

The repository tracks all data used for the current reconstruction and run:

- `data/cache/sources/`: byte-for-byte upstream snapshots.
- `data/processed/`: reconstructed benchmark, transformed prompts, Parquet
  output, and the build report.
- `results/evaluations.sqlite3`: resumable sample-level generation and judge
  records, including usage and latency metadata.
- `results/gpt-5-mini-paper-reproduction/`: exported GPT-5-mini results and
  paper-style tables.

Provenance is retained at three levels:

1. `sources.lock.json` records every upstream URL, immutable revision, SHA-256,
   byte count, homepage, and known licence.
2. `data/processed/build_report.json` records the source files actually used,
   output hashes, annotation audit, and validation counts.
3. Each evaluation `run_manifest.json` records the benchmark hash, complete
   model configuration, returned model snapshots, call completeness, token
   usage, pricing assumptions, and scoring contract.

`ARTIFACTS.sha256` provides a repository-wide integrity inventory for committed
data and result files. No API credentials or `.env` contents are tracked.

Some inputs are CC BY-NC and the AcuityBench annotation release still lacks a
final upstream licence. Keep the repository private and treat the data as
non-commercial research material unless those terms change; see [NOTICE.md](NOTICE.md).

## What is reproduced

The expected flow is 998 source cases to 914 benchmark cases: 247 direct
mappings, 450 physician-consensus cases, and 217 physician-confirmed ambiguous
cases. The validation step checks every released reference case ID as well as
all published source, split, method, and label distributions.

The build commands do not call a model API. Only the explicit `infer`, `judge`,
or `evaluate` commands spend provider credits.

See [NOTICE.md](NOTICE.md) before publishing or using the generated artifacts
commercially. [AUDIT.md](AUDIT.md) records inconsistencies found between the
paper, released annotations, and upstream regeneration utilities.
