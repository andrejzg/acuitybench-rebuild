# AcuityBench rebuild

This project reconstructs the 914-case benchmark introduced in
[AcuityBench: Evaluating Clinical Acuity Identification and Uncertainty Alignment](https://arxiv.org/abs/2605.11398)
([arXiv PDF](https://arxiv.org/pdf/2605.11398)) from pinned public source
snapshots and the authors' released anonymised physician annotations. Every
download is checked against a SHA-256 digest before it is used.

This public repository includes the pinned source snapshots, reconstructed
benchmark and labels, and completed evaluation artifacts so a checkout is
self-contained. Public visibility is an explicit project choice, not a grant
of rights to third-party material. See
[Data, provenance, and licensing](#data-provenance-and-licensing) before reuse,
redistribution, or commercial use.

For continuation by another human or AI, begin with
[`HANDOVER.md`](HANDOVER.md) and the agent-readable
[OKF knowledge bundle](docs/knowledge/index.md). Repository-specific operating
constraints are in [`AGENTS.md`](AGENTS.md).

## Goal

The rebuild is the foundation for training and evaluating a specialised,
low-latency clinical-acuity research model. A successful model should:

- retain or improve agreement with the physician/reference labels, including
  uncertainty-aware and under-triage evaluation;
- materially reduce p95 service latency, our primary optimisation target; and
- reduce inference cost without trading away accuracy.

On both frontier charts, success means moving a measured result toward the
top-left: higher accuracy with lower latency or cost. The green
`Our trained model?` point is the goal, not a claimed result; it will remain
explicitly aspirational until it can be replaced by a reproducible benchmark
run.

Large data and result files use Git LFS. After cloning, install Git LFS and
materialize the tracked objects:

```bash
git lfs install
git lfs pull
```

## Static-first student plan

The accepted first training target is deliberately simpler than a GP chat:
one complete case goes in and one A/B/C/D acuity label comes out. QA exact
agreement and under-triage are primary; the paper's one-shot conversational
response plus GPT-4.1 judge is a secondary paired comparison. True multi-turn
`ASK`/`DISPOSE`/`HANDOFF` work comes later, after a measured static student
earns progression.

The repository now has the versioned plan, strict separate-data schema, an
exact source-identity contamination guard, family-split checks, an evaluation
wrapper and OpenAI-compatible endpoint support. Semantic and paraphrase
deduplication remains a required external data-preparation step:

```bash
uv run python -m acuitybench static-plan
uv run python -m acuitybench static-data-validate --input <separate-training.jsonl>
# Paid; paired by default. Add --qa-only for classification without a judge.
uv run python -m acuitybench static-evaluate \
  --model <student-model-id> \
  --run-id <new-descriptive-run-id>
```

No training corpus, training loop or student checkpoint is included yet. The
914 reconstructed benchmark cases—and every derivative or paraphrase—remain
held-out evaluation only. The detailed contract and progression gate are in
the [static-first OKF concept](docs/knowledge/static-first.md).

## Later interactive triage pilot

The repository now includes an executable v1 contract for the proposed
multi-turn GP-style task:

- strict, versioned [case-card](schemas/interactive-case-card-v1.schema.json)
  and [action](schemas/interactive-action-v1.schema.json) schemas;
- a closed 33-question catalog with `ASK`, `DISPOSE`, and safe `HANDOFF`
  actions;
- a deterministic patient simulator and trajectory evaluator; and
- a provenance-linked [100-case seed](data/interactive/README.md), balanced at
  25 clear reference cases per A/B/C/D acuity class.

Of the 100 reference labels, 87 are released five-physician panel medians and
13 are direct HealthBench physician-agreed emergency categories. Those gold
labels remain the evaluation reference for this seed, but they are not a
claim of clinical ground truth, and the newly routed interactive content has
not been clinician reviewed. The cards are therefore explicitly
`evaluation_only`, require review, and cannot be used as training data.

```bash
python -m acuitybench interactive-build
python -m acuitybench interactive-validate
python -m acuitybench interactive-simulate \
  --case-id acuity-interactive-v1-038 \
  --actions examples/interactive_actions.json
python -m acuitybench interactive-cost
```

The deterministic build makes no API calls. Under the versioned planning
assumptions, clinical preparation of the 100 cases is estimated at **$6,300**;
the subsequent illustrative 500-case/2,000-teacher-trace training pilot is
**$36,062.29**, of which only **$62.29** is Tinker token compute and **$36,000**
is clinician review. These are assumptions, not observed billing. See the
[interactive v1 design and next review gate](docs/interactive-triage-v1.md)
and the committed [cost report](results/interactive-pilot-v1/cost_estimate.md).

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

The historical non-streaming reproduction committed under
`gpt-5-mini-paper-reproduction` used the paper's five-sample contract but
predates definitive client-side latency instrumentation. The paired,
instrumented paper-contract runs use fresh IDs so no legacy response can be
mistaken for a streamed call:

```bash
uv run python -m acuitybench evaluate \
  --model gpt-5-mini \
  --samples 5 \
  --run-id gpt-5-mini-paper-stream-medium-20260711 \
  --concurrency 20 \
  --judge-concurrency 20

uv run python -m acuitybench evaluate \
  --model gpt-5.4 \
  --samples 5 \
  --run-id gpt-5.4-paper-stream-none-20260711 \
  --concurrency 20 \
  --judge-concurrency 20
```

Each command makes 9,140 target-model calls (914 cases × two formats × five
samples) and 4,570 GPT-4.1 rubric-judge calls. Run them sequentially: identical
streaming, concurrency, and `default` service-tier settings make the latency
points comparable. Every result is persisted immediately to
`results/evaluations.sqlite3`; rerunning the identical command skips completed
work. Use `acuitybench infer`, `acuitybench judge`, and `acuitybench report` to
run those stages separately, `acuitybench runs` to inspect cached runs, or
`acuitybench compare --run-ids <run-a> <run-b>` to build a cross-model table.
The comparison output includes `frontier.csv`, with average exact accuracy,
target-model cost per 1,000 successful calls, task-specific latency, and
explicitly labelled macro-averages of QA and conversational p50/p95 service
latency, TTFT, and provider processing. It also writes two deterministic,
accessible SVGs:
`accuracy-vs-cost.svg` and `accuracy-vs-latency.svg`. A hollow diamond on the
latency chart is a legacy provider-processing proxy, never a true client
service-latency measurement. Gray circles are the latest measured model runs;
the green `Our trained model?` circle is deliberately aspirational and is not a
measured result or a row in `frontier.csv`.

The paper explicitly specifies temperature 1, five samples, and a maximum
4,096 completion tokens, but it does not report reasoning effort or a separate
reasoning-token budget. For reproducibility, the current profiles resolve that
omission to the documented provider defaults used by the paper-era aliases:
`medium` for GPT-5-mini and `none` for GPT-5.4. The 4,096 cap includes hidden
reasoning plus visible output and is now a hard retry cap. Manifests label the
efforts as inferred rather than paper-reported, retain observed reasoning-token
usage, and record the deliberate streaming/concurrency divergence needed for
the latency study. The resolution is pinned to the official
[reasoning guide](https://developers.openai.com/api/docs/guides/reasoning) for
GPT-5-mini and the [GPT-5.4 model page](https://developers.openai.com/api/docs/models/gpt-5.4),
with an access date in each run manifest.

The completed paired runs produced:

| Model | Reasoning | Avg exact | Paper delta | Target cost / 1K successful calls | p95 service latency | p95 TTFT |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5 mini | medium (5.429M observed tokens) | 73.719% | +0.869 pp | $2.09 | 17.234s | 10.909s |
| GPT-5.4 | none (0 observed tokens) | 77.324% | +0.124 pp | $5.09 | 6.983s | 1.100s |

Accuracy is the macro-average of QA and conversational exact agreement. Paper
delta means fresh run minus the [AcuityBench Table 2](https://arxiv.org/pdf/2605.11398)
value. The
latency values are macro-averages of the two formats' client-side p95s. Paper
deltas use Table 2's three-decimal values, so they inherit ±0.05 percentage
point rounding uncertainty. Including the GPT-4.1 judge, the complete runs cost
$36.64 and $61.83 respectively. GPT-5.4 is 2.43× the target-model cost per call,
but is 59.5% lower-latency at p95 and 3.61 percentage points more accurate in
this run.

![Average accuracy vs target-model cost](results/model-comparison/accuracy-vs-cost.svg)

![Average accuracy vs latency](results/model-comparison/accuracy-vs-latency.svg)

The historical paper-reproduction run above used the original non-streaming
transport. New calls stream by default so true time to first visible token
(TTFT) can be measured. Transport is recorded per invocation, but is not part
of the output-cache identity; use a new run ID when deliberately rerunning
cached samples under a different transport. Pass `--no-stream` for a provider
or model that cannot stream; TTFT will then be null.

Reports are written to `results/<run-id>/`:

- `tables/table2.csv` and `.md`: the paper-style main row.
- `tables/metrics_long.csv`: overall, per-dataset, and per-acuity metrics.
- `tables/boundary_metrics.csv`: results on the 170 boundary-label cases.
- `tables/distributional_metrics.csv`: physician-panel JSD, ordinal
  Wasserstein, consensus leave-one-out, and reported custom alpha metrics for
  the 450 panel-consensus and 217 ambiguous cases.
- `tables/confusion_*.csv`: QA and conversational confusion matrices.
- `tables/usage_and_cost.csv`: attempt-aware token usage, configured reasoning
  effort and completion caps, observed reasoning-token coverage, price-based
  cost, explicit token/cache-detail coverage, and labeled partial estimates.
- `tables/latency_summary.csv`: p50/p90/p95/p99 service latency, terminal
  request time, TTFT, stream tail, provider processing, queueing, and backoff,
  with measurement coverage and clock source kept explicit.
- `tables/execution_summary.csv`: per-invocation elapsed time, throughput,
  concurrency, retry rate, failures, cancellations, and unpersisted work.
- `exports/raw_samples.*`, `judged_samples.*`, and `case_predictions.*`:
  auditable sample- and case-level results.
- `exports/run_executions.csv` and `request_attempts.*`: invocation concurrency,
  runtime metadata, every API attempt, retries, response headers, and attempt
  usage for latency and billing audits.
- `run_manifest.json`: exact configuration, paper-vs-run inference contract,
  reasoning provenance, data digest, returned model/service tier, completeness,
  and aggregation contract.
- `SUMMARY.md`: compact main results, cost, paper comparison, and ambiguous-case
  physician-panel summary.

The main table follows the released analysis: only the 527 primary cases with
clear A/B/C/D gold labels are scored. It takes the mode of valid labels across
five samples and resolves ties toward the more severe label. Boundary and
ambiguous cases remain in the raw exports for separate analyses.

To add another OpenAI model, copy an entry in `configs/models.yaml` and change
its ID and API settings. A local or hosted student with the same API shape can
use `provider: openai_compatible` plus a `base_url_env` and stable `deployment`
description; endpoint values and credentials remain in `.env`. A configuration
change creates a different run fingerprint, preventing stale cached responses
from being reused. A different provider protocol needs an adapter in
`acuitybench/providers/` plus one registry entry in `get_provider()`.

Model aliases are not immutable. The runner records both the requested alias
and the exact model string returned by the API; a new run should be described
as a fresh replication rather than assumed byte-for-byte identical to the
authors' April 2026 run.

### Latency contract

Latency fields deliberately separate clocks that must not be conflated:

- `ttft_ms`: terminal request dispatch to the first non-empty visible text
  delta. It is only available for streaming calls.
- `request_wall_ms`: terminal attempt from dispatch through stream EOF.
- `request_wall_total_ms`: provider time summed across all attempts.
- `service_latency_ms`: all provider request time plus retry backoff, excluding
  the benchmark runner's local concurrency queue. This is the primary serving
  metric for latency comparisons.
- `queue_wait_ms`: cumulative local semaphore wait. This describes benchmark
  load, not intrinsic model latency.
- `total_duration_ms` (and compatibility alias `latency_ms`): full logical task
  residence time including queueing, requests, and backoff.
- `server_processing_ms`: the provider's `openai-processing-ms` response header
  when present. It is provider-defined and is not treated as TTFT, end-to-end
  latency, or pure GPU inference time.

Each inference or judge invocation records its configured concurrency and
streaming mode. Each request attempt records its own timing, outcome, retry
sleep, IDs, usage, and allowlisted server metadata. Legacy rows retain their
historical queue-inclusive duration and OpenAI processing header, but reporting
marks request-wall, queue, backoff, and TTFT as unavailable rather than
reconstructing them. Latency percentiles describe the latest successful parent
row for each logical request; the execution and attempt exports separately show
failed work, retries, cancellations, and whole-batch throughput.

## Data, provenance, and licensing

The repository tracks all data used for the current reconstruction and run:

- `data/cache/sources/`: byte-for-byte upstream snapshots.
- `data/processed/`: reconstructed benchmark, transformed prompts, Parquet
  output, and the build report.
- `data/interactive/seed_v1/`: evaluation-only interactive case cards and a
  manifest binding the source, configuration, schema, and artifact hashes.
- `results/evaluations.sqlite3`: resumable sample-level generation and judge
  records, including usage and latency metadata.
- `results/gpt-5-mini-paper-reproduction/`: exported GPT-5-mini results and
  paper-style tables.
- `results/gpt-5-mini-paper-stream-medium-20260711/` and
  `results/gpt-5.4-paper-stream-none-20260711/`: the complete paired,
  instrumented runs used by the frontier charts.
- `results/model-comparison/`: the combined frontier table and README graphs.
- `results/interactive-pilot-v1/`: versioned seed-review and illustrative
  training-pilot cost assumptions and reports.

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
final upstream licence. The repository is intentionally public, but downstream
users must still treat the data as non-commercial research material unless the
upstream terms change and must independently review redistribution rights; see
[NOTICE.md](NOTICE.md).

## What is reproduced

The expected flow is 998 source cases to 914 benchmark cases: 247 direct
mappings, 450 physician-consensus cases, and 217 physician-confirmed ambiguous
cases. The validation step checks every released reference case ID as well as
all published source, split, method, and label distributions.

The build commands do not call a model API. Only the explicit `infer`, `judge`,
`evaluate`, or `static-evaluate` commands can spend provider credits.

See [NOTICE.md](NOTICE.md) before publishing or using the generated artifacts
commercially. [AUDIT.md](AUDIT.md) records inconsistencies found between the
paper, released annotations, and upstream regeneration utilities.
