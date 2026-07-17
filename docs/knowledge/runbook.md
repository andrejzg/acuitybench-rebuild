---
type: Runbook
title: AcuityBench Setup, Validation and Execution Runbook
description: Safe operational commands for Git LFS, local setup, deterministic validation, interactive replay and authorised model runs.
tags: [runbook, setup, validation, git-lfs, evaluation]
timestamp: 2026-07-17T00:00:00+01:00
status: active
last_verified: 2026-07-17
---

# Safety and repository terms

The GitHub repository is intentionally **public** at this handover. Public
visibility does not grant rights to third-party material. PMR data is
CC BY-NC 4.0, the AcuityBench physician-annotation licence remains pending in
the pinned upstream README, and other sources have mixed or qualified terms.
Read [`../../NOTICE.md`](../../NOTICE.md) and
[`../../sources.lock.json`](../../sources.lock.json) before redistribution,
commercial use or adding new source-derived data.

This is medical-AI research infrastructure, not a clinical service. Never
describe reference-label agreement as clinical validation.

# Fresh checkout

The package requires Python 3.10 or newer. Install `uv` and Git LFS, then:

```bash
git lfs install
git lfs pull
uv sync --extra dev
uv run python -m acuitybench --version
uv run python -m acuitybench models
```

Large source snapshots, processed tables, the SQLite database and sample-level
exports use Git LFS. Do not interpret pointer files as real datasets, and do
not stage materialized LFS changes until their status and intended scope have
been checked.

# Secrets

Provider profiles read secrets from environment variables. The OpenAI profiles
expect `OPENAI_API_KEY`; a local `.env` is loaded by the runner and ignored by
Git.

```bash
cp .env.example .env
# Edit .env locally; never paste its contents into logs, issues, prompts or commits.
```

Never print or inspect the value during routine validation. Confirm only that
the required variable is present when an authorised provider run is about to
start.

# Read-only orientation

Start every work session with:

```bash
git status --short
git branch --show-current
git log -5 --oneline --decorate
git lfs status
uv run python -m acuitybench runs
```

The handover snapshot is dated. Trust the current worktree, manifests and
versioned configs over prose when they disagree.

# Deterministic benchmark checks

Validate the already materialized benchmark without downloading or rebuilding:

```bash
uv run python -m acuitybench validate
```

Verify the committed source cache explicitly:

```bash
uv run python -m acuitybench fetch --offline
```

Reconstructing the outputs is deterministic but writes tracked files. Use it
only when regeneration is intended:

```bash
uv run python -m acuitybench build --offline
```

Run the complete local test suite:

```bash
uv run --extra dev pytest
```

The dated test count in `HANDOVER.md` is only a historical baseline, not a
substitute for the current run.

For data or result changes, verify the artifact inventory:

```bash
shasum -a 256 -c ARTIFACTS.sha256
```

Transient SQLite `-wal` and `-shm` sidecars are intentionally excluded from the
inventory because Git ignores them and a fresh checkout may not create them
until SQLite is opened. Do not add runtime sidecars to the durable checksum
contract.

# Interactive seed and simulator

Validate the existing seed and its deterministic rebuild identity:

```bash
uv run python -m acuitybench interactive-validate
```

Rebuild the seed only when regeneration is intended:

```bash
uv run python -m acuitybench interactive-build
```

Replay the checked-in action example:

```bash
uv run python -m acuitybench interactive-simulate \
  --case-id acuity-interactive-v1-038 \
  --actions examples/interactive_actions.json
```

Regenerate the planning-only cost report without making provider calls:

```bash
uv run python -m acuitybench interactive-cost
```

Never train on `data/interactive/seed_v1/`. It is held-out,
`training_allowed: false`, and its transformed facts have not yet received
clinician review.

# Static-first student workflow

Inspect the accepted plan and benchmark arithmetic without making provider
calls:

```bash
uv run python -m acuitybench static-plan
```

Validate a separately sourced JSONL training pool before teacher generation or
training:

```bash
uv run python -m acuitybench static-data-validate \
  --input <separate-training.jsonl>
```

The validator checks the strict schema, duplicate example IDs, family-group
split crossings and exact `(source_dataset, source_id)` reuse from
AcuityBench. Also perform semantic/near-duplicate screening; the exact-identity
guard cannot detect paraphrases by itself.

`static-evaluate` makes paid model calls. Its default is the paired QA and
one-shot conversational contract (9,140 target and 4,570 judge calls at five
samples). `--qa-only` makes 4,570 target calls and no judge calls:

```bash
uv run python -m acuitybench static-evaluate \
  --model <student-model-id> \
  --run-id <new-descriptive-run-id> \
  --qa-only
```

All ordinary spend-authorisation, smoke-run and fresh-run-ID rules still
apply. No command in this repository trains a model yet.

# Fictional static-pilot workflow

Inspect and initialize the deterministic 20-case scaffold without provider
calls:

```bash
uv run python -m acuitybench synthetic-plan
uv run python -m acuitybench synthetic-init
uv run python -m acuitybench synthetic-validate --allow-incomplete
```

The initialized manifest must report zero paid calls and
`training_ready: false`. Generation and blinded double-labelling are paid and
must not be used as routine verification:

```bash
uv run python -m acuitybench synthetic-generate \
  --model <generator-profile> --confirm-spend --terms-reviewed
uv run python -m acuitybench synthetic-label \
  --model <labeler-profile> --confirm-spend --terms-reviewed
```

The authorised 200-case v1 contract pins its models in configuration, so no
model flags are required:

```bash
uv run python -m acuitybench synthetic-generate \
  --config configs/static/synthetic_pilot.v1.yaml \
  --concurrency 1 --confirm-spend --terms-reviewed
uv run python -m acuitybench synthetic-label \
  --config configs/static/synthetic_pilot.v1.yaml \
  --confirm-spend --terms-reviewed
```

Choose the exact model profiles, review provider terms, calculate a cost
ceiling and obtain explicit spend authorization first. After completion, run
`synthetic-validate` without `--allow-incomplete` and manually review all 20
candidates. Machine acceptance does not authorize training use.

# Model evaluation

The following commands can make paid external API calls:

- `infer`
- `evaluate`
- `judge`
- `static-evaluate`
- `synthetic-generate`
- `synthetic-label`

Do not launch even a smoke evaluation without explicit API-spend
authorisation. Before a paid run, record the model, sample count, selected
cases, task formats, concurrency, judge, expected call count and a new run ID.
A complete five-sample run makes 9,140 target calls and 4,570 GPT-4.1 judge
calls.

An authorised small smoke run looks like:

```bash
uv run python -m acuitybench evaluate \
  --model gpt-5-mini \
  --samples 1 \
  --limit 2 \
  --run-id <new-descriptive-smoke-id>
```

An authorised paper-contract run uses five samples:

```bash
uv run python -m acuitybench evaluate \
  --model <configured-model-id> \
  --samples 5 \
  --run-id <new-descriptive-run-id> \
  --concurrency 20 \
  --judge-concurrency 20
```

Every target result is persisted immediately, and rerunning the same compatible
run resumes missing work. However, streaming transport is not part of the
output-cache identity. Use a new run ID whenever transport or experimental
provenance changes; otherwise old responses can be reused unexpectedly.

Local reporting does not call a provider:

```bash
uv run python -m acuitybench report --run-id <run-id>
uv run python -m acuitybench compare \
  --run-ids <run-a> <run-b> \
  --output results/model-comparison
```

Only compare latency frontiers when streaming, concurrency and service-tier
profiles are compatible. A non-streaming legacy processing-header proxy is not
equivalent to client service latency or TTFT.

# Common traps

- **Public repository does not mean unrestricted data.** Apply every upstream
  licence and privacy constraint.
- **Reference labels are not ground truth.** Preserve “physician/reference
  agreement” wording.
- **The interactive seed is not training data.** Use a separate, grouped
  training pool.
- **The core benchmark is not training data either.** Do not train from any of
  its 914 cases, labels, paraphrases or derived teacher outputs.
- **Static and interactive latency differ.** Current measured charts cover
  one-shot benchmark calls, not multi-turn consultations.
- **The green frontier point is aspirational.** It is not a measured model.
- **Do not mix clocks.** TTFT, provider processing, service latency, queue wait
  and total logical duration are distinct.
- **Do not silently rewrite versioned artifacts.** Create a new seed,
  assumptions or schema version when semantics change.
- **Do not trust stale version prose.** For example, the processed build report
  records its historical builder version even when the package has since
  advanced.
- **Do not overwrite unrelated changes.** Review `git diff`, LFS status and the
  exact staging set before any commit.
