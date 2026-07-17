# AcuityBench AI handover

Snapshot date: **2026-07-17**.

Start with the [agent instructions](AGENTS.md), then use the
[Open Knowledge Format index](docs/knowledge/index.md) for progressive
disclosure. The knowledge bundle is deliberately split into focused Markdown
concepts so another AI can load only the context it needs.

## Mission

Reconstruct AcuityBench faithfully, measure frontier-model accuracy, cost and
latency, first develop a specialised low-latency **static** acuity classifier,
and only then progress to an interactive acuity policy without contaminating
held-out evaluation data.

Success means moving toward the top-left of both frontier plots:

- higher agreement with the released physician/reference labels;
- lower under-triage;
- lower p95 service latency and TTFT; and
- lower inference cost.

## What is complete

- The checksum-pinned 914-case AcuityBench reconstruction and provenance audit.
- A resumable multi-model evaluation pipeline with GPT-4.1 rubric judging,
  attempt-aware cost accounting, streaming TTFT, service-latency
  instrumentation, exports, manifests, and frontier plots.
- Full paired GPT-5-mini and GPT-5.4 paper-contract runs.
- A versioned static-student plan, strict separate-training-data schema,
  contamination checks, a paid evaluation wrapper, and an
  OpenAI-compatible serving path for local or hosted student endpoints.
- A versioned 20-case fictional static-pilot scaffold with deterministic
  requests, strict generation/label schemas, resumable double-blinded teacher
  labelling, leakage checks, fake-provider tests and a zero-call manifest.
- A separate completed 200-case fictional v1 run: Fable 5 generated all 200
  balanced static vignettes, Terra and GPT-5.4 supplied 400 blinded labels,
  and 131 candidates passed machine gates. All remain training-blocked.
- A strict interactive case-card/action contract, deterministic simulator,
  trajectory evaluator, and balanced 100-case evaluation seed.
- A versioned cost model for the 100-case clinician review gate and an
  illustrative 500-case Tinker training pilot.
- An OKF v0.1 knowledge bundle and root agent instructions for future handover.

## Immediate next move

Start with the completed [fictional-pilot evidence](docs/knowledge/synthetic-pilot.md).
Review all 200 v1 cases (including the 69 machine rejections), add the missing
semantic-similarity gate, and decide whether the 68 ambiguity-flag rejections
reflect real data defects or an overly broad flag policy. Do not promote the
131 machine-screened candidates until that review is recorded.

If that check succeeds, build a separately sourced, family-grouped pool of
roughly 500–1,000 cases, validate it with `static-data-validate`, generate
reviewed teacher targets, and train a small single-shot A/B/C/D student.
Evaluate QA agreement and latency first; then run the paired one-shot
conversational response plus GPT-4.1 judge for a paper-style comparison.

Do **not** train on any of the 914 AcuityBench cases or the 100-case
interactive seed. Clinician review of that seed remains necessary before a
future interactive benchmark, but it is no longer the immediate build gate.
Interactive `ASK`/`DISPOSE`/`HANDOFF` training starts only after the static
student demonstrates a worthwhile accuracy/latency trade-off and the
progression decision is recorded.

## Snapshot facts

| Item | State at handover |
| --- | --- |
| Repository | `andrejzg/acuitybench-rebuild` |
| Visibility | **Public by explicit owner decision (2026-07-17)** |
| Branch/remote baseline | `main` at `ea96b00`; current v1 run and adapter work are local until explicitly published |
| Package | `acuitybench-rebuild` 0.4.0 |
| Benchmark | 914 cases; 527 clear primary cases in the paper-style main score |
| Static-student capability | Plan/schema/validator/evaluation/serving adapter ready; no training pool or checkpoint yet |
| Fictional pilot | v0 remains a zero-call 20-case scaffold; v1 completed 200 generations + 400 labels, 131 machine passes, not training-ready |
| Interactive seed | 100 cases, 25 per A/B/C/D, evaluation-only |
| Seed label basis | 87 five-physician medians; 13 direct HealthBench emergency labels |
| Seed content review | 0 clinician-reviewed cards |
| Latest full tests | 201 passing after the completed v1 provider/data run |
| Best measured paired run | GPT-5.4: 77.324% average exact, 6.983s p95 service latency |
| Fast/cheap comparison | GPT-5-mini: 73.719% average exact, 17.234s p95, $2.09/1K target calls |
| Standalone 100-case review estimate | $6,300, excluding engineering |
| Illustrative 500-case pilot | $36,062.29: $62.29 compute + $36,000 human review |

Always re-run `git status` and the validation commands: this table is a dated
snapshot, not live state.

## Critical cautions

- The released labels are evaluation references, not clinical ground truth.
- The interactive facts are source-grounded but mechanically routed. In the
  current seed, duplicate answers occur across question IDs in 86 of 100 cards.
- The seed is useful for pipeline development and reviewer preparation, but
  not yet for defensible natural-conversation latency or clinical-performance
  claims.
- PMR data is non-commercial, and the physician-label license is still pending.
- Public repository visibility is intentional, but it is not a licence grant:
  upstream attribution, non-commercial restrictions and pending annotation
  terms still apply to reuse and redistribution.
- Existing materialized LFS result exports may appear dirty locally even when
  their Git pointers are unchanged.

## Where to continue

- [Current state](docs/knowledge/current-state.md)
- [Prioritised next steps](docs/knowledge/next-steps.md)
- [Static-first student plan](docs/knowledge/static-first.md)
- [Fictional static pilot](docs/knowledge/synthetic-pilot.md)
- [Repository map](docs/knowledge/repository-map.md)
- [Data and labels](docs/knowledge/data-and-labels.md)
- [Conversation-data landscape](docs/knowledge/conversation-data-landscape.md)
- [Model evaluation and latency](docs/knowledge/model-evaluation.md)
- [Interactive triage design](docs/knowledge/interactive-triage.md)
- [Clinical-review protocol](docs/knowledge/clinical-review-protocol.md)
- [Training and distillation strategy](docs/knowledge/training-strategy.md)
- [Cost model](docs/knowledge/costs.md)
- [Known limitations](docs/knowledge/known-limitations.md)
- [Runbook](docs/knowledge/runbook.md)
- [Decision record](docs/knowledge/decisions.md)
- [Glossary](docs/knowledge/glossary.md)
