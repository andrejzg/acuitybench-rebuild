---
type: Glossary
title: AcuityBench Project Glossary
description: Concise meanings for benchmark, interactive-policy, safety, latency and training terms.
tags: [glossary, terminology, acuity, latency]
timestamp: 2026-07-17T00:00:00+01:00
status: active
---

# Benchmark and labels

| Term | Meaning in this repository |
| --- | --- |
| **Acuity** | Ordinal urgency of the recommended care disposition, not a diagnosis. |
| **A** | Monitor at home: self-care/home monitoring is the main recommendation. |
| **B** | See a doctor within weeks: routine, non-urgent outpatient assessment. |
| **C** | See a doctor within 24-48 hours: time-sensitive outpatient assessment without default ED referral. |
| **D** | Go to the emergency department now or seek equivalent immediate emergency evaluation. |
| **Boundary label** | Adjacent released categories such as `A|B`, `B|C` or `C|D`. |
| **Reference label** | Released physician/source-derived evaluation target. It is not objective clinical truth. |
| **Gold acuity** | Schema/evaluator name for the fixed reference label hidden from a policy. |
| **Primary split** | 697 cases: 247 direct mappings plus 450 physician-panel consensus cases. |
| **Ambiguous split** | 217 accepted physician-panel cases with `avg_sd > 0.75`. |
| **Clear primary** | The 527 primary cases with one A-D label used by the paper-style main score. |
| **`avg_sd`** | Released panel-disagreement statistic used for the strict consensus/ambiguity split. |
| **Remove** | Physician annotation indicating that a case should be excluded; eight majority-Remove cases are excluded after survey screening. |
| **Released CSV** | Checksum-pinned `physician_labels.csv`; canonical annotation artifact for this rebuild. |

# Interactive evaluation

| Term | Meaning in this repository |
| --- | --- |
| **Case card** | Strict hidden simulator/evaluator record containing provenance, opening, facts, reference acuity and safety fields. |
| **Opening utterance** | Only case-specific clinical text visible to a policy before its first action. |
| **Queryable fact** | One of 33 versioned question slots. A known fact returns routed evidence; an unknown fact returns `I'm not sure.` |
| **Red flag** | Unreviewed v1 lexical-rule marker attached to source evidence; not an exhaustive clinical assessment. |
| **Required question** | Evaluator-marked question whose omission makes a trajectory incomplete; in v1 these are detected red flags not already visible in the opening. |
| **Unsafe disposition** | Any A-D disposition ordinally below the card's reference acuity. |
| **Interactive seed v1** | Deterministic 100-card, 25-per-class held-out artifact with zero clinician-reviewed cards. |
| **`ASK`** | Non-terminal action requesting one catalogued `question_id` in patient-facing wording. |
| **`DISPOSE`** | Terminal action selecting A-D with a rationale grounded in revealed facts. |
| **`HANDOFF`** | Terminal action ending autonomous control and targeting a human clinician. |
| **Deterministic simulator** | Exact question-ID lookup; no LLM, inference, spontaneous paraphrase or invented facts. |
| **Under-triage** | Predicted disposition below the reference label. |
| **Over-triage** | Predicted disposition above the reference label. |
| **Autonomous accuracy** | Accuracy among cases receiving `DISPOSE`; report with handoff rate so selective handoff is visible. |

# Evaluation, latency and cost

| Term | Meaning in this repository |
| --- | --- |
| **TTFT** | Terminal request dispatch to the first non-empty visible text delta; requires streaming. |
| **Request wall time** | One terminal provider attempt from dispatch through response/stream completion. |
| **Service latency** | Provider request time plus retry backoff, excluding local queue wait; p95 is the primary serving-latency comparison. |
| **Queue wait** | Local time before a request is dispatched, including concurrency/semaphore waiting. |
| **Provider processing** | Provider-reported timing header; provider-defined and not a replacement for client-observed latency. |
| **Total logical duration** | End-to-end local duration for a logical target, including queueing, retries and backoff where applicable. |
| **Run manifest** | Immutable record of data digest, configuration, model snapshot, transport, completeness, usage and scoring contract. |
| **Static frontier** | Deterministic SVG generated from committed comparison tables; not a live dashboard. |
| **Static student** | Small specialised model that receives one complete case and emits one A/B/C/D label; it does not select follow-up questions. |
| **One-shot conversational task** | AcuityBench response-writing format scored by a GPT-4.1 rubric judge; despite the name, it is not a multi-turn consultation. |

# Training and governance

| Term | Meaning in this repository |
| --- | --- |
| **Teacher target** | Reviewed label and optional rationale generated for a separately sourced static training case. |
| **Teacher trace** | Multi-turn action trajectory generated by a strong policy for reviewed interactive training-pool cases, never from held-out evaluation. |
| **Student** | Smaller specialised model trained/adapted from accepted separate data and evaluated on frozen held-out benchmarks. |
| **Grouped split** | Split assigned at source-family/group level so related cases cannot cross train, validation and evaluation. |
| **Deduplication** | Exact and near-duplicate checks across original, normalised, rewritten and generated case forms. |
| **OKF bundle** | `docs/knowledge/`: Open Knowledge Format v0.1 linked Markdown concepts used for durable handover context. |
| **Public visibility** | Owner-selected GitHub state; it grants no additional third-party data licence. |

# Related concepts

- [Data and labels](data-and-labels.md)
- [Known limitations](known-limitations.md)
- [Decision record](decisions.md)
