---
okf_version: "0.1"
title: AcuityBench Rebuild Knowledge Bundle
description: Curated project context and handover knowledge for humans and AI agents.
timestamp: 2026-07-17T00:00:00+01:00
---

# Start here

* [Current state](current-state.md) - Dated implementation, repository and risk snapshot.
* [Next steps](next-steps.md) - Prioritised work queue and acceptance gates.
* [Static-first student plan](static-first.md) - Accepted first-stage objective, executable contract, training boundary and progression gate.
* [Fictional static pilot](synthetic-pilot.md) - Zero-call 20-case generation/label pipeline check, review gates and paid-run boundary.
* [Project overview](project-overview.md) - Mission, scope, success criteria and non-goals.
* [Known limitations](known-limitations.md) - Claims that must not be made and unresolved risks.

# System and data

* [Repository map](repository-map.md) - Code modules, configurations, artifacts and source-of-truth hierarchy.
* [Data and labels](data-and-labels.md) - Dataset lineage, benchmark counts, label semantics and licensing.
* [Conversation-data landscape](conversation-data-landscape.md) - What AcuityBench, HealthBench and DDXPlus can and cannot supply for a live GP-style interaction.
* [Interactive triage](interactive-triage.md) - Case-card protocol, simulator, seed design and review gate.
* [Case-card example](case-card-example.md) - One abridged, real seed card with visible and hidden state.
* [Clinical-review protocol](clinical-review-protocol.md) - Proposed blinded content review, two-GP labelling, adjudication and replacement workflow.
* [Glossary](glossary.md) - Stable meanings for project-specific terms and metrics.

# Evaluation and economics

* [Model evaluation](model-evaluation.md) - Paper contract, completed runs, metrics, latency and extensibility.
* [Cost model](costs.md) - Reproducible arithmetic for the seed-review and training-pilot estimates.
* [Training strategy](training-strategy.md) - Proposed separate-data distillation path, scale heuristics, learning curves and Tinker status.

# Operations and governance

* [Runbook](runbook.md) - Setup, validation, model execution and artifact-integrity commands.
* [Decision record](decisions.md) - Important design choices and their rationale.
* [Knowledge update log](log.md) - Newest-first history of this bundle.

# About this bundle

This directory targets Google Cloud's **Open Knowledge Format v0.1 Draft**.
Every concept document has YAML frontmatter with a non-empty `type`; this
reserved `index.md` provides progressive disclosure, and `log.md` records dated
changes. Ordinary Markdown links form the knowledge graph.

OKF is used here as a portable knowledge layer, not as a replacement for
`AGENTS.md`, executable schemas, manifests or tests. A new agent should read
the root [`AGENTS.md`](../../AGENTS.md) and [`HANDOVER.md`](../../HANDOVER.md)
before acting.

# Citations

[1] [Google Cloud announcement: Introducing the Open Knowledge Format](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing)

[2] [Open Knowledge Format v0.1 Draft specification](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
