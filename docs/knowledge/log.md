# Knowledge Bundle Update Log

## 2026-07-17

* **Pipeline**: Added a versioned 20-case fictional static-pilot scaffold with deterministic slots, strict prompts/schemas, resumable generation and blinded double-labelling, lexical leakage checks and fake-provider tests.
* **Safety**: Initialized the scaffold with zero provider calls and kept training blocked pending paid generation, semantic screening and manual review of all 20 candidates.
* **Operations**: Added free `synthetic-plan`, `synthetic-init` and incomplete-validation commands plus explicit spend/terms gates for `synthetic-generate` and `synthetic-label`.
* **Decision**: Accepted the static-first sequence: train a complete-case-to-A/B/C/D student before undertaking multi-turn `ASK`/`DISPOSE`/`HANDOFF` training.
* **Contract**: Added a versioned static-student plan, separate-example JSON Schema, contamination-aware validator and persisted evaluation contract.
* **Serving**: Added model-registry support for OpenAI-compatible student endpoints with environment-based base URLs and stable deployment provenance.
* **Evaluation**: Added `static-plan`, `static-data-validate` and paid `static-evaluate` commands; QA is primary and paired one-shot conversation remains available for paper/chart comparison.
* **Creation**: Added an OKF v0.1 knowledge bundle for durable human/AI handover.
* **Creation**: Added project, data, evaluation, interactive-triage, cost, decision, limitation and runbook concepts.
* **Creation**: Added the conversation-data landscape, training strategy, concrete clinical-review protocol and a real abridged case-card example.
* **Verification**: Ran 195 tests, including the complete fake-provider fictional pilot, plus the core/interactive/static/synthetic validators successfully.
* **Decision**: Recorded the owner's explicit instruction to keep the GitHub repository public and updated stale private-repository wording.
* **Risk**: Retained the distinction that public visibility does not waive CC BY-NC restrictions or resolve the pending physician-label licence.
* **Integrity**: Removed ignored SQLite `-wal` and `-shm` runtime sidecars from the durable artifact checksum inventory.
