"""Versioned fictional static-acuity pilot construction.

The module separates free deterministic planning from paid provider calls.
Generated and labelled records remain research candidates until manual review;
they are never treated as clinical ground truth.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from acuitybench.interactive.schema_validation import load_json_schema, validate_instance
from acuitybench.models import ModelConfig
from acuitybench.providers.base import CompletionResult, Provider
from acuitybench.sources import project_root, sha256_file
from acuitybench.static_student import LABELS


PLAN_SCHEMA_VERSION = "synthetic-static-pilot/v0"
REQUEST_SCHEMA_VERSION = "synthetic-generation-request/v0"
MANIFEST_SCHEMA_VERSION = "synthetic-pilot-manifest/v0"
GENERATION_SCHEMA_VERSION = "synthetic-acuity-generation/v0"
LABEL_SCHEMA_VERSION = "synthetic-acuity-label/v0"


@dataclass(frozen=True)
class SyntheticPaths:
    output_dir: Path
    generation_requests: Path
    generated_raw: Path
    labels_raw: Path
    examples: Path
    rejected: Path
    contamination_report: Path
    manifest: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def default_synthetic_plan_path() -> Path:
    return project_root() / "configs/static/synthetic_pilot.v0.yaml"


def _resolve_root_path(value: str) -> Path:
    return (project_root() / value).resolve()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root()))
    except ValueError:
        return str(path.resolve())


def load_synthetic_plan(path: Path | None = None) -> dict[str, Any]:
    plan_path = path or default_synthetic_plan_path()
    raw = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Synthetic pilot plan must be a mapping: {plan_path}")
    if raw.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported synthetic pilot schema: {plan_path}")
    if raw.get("purpose", {}).get("fictional_only") is not True:
        raise ValueError("Synthetic pilot must declare fictional_only: true")
    if raw.get("purpose", {}).get("benchmark_role") != (
        "acuitybench_held_out_evaluation_only"
    ):
        raise ValueError("Synthetic pilot must preserve AcuityBench as held-out")

    design = raw.get("design")
    generation = raw.get("generation")
    labeling = raw.get("labeling")
    acceptance = raw.get("acceptance")
    leakage = raw.get("leakage")
    outputs = raw.get("outputs")
    if not all(
        isinstance(item, dict)
        for item in (design, generation, labeling, acceptance, leakage, outputs)
    ):
        raise ValueError("Synthetic plan is missing a required mapping")
    if tuple(design.get("labels", ())) != LABELS:
        raise ValueError("Synthetic pilot labels must be ordered A, B, C, D")
    groups = design.get("presentation_groups")
    if not isinstance(groups, list) or not groups or len(groups) != len(set(groups)):
        raise ValueError("Synthetic presentation groups must be unique and non-empty")
    if int(design.get("cases_per_cell", 0)) < 1:
        raise ValueError("Synthetic cases_per_cell must be positive")
    expected_per_label = len(groups) * int(design["cases_per_cell"])
    if int(design.get("cases_per_label", 0)) != expected_per_label:
        raise ValueError(
            "cases_per_label must equal presentation groups x cases_per_cell"
        )
    development = int(design.get("development_cases_per_label", -1))
    if development < 0 or development >= expected_per_label:
        raise ValueError("development_cases_per_label must be within the pilot cell count")
    if generation.get("include_acuitybench_content") is not False:
        raise ValueError("Generator must prohibit AcuityBench content")
    if int(generation.get("samples_per_slot", 0)) != 1:
        raise ValueError("Synthetic pilot v0 requires exactly one generation per slot")
    if labeling.get("reveal_intended_label") is not False:
        raise ValueError("Independent labeler must remain blinded to generator intent")
    if int(labeling.get("independent_samples_per_case", 0)) < 2:
        raise ValueError("Synthetic pilot requires at least two label samples per case")
    for key in (
        "require_generation_intended_label_match",
        "require_unanimous_teacher_labels",
        "require_teacher_intended_label_match",
        "require_no_lexical_leakage",
    ):
        if acceptance.get(key) is not True:
            raise ValueError(f"Synthetic acceptance must set {key}: true")
    for relative in (
        generation.get("prompt"),
        generation.get("output_schema"),
        labeling.get("prompt"),
        labeling.get("output_schema"),
        outputs.get("candidate_schema"),
        leakage.get("benchmark"),
    ):
        if not isinstance(relative, str) or not _resolve_root_path(relative).exists():
            raise FileNotFoundError(f"Synthetic plan dependency is missing: {relative}")
    return raw


def build_generation_requests(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    design = plan["design"]
    groups = list(design["presentation_groups"])
    cases_per_cell = int(design["cases_per_cell"])
    development_per_label = int(design["development_cases_per_label"])
    base_seed = int(plan["seed"])
    requests: list[dict[str, Any]] = []
    counter = 0
    for label_index, label in enumerate(LABELS):
        development_cells = {
            (label_index + offset) % len(groups)
            for offset in range(development_per_label)
        }
        for group_index, group in enumerate(groups):
            for cell_index in range(cases_per_cell):
                counter += 1
                case_id = f"fictional-static-v0-{counter:03d}"
                requests.append(
                    {
                        "schema_version": REQUEST_SCHEMA_VERSION,
                        "case_id": case_id,
                        "family_id": case_id,
                        "split": (
                            "development"
                            if group_index in development_cells and cell_index == 0
                            else "train"
                        ),
                        "intended_acuity": label,
                        "presentation_group": str(group),
                        "seed": base_seed + counter,
                        "fictional_only": True,
                    }
                )
    return requests


def _paths(plan: Mapping[str, Any], output_dir: Path | None = None) -> SyntheticPaths:
    outputs = plan["outputs"]
    directory = output_dir or _resolve_root_path(str(outputs["directory"]))
    return SyntheticPaths(
        output_dir=directory,
        generation_requests=directory / str(outputs["generation_requests"]),
        generated_raw=directory / str(outputs["generated_raw"]),
        labels_raw=directory / str(outputs["labels_raw"]),
        examples=directory / str(outputs["examples"]),
        rejected=directory / str(outputs["rejected"]),
        contamination_report=directory / str(outputs["contamination_report"]),
        manifest=directory / str(outputs["manifest"]),
    )


def inspect_synthetic_plan(path: Path | None = None) -> dict[str, Any]:
    plan = load_synthetic_plan(path)
    requests = build_generation_requests(plan)
    label_samples = int(plan["labeling"]["independent_samples_per_case"])
    split_counts = Counter(str(item["split"]) for item in requests)
    label_counts = Counter(str(item["intended_acuity"]) for item in requests)
    group_counts = Counter(str(item["presentation_group"]) for item in requests)
    generation_calls = len(requests) * int(plan["generation"]["samples_per_slot"])
    labeling_calls = len(requests) * label_samples
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "pilot_id": plan["pilot_id"],
        "status": plan["status"],
        "fictional_only": True,
        "planned_cases": len(requests),
        "label_counts": {label: label_counts[label] for label in LABELS},
        "presentation_group_counts": dict(sorted(group_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "planned_generation_calls": generation_calls,
        "planned_label_calls": labeling_calls,
        "planned_total_provider_calls": generation_calls + labeling_calls,
        "independent_label_samples_per_case": label_samples,
        "acuitybench_content_in_generation": False,
        "semantic_embedding_check": plan["leakage"]["semantic_embedding_check"],
        "manual_review": plan["leakage"]["manual_review"],
        "ready_for_paid_generation": False,
        "paid_generation_blockers": [
            "select and fingerprint generator/labeler models",
            "review provider terms and data handling",
            "estimate spend",
            "obtain explicit spend confirmation",
        ],
    }


def _jsonl_text(values: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n"
        for value in values
    )


def _write_stable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Refusing to rewrite an incompatible pilot artifact: {path}")
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_jsonl(path: Path, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    if allow_missing and not path.exists():
        return []
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        values.append(value)
    return values


def initialize_synthetic_pilot(
    *, config_path: Path | None = None, output_dir: Path | None = None
) -> SyntheticPaths:
    plan_path = config_path or default_synthetic_plan_path()
    plan = load_synthetic_plan(plan_path)
    paths = _paths(plan, output_dir)
    requests = build_generation_requests(plan)
    _write_stable(paths.generation_requests, _jsonl_text(requests))

    generation_prompt = _resolve_root_path(str(plan["generation"]["prompt"]))
    label_prompt = _resolve_root_path(str(plan["labeling"]["prompt"]))
    generation_schema = _resolve_root_path(str(plan["generation"]["output_schema"]))
    label_schema = _resolve_root_path(str(plan["labeling"]["output_schema"]))
    candidate_schema = _resolve_root_path(str(plan["outputs"]["candidate_schema"]))
    benchmark = _resolve_root_path(str(plan["leakage"]["benchmark"]))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pilot_id": plan["pilot_id"],
        "status": "scaffold_initialized",
        "fictional_only": True,
        "training_ready": False,
        "training_blockers": [
            "generation not run",
            "independent labeling not run",
            "lexical leakage report not complete",
            "semantic similarity review not implemented",
            "manual review of all 20 cases not recorded",
        ],
        "config": {
            "path": _portable_path(plan_path),
            "sha256": sha256_file(plan_path),
        },
        "prompts": {
            "generation": {
                "path": _portable_path(generation_prompt),
                "sha256": sha256_file(generation_prompt),
            },
            "labeling": {
                "path": _portable_path(label_prompt),
                "sha256": sha256_file(label_prompt),
            },
        },
        "schemas": {
            "generation": {
                "path": _portable_path(generation_schema),
                "sha256": sha256_file(generation_schema),
            },
            "labeling": {
                "path": _portable_path(label_schema),
                "sha256": sha256_file(label_schema),
            },
            "candidate": {
                "path": _portable_path(candidate_schema),
                "sha256": sha256_file(candidate_schema),
            },
        },
        "held_out_benchmark": {
            "path": _portable_path(benchmark),
            "sha256": sha256_file(benchmark),
            "used_for_generation": False,
            "used_for_lexical_screening_only": True,
        },
        "artifacts": {
            "generation_requests": {
                "path": _portable_path(paths.generation_requests),
                "sha256": sha256_file(paths.generation_requests),
            }
        },
        "counts": {
            "planned_cases": len(requests),
            "successful_generations": 0,
            "successful_label_calls": 0,
            "accepted_examples": 0,
            "rejected_cases": 0,
        },
        "paid_provider_calls_recorded": 0,
    }
    if paths.manifest.exists():
        existing = json.loads(paths.manifest.read_text(encoding="utf-8"))
        expected_hashes = {
            "config": manifest["config"]["sha256"],
            "generation_prompt": manifest["prompts"]["generation"]["sha256"],
            "labeling_prompt": manifest["prompts"]["labeling"]["sha256"],
            "generation_schema": manifest["schemas"]["generation"]["sha256"],
            "labeling_schema": manifest["schemas"]["labeling"]["sha256"],
            "candidate_schema": manifest["schemas"]["candidate"]["sha256"],
            "held_out_benchmark": manifest["held_out_benchmark"]["sha256"],
            "generation_requests": manifest["artifacts"]["generation_requests"][
                "sha256"
            ],
        }
        actual_hashes = {
            "config": existing.get("config", {}).get("sha256"),
            "generation_prompt": existing.get("prompts", {})
            .get("generation", {})
            .get("sha256"),
            "labeling_prompt": existing.get("prompts", {})
            .get("labeling", {})
            .get("sha256"),
            "generation_schema": existing.get("schemas", {})
            .get("generation", {})
            .get("sha256"),
            "labeling_schema": existing.get("schemas", {})
            .get("labeling", {})
            .get("sha256"),
            "candidate_schema": existing.get("schemas", {})
            .get("candidate", {})
            .get("sha256"),
            "held_out_benchmark": existing.get("held_out_benchmark", {}).get(
                "sha256"
            ),
            "generation_requests": existing.get("artifacts", {})
            .get("generation_requests", {})
            .get("sha256"),
        }
        mismatches = sorted(
            name
            for name, expected in expected_hashes.items()
            if actual_hashes.get(name) != expected
        )
        if mismatches:
            raise ValueError(
                "Existing synthetic manifest has incompatible frozen inputs: "
                f"{mismatches}"
            )
    else:
        _write_json(paths.manifest, manifest)
    return paths


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Provider output did not contain a JSON object")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()


def _result_metadata(result: CompletionResult) -> dict[str, Any]:
    return {
        "response_id": result.response_id,
        "request_id": result.request_id,
        "returned_model": result.returned_model,
        "finish_reason": result.finish_reason,
        "input_tokens": result.input_tokens,
        "cached_input_tokens": result.cached_input_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "total_tokens": result.total_tokens,
        "server_processing_ms": result.server_processing_ms,
        "provider_metadata": result.provider_metadata,
    }


def _attempt_number(records: list[dict[str, Any]], key: tuple[Any, ...]) -> int:
    return 1 + sum(
        1
        for record in records
        if tuple(record.get(name) for name in ("case_id", "sample_index")[: len(key)])
        == key
    )


def _latest_successes(
    records: Iterable[dict[str, Any]], key_fields: tuple[str, ...]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if record.get("status") == "success":
            latest[tuple(record.get(field) for field in key_fields)] = record
    return latest


def _render_generation_prompt(template: str, request: Mapping[str, Any]) -> str:
    public_slot = {
        "case_id": request["case_id"],
        "intended_acuity": request["intended_acuity"],
        "presentation_group": request["presentation_group"],
        "seed": request["seed"],
    }
    return template.replace(
        "{{SLOT_JSON}}", json.dumps(public_slot, indent=2, sort_keys=True)
    )


def _render_label_prompt(template: str, vignette: str) -> str:
    return template.replace("{{VIGNETTE}}", vignette)


async def generate_synthetic_cases(
    *,
    provider: Provider,
    model: ModelConfig,
    config_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    plan = load_synthetic_plan(config_path)
    paths = initialize_synthetic_pilot(config_path=config_path, output_dir=output_dir)
    requests = _load_jsonl(paths.generation_requests)
    records = _load_jsonl(paths.generated_raw, allow_missing=True)
    complete = _latest_successes(records, ("case_id",))
    template = _resolve_root_path(str(plan["generation"]["prompt"])).read_text(
        encoding="utf-8"
    )
    schema = load_json_schema(
        _resolve_root_path(str(plan["generation"]["output_schema"]))
    )
    model_hash = _sha256_text(_canonical_json(model.as_dict()))
    new_successes = 0
    new_failures = 0
    for request in requests:
        key = (request["case_id"],)
        if key in complete:
            continue
        started_at = _utc_now()
        started = time.perf_counter()
        record: dict[str, Any] = {
            "schema_version": "synthetic-generation-attempt/v0",
            "case_id": request["case_id"],
            "attempt": _attempt_number(records, key),
            "status": "failed",
            "started_at": started_at,
            "requested_model_id": model.id,
            "requested_api_model": model.api_model,
            "model_config_sha256": model_hash,
            "prompt_sha256": _sha256_text(_render_generation_prompt(template, request)),
        }
        try:
            result = await provider.complete(
                config=model,
                messages=[
                    {
                        "role": "user",
                        "content": _render_generation_prompt(template, request),
                    }
                ],
                max_output_tokens=int(plan["generation"]["max_output_tokens"]),
                stream=bool(plan["generation"]["stream"]),
            )
            output = _parse_json_object(result.text)
            validate_instance(output, schema)
            if output.get("schema_version") != GENERATION_SCHEMA_VERSION:
                raise ValueError("Unsupported generation output version")
            if output["intended_acuity"] != request["intended_acuity"]:
                raise ValueError("Generation output changed the intended acuity")
            if output["presentation_group"] != request["presentation_group"]:
                raise ValueError("Generation output changed the presentation group")
            record.update(
                {
                    "status": "success",
                    "raw_text": result.text,
                    "output": output,
                    "response": _result_metadata(result),
                }
            )
            new_successes += 1
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            new_failures += 1
        record["duration_ms"] = (time.perf_counter() - started) * 1000
        _append_jsonl(paths.generated_raw, record)
        records.append(record)
    _update_manifest(paths, plan)
    return {
        "planned_cases": len(requests),
        "new_successes": new_successes,
        "new_failures": new_failures,
        "successful_cases_total": len(
            _latest_successes(records, ("case_id",))
        ),
        "generated_raw": str(paths.generated_raw),
    }


async def label_synthetic_cases(
    *,
    provider: Provider,
    model: ModelConfig,
    config_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    plan = load_synthetic_plan(config_path)
    paths = initialize_synthetic_pilot(config_path=config_path, output_dir=output_dir)
    generations = _latest_successes(
        _load_jsonl(paths.generated_raw, allow_missing=True), ("case_id",)
    )
    if not generations:
        raise ValueError("No successful fictional generations are available to label")
    records = _load_jsonl(paths.labels_raw, allow_missing=True)
    complete = _latest_successes(records, ("case_id", "sample_index"))
    template = _resolve_root_path(str(plan["labeling"]["prompt"])).read_text(
        encoding="utf-8"
    )
    schema = load_json_schema(
        _resolve_root_path(str(plan["labeling"]["output_schema"]))
    )
    samples = int(plan["labeling"]["independent_samples_per_case"])
    model_hash = _sha256_text(_canonical_json(model.as_dict()))
    new_successes = 0
    new_failures = 0
    for (case_id,), generation in sorted(generations.items()):
        vignette = str(generation["output"]["vignette"])
        for sample_index in range(samples):
            key = (case_id, sample_index)
            if key in complete:
                continue
            prompt = _render_label_prompt(template, vignette)
            started = time.perf_counter()
            record: dict[str, Any] = {
                "schema_version": "synthetic-label-attempt/v0",
                "case_id": case_id,
                "sample_index": sample_index,
                "attempt": _attempt_number(records, key),
                "status": "failed",
                "started_at": _utc_now(),
                "requested_model_id": model.id,
                "requested_api_model": model.api_model,
                "model_config_sha256": model_hash,
                "prompt_sha256": _sha256_text(prompt),
                "intended_label_revealed": False,
            }
            try:
                result = await provider.complete(
                    config=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_output_tokens=int(plan["labeling"]["max_output_tokens"]),
                    stream=bool(plan["labeling"]["stream"]),
                )
                output = _parse_json_object(result.text)
                validate_instance(output, schema)
                if output.get("schema_version") != LABEL_SCHEMA_VERSION:
                    raise ValueError("Unsupported label output version")
                record.update(
                    {
                        "status": "success",
                        "raw_text": result.text,
                        "output": output,
                        "response": _result_metadata(result),
                    }
                )
                new_successes += 1
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                new_failures += 1
            record["duration_ms"] = (time.perf_counter() - started) * 1000
            _append_jsonl(paths.labels_raw, record)
            records.append(record)
    finalize = finalize_synthetic_pilot(config_path=config_path, output_dir=output_dir)
    return {
        "generated_cases": len(generations),
        "new_successes": new_successes,
        "new_failures": new_failures,
        "successful_label_calls_total": len(
            _latest_successes(records, ("case_id", "sample_index"))
        ),
        "finalize": finalize,
    }


def _normalise_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _token_ngrams(value: str, size: int = 3) -> set[tuple[str, ...]]:
    tokens = _normalise_text(value).split()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _trigram_containment(left: str, right: str) -> float:
    left_grams = _token_ngrams(left)
    right_grams = _token_ngrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / min(len(left_grams), len(right_grams))


def _benchmark_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, list):
        return "\n".join(
            str(item.get("content", ""))
            for item in parsed
            if isinstance(item, dict)
        )
    return value


def build_contamination_report(
    candidates: Mapping[str, str],
    *,
    plan: Mapping[str, Any],
    benchmark_path: Path | None = None,
) -> dict[str, Any]:
    leakage = plan["leakage"]
    benchmark = benchmark_path or _resolve_root_path(str(leakage["benchmark"]))
    frame = pd.read_csv(benchmark, dtype={"source_id": str})
    field = str(leakage["benchmark_text_field"])
    if field not in frame.columns:
        raise ValueError(f"Leakage benchmark is missing text field {field!r}")
    benchmark_rows = [
        {
            "dataset": str(row.dataset),
            "source_id": str(row.source_id),
            "text": _benchmark_text(getattr(row, field)),
        }
        for row in frame.itertuples(index=False)
    ]
    sequence_threshold = float(leakage["sequence_similarity_threshold"])
    containment_threshold = float(leakage["token_trigram_containment_threshold"])
    minimum_characters = int(leakage["minimum_normalized_characters"])
    results: list[dict[str, Any]] = []
    blocked_cases: set[str] = set()
    candidate_items = sorted(candidates.items())
    for case_id, text in candidate_items:
        normalized = _normalise_text(text)
        best: dict[str, Any] | None = None
        blocked_reasons: list[str] = []
        if len(normalized) < minimum_characters:
            blocked_reasons.append("too_short_for_reliable_lexical_screen")
        for row in benchmark_rows:
            benchmark_normalized = _normalise_text(row["text"])
            exact = normalized == benchmark_normalized and bool(normalized)
            sequence = SequenceMatcher(None, normalized, benchmark_normalized).ratio()
            containment = _trigram_containment(text, row["text"])
            score = max(
                sequence / sequence_threshold,
                containment / containment_threshold,
                2.0 if exact else 0.0,
            )
            if best is None or score > best["relative_score"]:
                best = {
                    "dataset": row["dataset"],
                    "source_id": row["source_id"],
                    "sequence_similarity": sequence,
                    "token_trigram_containment": containment,
                    "exact_normalized_match": exact,
                    "relative_score": score,
                }
        assert best is not None
        if best["exact_normalized_match"]:
            blocked_reasons.append("exact_benchmark_match")
        if best["sequence_similarity"] >= sequence_threshold:
            blocked_reasons.append("benchmark_sequence_similarity")
        if best["token_trigram_containment"] >= containment_threshold:
            blocked_reasons.append("benchmark_token_trigram_containment")
        label_leak = re.search(
            r"\b(?:acuity|disposition)\s*(?:label|category)?\s*[:=]\s*[abcd]\b",
            text,
            flags=re.I,
        )
        if label_leak:
            blocked_reasons.append("acuity_label_leaked_into_vignette")
        for other_case_id, other_text in candidate_items:
            if other_case_id >= case_id:
                continue
            other_normalized = _normalise_text(other_text)
            exact = normalized == other_normalized and bool(normalized)
            sequence = SequenceMatcher(None, normalized, other_normalized).ratio()
            containment = _trigram_containment(text, other_text)
            if exact or sequence >= sequence_threshold or containment >= containment_threshold:
                blocked_reasons.append(f"generated_near_duplicate:{other_case_id}")
        blocked_reasons = sorted(set(blocked_reasons))
        if blocked_reasons:
            blocked_cases.add(case_id)
        best.pop("relative_score", None)
        results.append(
            {
                "case_id": case_id,
                "blocked": bool(blocked_reasons),
                "blocked_reasons": blocked_reasons,
                "best_benchmark_match": best,
                "normalized_characters": len(normalized),
            }
        )
    return {
        "schema_version": "synthetic-contamination-report/v0",
        "benchmark": {
            "path": str(benchmark.resolve()),
            "sha256": sha256_file(benchmark),
            "cases": len(benchmark_rows),
            "used_for_generation": False,
        },
        "thresholds": {
            "sequence_similarity": sequence_threshold,
            "token_trigram_containment": containment_threshold,
            "minimum_normalized_characters": minimum_characters,
        },
        "checks": {
            "normalized_exact": "complete",
            "sequence_similarity": "complete",
            "token_trigram_containment": "complete",
            "generated_internal_duplicates": "complete",
            "semantic_embedding_similarity": leakage["semantic_embedding_check"],
            "manual_review": leakage["manual_review"],
        },
        "candidate_cases": len(candidates),
        "blocked_cases": len(blocked_cases),
        "results": results,
    }


def finalize_synthetic_pilot(
    *, config_path: Path | None = None, output_dir: Path | None = None
) -> dict[str, Any]:
    plan = load_synthetic_plan(config_path)
    paths = initialize_synthetic_pilot(config_path=config_path, output_dir=output_dir)
    requests = {item["case_id"]: item for item in _load_jsonl(paths.generation_requests)}
    generations = _latest_successes(
        _load_jsonl(paths.generated_raw, allow_missing=True), ("case_id",)
    )
    labels = _latest_successes(
        _load_jsonl(paths.labels_raw, allow_missing=True),
        ("case_id", "sample_index"),
    )
    candidates = {
        case_id: str(record["output"]["vignette"])
        for (case_id,), record in generations.items()
    }
    contamination = build_contamination_report(candidates, plan=plan)
    _write_json(paths.contamination_report, contamination)
    blocked_by_case = {
        str(item["case_id"]): list(item["blocked_reasons"])
        for item in contamination["results"]
    }
    samples_per_case = int(plan["labeling"]["independent_samples_per_case"])
    labels_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (case_id, _), record in labels.items():
        labels_by_case[str(case_id)].append(record)
    for records in labels_by_case.values():
        records.sort(key=lambda item: int(item["sample_index"]))

    examples: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for case_id, request in requests.items():
        reasons: list[str] = []
        generation = generations.get((case_id,))
        label_records = labels_by_case.get(case_id, [])
        if generation is None:
            reasons.append("missing_successful_generation")
        if len(label_records) != samples_per_case:
            reasons.append("incomplete_independent_labeling")
        if generation is not None:
            generated_output = generation["output"]
            if generated_output.get("intended_acuity") != request["intended_acuity"]:
                reasons.append("generator_intended_label_mismatch")
            if generated_output.get("presentation_group") != request["presentation_group"]:
                reasons.append("generator_presentation_group_mismatch")
        outputs = [record["output"] for record in label_records]
        teacher_labels = [str(output["acuity"]) for output in outputs]
        if len(set(teacher_labels)) > 1:
            reasons.append("teacher_label_disagreement")
        if teacher_labels and any(
            label != str(request["intended_acuity"]) for label in teacher_labels
        ):
            reasons.append("teacher_generator_intent_disagreement")
        if plan["acceptance"]["reject_any_ambiguity_flags"] and any(
            output.get("ambiguity_flags") for output in outputs
        ):
            reasons.append("teacher_reported_ambiguity")
        reasons.extend(blocked_by_case.get(case_id, []))
        reasons = sorted(set(reasons))
        if not reasons and generation is not None and outputs:
            vignette = str(generation["output"]["vignette"])
            label_record = label_records[0]
            model_hash = str(label_record["model_config_sha256"])
            examples.append(
                {
                    "schema_version": "synthetic-acuity-candidate/v0",
                    "candidate_id": case_id,
                    "family_id": request["family_id"],
                    "intended_split": request["split"],
                    "training_allowed": False,
                    "review_status": "machine_screened_manual_review_pending",
                    "vignette": vignette,
                    "reference_acuity": teacher_labels[0],
                    "target_rationale": outputs[0]["rationale"],
                    "teacher": {
                        "basis": (
                            f"{samples_per_case} independent teacher samples "
                            "agreed with fictional generator intent; manual review pending"
                        ),
                        "sample_count": samples_per_case,
                        "model_id": label_record["requested_model_id"],
                        "config_sha256": model_hash,
                    },
                    "provenance": {
                        "source_dataset": plan["provenance"]["source_dataset"],
                        "source_id": case_id,
                        "source_revision": plan["provenance"]["source_revision"],
                        "source_text_sha256": _sha256_text(vignette),
                        "transformation": plan["provenance"]["transformation"],
                        "license": plan["provenance"]["license_note"],
                    },
                }
            )
        else:
            rejected.append(
                {
                    "schema_version": "synthetic-rejection/v0",
                    "case_id": case_id,
                    "reasons": reasons,
                    "request": request,
                    "generation": None if generation is None else generation["output"],
                    "teacher_outputs": outputs,
                }
            )
    paths.examples.write_text(_jsonl_text(examples), encoding="utf-8")
    paths.rejected.write_text(_jsonl_text(rejected), encoding="utf-8")
    _update_manifest(paths, plan)
    return {
        "planned_cases": len(requests),
        "successful_generations": len(generations),
        "successful_label_calls": len(labels),
        "accepted_examples": len(examples),
        "rejected_cases": len(rejected),
        "lexically_blocked_cases": contamination["blocked_cases"],
        "training_ready": False,
        "training_blocker": "manual review of all 20 cases is not recorded",
        "semantic_similarity_status": plan["leakage"]["semantic_embedding_check"],
    }


def _update_manifest(paths: SyntheticPaths, plan: Mapping[str, Any]) -> None:
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    generations = _latest_successes(
        _load_jsonl(paths.generated_raw, allow_missing=True), ("case_id",)
    )
    labels = _latest_successes(
        _load_jsonl(paths.labels_raw, allow_missing=True),
        ("case_id", "sample_index"),
    )
    examples = _load_jsonl(paths.examples, allow_missing=True)
    rejected = _load_jsonl(paths.rejected, allow_missing=True)
    planned = len(_load_jsonl(paths.generation_requests))
    label_expected = planned * int(plan["labeling"]["independent_samples_per_case"])
    if examples or rejected:
        status = "machine_screened_manual_review_required"
    elif len(generations) == planned and len(labels) == label_expected:
        status = "labeling_complete_not_finalized"
    elif generations:
        status = "generation_in_progress_or_complete"
    else:
        status = "scaffold_initialized"
    manifest["status"] = status
    manifest["updated_at"] = _utc_now()
    manifest["counts"] = {
        "planned_cases": planned,
        "successful_generations": len(generations),
        "successful_label_calls": len(labels),
        "accepted_examples": len(examples),
        "rejected_cases": len(rejected),
    }
    manifest["paid_provider_calls_recorded"] = len(
        _load_jsonl(paths.generated_raw, allow_missing=True)
    ) + len(_load_jsonl(paths.labels_raw, allow_missing=True))
    manifest["training_ready"] = False
    manifest["training_blockers"] = [
        "manual review of all 20 cases not recorded",
        "semantic embedding similarity is not implemented for this scaffold",
    ]
    for name, artifact_path in (
        ("generated_raw", paths.generated_raw),
        ("labels_raw", paths.labels_raw),
        ("examples", paths.examples),
        ("rejected", paths.rejected),
        ("contamination_report", paths.contamination_report),
    ):
        if artifact_path.exists():
            manifest["artifacts"][name] = {
                "path": _portable_path(artifact_path),
                "sha256": sha256_file(artifact_path),
            }
    _write_json(paths.manifest, manifest)


def validate_synthetic_pilot(
    *,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    plan_path = config_path or default_synthetic_plan_path()
    plan = load_synthetic_plan(plan_path)
    paths = _paths(plan, output_dir)
    if not paths.manifest.exists() or not paths.generation_requests.exists():
        raise FileNotFoundError("Synthetic pilot scaffold is not initialized")
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported synthetic pilot manifest")
    if manifest["config"]["sha256"] != sha256_file(plan_path):
        raise ValueError("Synthetic pilot config hash does not match manifest")
    for phase, prompt_key in (("generation", "generation"), ("labeling", "labeling")):
        prompt_path = _resolve_root_path(str(plan[phase]["prompt"]))
        if manifest["prompts"][prompt_key]["sha256"] != sha256_file(prompt_path):
            raise ValueError(f"Synthetic {prompt_key} prompt hash does not match manifest")
    for schema_key, relative in (
        ("generation", plan["generation"]["output_schema"]),
        ("labeling", plan["labeling"]["output_schema"]),
        ("candidate", plan["outputs"]["candidate_schema"]),
    ):
        schema_path = _resolve_root_path(str(relative))
        if manifest["schemas"][schema_key]["sha256"] != sha256_file(schema_path):
            raise ValueError(f"Synthetic {schema_key} schema hash does not match manifest")
    benchmark_path = _resolve_root_path(str(plan["leakage"]["benchmark"]))
    if manifest["held_out_benchmark"]["sha256"] != sha256_file(benchmark_path):
        raise ValueError("Synthetic leakage benchmark hash does not match manifest")
    requests = _load_jsonl(paths.generation_requests)
    expected_requests = build_generation_requests(plan)
    if requests != expected_requests:
        raise ValueError("Generation requests do not match the versioned plan")
    if manifest["artifacts"]["generation_requests"]["sha256"] != sha256_file(
        paths.generation_requests
    ):
        raise ValueError("Generation request checksum does not match manifest")
    if allow_incomplete and not paths.examples.exists():
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "pilot_id": plan["pilot_id"],
            "status": manifest["status"],
            "planned_cases": len(requests),
            "scaffold_valid": True,
            "pipeline_complete": False,
            "training_ready": False,
            "paid_provider_calls_recorded": manifest["paid_provider_calls_recorded"],
        }
    required = (
        paths.generated_raw,
        paths.labels_raw,
        paths.examples,
        paths.rejected,
        paths.contamination_report,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Synthetic pilot is incomplete: {missing}")
    for name, artifact_path in (
        ("generated_raw", paths.generated_raw),
        ("labels_raw", paths.labels_raw),
        ("examples", paths.examples),
        ("rejected", paths.rejected),
        ("contamination_report", paths.contamination_report),
    ):
        if manifest["artifacts"][name]["sha256"] != sha256_file(artifact_path):
            raise ValueError(f"Synthetic artifact checksum mismatch: {name}")
    examples = _load_jsonl(paths.examples)
    candidate_schema = load_json_schema(
        _resolve_root_path(str(plan["outputs"]["candidate_schema"]))
    )
    for example in examples:
        validate_instance(example, candidate_schema)
        if example["training_allowed"] is not False:
            raise ValueError("Machine-screened candidate unexpectedly allows training")
    contamination = json.loads(paths.contamination_report.read_text(encoding="utf-8"))
    accepted_ids = {str(item["candidate_id"]) for item in examples}
    blocked_ids = {
        str(item["case_id"])
        for item in contamination["results"]
        if item["blocked"]
    }
    if accepted_ids & blocked_ids:
        raise ValueError("A lexically blocked case appears in accepted examples")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pilot_id": plan["pilot_id"],
        "status": manifest["status"],
        "planned_cases": len(requests),
        "accepted_examples": len(examples),
        "rejected_cases": len(_load_jsonl(paths.rejected)),
        "candidate_validation": {
            "schema": "synthetic-acuity-candidate/v0",
            "candidates": len(examples),
            "training_allowed": False,
        },
        "lexically_blocked_cases": contamination["blocked_cases"],
        "semantic_similarity_status": plan["leakage"]["semantic_embedding_check"],
        "pipeline_complete": True,
        "training_ready": False,
        "training_blockers": manifest["training_blockers"],
    }
