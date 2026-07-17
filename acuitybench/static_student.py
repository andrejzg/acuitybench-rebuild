"""Static-first student plan, evaluation contract, and data safeguards."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from acuitybench.interactive.schema_validation import (
    load_json_schema,
    validate_instance,
)
from acuitybench.sources import project_root


PLAN_SCHEMA_VERSION = "static-student-plan/v1"
EXAMPLE_SCHEMA_VERSION = "static-acuity-example/v1"
LABELS = ("A", "B", "C", "D")


def default_static_plan_path() -> Path:
    return project_root() / "configs/static_student.v1.yaml"


def default_static_example_schema_path() -> Path:
    return project_root() / "schemas/static-acuity-example-v1.schema.json"


def load_static_plan(path: Path | None = None) -> dict[str, Any]:
    plan_path = path or default_static_plan_path()
    raw = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Static plan must be a mapping: {plan_path}")
    if raw.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"Unsupported static plan schema in {plan_path}")
    if raw.get("strategy") != "static_first":
        raise ValueError("Static student plan must declare strategy: static_first")

    evaluation = raw.get("evaluation")
    training = raw.get("training_data")
    serving = raw.get("serving")
    if not all(isinstance(value, dict) for value in (evaluation, training, serving)):
        raise ValueError("Static plan must define evaluation, training_data, and serving")
    if evaluation.get("primary_task") != "qa":
        raise ValueError("Static-first primary task must be qa")
    if evaluation.get("secondary_task") != "conv":
        raise ValueError("Static-first secondary task must be conv")
    if evaluation.get("output_contract") != "ACUITY: <A|B|C|D>":
        raise ValueError("Static QA output contract must be ACUITY: <A|B|C|D>")
    if int(evaluation.get("samples_per_case", 0)) < 1:
        raise ValueError("Static plan samples_per_case must be positive")
    main_score = evaluation.get("main_score")
    if not isinstance(main_score, dict) or tuple(main_score.get("labels", ())) != LABELS:
        raise ValueError("Static plan main score must use ordered labels A, B, C, D")
    if training.get("status") not in {"not_built", "built"}:
        raise ValueError("Static training_data.status must be not_built or built")
    if "openai_compatible" not in serving.get("supported_provider_paths", []):
        raise ValueError("Static plan must retain the openai_compatible student path")
    return raw


def inspect_static_plan(
    *,
    plan_path: Path | None = None,
    benchmark_path: Path | None = None,
) -> dict[str, Any]:
    plan = load_static_plan(plan_path)
    configured_benchmark = project_root() / str(plan["evaluation"]["benchmark"])
    benchmark = benchmark_path or configured_benchmark
    if not benchmark.exists():
        raise FileNotFoundError(f"Static evaluation benchmark not found: {benchmark}")
    frame = pd.read_csv(benchmark, dtype={"source_id": str})
    required = {"dataset", "source_id", "normalized_label", "split", "is_edge_case"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Static evaluation benchmark is missing: {sorted(missing)}")
    clear = frame[
        (frame["split"] == "primary")
        & frame["normalized_label"].isin(LABELS)
        & ~frame["is_edge_case"].astype(bool)
    ]
    expected_cases = int(plan["evaluation"]["expected_cases"])
    expected_main = int(plan["evaluation"]["main_score"]["expected_cases"])
    if len(frame) != expected_cases or len(clear) != expected_main:
        raise ValueError(
            "Static evaluation cohort mismatch: "
            f"expected {expected_cases}/{expected_main} total/main, "
            f"found {len(frame)}/{len(clear)}"
        )
    samples = int(plan["evaluation"]["samples_per_case"])
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "strategy": "static_first",
        "plan_path": str((plan_path or default_static_plan_path()).resolve()),
        "benchmark_path": str(benchmark.resolve()),
        "benchmark_role": "held_out_evaluation_only",
        "benchmark_cases": len(frame),
        "main_score_cases": len(clear),
        "main_score_label_counts": {
            label: int((clear["normalized_label"] == label).sum()) for label in LABELS
        },
        "primary_task": "qa",
        "secondary_task": "conv",
        "samples_per_case": samples,
        "qa_target_calls": len(frame) * samples,
        "paired_target_calls": len(frame) * samples * 2,
        "paired_judge_calls": len(frame) * samples,
        "training_data_status": plan["training_data"]["status"],
        "ready_for_student_evaluation": True,
        "ready_for_student_training": plan["training_data"]["status"] == "built",
        "interactive_phase_deferred": True,
    }


def static_evaluation_contract(
    plan: Mapping[str, Any], *, include_conversation: bool
) -> dict[str, Any]:
    evaluation = plan["evaluation"]
    tasks = ["qa", "conv"] if include_conversation else ["qa"]
    return {
        "schema_version": "static-student-evaluation/v1",
        "strategy": "static_first",
        "benchmark_role": evaluation["benchmark_role"],
        "tasks": tasks,
        "primary_task": evaluation["primary_task"],
        "secondary_task": evaluation["secondary_task"] if include_conversation else None,
        "output_contract": evaluation["output_contract"],
        "main_score": evaluation["main_score"],
        "interactive_policy": "deferred",
    }


def load_static_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Static example at {path}:{line_number} is not an object")
        examples.append(value)
    if not examples:
        raise ValueError(f"Static example file is empty: {path}")
    return examples


def validate_static_examples(
    path: Path,
    *,
    benchmark_path: Path | None = None,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    examples = load_static_examples(path)
    schema = load_json_schema(schema_path or default_static_example_schema_path())
    benchmark = pd.read_csv(
        benchmark_path or project_root() / "data/processed/acuitybench_transformed.csv",
        dtype={"source_id": str},
    )
    benchmark_ids = {
        (str(row.dataset), str(row.source_id)) for row in benchmark.itertuples(index=False)
    }
    example_ids: set[str] = set()
    family_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    source_ids: set[tuple[str, str]] = set()
    for index, example in enumerate(examples, 1):
        validate_instance(example, schema)
        if example.get("schema_version") != EXAMPLE_SCHEMA_VERSION:
            raise ValueError(f"Example {index} has an unsupported schema version")
        example_id = str(example["example_id"])
        if example_id in example_ids:
            raise ValueError(f"Duplicate static example_id: {example_id}")
        example_ids.add(example_id)
        split = str(example["split"])
        family_splits[str(example["family_id"])].add(split)
        split_counts[split] += 1
        label_counts[str(example["reference_acuity"])] += 1
        provenance = example["provenance"]
        source_key = (str(provenance["source_dataset"]), str(provenance["source_id"]))
        source_ids.add(source_key)
        if example["training_allowed"] and source_key in benchmark_ids:
            raise ValueError(
                f"Training contamination: {example_id} reuses AcuityBench source "
                f"{source_key[0]}:{source_key[1]}"
            )
    crossed = {
        family: sorted(splits)
        for family, splits in family_splits.items()
        if len(splits) > 1
    }
    if crossed:
        first_family = sorted(crossed)[0]
        raise ValueError(
            f"Family {first_family!r} crosses splits: {crossed[first_family]}"
        )
    return {
        "schema_version": EXAMPLE_SCHEMA_VERSION,
        "examples": len(examples),
        "unique_families": len(family_splits),
        "unique_sources": len(source_ids),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": {label: label_counts[label] for label in LABELS},
        "acuitybench_training_contamination": 0,
        "family_split_crossings": 0,
    }
