from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from acuitybench.evaluation import prepare_run
from acuitybench.models import ModelRegistry
from acuitybench.static_student import (
    inspect_static_plan,
    load_static_plan,
    static_evaluation_contract,
    validate_static_examples,
)
from acuitybench.store import EvaluationStore


def _example(
    example_id: str,
    *,
    family_id: str = "family-1",
    split: str = "train",
    source_dataset: str = "separate_source",
    source_id: str = "source-1",
) -> dict[str, object]:
    return {
        "schema_version": "static-acuity-example/v1",
        "example_id": example_id,
        "family_id": family_id,
        "split": split,
        "training_allowed": split != "evaluation",
        "prompt": "A separate synthetic patient presentation.",
        "reference_acuity": "C",
        "target_rationale": "Time-sensitive assessment is the reference disposition.",
        "label": {
            "basis": "teacher label for pipeline testing",
            "source": "teacher",
            "review_status": "unreviewed",
            "teacher_model": "teacher-fixture",
            "teacher_config_sha256": "b" * 64,
        },
        "provenance": {
            "source_dataset": source_dataset,
            "source_id": source_id,
            "source_revision": "fixture-v1",
            "source_text_sha256": "a" * 64,
            "transformation": "fixture/v1",
            "license": "fixture-only",
        },
    }


def _write_jsonl(path: Path, examples: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(example, sort_keys=True) + "\n" for example in examples),
        encoding="utf-8",
    )
    return path


def test_static_plan_matches_committed_held_out_benchmark() -> None:
    report = inspect_static_plan()

    assert report["strategy"] == "static_first"
    assert report["benchmark_cases"] == 914
    assert report["main_score_cases"] == 527
    assert report["qa_target_calls"] == 4570
    assert report["paired_target_calls"] == 9140
    assert report["paired_judge_calls"] == 4570
    assert report["ready_for_student_evaluation"] is True
    assert report["ready_for_student_training"] is False


def test_static_example_validator_accepts_separate_grouped_data(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "training.jsonl",
        [
            _example("example-1", source_id="source-1"),
            _example("example-2", source_id="source-2"),
        ],
    )

    report = validate_static_examples(path)

    assert report["examples"] == 2
    assert report["unique_families"] == 1
    assert report["split_counts"] == {"train": 2}
    assert report["acuitybench_training_contamination"] == 0


def test_static_example_validator_rejects_acuitybench_source_reuse(
    tmp_path: Path,
) -> None:
    benchmark = pd.read_csv(
        Path(__file__).resolve().parents[1]
        / "data/processed/acuitybench_transformed.csv",
        dtype={"source_id": str},
    )
    row = benchmark.iloc[0]
    path = _write_jsonl(
        tmp_path / "contaminated.jsonl",
        [
            _example(
                "contaminated",
                source_dataset=str(row["dataset"]),
                source_id=str(row["source_id"]),
            )
        ],
    )

    with pytest.raises(ValueError, match="Training contamination"):
        validate_static_examples(path)


def test_static_example_validator_rejects_family_split_crossing(
    tmp_path: Path,
) -> None:
    path = _write_jsonl(
        tmp_path / "crossed.jsonl",
        [
            _example("train-example", source_id="train-source"),
            _example(
                "eval-example",
                split="evaluation",
                source_id="eval-source",
            ),
        ],
    )

    with pytest.raises(ValueError, match="crosses splits"):
        validate_static_examples(path)


def test_static_contract_is_persisted_and_changes_run_identity(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.csv"
    pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "source_id": "1",
                "normalized_label": "A",
                "split": "primary",
                "mapping_method": "fixture",
                "is_edge_case": False,
                "qa_prompt": "case",
                "conversational_prompt": json.dumps(
                    [{"role": "user", "content": "case"}]
                ),
            }
        ]
    ).to_csv(benchmark, index=False)
    plan = load_static_plan()
    contract = static_evaluation_contract(plan, include_conversation=False)
    model = ModelRegistry().get("gpt-5-mini")
    with EvaluationStore(tmp_path / "runs.sqlite3") as store:
        base_id, _ = prepare_run(
            store=store,
            model=model,
            benchmark_path=benchmark,
            tasks=("qa",),
            samples=1,
            datasets=None,
            limit=None,
            run_id="base",
        )
        static_id, _ = prepare_run(
            store=store,
            model=model,
            benchmark_path=benchmark,
            tasks=("qa",),
            samples=1,
            datasets=None,
            limit=None,
            run_id="static",
            experiment_contract=contract,
        )
        base = store.get_run(base_id)
        static = store.get_run(static_id)

    assert base["manifest_fingerprint"] != static["manifest_fingerprint"]
    assert static["experiment_contract"] == contract
