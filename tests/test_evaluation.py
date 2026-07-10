from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from acuitybench.distributional import jensen_shannon, wasserstein_1
from acuitybench.evaluation import extract_label, extract_reasoning, sha256_text
from acuitybench.models import ModelConfig, ModelRegistry
from acuitybench.reporting import (
    _metrics_long,
    _mode_severe,
    _paper_value,
    _table2,
    generate_report,
)
from acuitybench.store import EvaluationStore


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("REASONING: stable\nACUITY: c", "C"),
        ("A preface\nacuity - D\nACUITY: A", "D"),
        ("ACUITY: E", None),
        ("No machine-readable answer", None),
        (None, None),
    ],
)
def test_extract_label_uses_first_valid_case_insensitive_match(
    response: str | None, expected: str | None
) -> None:
    assert extract_label(response) == expected


def test_extract_reasoning_stops_before_label() -> None:
    response = "REASONING: first line\nsecond line\nACUITY: B"
    assert extract_reasoning(response) == "first line\nsecond line"


def test_mode_ties_resolve_to_more_severe_label() -> None:
    assert _mode_severe(["A", "B", "A", "B", None]) == "B"
    assert _mode_severe(["C", "D"]) == "D"
    assert _mode_severe([None, "not-a-label"]) is None


def test_distributional_distances_use_paper_conventions() -> None:
    at_home = [1.0, 0.0, 0.0, 0.0]
    emergency = [0.0, 0.0, 0.0, 1.0]
    assert jensen_shannon(at_home, at_home) == pytest.approx(0)
    assert jensen_shannon(at_home, emergency) == pytest.approx(0.69314718056)
    assert wasserstein_1(at_home, emergency) == pytest.approx(3.0)


def test_paper_metrics_exclude_boundary_and_ambiguous_cases() -> None:
    rows = [
        # The three clear primary cases produce one exact, one over, one under.
        ("primary", "A", "A", 1, 0, 0, 0),
        ("primary", "B", "C", 0, 1, 0, 1),
        ("primary", "D", "C", 0, 0, 1, 1),
        # Neither row belongs in the paper-compatible main table.
        ("primary", "B|C", "C", None, None, None, None),
        ("ambiguous", "D", "D", 1, 0, 0, 0),
    ]
    predictions = pd.DataFrame(
        [
            {
                "task_type": "qa",
                "dataset": "synthetic",
                "split": split,
                "normalized_label": gold,
                "mode_prediction": prediction,
                "n_expected": 5,
                "n_valid": 5,
                "exact": exact,
                "over": over,
                "under": under,
                "ordinal_distance": distance,
            }
            for split, gold, prediction, exact, over, under, distance in rows
        ]
    )

    metrics = _metrics_long(predictions)
    overall = metrics[
        (metrics["task_type"] == "qa")
        & (metrics["group_type"] == "overall")
    ].iloc[0]
    assert overall["n_total"] == 3
    assert overall["n_evaluable"] == 3
    assert overall["exact"] == pytest.approx(1 / 3)
    assert overall["over"] == pytest.approx(1 / 3)
    assert overall["under"] == pytest.approx(1 / 3)

    table = _table2({"model_id": "test-model"}, metrics, scope="unit test")
    assert table.loc[0, "QA N"] == 3
    assert table.loc[0, "QA Exact"] == "0.333"
    assert table.loc[0, "QA Over"] == "0.333"
    assert table.loc[0, "QA Under"] == "0.333"
    assert _paper_value(2 / 3) == "0.667"


def test_model_registry_exposes_paper_configs_and_stable_fingerprints() -> None:
    registry = ModelRegistry()
    target = registry.get("gpt-5-mini")
    judge = registry.get_judge("paper-gpt-4.1")

    assert target.api_model == "gpt-5-mini"
    assert target.endpoint == "chat_completions"
    assert target.token_parameter == "max_completion_tokens"
    assert target.send_temperature is False
    assert judge.model.api_model == "gpt-4.1"
    assert judge.model.temperature == 0
    assert judge.model.max_output_tokens == 1024
    assert replace(target, max_output_tokens=target.max_output_tokens + 1).fingerprint != target.fingerprint
    with pytest.raises(ValueError, match="Unknown model"):
        registry.get("not-configured")


def _manifest(
    model: ModelConfig,
    *,
    run_id: str,
    selected_cases: int = 1,
    samples: int = 1,
    tasks: tuple[str, ...] = ("qa", "conv"),
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "manifest_fingerprint": f"fingerprint-{run_id}",
        "model_id": model.id,
        "provider": model.provider,
        "api_model": model.api_model,
        "model_config": model.as_dict(),
        "benchmark_path": "/synthetic/benchmark.csv",
        "benchmark_sha256": "benchmark-sha",
        "tasks": list(tasks),
        "samples": samples,
        "selected_cases": selected_cases,
        "expected_generations": selected_cases * samples * len(tasks),
        "selection": {"fixture": True},
    }


def _generation_row(
    *,
    run_id: str,
    model: ModelConfig,
    source_id: str,
    task_type: str,
    sample_idx: int,
    gold: str,
    label: str,
) -> dict[str, object]:
    response = (
        f"REASONING: fixture\nACUITY: {label}"
        if task_type == "qa"
        else f"Synthetic recommendation {source_id}-{sample_idx}"
    )
    prompt = (
        "Synthetic QA prompt"
        if task_type == "qa"
        else json.dumps([{"role": "user", "content": "Synthetic case"}])
    )
    return {
        "run_id": run_id,
        "case_id": f"synthetic:{source_id}",
        "dataset": "synthetic",
        "source_id": source_id,
        "task_type": task_type,
        "sample_idx": sample_idx,
        "normalized_label": gold,
        "split": "primary",
        "mapping_method": "fixture",
        "is_edge_case": 0,
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "model_config_sha256": model.fingerprint,
        "response": response,
        "response_sha256": sha256_text(response),
        "parsed_label": label if task_type == "qa" else None,
        "parse_ok": 1,
        "status": "ok",
        "error": None,
        "attempts": 1,
        "finish_reason": "stop",
        "response_id": f"response-{source_id}-{task_type}-{sample_idx}",
        "returned_model": model.api_model,
        "input_tokens": 10,
        "cached_input_tokens": 2,
        "output_tokens": 4,
        "reasoning_tokens": 1,
        "total_tokens": 14,
        "latency_ms": 1.5,
        "rate_limit_json": "{}",
        "provider_metadata_json": "{}",
    }


def _judgment_row(
    generation: dict[str, object], *, judge_id: str, judge_label: str
) -> dict[str, object]:
    response = f"REASONING: fixture judge\nACUITY: {judge_label}"
    prompt = "Synthetic judge prompt"
    return {
        "run_id": generation["run_id"],
        "case_id": generation["case_id"],
        "dataset": generation["dataset"],
        "source_id": generation["source_id"],
        "sample_idx": generation["sample_idx"],
        "judge_id": judge_id,
        "judge_config_sha256": ModelRegistry().get_judge(judge_id).fingerprint,
        "generation_response_sha256": generation["response_sha256"],
        "judge_prompt": prompt,
        "judge_prompt_sha256": sha256_text(prompt),
        "response": response,
        "response_sha256": sha256_text(response),
        "judge_label": judge_label,
        "judge_reasoning": "fixture judge",
        "parse_ok": 1,
        "status": "ok",
        "error": None,
        "attempts": 1,
        "finish_reason": "stop",
        "response_id": f"judge-{generation['source_id']}-{generation['sample_idx']}",
        "returned_model": "gpt-4.1",
        "input_tokens": 12,
        "cached_input_tokens": 0,
        "output_tokens": 3,
        "reasoning_tokens": 0,
        "total_tokens": 15,
        "latency_ms": 2.0,
        "rate_limit_json": "{}",
        "provider_metadata_json": "{}",
    }


def test_sqlite_store_resumes_successes_and_upserts_judge_cache(tmp_path: Path) -> None:
    model = ModelRegistry().get("gpt-5-mini")
    database = tmp_path / "evaluation.sqlite3"
    manifest = _manifest(model, run_id="resume-test")

    with EvaluationStore(database) as store:
        assert store.ensure_run(manifest) is True
        assert store.ensure_run(manifest) is False
        generation = _generation_row(
            run_id="resume-test",
            model=model,
            source_id="1",
            task_type="qa",
            sample_idx=0,
            gold="A",
            label="A",
        )
        store.upsert_generation(generation)
        assert store.successful_generation_keys("resume-test") == {
            ("synthetic", "1", "qa", 0)
        }

        judgment = _judgment_row(
            generation, judge_id="paper-gpt-4.1", judge_label="A"
        )
        store.upsert_judgment(judgment)
        judgment["judge_label"] = "B"
        store.upsert_judgment(judgment)
        cached = store.judgment_records("resume-test", "paper-gpt-4.1")
        assert len(cached) == 1
        assert cached[("synthetic", "1", 0)]["judge_label"] == "B"

        conflicting = dict(manifest)
        conflicting["manifest_fingerprint"] = "different"
        with pytest.raises(ValueError, match="already exists with a different"):
            store.ensure_run(conflicting)

    with EvaluationStore(database) as reopened:
        assert reopened.get_run("resume-test")["model_id"] == "gpt-5-mini"


def test_generate_report_from_complete_synthetic_run(tmp_path: Path) -> None:
    registry = ModelRegistry()
    model = registry.get("gpt-5-mini")
    judge_id = "paper-gpt-4.1"
    run_id = "synthetic-report"
    database = tmp_path / "evaluation.sqlite3"
    samples = 2
    cases = {
        "a": ("A", "A", "A"),
        "b": ("B", "C", "A"),
        "c": ("D", "C", "D"),
    }

    with EvaluationStore(database) as store:
        store.ensure_run(
            _manifest(
                model,
                run_id=run_id,
                selected_cases=len(cases),
                samples=samples,
            )
        )
        for source_id, (gold, qa_label, conv_label) in cases.items():
            for sample_idx in range(samples):
                qa = _generation_row(
                    run_id=run_id,
                    model=model,
                    source_id=source_id,
                    task_type="qa",
                    sample_idx=sample_idx,
                    gold=gold,
                    label=qa_label,
                )
                conv = _generation_row(
                    run_id=run_id,
                    model=model,
                    source_id=source_id,
                    task_type="conv",
                    sample_idx=sample_idx,
                    gold=gold,
                    label=conv_label,
                )
                store.upsert_generation(qa)
                store.upsert_generation(conv)
                store.upsert_judgment(
                    _judgment_row(conv, judge_id=judge_id, judge_label=conv_label)
                )

    destination = generate_report(
        run_id=run_id,
        store_path=database,
        output_root=tmp_path / "reports",
        judge_id=judge_id,
    )
    table = pd.read_csv(destination / "tables/table2.csv", dtype=str)
    assert table.loc[0, "QA N"] == "3"
    assert table.loc[0, "QA Exact"] == "0.333"
    assert table.loc[0, "QA Over"] == "0.333"
    assert table.loc[0, "QA Under"] == "0.333"
    assert table.loc[0, "Conv Exact"] == "0.667"
    assert table.loc[0, "Conv Over"] == "0.000"
    assert table.loc[0, "Conv Under"] == "0.333"

    report_manifest = json.loads(
        (destination / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert report_manifest["report_complete"] is True
    assert report_manifest["generation_status"] == {"expected": 12, "ok": 12}
    assert report_manifest["judge_status"] == {"expected": 6, "ok": 6}
    assert (destination / "exports/case_predictions.parquet").exists()
    assert (destination / "tables/usage_and_cost.csv").exists()
