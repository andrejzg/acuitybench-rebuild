from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import acuitybench.evaluation as evaluation_module
from acuitybench.distributional import (
    evaluate_panel_distribution,
    jensen_shannon,
    wasserstein_1,
)
from acuitybench.evaluation import (
    GenerationTask,
    build_judge_prompt,
    extract_label,
    extract_reasoning,
    load_judge_assets,
    prepare_run,
    run_inference_async,
    sha256_text,
)
from acuitybench.models import ModelConfig, ModelRegistry
from acuitybench.providers.base import CompletionResult
from acuitybench.reporting import (
    _metrics_long,
    _mode_severe,
    _paper_comparison,
    _paper_value,
    _table2,
    combine_reports,
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


def test_paper_comparison_uses_model_specific_published_baseline() -> None:
    metrics = pd.DataFrame(
        [
            {
                "group_type": "overall",
                "task_type": task,
                "exact": 0.8,
                "over": 0.1,
                "under": 0.1,
            }
            for task in ("qa", "conv")
        ]
    )

    comparison = _paper_comparison(
        {
            "model_id": "gpt-5.4",
            "selected_cases": 914,
            "samples": 5,
            "tasks": ["qa", "conv"],
        },
        metrics,
    )

    exact = comparison[comparison["metric"] == "exact"].set_index("task_type")
    assert exact.loc["qa", "published"] == pytest.approx(0.772)
    assert exact.loc["conv", "published"] == pytest.approx(0.772)
    assert exact.loc["qa", "delta"] == pytest.approx(0.028)

    smoke = _paper_comparison(
        {
            "model_id": "gpt-5.4",
            "selected_cases": 1,
            "samples": 1,
            "tasks": ["qa", "conv"],
        },
        metrics,
    )
    assert smoke.empty


def test_distributional_distances_use_paper_conventions() -> None:
    at_home = [1.0, 0.0, 0.0, 0.0]
    emergency = [0.0, 0.0, 0.0, 1.0]
    assert jensen_shannon(at_home, at_home) == pytest.approx(0)
    assert jensen_shannon(at_home, emergency) == pytest.approx(0.69314718056)
    assert wasserstein_1(at_home, emergency) == pytest.approx(3.0)


def test_empty_distributional_subset_returns_typed_empty_result() -> None:
    generations = pd.DataFrame(
        [
            {
                "run_id": "run",
                "dataset": "synthetic",
                "source_id": "1",
                "sample_idx": 0,
                "task_type": "qa",
                "split": "primary",
                "parsed_label": "A",
            }
        ]
    )
    benchmark = pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "source_id": "1",
                "normalized_label": "A",
                **{f"anon_label_{index}": "A" for index in range(1, 6)},
            }
        ]
    )
    result, summary = evaluate_panel_distribution(
        generations=generations,
        judgments=pd.DataFrame(),
        benchmark=benchmark,
        task_type="qa",
        split="ambiguous",
    )
    assert result.empty
    assert "mode_prediction" in result.columns
    assert summary["n"] == 0
    assert summary["n_evaluable"] == 0


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
    frontier = registry.get("gpt-5.4")
    judge = registry.get_judge("paper-gpt-4.1")

    assert target.api_model == "gpt-5-mini"
    assert target.endpoint == "chat_completions"
    assert target.token_parameter == "max_completion_tokens"
    assert target.send_temperature is False
    assert target.reasoning_effort == "medium"
    assert target.max_retry_output_tokens == 4096
    assert target.service_tier == "default"
    assert frontier.api_model == "gpt-5.4"
    assert frontier.temperature == 1
    assert frontier.send_temperature is True
    assert frontier.reasoning_effort == "none"
    assert frontier.max_output_tokens == 4096
    assert frontier.max_retry_output_tokens == 4096
    assert frontier.service_tier == "default"
    assert judge.model.api_model == "gpt-4.1"
    assert judge.model.temperature == 0
    assert judge.model.max_output_tokens == 1024
    assert replace(target, max_output_tokens=target.max_output_tokens + 1).fingerprint != target.fingerprint
    assert replace(target, reasoning_effort="low").fingerprint != target.fingerprint
    with pytest.raises(ValueError, match="Unknown model"):
        registry.get("not-configured")


def test_stream_transport_is_execution_provenance_not_run_identity(
    tmp_path: Path,
) -> None:
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
    database = tmp_path / "runs.sqlite3"
    model = ModelRegistry().get("gpt-5-mini")
    with EvaluationStore(database) as store:
        first_id, _ = prepare_run(
            store=store,
            model=model,
            benchmark_path=benchmark,
            tasks=("qa",),
            samples=1,
            datasets=None,
            limit=None,
            run_id="legacy-compatible",
            stream=False,
        )
        fingerprint = store.get_run(first_id)["manifest_fingerprint"]
        second_id, _ = prepare_run(
            store=store,
            model=model,
            benchmark_path=benchmark,
            tasks=("qa",),
            samples=1,
            datasets=None,
            limit=None,
            run_id="legacy-compatible",
            stream=True,
        )
        assert store.get_run(second_id)["manifest_fingerprint"] == fingerprint

    assert first_id == second_id == "legacy-compatible"


def test_cancelled_generation_finalizes_execution_and_parent_run_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CancelledProvider:
        async def complete(self, **_: object) -> object:
            raise asyncio.CancelledError

        async def close(self) -> None:
            return None

    model = ModelRegistry().get("gpt-5-mini")
    database = tmp_path / "cancelled.sqlite3"
    with EvaluationStore(database) as store:
        store.ensure_run(
            _manifest(
                model,
                run_id="cancelled-run",
                selected_cases=1,
                samples=1,
                tasks=("qa",),
            )
        )
        task = GenerationTask(
            run_id="cancelled-run",
            case_id="synthetic:1",
            dataset="synthetic",
            source_id="1",
            task_type="qa",
            sample_idx=0,
            normalized_label="A",
            split="primary",
            mapping_method="fixture",
            is_edge_case=False,
            prompt="case",
            prompt_sha256=sha256_text("case"),
        )
        monkeypatch.setattr(
            evaluation_module,
            "get_provider",
            lambda _: CancelledProvider(),
        )

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                run_inference_async(
                    store=store,
                    run_id="cancelled-run",
                    tasks=[task],
                    model=model,
                    concurrency=1,
                    stream=True,
                )
            )

        run = store.get_run("cancelled-run")
        execution = store.connection.execute(
            "SELECT * FROM run_executions WHERE run_id='cancelled-run'"
        ).fetchone()
    assert run["status"] == "generation_cancelled"
    assert execution["status"] == "cancelled"
    assert execution["cancelled_count"] == 1


def test_partial_generation_task_list_cannot_mark_whole_run_generated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SuccessProvider:
        async def complete(self, **_: object) -> CompletionResult:
            return CompletionResult(text="ACUITY: A", finish_reason="stop")

        async def close(self) -> None:
            return None

    model = ModelRegistry().get("gpt-5-mini")
    database = tmp_path / "partial.sqlite3"
    with EvaluationStore(database) as store:
        store.ensure_run(
            _manifest(
                model,
                run_id="partial-run",
                selected_cases=1,
                samples=1,
                tasks=("qa", "conv"),
            )
        )
        task = GenerationTask(
            run_id="partial-run",
            case_id="synthetic:1",
            dataset="synthetic",
            source_id="1",
            task_type="qa",
            sample_idx=0,
            normalized_label="A",
            split="primary",
            mapping_method="fixture",
            is_edge_case=False,
            prompt="case",
            prompt_sha256=sha256_text("case"),
        )
        monkeypatch.setattr(
            evaluation_module,
            "get_provider",
            lambda _: SuccessProvider(),
        )
        result = asyncio.run(
            run_inference_async(
                store=store,
                run_id="partial-run",
                tasks=[task],
                model=model,
                concurrency=1,
                stream=True,
            )
        )
        run = store.get_run("partial-run")

    assert result == {"pending": 1, "completed": 1, "failed": 0}
    assert run["status"] == "generation_incomplete"


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
        "selection": {"fixture": True, "case_ids_sha256": "fixture-cases-sha"},
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
    latest_execution_id: str | None = None,
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
        "latest_execution_id": latest_execution_id,
        "timing_version": 2,
        "timing_source": "instrumented_stream",
        "queued_at": "2026-01-01T00:00:00+00:00",
        "request_started_at": "2026-01-01T00:00:00.001000+00:00",
        "first_token_at": "2026-01-01T00:00:00.001500+00:00",
        "response_completed_at": "2026-01-01T00:00:00.002000+00:00",
        "queue_wait_ms": 0.5,
        "request_wall_ms": 1.0,
        "request_wall_total_ms": 1.0,
        "service_latency_ms": 1.0,
        "retry_backoff_ms": 0.0,
        "first_event_ms": 0.25,
        "ttft_ms": 0.5,
        "time_after_first_token_ms": 0.5,
        "total_duration_ms": 1.5,
        "server_processing_ms": 0.4,
        "rate_limit_json": "{}",
        "provider_metadata_json": json.dumps(
            {"returned_service_tier": "default"}
            if latest_execution_id is not None
            else {}
        ),
    }


def _judgment_row(
    generation: dict[str, object], *, judge_id: str, judge_label: str
) -> dict[str, object]:
    response = f"REASONING: fixture judge\nACUITY: {judge_label}"
    if generation["task_type"] == "conv":
        rubric, template = load_judge_assets()
        prompt = build_judge_prompt(generation, rubric, template)
    else:
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
        "timing_version": 2,
        "timing_source": "instrumented_stream",
        "queued_at": "2026-01-01T00:00:00+00:00",
        "request_started_at": "2026-01-01T00:00:00.001000+00:00",
        "first_token_at": "2026-01-01T00:00:00.002000+00:00",
        "response_completed_at": "2026-01-01T00:00:00.003000+00:00",
        "queue_wait_ms": 0.5,
        "request_wall_ms": 1.5,
        "request_wall_total_ms": 1.5,
        "service_latency_ms": 1.5,
        "retry_backoff_ms": 0.0,
        "first_event_ms": 0.5,
        "ttft_ms": 1.0,
        "time_after_first_token_ms": 0.5,
        "total_duration_ms": 2.0,
        "server_processing_ms": 0.7,
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
        generation_execution_id = store.start_execution(
            run_id=run_id,
            phase="generation",
            profile_id=model.id,
            judge_id=None,
            provider=model.provider,
            api_model=model.api_model,
            endpoint=model.endpoint,
            config_sha256=model.fingerprint,
            concurrency=20,
            streaming=True,
            max_attempts=6,
            retry_policy={},
            runner_metadata={},
            task_count=len(cases) * samples * 2,
            cache_hit_count=0,
            pending_count=len(cases) * samples * 2,
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
                    latest_execution_id=generation_execution_id,
                )
                conv = _generation_row(
                    run_id=run_id,
                    model=model,
                    source_id=source_id,
                    task_type="conv",
                    sample_idx=sample_idx,
                    gold=gold,
                    label=conv_label,
                    latest_execution_id=generation_execution_id,
                )
                store.upsert_generation(qa)
                store.upsert_generation(conv)
                store.upsert_judgment(
                    _judgment_row(conv, judge_id=judge_id, judge_label=conv_label)
                )
        store.finish_execution(
            generation_execution_id,
            status="complete",
            success_count=len(cases) * samples * 2,
            failure_count=0,
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
    assert report_manifest["judge_config"]["fingerprint"] == (
        ModelRegistry().get_judge(judge_id).fingerprint
    )
    assert len(report_manifest["judge_assets"]["fingerprint"]) == 64
    assert len(report_manifest["judge_assets"]["rubric_sha256"]) == 64
    assert len(report_manifest["judge_assets"]["template_sha256"]) == 64
    inference_contract = report_manifest["inference_contract"]
    assert (
        inference_contract["paper_reported"]["api_identifier"]
        == "gpt-5-mini"
    )
    assert inference_contract["paper_reported"]["reasoning_effort"] is None
    assert inference_contract["paper_reported"]["service_tier"] is None
    assert inference_contract["paper_reported"]["streaming"] is None
    assert "Not reported" in (
        inference_contract["paper_reported"]["service_tier_note"]
    )
    assert (
        inference_contract["configured"]["reasoning_effort"] == "medium"
    )
    assert (
        inference_contract["configured"]["reasoning_effort_provenance"][
            "source_url"
        ]
        == "https://developers.openai.com/api/docs/guides/reasoning"
    )
    assert report_manifest["paper_baseline"]["arxiv_id"] == "2605.11398"
    assert report_manifest["paper_baseline"]["applicable_to_run"] is False
    assert (
        inference_contract["configured"]["max_retry_completion_tokens"]
        == 4096
    )
    assert inference_contract["configured"]["streaming_values"] == [True]
    assert inference_contract["configured"]["concurrency_values"] == [20]
    assert (
        inference_contract["observed"]["execution_profile_coverage"]
        == 1
    )
    assert inference_contract["observed"]["returned_service_tiers"] == [
        "default"
    ]
    assert (
        inference_contract["observed"]["returned_service_tier_coverage"]
        == 1
    )
    assert (destination / "exports/case_predictions.parquet").exists()
    assert (destination / "tables/usage_and_cost.csv").exists()
    assert (destination / "tables/latency_summary.csv").exists()
    assert (destination / "exports/run_executions.csv").exists()
    assert (destination / "exports/request_attempts.csv").exists()
    latency = pd.read_csv(destination / "tables/latency_summary.csv")
    qa_ttft = latency[
        (latency["phase"] == "target")
        & (latency["task_type"] == "qa")
        & (latency["metric"] == "ttft_ms")
    ].iloc[0]
    assert qa_ttft["n_measured"] == 6
    assert qa_ttft["p50_ms"] == pytest.approx(0.5)
    assert report_manifest["timing"]["primary_latency_metric"] == "service_latency_ms"
    assert report_manifest["timing"]["legacy_rows"] == 0

    comparison = combine_reports(
        run_ids=[run_id],
        results_root=tmp_path / "reports",
        destination=tmp_path / "comparison",
    )
    frontier = pd.read_csv(comparison / "frontier.csv")
    assert frontier.loc[0, "run_id"] == run_id
    assert frontier.loc[0, "average_exact"] == pytest.approx(0.5)
    assert frontier.loc[0, "accuracy_complete"]
    assert frontier.loc[0, "target_cost_per_1000_successful_calls_usd"] > 0
    assert frontier.loc[0, "reasoning_effort"] == "medium"
    assert frontier.loc[0, "max_completion_tokens"] == 4096
    assert frontier.loc[0, "max_retry_completion_tokens"] == 4096
    assert (
        frontier.loc[0, "max_retry_completion_tokens_basis"]
        == "inference_contract"
    )
    assert frontier.loc[0, "target_reasoning_tokens"] > 0
    assert frontier.loc[0, "latency_profile_execution_coverage"] == 1
    assert (
        frontier.loc[0, "latency_profile_returned_service_tier_coverage"]
        == 1
    )
    assert frontier.loc[0, "service_latency_p95_macro_ms"] == pytest.approx(1.0)
    assert frontier.loc[0, "qa_service_latency_p95_ms"] == pytest.approx(1.0)
    assert frontier.loc[0, "conv_service_latency_p95_ms"] == pytest.approx(1.0)
    assert frontier.loc[0, "latency_plot_p95_ms"] == pytest.approx(1.0)
    assert frontier.loc[0, "latency_plot_source"] == "client_service_latency"
    assert not frontier.loc[0, "latency_plot_is_proxy"]
    assert (comparison / "usage_and_cost.csv").exists()
    assert (comparison / "latency_summary.csv").exists()
    assert (comparison / "execution_summary.csv").exists()
    assert (comparison / "accuracy-vs-cost.svg").exists()
    assert (comparison / "accuracy-vs-latency.svg").exists()

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE judgments SET judge_prompt=?, judge_prompt_sha256=? "
            "WHERE rowid=(SELECT rowid FROM judgments LIMIT 1)",
            ("stale prompt", sha256_text("stale prompt")),
        )
    with pytest.raises(RuntimeError, match="stale rubric or judge prompt"):
        generate_report(
            run_id=run_id,
            store_path=database,
            output_root=tmp_path / "stale-reports",
            judge_id=judge_id,
        )


def test_generate_report_supports_qa_only_static_run(tmp_path: Path) -> None:
    model = ModelRegistry().get("gpt-5-mini")
    run_id = "static-qa-only"
    database = tmp_path / "static.sqlite3"
    manifest = _manifest(
        model,
        run_id=run_id,
        selected_cases=1,
        samples=1,
        tasks=("qa",),
    )
    manifest["experiment_contract"] = {
        "schema_version": "static-student-evaluation/v1",
        "strategy": "static_first",
    }
    with EvaluationStore(database) as store:
        store.ensure_run(manifest)
        store.upsert_generation(
            _generation_row(
                run_id=run_id,
                model=model,
                source_id="1",
                task_type="qa",
                sample_idx=0,
                gold="A",
                label="A",
            )
        )

    destination = generate_report(
        run_id=run_id,
        store_path=database,
        output_root=tmp_path / "reports",
    )

    table = pd.read_csv(destination / "tables/table2.csv", dtype=str)
    report_manifest = json.loads(
        (destination / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert table.loc[0, "QA Exact"] == "1.000"
    assert pd.isna(table.loc[0, "Conv Exact"])
    assert report_manifest["run"]["experiment_contract"]["strategy"] == "static_first"
    assert report_manifest["judge_status"] == {"expected": 0, "ok": 0}
