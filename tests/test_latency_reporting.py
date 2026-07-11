from __future__ import annotations

import json

import pandas as pd
import pytest

from acuitybench.models import ModelRegistry
from acuitybench.reporting import (
    PAPER_TARGET_API_IDENTIFIERS,
    _assert_comparable,
    _assert_latency_profiles_comparable,
    _comparison_contract,
    _effective_retry_completion_cap,
    _execution_summary,
    _frontier_latency_macro,
    _latency_summary,
    _returned_service_tier_provenance,
    _successful_generation_latency_profile,
    _usage_summary,
    _validated_client_latency_profile,
    _verified_reasoning_total,
)


def test_paper_api_identifiers_are_pinned_independently_of_run_aliases() -> None:
    assert PAPER_TARGET_API_IDENTIFIERS == {
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5.4": "gpt-5.4",
        "gpt-4.1": "gpt-4.1",
    }


def test_latency_summary_uses_explicit_linear_percentiles_and_coverage() -> None:
    generations = pd.DataFrame(
        [
            {
                "task_type": "qa",
                "status": "ok",
                "timing_source": "instrumented_stream",
                "request_wall_ms": value,
                "service_latency_ms": value + 1,
                "ttft_ms": value / 2,
                "server_processing_ms": value - 1,
            }
            for value in (10, 20, 30, 40, 50)
        ]
    )
    summary = _latency_summary(generations, pd.DataFrame())
    request = summary[summary["metric"] == "request_wall_ms"].iloc[0]
    ttft = summary[summary["metric"] == "ttft_ms"].iloc[0]

    assert request["n_total"] == 5
    assert request["n_success"] == 5
    assert request["n_measured"] == 5
    assert request["coverage"] == 1
    assert request["p50_ms"] == 30
    assert request["p95_ms"] == pytest.approx(48)
    assert request["p99_ms"] == pytest.approx(49.6)
    assert ttft["p50_ms"] == 15


def test_legacy_provider_processing_is_not_mislabeled_as_request_latency() -> None:
    generations = pd.DataFrame(
        [
            {
                "task_type": "qa",
                "status": "ok",
                "timing_source": "legacy_aggregate",
                "total_duration_ms": 900_000,
                "provider_metadata_json": json.dumps(
                    {"openai_processing_ms": value}
                ),
            }
            for value in (800, 1200)
        ]
    )
    summary = _latency_summary(generations, pd.DataFrame())
    request = summary[summary["metric"] == "request_wall_ms"].iloc[0]
    processing = summary[summary["metric"] == "server_processing_ms"].iloc[0]
    total = summary[summary["metric"] == "total_duration_ms"].iloc[0]

    assert request["n_measured"] == 0
    assert pd.isna(request["p50_ms"])
    assert processing["n_measured"] == 2
    assert processing["p50_ms"] == 1000
    assert processing["p95_ms"] == pytest.approx(1180)
    assert processing["source"] == "provider_header"
    assert total["p50_ms"] == 900_000
    assert total["timing_sources"] == "legacy_aggregate"


def test_target_and_judge_latency_remain_separate() -> None:
    generations = pd.DataFrame(
        [
            {
                "task_type": task,
                "status": "ok",
                "timing_source": "instrumented_stream",
                "service_latency_ms": value,
            }
            for task, value in (("qa", 10), ("conv", 20))
        ]
    )
    judgments = pd.DataFrame(
        [
            {
                "status": "ok",
                "timing_source": "instrumented_stream",
                "service_latency_ms": 30,
            }
        ]
    )
    summary = _latency_summary(generations, judgments)
    service = summary[summary["metric"] == "service_latency_ms"]
    assert {
        (row.phase, row.task_type, row.p50_ms)
        for row in service.itertuples(index=False)
    } == {("target", "qa", 10), ("target", "conv", 20), ("judge", "conv", 30)}


def test_usage_summary_counts_all_attempts_and_flags_missing_usage() -> None:
    generations = pd.DataFrame(
        [
            {
                "status": "ok",
                "latest_execution_id": "execution-1",
                "returned_model": "snapshot",
                # This terminal parent duplicates attempt 2 and must not be added.
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "output_tokens": 50,
                "reasoning_tokens": 0,
                "finish_reason": "stop",
            },
            {
                "status": "ok",
                "latest_execution_id": None,
                "returned_model": "legacy-snapshot",
                "input_tokens": 7,
                "cached_input_tokens": 1,
                "output_tokens": 3,
                "reasoning_tokens": 0,
                "finish_reason": "stop",
            },
        ]
    )
    attempts = pd.DataFrame(
        [
            {
                "phase": "generation",
                "execution_id": "execution-1",
                "outcome": "retry_length",
                "finish_reason": "length",
                "response_id": "retry",
                "input_tokens": 10,
                "cached_input_tokens": 2,
                "output_tokens": 4,
                "reasoning_tokens": 1,
            },
            {
                "phase": "generation",
                "execution_id": "execution-1",
                "outcome": "success",
                "finish_reason": "stop",
                "response_id": "final",
                "input_tokens": None,
                "cached_input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
            },
        ]
    )
    registry = ModelRegistry()
    summary = _usage_summary(
        generations,
        pd.DataFrame(),
        attempts,
        target=registry.get("gpt-5-mini"),
        judge=registry.get_judge("paper-gpt-4.1").model,
        judge_id="paper-gpt-4.1",
    )
    target = summary.iloc[0]

    assert target["attempts_tracked"] == 2
    assert target["legacy_parent_records"] == 1
    assert target["input_tokens"] == 17
    assert target["output_tokens"] == 7
    assert target["usage_records_expected"] == 3
    assert target["usage_records_complete"] == 2
    assert target["missing_usage_records"] == 1
    assert target["usage_coverage"] == pytest.approx(2 / 3)
    assert target["reasoning_token_coverage"] == pytest.approx(2 / 3)
    assert pd.isna(target["reasoning_tokens"])
    assert target["reasoning_tokens_observed"] == 1
    assert pd.isna(target["reasoning_tokens_per_successful_call"])
    assert target["cost_completeness"] == (
        "partial_missing_usage_or_cache_breakdown"
    )


def test_cost_completeness_requires_cache_breakdown_and_post_dispatch_usage() -> None:
    generations = pd.DataFrame(
        [
            {
                "status": "ok",
                "latest_execution_id": "execution-1",
                "returned_model": "snapshot",
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "output_tokens": 2,
                "reasoning_tokens": 0,
                "finish_reason": "stop",
            }
        ]
    )
    attempts = pd.DataFrame(
        [
            {
                "phase": "generation",
                "execution_id": "execution-1",
                "outcome": "success",
                "finish_reason": "stop",
                "response_id": "with-cache-gap",
                "http_status": 200,
                "first_event_ms": 1.0,
                "input_tokens": 10,
                "cached_input_tokens": None,
                "output_tokens": 2,
                "reasoning_tokens": 0,
            },
            {
                "phase": "generation",
                "execution_id": "execution-1",
                "outcome": "retry_error",
                "finish_reason": None,
                "response_id": None,
                "http_status": 200,
                "first_event_ms": None,
                "input_tokens": None,
                "cached_input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
            },
        ]
    )
    registry = ModelRegistry()
    row = _usage_summary(
        generations,
        pd.DataFrame(),
        attempts,
        target=registry.get("gpt-5-mini"),
        judge=registry.get_judge("paper-gpt-4.1").model,
        judge_id="paper-gpt-4.1",
    ).iloc[0]

    assert row["usage_records_expected"] == 2
    assert row["missing_usage_records"] == 1
    assert row["missing_cache_breakdown_records"] == 2
    assert row["usage_coverage"] == 0.5
    assert row["cache_breakdown_coverage"] == 0
    assert row["reasoning_token_coverage"] == 0.5
    assert pd.isna(row["reasoning_tokens"])
    assert row["reasoning_tokens_observed"] == 0
    assert row["cost_completeness"] == (
        "partial_missing_usage_or_cache_breakdown"
    )


def test_execution_summary_reports_throughput_retries_and_abnormal_work() -> None:
    executions = pd.DataFrame(
        [
            {
                "execution_id": "execution-1",
                "phase": "generation",
                "profile_id": "model",
                "status": "cancelled",
                "streaming": 1,
                "configured_concurrency": 4,
                "pending_count": 4,
                "success_count": 2,
                "failure_count": 0,
                "cancelled_count": 1,
                "unpersisted_count": 1,
                "started_at": "2026-01-01T00:00:00+00:00",
                "ended_at": "2026-01-01T00:00:02+00:00",
            }
        ]
    )
    attempts = pd.DataFrame(
        [
            {
                "execution_id": "execution-1",
                "request_key": "a",
                "attempt_index": 1,
                "outcome": "retry_error",
            },
            {
                "execution_id": "execution-1",
                "request_key": "a",
                "attempt_index": 2,
                "outcome": "success",
            },
            {
                "execution_id": "execution-1",
                "request_key": "b",
                "attempt_index": 1,
                "outcome": "terminal_error",
            },
        ]
    )
    row = _execution_summary(executions, attempts).iloc[0]
    assert row["elapsed_seconds"] == 2
    assert row["persisted_requests_per_second"] == 1
    assert row["requests_observed"] == 2
    assert row["attempts_total"] == 3
    assert row["retry_attempts"] == 1
    assert row["attempts_per_request"] == 1.5
    assert row["terminal_error_attempts"] == 1


def test_judge_usage_is_profile_scoped_even_without_parent_rows() -> None:
    attempts = pd.DataFrame(
        [
            {
                "phase": "judge",
                "judge_id": "paper-gpt-4.1",
                "execution_id": "selected",
                "outcome": "success",
                "finish_reason": "stop",
                "response_id": "selected-response",
                "returned_model": "gpt-4.1-snapshot",
                "http_status": 200,
                "first_event_ms": 1.0,
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "output_tokens": 2,
                "reasoning_tokens": 0,
            },
            {
                "phase": "judge",
                "judge_id": "other-judge",
                "execution_id": "other",
                "outcome": "success",
                "finish_reason": "stop",
                "response_id": "other-response",
                "returned_model": "other-snapshot",
                "http_status": 200,
                "first_event_ms": 1.0,
                "input_tokens": 1000,
                "cached_input_tokens": 0,
                "output_tokens": 1000,
                "reasoning_tokens": 0,
            },
        ]
    )
    registry = ModelRegistry()
    summary = _usage_summary(
        pd.DataFrame(),
        pd.DataFrame(),
        attempts,
        target=registry.get("gpt-5-mini"),
        judge=registry.get_judge("paper-gpt-4.1").model,
        judge_id="paper-gpt-4.1",
    )

    row = summary.iloc[0]
    assert row["phase"] == "judge"
    assert row["calls"] == 0
    assert row["attempts_tracked"] == 1
    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 2
    assert row["returned_models"] == "gpt-4.1-snapshot"


def test_latency_profile_comes_only_from_successful_parent_execution_ids() -> None:
    generations = pd.DataFrame(
        [
            {"status": "ok", "latest_execution_id": "used-a"},
            {"status": "ok", "latest_execution_id": "used-b"},
            {"status": "failed", "latest_execution_id": "unused"},
        ]
    )
    executions = pd.DataFrame(
        [
            {
                "execution_id": "used-a",
                "phase": "generation",
                "streaming": 1,
                "configured_concurrency": 20,
            },
            {
                "execution_id": "used-b",
                "phase": "generation",
                "streaming": 1,
                "configured_concurrency": 20,
            },
            {
                "execution_id": "unused",
                "phase": "generation",
                "streaming": 0,
                "configured_concurrency": 1,
            },
            {
                "execution_id": "judge",
                "phase": "judge",
                "streaming": 0,
                "configured_concurrency": 99,
            },
        ]
    )

    profile = _successful_generation_latency_profile(generations, executions)

    assert profile == {
        "source": "successful_generation_latest_execution_id",
        "successful_generation_count": 2,
        "profiled_successful_generation_count": 2,
        "execution_profile_coverage": 1.0,
        "streaming_values": [True],
        "concurrency_values": [20],
    }

    executions.loc[executions["execution_id"] == "used-b", "streaming"] = 0
    executions.loc[
        executions["execution_id"] == "used-b", "configured_concurrency"
    ] = 10
    mixed = _successful_generation_latency_profile(generations, executions)
    assert mixed["streaming_values"] == [False, True]
    assert mixed["concurrency_values"] == [10, 20]


def test_returned_service_tier_provenance_is_success_scoped_and_covered() -> None:
    generations = pd.DataFrame(
        [
            {
                "status": "ok",
                "provider_metadata_json": json.dumps(
                    {"returned_service_tier": "default"}
                ),
            },
            {
                "status": "ok",
                "provider_metadata_json": "{}",
            },
            {
                "status": "failed",
                "provider_metadata_json": json.dumps(
                    {"returned_service_tier": "priority"}
                ),
            },
        ]
    )

    provenance = _returned_service_tier_provenance(generations)

    assert provenance["returned_service_tiers"] == ["default"]
    assert provenance["returned_service_tier_records"] == 1
    assert provenance["returned_service_tier_coverage"] == 0.5


def test_reasoning_total_distinguishes_verified_zero_from_missing() -> None:
    assert _verified_reasoning_total(
        pd.DataFrame(
            [{"reasoning_token_coverage": 1.0, "reasoning_tokens": 0}]
        )
    ) == (0.0, 1.0)
    assert _verified_reasoning_total(
        pd.DataFrame(
            [{"reasoning_token_coverage": 0.5, "reasoning_tokens": 12}]
        )
    ) == (None, 0.5)
    assert _verified_reasoning_total(
        pd.DataFrame([{"reasoning_tokens": 0}])
    ) == (None, None)


def test_retry_completion_cap_preserves_legacy_runner_default() -> None:
    assert _effective_retry_completion_cap({}, {}) == (
        8192,
        "legacy_runner_default",
    )
    assert _effective_retry_completion_cap(
        {}, {"max_retry_output_tokens": 4096}
    ) == (4096, "run_model_config")
    assert _effective_retry_completion_cap(
        {"max_retry_completion_tokens": 2048},
        {"max_retry_output_tokens": 4096},
    ) == (2048, "inference_contract")


def test_client_latency_profile_requires_complete_singleton_provenance() -> None:
    valid = {
        "run_id": "valid",
        "streaming_values": (True,),
        "concurrency_values": (20,),
        "execution_profile_coverage": 1.0,
        "configured_service_tier": "default",
        "returned_service_tiers": ("default",),
        "returned_service_tier_coverage": 1.0,
    }
    assert _validated_client_latency_profile(**valid) == (True, 20, "default")

    invalid_cases = [
        ({"streaming_values": ()}, "streaming mode"),
        ({"streaming_values": (True, False)}, "streaming mode"),
        ({"concurrency_values": ()}, "configured concurrency"),
        ({"concurrency_values": (20, 40)}, "configured concurrency"),
        ({"execution_profile_coverage": 0.99}, "execution-profile coverage"),
        ({"configured_service_tier": None}, "configured service tier"),
        ({"returned_service_tiers": ()}, "returned service tier"),
        (
            {"returned_service_tier_coverage": 0.5},
            "service-tier coverage",
        ),
        (
            {"returned_service_tiers": ("priority",)},
            "configured and returned service tiers",
        ),
    ]
    for changes, message in invalid_cases:
        candidate = {**valid, **changes, "run_id": message}
        with pytest.raises(ValueError, match=message):
            _validated_client_latency_profile(**candidate)


def test_client_latency_profiles_must_match_across_runs() -> None:
    _assert_latency_profiles_comparable(
        {"mini": (True, 20, "default"), "frontier": (True, 20, "default")}
    )
    with pytest.raises(ValueError, match="share streaming mode"):
        _assert_latency_profiles_comparable(
            {
                "mini": (True, 20, "default"),
                "frontier": (True, 40, "default"),
            }
        )


def test_frontier_latency_requires_full_two_format_coverage_and_valid_source() -> None:
    latency = pd.DataFrame(
        [
            {
                "phase": "target",
                "task_type": task_type,
                "metric": "service_latency_ms",
                "coverage": 1.0,
                "n_success": 10,
                "n_measured": 10,
                "timing_sources": "instrumented_stream",
                "p95_ms": value,
            }
            for task_type, value in (("qa", 100), ("conv", 200))
        ]
    )
    assert _frontier_latency_macro(
        latency,
        metric="service_latency_ms",
        percentile="p95_ms",
        required_sources={"instrumented_stream", "instrumented_nonstream"},
    ) == pytest.approx(150)

    latency.loc[latency["task_type"] == "conv", "coverage"] = 0.9
    assert _frontier_latency_macro(
        latency,
        metric="service_latency_ms",
        percentile="p95_ms",
        required_sources={"instrumented_stream", "instrumented_nonstream"},
    ) is None

    latency.loc[:, "coverage"] = 1.0
    assert _frontier_latency_macro(
        latency,
        metric="service_latency_ms",
        percentile="p95_ms",
        required_sources={"legacy_aggregate"},
    ) is None

    latency.loc[:, "timing_sources"] = (
        "instrumented_stream,instrumented_nonstream"
    )
    assert _frontier_latency_macro(
        latency,
        metric="service_latency_ms",
        percentile="p95_ms",
        required_sources={"instrumented_stream", "instrumented_nonstream"},
    ) is None


def test_comparison_contract_rejects_mixed_benchmark_scope() -> None:
    reference = {
        "benchmark_sha256": "benchmark",
        "case_ids_sha256": "cases",
        "selected_cases": 914,
        "tasks": ("conv", "qa"),
        "samples": 5,
        "expected_generations": 9140,
        "main_table_filter": "primary-clear",
        "aggregation": "mode",
        "parser": "acuity-parser-v1",
        "judge_id": "judge",
        "judge_config_fingerprint": "judge-config",
        "judge_assets_fingerprint": "judge-assets",
        "returned_judge_models": ("judge-snapshot",),
    }
    _assert_comparable(
        reference,
        dict(reference),
        reference_run_id="a",
        candidate_run_id="b",
    )

    candidate = dict(reference)
    candidate.update(
        {
            "case_ids_sha256": "smoke-cases",
            "selected_cases": 2,
            "samples": 1,
        }
    )
    with pytest.raises(ValueError, match=r"case_ids_sha256.*selected_cases.*samples"):
        _assert_comparable(
            reference,
            candidate,
            reference_run_id="full",
            candidate_run_id="smoke",
        )


def test_comparison_contract_pins_judge_snapshot_parser_and_assets() -> None:
    manifest = {
        "run": {
            "benchmark_sha256": "benchmark",
            "selection": {"case_ids_sha256": "cases"},
            "selected_cases": 914,
            "tasks": ["qa", "conv"],
            "samples": 5,
            "expected_generations": 9140,
        },
        "paper_contract": {
            "main_table_filter": "primary-clear",
            "aggregation": "mode",
            "parser": {"version": 1, "pattern": "acuity-parser-v1"},
            "judge_id": "judge",
        },
        "judge_config": {"fingerprint": "judge-config"},
        "judge_assets": {"fingerprint": "judge-assets"},
        "returned_judge_models": ["judge-snapshot"],
    }
    contract = _comparison_contract(manifest)
    assert contract["parser"] == {
        "version": 1,
        "pattern": "acuity-parser-v1",
    }
    assert contract["judge_config_fingerprint"] == "judge-config"
    assert contract["judge_assets_fingerprint"] == "judge-assets"
    assert contract["returned_judge_models"] == ("judge-snapshot",)

    changed = dict(contract)
    changed["returned_judge_models"] = ("different-snapshot",)
    with pytest.raises(ValueError, match="returned_judge_models"):
        _assert_comparable(
            contract,
            changed,
            reference_run_id="a",
            candidate_run_id="b",
        )

    del manifest["judge_assets"]
    with pytest.raises(ValueError, match="judge_assets_fingerprint"):
        _comparison_contract(manifest)

    manifest["judge_assets"] = {"fingerprint": "judge-assets"}
    manifest["returned_judge_models"] = []
    with pytest.raises(ValueError, match="exactly one returned judge"):
        _comparison_contract(manifest)
