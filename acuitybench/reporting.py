"""Audit-friendly exports and paper-compatible AcuityBench tables."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from acuitybench.distributional import evaluate_panel_distribution
from acuitybench.evaluation import (
    build_judge_prompt,
    default_store_path,
    extract_label,
    extract_reasoning,
    label_parser_contract,
    load_judge_assets,
    sha256_text,
)
from acuitybench.models import ModelConfig, ModelRegistry
from acuitybench.sources import project_root, sha256_file
from acuitybench.store import EvaluationStore


LABELS = ("A", "B", "C", "D")
RANK = {label: index for index, label in enumerate(LABELS)}
PAPER_GPT5_MINI = {
    "qa": {"exact": 0.780, "over": 0.055, "under": 0.165},
    "conv": {"exact": 0.677, "over": 0.036, "under": 0.286},
}
PAPER_GPT5_4 = {
    "qa": {"exact": 0.772, "over": 0.142, "under": 0.085},
    "conv": {"exact": 0.772, "over": 0.049, "under": 0.178},
}
PAPER_MODEL_RESULTS = {
    "gpt-5-mini": PAPER_GPT5_MINI,
    "gpt-5.4": PAPER_GPT5_4,
}
PAPER_BASELINE_PROVENANCE = {
    "arxiv_id": "2605.11398",
    "source_url": "https://arxiv.org/pdf/2605.11398",
    "table": "Table 2, printed page 5",
    "published_decimal_places": 3,
    "accessed_at": "2026-07-11",
    "delta_definition": "fresh_run - published",
}
PAPER_TARGET_API_IDENTIFIERS = {
    # Keep paper provenance independent from the configured request alias. A
    # report must not make an arbitrary configured model look paper-reported.
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5.4": "gpt-5.4",
    "gpt-4.1": "gpt-4.1",
}
REASONING_EFFORT_PROVENANCE = {
    "gpt-5-mini": {
        "source_url": "https://developers.openai.com/api/docs/guides/reasoning",
        "accessed_at": "2026-07-11",
        "resolution": (
            "The paper and companion adapter omitted reasoning effort. The "
            "pre-GPT-5.1 documented default was medium, so this reconstruction "
            "pins medium rather than relying on a mutable alias default."
        ),
    },
    "gpt-5.4": {
        "source_url": "https://developers.openai.com/api/docs/models/gpt-5.4",
        "accessed_at": "2026-07-11",
        "resolution": (
            "The paper omitted reasoning effort. The GPT-5.4 model page "
            "documents none as the default, and temperature is supported only "
            "at none, so this reconstruction pins none."
        ),
    },
}


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _successful_generation_latency_profile(
    generations: pd.DataFrame,
    executions: pd.DataFrame,
) -> dict[str, Any]:
    """Resolve transport provenance from the rows that supply latency values."""
    successful = (
        generations[generations["status"] == "ok"].copy()
        if not generations.empty and "status" in generations
        else pd.DataFrame()
    )
    successful_count = len(successful)
    if successful_count == 0:
        return {
            "source": "successful_generation_latest_execution_id",
            "successful_generation_count": 0,
            "profiled_successful_generation_count": 0,
            "execution_profile_coverage": None,
            "streaming_values": [],
            "concurrency_values": [],
        }

    target_executions = (
        executions[executions["phase"] == "generation"].copy()
        if not executions.empty
        and {"phase", "execution_id"} <= set(executions.columns)
        else pd.DataFrame()
    )
    known_execution_ids = set(
        target_executions.get("execution_id", pd.Series(dtype=object))
        .dropna()
        .astype(str)
    )
    latest_execution_ids = successful.get(
        "latest_execution_id", pd.Series(index=successful.index, dtype=object)
    )
    linked = latest_execution_ids.notna() & latest_execution_ids.astype(str).isin(
        known_execution_ids
    )
    profiled_count = int(linked.sum())
    selected_ids = set(latest_execution_ids[linked].astype(str))
    selected_executions = (
        target_executions[
            target_executions["execution_id"].astype(str).isin(selected_ids)
        ]
        if selected_ids
        else pd.DataFrame()
    )
    raw_streaming_values = {
        int(value)
        for value in pd.to_numeric(
            selected_executions.get("streaming", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
    }
    streaming_values = [
        bool(value) if value in (0, 1) else value
        for value in sorted(raw_streaming_values)
    ]
    concurrency_values = sorted(
        {
            int(value)
            for value in pd.to_numeric(
                selected_executions.get(
                    "configured_concurrency", pd.Series(dtype=float)
                ),
                errors="coerce",
            ).dropna()
        }
    )
    return {
        "source": "successful_generation_latest_execution_id",
        "successful_generation_count": successful_count,
        "profiled_successful_generation_count": profiled_count,
        "execution_profile_coverage": profiled_count / successful_count,
        "streaming_values": streaming_values,
        "concurrency_values": concurrency_values,
    }


def _returned_service_tier_provenance(
    generations: pd.DataFrame,
) -> dict[str, Any]:
    """Return service tiers observed on successful target generations only."""
    successful = (
        generations[generations["status"] == "ok"]
        if not generations.empty and "status" in generations
        else pd.DataFrame()
    )
    successful_count = len(successful)
    returned_tiers: set[str] = set()
    measured = 0
    for raw_metadata in successful.get(
        "provider_metadata_json", pd.Series(dtype=object)
    ).dropna():
        try:
            metadata = json.loads(str(raw_metadata))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        returned_tier = metadata.get("returned_service_tier")
        if returned_tier is None or not str(returned_tier).strip():
            continue
        returned_tiers.add(str(returned_tier).strip())
        measured += 1
    return {
        "returned_service_tiers": sorted(returned_tiers),
        "returned_service_tier_records": measured,
        "returned_service_tier_coverage": (
            measured / successful_count if successful_count else None
        ),
    }


def _mode_severe(values: Iterable[str | None]) -> str | None:
    counts: dict[str, int] = {}
    for value in values:
        if value in RANK:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    maximum = max(counts.values())
    return max(
        (label for label, count in counts.items() if count == maximum),
        key=RANK.__getitem__,
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _paper_value(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    # The released analysis rounded to four decimals before formatting Table 2.
    return f"{round(float(value), 4):.3f}"


def _markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.fillna("").itertuples(index=False, name=None)]
    widths = [len(value) for value in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"

    lines = [render(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines) + "\n"


def _case_predictions(
    generations: pd.DataFrame,
    judgments: pd.DataFrame,
    *,
    samples: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for task_type in ("qa", "conv"):
        task_rows = generations[generations["task_type"] == task_type].copy()
        if task_rows.empty:
            continue
        if task_type == "qa":
            task_rows["evaluation_label"] = task_rows["parsed_label"]
            task_rows["evaluation_status"] = task_rows["status"]
        else:
            judge_columns = [
                "run_id", "dataset", "source_id", "sample_idx", "judge_label",
                "status", "parse_ok",
            ]
            judge_rows = judgments[judge_columns].rename(
                columns={"status": "judge_status", "parse_ok": "judge_parse_ok"}
            ) if not judgments.empty else pd.DataFrame(columns=judge_columns)
            task_rows = task_rows.merge(
                judge_rows,
                on=["run_id", "dataset", "source_id", "sample_idx"],
                how="left",
            )
            task_rows["evaluation_label"] = task_rows.get("judge_label")
            task_rows["evaluation_status"] = task_rows.get("judge_status")
        group_columns = [
            "run_id", "case_id", "dataset", "source_id", "normalized_label",
            "split", "mapping_method", "is_edge_case",
        ]
        for keys, group in task_rows.groupby(group_columns, dropna=False, sort=True):
            base = dict(zip(group_columns, keys))
            valid = [value for value in group["evaluation_label"] if value in RANK]
            prediction = _mode_severe(valid)
            gold = str(base["normalized_label"])
            pred_rank = RANK.get(prediction) if prediction else None
            gold_rank = RANK.get(gold)
            base.update(
                {
                    "task_type": task_type,
                    "n_expected": samples,
                    "n_present": int(len(group)),
                    "n_target_success": int((group["status"] == "ok").sum()),
                    "n_evaluation_success": int((group["evaluation_status"] == "ok").sum()),
                    "n_valid": len(valid),
                    "mode_prediction": prediction,
                    "exact": (
                        int(prediction == gold)
                        if prediction is not None and gold_rank is not None
                        else None
                    ),
                    "over": (
                        int(pred_rank > gold_rank)
                        if pred_rank is not None and gold_rank is not None
                        else None
                    ),
                    "under": (
                        int(pred_rank < gold_rank)
                        if pred_rank is not None and gold_rank is not None
                        else None
                    ),
                    "ordinal_distance": (
                        abs(pred_rank - gold_rank)
                        if pred_rank is not None and gold_rank is not None
                        else None
                    ),
                }
            )
            records.append(base)
    return pd.DataFrame(records)


def _metric_row(
    group: pd.DataFrame,
    *,
    task_type: str,
    group_type: str,
    group_value: str,
) -> dict[str, Any]:
    evaluable = group[group["mode_prediction"].isin(LABELS)].copy()
    n_evaluable = len(evaluable)
    return {
        "task_type": task_type,
        "group_type": group_type,
        "group_value": group_value,
        "n_total": len(group),
        "n_evaluable": n_evaluable,
        "n_all_invalid": int(len(group) - n_evaluable),
        "invalid_samples": int((group["n_expected"] - group["n_valid"]).sum()),
        "exact": _rate(int(evaluable["exact"].sum()), n_evaluable),
        "over": _rate(int(evaluable["over"].sum()), n_evaluable),
        "under": _rate(int(evaluable["under"].sum()), n_evaluable),
        "within_1": _rate(int((evaluable["ordinal_distance"] <= 1).sum()), n_evaluable),
    }


def _metrics_long(case_predictions: pd.DataFrame) -> pd.DataFrame:
    clear = case_predictions[
        (case_predictions["split"] == "primary")
        & case_predictions["normalized_label"].isin(LABELS)
    ].copy()
    records: list[dict[str, Any]] = []
    for task_type, task_group in clear.groupby("task_type", sort=True):
        records.append(
            _metric_row(
                task_group,
                task_type=task_type,
                group_type="overall",
                group_value="all",
            )
        )
        for dataset, group in task_group.groupby("dataset", sort=True):
            records.append(
                _metric_row(
                    group,
                    task_type=task_type,
                    group_type="dataset",
                    group_value=str(dataset),
                )
            )
        for acuity, group in task_group.groupby("normalized_label", sort=True):
            records.append(
                _metric_row(
                    group,
                    task_type=task_type,
                    group_type="acuity",
                    group_value=str(acuity),
                )
            )
    return pd.DataFrame(records)


def _table2(
    run: dict[str, Any], metrics: pd.DataFrame, *, scope: str
) -> pd.DataFrame:
    overall = metrics[metrics["group_type"] == "overall"].set_index("task_type")

    def get(task: str, metric: str) -> Any:
        return overall.loc[task, metric] if task in overall.index else None

    return pd.DataFrame(
        [
            {
                "Model": run["model_id"],
                "Scope": scope,
                "QA N": int(get("qa", "n_evaluable")) if get("qa", "n_evaluable") is not None else "",
                "QA Exact": _paper_value(get("qa", "exact")),
                "QA Over": _paper_value(get("qa", "over")),
                "QA Under": _paper_value(get("qa", "under")),
                "Conv N": int(get("conv", "n_evaluable")) if get("conv", "n_evaluable") is not None else "",
                "Conv Exact": _paper_value(get("conv", "exact")),
                "Conv Over": _paper_value(get("conv", "over")),
                "Conv Under": _paper_value(get("conv", "under")),
            }
        ]
    )


def _usage_summary(
    generations: pd.DataFrame,
    judgments: pd.DataFrame,
    attempts: pd.DataFrame,
    *,
    target: ModelConfig,
    judge: ModelConfig,
    judge_id: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for phase, attempt_phase, frame, config, attempt_judge_id in (
        ("target", "generation", generations, target, None),
        ("judge", "judge", judgments, judge, judge_id),
    ):
        phase_attempts = (
            attempts[attempts["phase"] == attempt_phase].copy()
            if not attempts.empty and "phase" in attempts
            else pd.DataFrame()
        )
        if (
            attempt_judge_id is not None
            and not phase_attempts.empty
            and "judge_id" in phase_attempts
        ):
            phase_attempts = phase_attempts[
                phase_attempts["judge_id"] == attempt_judge_id
            ].copy()
        if frame.empty and phase_attempts.empty:
            continue
        if phase_attempts.empty:
            legacy_parents = frame.copy()
        else:
            tracked_execution_ids = set(
                str(value)
                for value in phase_attempts["execution_id"].dropna().unique()
            )
            latest_execution = frame.get(
                "latest_execution_id",
                pd.Series(index=frame.index, dtype=object),
            )
            legacy_parents = frame[
                ~latest_execution.astype(str).isin(tracked_execution_ids)
            ].copy()

        usage_frames: list[pd.DataFrame] = []
        if not phase_attempts.empty:
            attempt_usage = phase_attempts.copy()
            first_event_seen = attempt_usage.get(
                "first_event_ms",
                pd.Series(index=attempt_usage.index, dtype=float),
            ).notna()
            http_status = pd.to_numeric(
                attempt_usage.get(
                    "http_status",
                    pd.Series(index=attempt_usage.index, dtype=float),
                ),
                errors="coerce",
            )
            attempt_usage["usage_expected"] = (
                attempt_usage["finish_reason"].notna()
                | attempt_usage["response_id"].notna()
                | first_event_seen
                | http_status.between(200, 299)
                | attempt_usage["outcome"].isin(
                    ["success", "retry_length"]
                )
            )
            usage_frames.append(attempt_usage)
        if not legacy_parents.empty:
            parent_usage = legacy_parents.copy()
            parent_usage["usage_expected"] = (
                parent_usage["status"].eq("ok")
                | parent_usage["finish_reason"].notna()
            )
            usage_frames.append(parent_usage)
        usage_rows = pd.concat(usage_frames, ignore_index=True, sort=False)
        usage_rows["base_usage_complete"] = (
            usage_rows["input_tokens"].notna()
            & usage_rows["output_tokens"].notna()
        )
        cache_breakdown_required = (
            config.cached_input_cost_per_million
            != config.input_cost_per_million
        )
        usage_rows["cache_breakdown_complete"] = (
            usage_rows["cached_input_tokens"].notna()
            if cache_breakdown_required
            else True
        )
        usage_rows["usage_complete"] = (
            usage_rows["base_usage_complete"]
            & usage_rows["cache_breakdown_complete"]
        )
        expected_usage = usage_rows[usage_rows["usage_expected"]]
        complete_usage = expected_usage[expected_usage["usage_complete"]]
        base_complete_usage = expected_usage[
            expected_usage["base_usage_complete"]
        ]
        cache_complete_usage = expected_usage[
            expected_usage["cache_breakdown_complete"]
        ]
        reasoning_complete_usage = expected_usage[
            expected_usage["reasoning_tokens"].notna()
        ]
        input_tokens = int(
            pd.to_numeric(usage_rows["input_tokens"], errors="coerce")
            .fillna(0)
            .sum()
        )
        cached_tokens = int(
            pd.to_numeric(usage_rows["cached_input_tokens"], errors="coerce")
            .fillna(0)
            .sum()
        )
        output_tokens = int(
            pd.to_numeric(usage_rows["output_tokens"], errors="coerce")
            .fillna(0)
            .sum()
        )
        reasoning_tokens_observed = int(
            pd.to_numeric(usage_rows["reasoning_tokens"], errors="coerce")
            .fillna(0)
            .sum()
        )
        reasoning_token_coverage = (
            len(reasoning_complete_usage) / len(expected_usage)
            if len(expected_usage)
            else None
        )
        reasoning_tokens = (
            reasoning_tokens_observed
            if reasoning_token_coverage is not None
            and math.isclose(reasoning_token_coverage, 1.0)
            else None
        )
        uncached_tokens = max(input_tokens - cached_tokens, 0)
        cost = (
            uncached_tokens * config.input_cost_per_million
            + cached_tokens * config.cached_input_cost_per_million
            + output_tokens * config.output_cost_per_million
        ) / 1_000_000
        returned_models = pd.concat(
            [
                frame.get("returned_model", pd.Series(dtype=object)),
                phase_attempts.get(
                    "returned_model", pd.Series(dtype=object)
                ),
            ],
            ignore_index=True,
        )
        successful_calls = int(
            frame.get("status", pd.Series(dtype=object)).eq("ok").sum()
        )
        records.append(
            {
                "phase": phase,
                "configured_model": config.api_model,
                "reasoning_effort": config.reasoning_effort,
                "reasoning_effort_basis": config.reasoning_effort_basis,
                "service_tier": config.service_tier,
                "max_output_tokens": config.max_output_tokens,
                "max_retry_output_tokens": (
                    config.max_retry_output_tokens or 8192
                ),
                "returned_models": ",".join(
                    sorted(str(value) for value in returned_models.dropna().unique())
                ),
                "calls": successful_calls,
                "attempts_tracked": len(phase_attempts),
                "legacy_parent_records": len(legacy_parents),
                "usage_records_expected": len(expected_usage),
                "usage_records_complete": len(complete_usage),
                "missing_usage_records": (
                    len(expected_usage) - len(base_complete_usage)
                ),
                "missing_cache_breakdown_records": (
                    len(expected_usage) - len(cache_complete_usage)
                ),
                "usage_coverage": (
                    len(base_complete_usage) / len(expected_usage)
                    if len(expected_usage)
                    else None
                ),
                "cache_breakdown_coverage": (
                    len(cache_complete_usage) / len(expected_usage)
                    if len(expected_usage)
                    else None
                ),
                "reasoning_token_coverage": reasoning_token_coverage,
                "cost_completeness": (
                    "complete"
                    if len(complete_usage) == len(expected_usage)
                    else "partial_missing_usage_or_cache_breakdown"
                ),
                "billing_basis": "all_tracked_attempts_plus_untracked_parent_rows",
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "reasoning_tokens_observed": reasoning_tokens_observed,
                "reasoning_tokens_per_successful_call": (
                    reasoning_tokens / successful_calls
                    if reasoning_tokens is not None and successful_calls
                    else None
                ),
                "estimated_cost_usd": cost,
                "input_cost_per_million": config.input_cost_per_million,
                "cached_input_cost_per_million": config.cached_input_cost_per_million,
                "output_cost_per_million": config.output_cost_per_million,
            }
        )
    return pd.DataFrame(records)


_LATENCY_METRICS = (
    ("service_latency_ms", "client", "All provider requests plus retry backoff; excludes local semaphore queue."),
    ("request_wall_ms", "client", "Terminal provider request from dispatch through stream EOF."),
    ("request_wall_total_ms", "client", "Cumulative provider request wall time across all attempts."),
    ("ttft_ms", "client_stream", "Terminal request dispatch to first non-empty visible text delta."),
    ("time_after_first_token_ms", "client_stream", "First visible text delta through stream EOF."),
    ("server_processing_ms", "provider_header", "Provider-reported processing duration; not pure model compute."),
    ("queue_wait_ms", "client", "Cumulative local semaphore wait across attempts."),
    ("retry_backoff_ms", "client", "Cumulative actual retry sleep."),
    ("total_duration_ms", "client", "Logical task residence time including local queue, provider calls, and backoff."),
)


def _processing_from_metadata(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        metadata = json.loads(value)
        raw = metadata.get("openai_processing_ms")
        if raw is None:
            raw = metadata.get("server_headers", {}).get("openai-processing-ms")
        return float(raw) if raw is not None else None
    except (TypeError, ValueError, json.JSONDecodeError, AttributeError):
        return None


def _latency_summary(
    generations: pd.DataFrame,
    judgments: pd.DataFrame,
) -> pd.DataFrame:
    """Return source-explicit latency percentiles without pooling unlike clocks."""
    records: list[dict[str, Any]] = []
    groups: list[tuple[str, str, pd.DataFrame]] = []
    if not generations.empty:
        for task_type, frame in generations.groupby("task_type", sort=True):
            groups.append(("target", str(task_type), frame.copy()))
    if not judgments.empty:
        groups.append(("judge", "conv", judgments.copy()))

    for phase, task_type, frame in groups:
        if "server_processing_ms" not in frame:
            frame["server_processing_ms"] = None
        if "provider_metadata_json" in frame:
            fallback = frame["provider_metadata_json"].map(_processing_from_metadata)
            frame["server_processing_ms"] = pd.to_numeric(
                frame["server_processing_ms"], errors="coerce"
            ).fillna(fallback)
        success = frame[frame["status"] == "ok"].copy()
        timing_sources = (
            ",".join(
                sorted(
                    str(value)
                    for value in success.get(
                        "timing_source", pd.Series(dtype=object)
                    ).dropna().unique()
                )
            )
            or "unknown"
        )
        for metric, source, definition in _LATENCY_METRICS:
            values = pd.to_numeric(
                success.get(metric, pd.Series(index=success.index, dtype=float)),
                errors="coerce",
            ).dropna()
            measured = len(values)
            row: dict[str, Any] = {
                "phase": phase,
                "task_type": task_type,
                "metric": metric,
                "source": source,
                "definition": definition,
                "population": "latest_parent_rows_with_status_ok",
                "unit": "ms",
                "n_total": len(frame),
                "n_success": len(success),
                "n_measured": measured,
                "coverage": measured / len(success) if len(success) else None,
                "timing_sources": timing_sources,
                "mean_ms": None,
                "p50_ms": None,
                "p90_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "min_ms": None,
                "max_ms": None,
            }
            if measured:
                row.update(
                    {
                        "mean_ms": float(values.mean()),
                        "p50_ms": float(values.quantile(0.50, interpolation="linear")),
                        "p90_ms": float(values.quantile(0.90, interpolation="linear")),
                        "p95_ms": float(values.quantile(0.95, interpolation="linear")),
                        "p99_ms": float(values.quantile(0.99, interpolation="linear")),
                        "min_ms": float(values.min()),
                        "max_ms": float(values.max()),
                    }
                )
            records.append(row)
    return pd.DataFrame(records)


def _execution_summary(
    executions: pd.DataFrame,
    attempts: pd.DataFrame,
) -> pd.DataFrame:
    """Return per-invocation throughput, retry, and failure diagnostics."""
    columns = [
        "execution_id",
        "phase",
        "profile_id",
        "status",
        "streaming",
        "configured_concurrency",
        "pending_count",
        "success_count",
        "failure_count",
        "cancelled_count",
        "unpersisted_count",
        "elapsed_seconds",
        "persisted_requests_per_second",
        "requests_observed",
        "attempts_total",
        "retry_attempts",
        "attempts_per_request",
        "terminal_error_attempts",
    ]
    if executions.empty:
        return pd.DataFrame(columns=columns)
    summary = executions.copy()
    for name in ("cancelled_count", "unpersisted_count"):
        if name not in summary:
            summary[name] = 0
    started = pd.to_datetime(summary["started_at"], errors="coerce", utc=True)
    ended = pd.to_datetime(summary["ended_at"], errors="coerce", utc=True)
    summary["elapsed_seconds"] = (ended - started).dt.total_seconds()
    persisted = summary["success_count"].fillna(0) + summary[
        "failure_count"
    ].fillna(0)
    summary["persisted_requests_per_second"] = persisted / summary[
        "elapsed_seconds"
    ].where(summary["elapsed_seconds"] > 0)

    if attempts.empty:
        summary["requests_observed"] = 0
        summary["attempts_total"] = 0
        summary["retry_attempts"] = 0
        summary["terminal_error_attempts"] = 0
    else:
        attempt_stats = attempts.groupby("execution_id", sort=False).agg(
            requests_observed=("request_key", "nunique"),
            attempts_total=("attempt_index", "size"),
            retry_attempts=("attempt_index", lambda values: int((values > 1).sum())),
            terminal_error_attempts=(
                "outcome",
                lambda values: int(values.eq("terminal_error").sum()),
            ),
        )
        summary = summary.merge(
            attempt_stats,
            left_on="execution_id",
            right_index=True,
            how="left",
        )
        for name in (
            "requests_observed",
            "attempts_total",
            "retry_attempts",
            "terminal_error_attempts",
        ):
            summary[name] = summary[name].fillna(0).astype(int)
    summary["attempts_per_request"] = summary["attempts_total"] / summary[
        "requests_observed"
    ].where(summary["requests_observed"] > 0)
    return summary.reindex(columns=columns)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    # pandas' JSON conversion normalizes numpy scalars and emits JSON null for NaN.
    return json.loads(frame.to_json(orient="records"))


def _boundary_metrics(case_predictions: pd.DataFrame) -> pd.DataFrame:
    boundary = case_predictions[
        (case_predictions["split"] == "primary")
        & case_predictions["normalized_label"].astype(str).str.match(r"^[A-D]\|[A-D]$")
    ].copy()
    records: list[dict[str, Any]] = []
    for (task, gold), group in boundary.groupby(
        ["task_type", "normalized_label"], sort=True
    ):
        valid = group[group["mode_prediction"].isin(LABELS)].copy()
        parts = str(gold).split("|")
        low, high = sorted(parts, key=RANK.__getitem__)
        constituent = valid["mode_prediction"].isin(parts)
        records.append(
            {
                "task_type": task,
                "boundary": gold,
                "n_total": len(group),
                "n_evaluable": len(valid),
                "constituent_rate": _rate(int(constituent.sum()), len(valid)),
                "upper_among_constituent": _rate(
                    int((valid.loc[constituent, "mode_prediction"] == high).sum()),
                    int(constituent.sum()),
                ),
                "outside_high_rate": _rate(
                    int(valid["mode_prediction"].map(RANK).gt(RANK[high]).sum()),
                    len(valid),
                ),
                "outside_low_rate": _rate(
                    int(valid["mode_prediction"].map(RANK).lt(RANK[low]).sum()),
                    len(valid),
                ),
            }
        )
    return pd.DataFrame(records)


def _confusion(case_predictions: pd.DataFrame, task: str) -> pd.DataFrame:
    clear = case_predictions[
        (case_predictions["task_type"] == task)
        & (case_predictions["split"] == "primary")
        & case_predictions["normalized_label"].isin(LABELS)
        & case_predictions["mode_prediction"].isin(LABELS)
    ]
    matrix = pd.crosstab(clear["normalized_label"], clear["mode_prediction"])
    return matrix.reindex(index=LABELS, columns=LABELS, fill_value=0).rename_axis(
        index="true", columns="predicted"
    )


def _is_paper_comparable_run(run: dict[str, Any]) -> bool:
    return (
        run.get("selected_cases") == 914
        and run.get("samples") == 5
        and set(run.get("tasks", [])) == {"qa", "conv"}
    )


def _paper_comparison(run: dict[str, Any], metrics: pd.DataFrame) -> pd.DataFrame:
    published_results = PAPER_MODEL_RESULTS.get(run["model_id"])
    if published_results is None or not _is_paper_comparable_run(run):
        return pd.DataFrame()
    overall = metrics[metrics["group_type"] == "overall"].set_index("task_type")
    records = []
    for task, published in published_results.items():
        if task not in overall.index:
            continue
        for metric, paper_value in published.items():
            fresh = float(overall.loc[task, metric])
            records.append(
                {
                    "task_type": task,
                    "metric": metric,
                    "published": paper_value,
                    "fresh_run": fresh,
                    "delta": fresh - paper_value,
                }
            )
    return pd.DataFrame(records)


def generate_report(
    *,
    run_id: str,
    store_path: Path | None = None,
    output_root: Path | None = None,
    judge_id: str = "paper-gpt-4.1",
    allow_incomplete: bool = False,
) -> Path:
    registry = ModelRegistry()
    judge_profile = registry.get_judge(judge_id)
    with EvaluationStore(store_path or default_store_path()) as store:
        run = store.get_run(run_id)
        generation_rows = store.dataframe_rows(
            "SELECT * FROM generations WHERE run_id=? ORDER BY dataset,source_id,task_type,sample_idx",
            (run_id,),
        )
        judgment_rows = store.dataframe_rows(
            "SELECT * FROM judgments WHERE run_id=? AND judge_id=? ORDER BY dataset,source_id,sample_idx",
            (run_id, judge_id),
        )
        execution_rows = store.dataframe_rows(
            "SELECT * FROM run_executions WHERE run_id=? ORDER BY started_at,execution_id",
            (run_id,),
        )
        attempt_rows = store.dataframe_rows(
            """SELECT a.*, e.phase AS phase, e.profile_id AS profile_id
               FROM request_attempts AS a
               JOIN run_executions AS e ON e.execution_id=a.execution_id
               WHERE e.run_id=? ORDER BY e.started_at,a.request_key,a.attempt_index""",
            (run_id,),
        )
    generations = pd.DataFrame(generation_rows)
    judgments = pd.DataFrame(judgment_rows)
    executions = pd.DataFrame(execution_rows)
    attempts = pd.DataFrame(attempt_rows)
    if generations.empty:
        raise ValueError(f"Run {run_id!r} has no generation results")

    rubric, judge_template = load_judge_assets()

    # Make every flat export self-describing rather than requiring a DB join.
    generations.insert(1, "model_id", run["model_id"])
    generations.insert(2, "provider", run["provider"])
    generations.insert(3, "configured_api_model", run["api_model"])
    if not judgments.empty:
        judgments.insert(1, "target_model_id", run["model_id"])
        judgments.insert(2, "judge_model_id", judge_profile.model.api_model)

        expected_judge_hash = judge_profile.fingerprint
        stale_config = judgments["judge_config_sha256"] != expected_judge_hash
        conv_hashes = generations[generations["task_type"] == "conv"][
            ["run_id", "dataset", "source_id", "sample_idx", "response_sha256"]
        ].rename(columns={"response_sha256": "current_generation_sha256"})
        checked = judgments.merge(
            conv_hashes,
            on=["run_id", "dataset", "source_id", "sample_idx"],
            how="left",
        )
        stale_response = (
            checked["generation_response_sha256"]
            != checked["current_generation_sha256"]
        )
        if stale_config.any() or stale_response.any():
            raise RuntimeError(
                "Stored conversational judgments are stale for the current judge "
                "configuration or target responses. Rerun `acuitybench judge`."
            )
        conv_generations = {
            (row["dataset"], row["source_id"], row["sample_idx"]): row
            for row in generation_rows
            if row["task_type"] == "conv"
        }
        stale_prompts = 0
        for judgment in judgment_rows:
            key = (
                judgment["dataset"],
                judgment["source_id"],
                judgment["sample_idx"],
            )
            generation = conv_generations.get(key)
            if generation is None:
                stale_prompts += 1
                continue
            expected_prompt = build_judge_prompt(
                generation, rubric, judge_template
            )
            if (
                judgment.get("judge_prompt") != expected_prompt
                or judgment.get("judge_prompt_sha256")
                != sha256_text(expected_prompt)
            ):
                stale_prompts += 1
        if stale_prompts:
            raise RuntimeError(
                f"{stale_prompts} stored conversational judgments use a stale "
                "rubric or judge prompt. Rerun `acuitybench judge`."
            )

    # Labels are report-time derivations of immutable raw responses. This pins
    # every score to the parser contract recorded below, even for older rows.
    def parsed_label(value: object) -> str | None:
        return extract_label(value if isinstance(value, str) else None)

    qa_mask = generations["task_type"] == "qa"
    generations.loc[qa_mask, "parsed_label"] = generations.loc[
        qa_mask, "response"
    ].map(parsed_label)
    generations.loc[qa_mask, "parse_ok"] = generations.loc[
        qa_mask, "parsed_label"
    ].notna().astype(int)
    if not judgments.empty:
        judgments["judge_label"] = judgments["response"].map(parsed_label)
        judgments["judge_reasoning"] = judgments["response"].map(
            lambda value: extract_reasoning(
                value if isinstance(value, str) else None
            )
        )
        judgments["parse_ok"] = judgments["judge_label"].notna().astype(int)
    expected_generations = run["expected_generations"]
    ok_generations = int((generations["status"] == "ok").sum())
    expected_judgments = (
        run["selected_cases"] * run["samples"] if "conv" in run["tasks"] else 0
    )
    ok_judgments = (
        int((judgments["status"] == "ok").sum()) if not judgments.empty else 0
    )
    incomplete = ok_generations != expected_generations or ok_judgments != expected_judgments
    if incomplete and not allow_incomplete:
        raise RuntimeError(
            f"Run is incomplete: generation {ok_generations}/{expected_generations}, "
            f"judge {ok_judgments}/{expected_judgments}. Rerun it or pass --allow-incomplete."
        )

    destination = (output_root or project_root() / "results") / run_id
    tables = destination / "tables"
    exports = destination / "exports"
    tables.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)

    case_predictions = _case_predictions(
        generations, judgments, samples=run["samples"]
    )
    case_predictions.insert(1, "model_id", run["model_id"])
    metrics = _metrics_long(case_predictions)
    metrics.insert(0, "model_id", run["model_id"])
    scope = (
        "full 914-case benchmark"
        if run["selected_cases"] == 914 and run["samples"] == 5
        else f"selected {run['selected_cases']}-case run"
    )
    table2 = _table2(run, metrics, scope=scope)
    boundary = _boundary_metrics(case_predictions)
    if not boundary.empty:
        boundary.insert(0, "model_id", run["model_id"])
    target_config = ModelConfig(**run["model_config"])
    judge_config = judge_profile.model
    usage = _usage_summary(
        generations,
        judgments,
        attempts,
        target=target_config,
        judge=judge_config,
        judge_id=judge_id,
    )
    latency = _latency_summary(generations, judgments)
    execution_summary = _execution_summary(executions, attempts)
    comparison = _paper_comparison(run, metrics)
    if not comparison.empty:
        comparison.insert(0, "model_id", run["model_id"])

    distributional_summaries: list[dict[str, Any]] = []
    benchmark_path = Path(run["benchmark_path"])
    if benchmark_path.exists():
        benchmark = pd.read_csv(benchmark_path, dtype={"source_id": str})
        rater_columns = {f"anon_label_{index}" for index in range(1, 6)}
        if rater_columns <= set(benchmark.columns):
            for panel_split in ("primary", "ambiguous"):
                for task in run["tasks"]:
                    cases, distributional_summary = evaluate_panel_distribution(
                        generations=generations,
                        judgments=judgments,
                        benchmark=benchmark,
                        task_type=task,
                        split=panel_split,
                    )
                    cases.insert(0, "model_id", run["model_id"])
                    cases.to_csv(
                        exports / f"distributional_{panel_split}_{task}.csv",
                        index=False,
                    )
                    cases.to_parquet(
                        exports / f"distributional_{panel_split}_{task}.parquet",
                        index=False,
                    )
                    distributional_summary["model_id"] = run["model_id"]
                    distributional_summaries.append(distributional_summary)
    distributional = pd.DataFrame(distributional_summaries)

    overall = metrics[metrics["group_type"] == "overall"].set_index("task_type")
    table_evaluable = {
        task: int(overall.loc[task, "n_evaluable"])
        for task in ("qa", "conv")
        if task in overall.index
    }
    full_contract = _is_paper_comparable_run(run)
    if full_contract:
        totals = {
            task: int(overall.loc[task, "n_total"])
            for task in ("qa", "conv")
            if task in overall.index
        }
        if totals != {"qa": 527, "conv": 527}:
            raise RuntimeError(
                f"Paper Table 2 cohort mismatch: expected 527 per task, found {totals}"
            )

    generations.to_csv(exports / "raw_samples.csv", index=False)
    generations.to_parquet(exports / "raw_samples.parquet", index=False)
    judgments.to_csv(exports / "judged_samples.csv", index=False)
    if not judgments.empty:
        judgments.to_parquet(exports / "judged_samples.parquet", index=False)
    executions.to_csv(exports / "run_executions.csv", index=False)
    attempts.to_csv(exports / "request_attempts.csv", index=False)
    if not attempts.empty:
        attempts.to_parquet(exports / "request_attempts.parquet", index=False)
    case_predictions.to_csv(exports / "case_predictions.csv", index=False)
    case_predictions.to_parquet(exports / "case_predictions.parquet", index=False)
    metrics.to_csv(tables / "metrics_long.csv", index=False)
    table2.to_csv(tables / "table2.csv", index=False)
    (tables / "table2.md").write_text(_markdown(table2), encoding="utf-8")
    boundary.to_csv(tables / "boundary_metrics.csv", index=False)
    usage.to_csv(tables / "usage_and_cost.csv", index=False)
    latency.to_csv(tables / "latency_summary.csv", index=False)
    execution_summary.to_csv(tables / "execution_summary.csv", index=False)
    distributional.to_csv(tables / "distributional_metrics.csv", index=False)
    published_comparison_path = tables / "published_comparison.csv"
    if not comparison.empty:
        comparison.to_csv(published_comparison_path, index=False)
    else:
        published_comparison_path.unlink(missing_ok=True)
    for task in run["tasks"]:
        _confusion(case_predictions, task).to_csv(
            tables / f"confusion_{task}.csv"
        )

    execution_manifest = _json_records(executions)
    for record in execution_manifest:
        for key in ("retry_policy_json", "runner_metadata_json"):
            raw = record.pop(key, None)
            if raw:
                record[key.removesuffix("_json")] = json.loads(raw)
    latency_records = _json_records(latency)
    all_call_frames = [generations]
    if not judgments.empty:
        all_call_frames.append(judgments)
    timing_sources = pd.concat(
        [
            frame.get("timing_source", pd.Series(index=frame.index, dtype=object))
            for frame in all_call_frames
        ],
        ignore_index=True,
    )
    legacy_rows = int((timing_sources == "legacy_aggregate").sum())
    cost_complete = bool(
        not usage.empty and usage["cost_completeness"].eq("complete").all()
    )
    estimated_total_cost = float(usage["estimated_cost_usd"].sum())
    returned_judge_models = sorted(
        str(value)
        for value in judgments.get(
            "returned_model", pd.Series(dtype=str)
        ).dropna().unique()
    )
    judge_config_manifest = {
        "id": judge_profile.id,
        "display_name": judge_profile.display_name,
        "model": judge_profile.model.as_dict(),
        "fingerprint": judge_profile.fingerprint,
    }
    rubric_path = project_root() / "configs/rubric.yaml"
    judge_template_path = project_root() / "configs/prompts/judge_rubric.txt"
    judge_assets_manifest = {
        "rubric_path": str(rubric_path.relative_to(project_root())),
        "rubric_sha256": sha256_file(rubric_path),
        "template_path": str(judge_template_path.relative_to(project_root())),
        "template_sha256": sha256_file(judge_template_path),
    }
    judge_assets_manifest["fingerprint"] = _json_fingerprint(
        judge_assets_manifest
    )
    latency_profile = _successful_generation_latency_profile(
        generations, executions
    )
    execution_concurrency = latency_profile["concurrency_values"]
    execution_streaming = latency_profile["streaming_values"]
    service_tier_provenance = _returned_service_tier_provenance(generations)
    transport_differences: list[str] = []
    if execution_streaming and execution_streaming != [False]:
        transport_differences.append(
            "streaming enabled to measure TTFT; companion paper adapter was non-streaming"
        )
    if execution_concurrency and execution_concurrency != [1]:
        transport_differences.append(
            "concurrent execution used for practical runtime; paper reports no concurrency"
        )
    if target_config.service_tier is not None:
        transport_differences.append(
            f"service tier explicitly pinned to {target_config.service_tier!r}; "
            "the paper did not report service tier"
        )
    paper_api_identifier = PAPER_TARGET_API_IDENTIFIERS.get(run["model_id"])
    inference_contract = {
        "paper_reported": {
            "api_identifier": paper_api_identifier,
            "api_identifier_source": (
                "pinned_project_paper_contract"
                if paper_api_identifier is not None
                else "no_paper_identifier_pinned_for_model"
            ),
            "samples_per_case_per_format": 5,
            "temperature": 1.0,
            "max_completion_tokens": 4096,
            "reasoning_effort": None,
            "reasoning_effort_note": (
                "Not reported by the paper; 4096 is the combined completion "
                "cap, not a separate reasoning-token budget."
            ),
            "service_tier": None,
            "service_tier_note": (
                "Not reported by the paper; the public companion adapter "
                "omitted the service-tier parameter."
            ),
            "streaming": None,
            "streaming_note": (
                "Not reported by the paper; the public companion adapter was "
                "non-streaming."
            ),
            "concurrency": 1,
            "concurrency_note": (
                "Appendix C reports no concurrency or distributed computation."
            ),
        },
        "configured": {
            "api_identifier": target_config.api_model,
            "endpoint": target_config.endpoint,
            "samples_per_case_per_format": run["samples"],
            "temperature": target_config.temperature,
            "temperature_parameter_sent": target_config.send_temperature,
            "max_completion_tokens": target_config.max_output_tokens,
            "max_retry_completion_tokens": (
                target_config.max_retry_output_tokens or 8192
            ),
            "reasoning_effort": target_config.reasoning_effort,
            "reasoning_effort_basis": target_config.reasoning_effort_basis,
            "reasoning_effort_provenance": REASONING_EFFORT_PROVENANCE.get(
                run["model_id"]
            ),
            "service_tier": target_config.service_tier,
            "streaming_values": execution_streaming,
            "concurrency_values": execution_concurrency,
        },
        "observed": {
            "returned_models": sorted(
                str(value)
                for value in generations["returned_model"].dropna().unique()
            ),
            **latency_profile,
            **service_tier_provenance,
        },
        "transport_differences_from_paper": transport_differences,
    }

    manifest = {
        "run": run,
        "report_scope": scope,
        "report_complete": not incomplete,
        "generation_status": {
            "expected": expected_generations,
            "ok": ok_generations,
        },
        "judge_status": {"expected": expected_judgments, "ok": ok_judgments},
        "returned_target_models": sorted(
            str(value) for value in generations["returned_model"].dropna().unique()
        ),
        "returned_judge_models": returned_judge_models,
        "judge_config": judge_config_manifest,
        "judge_assets": judge_assets_manifest,
        "inference_contract": inference_contract,
        "paper_baseline": {
            **PAPER_BASELINE_PROVENANCE,
            "model_id": run["model_id"],
            "published_values": PAPER_MODEL_RESULTS.get(run["model_id"]),
            "applicable_to_run": not comparison.empty,
            "inapplicable_reason": (
                None
                if not comparison.empty
                else "run is not the complete 914-case, five-sample, two-format contract or has no pinned paper baseline"
            ),
        },
        "estimated_total_cost_usd": estimated_total_cost,
        "cost_completeness": (
            "complete" if cost_complete else "partial_usage_telemetry"
        ),
        "usage": _json_records(usage),
        "pricing_note": (
            "Estimate from every tracked attempt plus legacy parent usage and "
            "prices in configs/models.yaml. Missing token totals can understate "
            "cost; missing cache detail is conservatively priced as uncached "
            "input and can overstate that input component."
        ),
        "timing": {
            "schema_version": EvaluationStore.SCHEMA_VERSION,
            "instrumentation_version": EvaluationStore.TIMING_VERSION,
            "primary_latency_metric": "service_latency_ms",
            "primary_latency_percentile": "p95_ms",
            "ttft_definition": (
                "Request dispatch to the first non-empty visible text delta on "
                "the terminal successful streaming attempt."
            ),
            "field_definitions": {
                "queue_wait_ms": "Cumulative local semaphore wait across attempts.",
                "request_wall_ms": "Terminal provider request dispatch through stream EOF.",
                "request_wall_total_ms": "Cumulative provider request wall time across attempts.",
                "service_latency_ms": "All provider request wall time plus retry backoff; excludes local semaphore queue.",
                "retry_backoff_ms": "Cumulative actual retry sleep.",
                "ttft_ms": "Terminal request dispatch to first non-empty visible text delta.",
                "time_after_first_token_ms": "First visible text delta through stream EOF.",
                "total_duration_ms": "Logical task time including local queue, provider requests, and backoff.",
                "server_processing_ms": "Provider-reported processing header; not pure model compute.",
                "latency_ms": "Compatibility alias for total_duration_ms; never use as API request latency.",
            },
            "legacy_rows": legacy_rows,
            "legacy_limitations": (
                "Legacy rows retain queue-inclusive total duration and provider "
                "processing when present; queue, request wall, backoff, and TTFT "
                "cannot be reconstructed."
            ),
            "summary": latency_records,
            "execution_summary": _json_records(execution_summary),
            "executions": execution_manifest,
        },
        "paper_contract": {
            "main_table_filter": "split=primary and gold in A/B/C/D",
            "main_table_expected_n": 527 if scope.startswith("full") else None,
            "samples_per_case": run["samples"],
            "aggregation": "valid-label mode; ties resolve to higher acuity",
            "parser": label_parser_contract(),
            "judge_id": judge_id,
            "full_contract_run": full_contract,
            "evaluable_cases": table_evaluable,
            "all_527_cases_evaluable": (
                table_evaluable == {"qa": 527, "conv": 527}
                if full_contract
                else None
            ),
        },
        "physician_panel_distributional_metrics": _json_records(distributional),
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_parts = [
        f"# AcuityBench report: {run_id}",
        "",
        f"Scope: {scope}. Requested model: `{run['api_model']}`.",
        f"Returned target snapshot(s): {', '.join(manifest['returned_target_models'])}.",
        "",
        "## Inference contract",
        "",
        (
            f"Reasoning effort: `{target_config.reasoning_effort or 'omitted'}` "
            f"({target_config.reasoning_effort_basis or 'no explicit basis'}). "
            "The paper did not report a reasoning effort or separate reasoning-token "
            "budget."
        ),
        (
            f"Completion cap: {target_config.max_output_tokens:,} tokens; retry cap: "
            f"{target_config.max_retry_output_tokens or 8192:,}. This combined cap "
            "includes hidden reasoning and visible output."
        ),
        (
            f"Configured temperature: {target_config.temperature}; parameter sent: "
            f"{target_config.send_temperature}. Service tier: "
            f"`{target_config.service_tier or 'provider default'}` "
            "(paper unreported)."
        ),
        (
            "Execution streaming value(s): "
            f"{execution_streaming or ['not recorded']}; concurrency value(s): "
            f"{execution_concurrency or ['not recorded']}."
        ),
        "",
        "## Paper-style main table",
        "",
        _markdown(table2).rstrip(),
        "",
        "The main table scores only primary cases with clear A/B/C/D gold labels; "
        "valid sample labels are aggregated by mode with severe tie-breaking.",
        "",
        "## Usage and estimated cost",
        "",
        _markdown(usage.round({"estimated_cost_usd": 4})).rstrip(),
        "",
        (
            "Estimated total: "
            if cost_complete
            else "Estimated tracked total (partial usage telemetry): "
        )
        + f"${estimated_total_cost:.4f} USD.",
    ]
    latency_display = latency[
        latency["metric"].isin(
            ["service_latency_ms", "ttft_ms", "server_processing_ms"]
        )
        & (latency["n_measured"] > 0)
    ][
        [
            "phase",
            "task_type",
            "metric",
            "n_measured",
            "coverage",
            "p50_ms",
            "p95_ms",
        ]
    ].copy()
    if not latency_display.empty:
        latency_display["coverage"] = latency_display["coverage"].map(
            lambda value: f"{value:.1%}"
        )
        latency_display[["p50_ms", "p95_ms"]] = latency_display[
            ["p50_ms", "p95_ms"]
        ].round(1)
        summary_parts.extend(
            [
                "",
                "## Latency",
                "",
                _markdown(latency_display).rstrip(),
                "",
                "The primary serving metric is p95 `service_latency_ms`; TTFT is "
                "the first non-empty visible text delta. Provider processing is "
                "reported separately and is not pure model compute.",
            ]
        )
    if not comparison.empty:
        display_comparison = comparison.copy()
        for column in ("published", "fresh_run", "delta"):
            display_comparison[column] = display_comparison[column].map(
                lambda value: f"{value:.3f}"
            )
        summary_parts.extend(
            [
                "",
                "## Published comparison",
                "",
                _markdown(display_comparison).rstrip(),
                "",
                (
                    "Baseline: [AcuityBench Table 2]"
                    "(https://arxiv.org/pdf/2605.11398), published to three "
                    "decimals. Delta is fresh run minus published."
                ),
                "",
                "This is a fresh stochastic run; the published aliases were not "
                "immutable experiment artifacts.",
            ]
        )
    ambiguous_display = (
        distributional[
            (distributional["split"] == "ambiguous")
            & (distributional["n"] > 0)
        ].copy()
        if not distributional.empty
        else pd.DataFrame()
    )
    if not ambiguous_display.empty:
        keep = [
            "task_type", "n", "n_evaluable", "jsd_mean", "wasserstein_mean",
            "rater_probability_mean", "consensus_loo_change_rate_mean",
            "consensus_loo_mean_delta_mean", "baseline_human_alpha",
            "mean_loo_alpha", "alpha_delta",
        ]
        ambiguous_display = ambiguous_display[keep].round(4)
        summary_parts.extend(
            [
                "",
                "## Physician-panel ambiguous cases",
                "",
                _markdown(ambiguous_display).rstrip(),
                "",
                "JSD uses natural logarithms for compatibility with the released "
                "analysis (its maximum is ln(2), not 1).",
            ]
        )
    (destination / "SUMMARY.md").write_text(
        "\n".join(summary_parts) + "\n", encoding="utf-8"
    )
    return destination


def _frontier_latency_value(
    latency: pd.DataFrame,
    *,
    metric: str,
    percentile: str,
    task_type: str,
    required_sources: set[str] | None = None,
) -> float | None:
    if latency.empty:
        return None
    selected = latency[
        (latency["phase"] == "target")
        & (latency["metric"] == metric)
        & (latency["task_type"] == task_type)
    ]
    if len(selected) != 1:
        return None
    row = selected.iloc[0]
    coverage = pd.to_numeric(
        pd.Series([row.get("coverage")]), errors="coerce"
    ).iloc[0]
    n_success = pd.to_numeric(
        pd.Series([row.get("n_success")]), errors="coerce"
    ).iloc[0]
    n_measured = pd.to_numeric(
        pd.Series([row.get("n_measured")]), errors="coerce"
    ).iloc[0]
    if (
        pd.isna(coverage)
        or not math.isclose(float(coverage), 1.0)
        or pd.isna(n_success)
        or pd.isna(n_measured)
        or int(n_success) <= 0
        or int(n_measured) != int(n_success)
    ):
        return None
    timing_sources = {
        source.strip()
        for source in str(row.get("timing_sources") or "").split(",")
        if source.strip()
    }
    if required_sources is not None and (
        len(timing_sources) != 1 or not timing_sources <= required_sources
    ):
        return None
    value = pd.to_numeric(
        pd.Series([row.get(percentile)]), errors="coerce"
    ).iloc[0]
    return float(value) if not pd.isna(value) else None


def _frontier_latency_macro(
    latency: pd.DataFrame,
    *,
    metric: str,
    percentile: str,
    required_sources: set[str] | None = None,
) -> float | None:
    qa_value = _frontier_latency_value(
        latency,
        metric=metric,
        percentile=percentile,
        task_type="qa",
        required_sources=required_sources,
    )
    conv_value = _frontier_latency_value(
        latency,
        metric=metric,
        percentile=percentile,
        task_type="conv",
        required_sources=required_sources,
    )
    if qa_value is None or conv_value is None:
        return None
    return (qa_value + conv_value) / 2


def _verified_reasoning_total(
    target_usage: pd.DataFrame,
) -> tuple[float | None, float | None]:
    """Return a total only when every expected call reported reasoning usage."""
    if target_usage.empty or "reasoning_token_coverage" not in target_usage:
        return None, None
    coverage_values = pd.to_numeric(
        target_usage["reasoning_token_coverage"], errors="coerce"
    ).dropna()
    if coverage_values.empty:
        return None, None
    coverage = float(coverage_values.min())
    if not math.isclose(coverage, 1.0):
        return None, coverage
    if "reasoning_tokens" not in target_usage:
        return None, coverage
    token_values = pd.to_numeric(
        target_usage["reasoning_tokens"], errors="coerce"
    )
    if token_values.isna().any() or token_values.empty:
        return None, coverage
    return float(token_values.sum()), coverage


def _effective_retry_completion_cap(
    configured_inference: dict[str, Any],
    run_model_config: dict[str, Any],
) -> tuple[Any, str]:
    configured_cap = configured_inference.get("max_retry_completion_tokens")
    if configured_cap is not None:
        return configured_cap, "inference_contract"
    model_cap = run_model_config.get("max_retry_output_tokens")
    if model_cap is not None:
        return model_cap, "run_model_config"
    return 8192, "legacy_runner_default"


def _validated_client_latency_profile(
    *,
    run_id: str,
    streaming_values: tuple[Any, ...],
    concurrency_values: tuple[Any, ...],
    execution_profile_coverage: Any,
    configured_service_tier: Any,
    returned_service_tiers: tuple[Any, ...],
    returned_service_tier_coverage: Any,
) -> tuple[bool, int, str]:
    """Fail closed unless one fully observed serving profile produced the dot."""
    problems: list[str] = []

    streaming: bool | None = None
    if len(streaming_values) != 1:
        problems.append("exactly one streaming mode is required")
    else:
        raw_streaming = streaming_values[0]
        if isinstance(raw_streaming, bool):
            streaming = raw_streaming
        elif raw_streaming in (0, 1):
            streaming = bool(raw_streaming)
        else:
            problems.append("streaming mode must be boolean")

    concurrency: int | None = None
    if len(concurrency_values) != 1:
        problems.append("exactly one configured concurrency is required")
    else:
        try:
            raw_concurrency = float(concurrency_values[0])
        except (TypeError, ValueError):
            problems.append("configured concurrency must be an integer")
        else:
            if not math.isfinite(raw_concurrency) or not raw_concurrency.is_integer():
                problems.append("configured concurrency must be an integer")
            else:
                concurrency = int(raw_concurrency)
            if concurrency is not None and concurrency < 1:
                problems.append("configured concurrency must be positive")

    profile_coverage = pd.to_numeric(
        pd.Series([execution_profile_coverage]), errors="coerce"
    ).iloc[0]
    if pd.isna(profile_coverage) or not math.isclose(
        float(profile_coverage), 1.0
    ):
        problems.append("successful-generation execution-profile coverage must be 1")

    configured_tier = (
        str(configured_service_tier).strip()
        if configured_service_tier is not None
        else ""
    )
    if not configured_tier:
        problems.append("configured service tier is required")

    returned_tiers = tuple(
        str(value).strip()
        for value in returned_service_tiers
        if value is not None and str(value).strip()
    )
    if len(returned_tiers) != 1:
        problems.append("exactly one returned service tier is required")
    returned_coverage = pd.to_numeric(
        pd.Series([returned_service_tier_coverage]), errors="coerce"
    ).iloc[0]
    if pd.isna(returned_coverage) or not math.isclose(
        float(returned_coverage), 1.0
    ):
        problems.append("returned service-tier coverage must be 1")
    if (
        configured_tier
        and len(returned_tiers) == 1
        and configured_tier != returned_tiers[0]
    ):
        problems.append(
            "configured and returned service tiers must match "
            f"({configured_tier!r} != {returned_tiers[0]!r})"
        )

    if problems:
        raise ValueError(
            f"Run {run_id!r} cannot supply a client service-latency frontier "
            f"point: {'; '.join(problems)}"
        )
    assert streaming is not None and concurrency is not None and configured_tier
    return streaming, concurrency, configured_tier


def _assert_latency_profiles_comparable(
    profiles: dict[str, tuple[bool, int, str]],
) -> None:
    if len(set(profiles.values())) > 1:
        rendered = ", ".join(
            f"{run_id}={profile}" for run_id, profile in sorted(profiles.items())
        )
        raise ValueError(
            "Client service-latency frontier points must share streaming mode, "
            f"configured concurrency, and service tier; found: {rendered}"
        )


def _comparison_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    run = manifest["run"]
    paper = manifest["paper_contract"]
    selection = run.get("selection", {})
    judge_config = manifest.get("judge_config", {})
    judge_assets = manifest.get("judge_assets", {})
    contract = {
        "benchmark_sha256": run.get("benchmark_sha256"),
        "case_ids_sha256": selection.get("case_ids_sha256"),
        "selected_cases": run.get("selected_cases"),
        "tasks": tuple(sorted(run.get("tasks", []))),
        "samples": run.get("samples"),
        "expected_generations": run.get("expected_generations"),
        "main_table_filter": paper.get("main_table_filter"),
        "aggregation": paper.get("aggregation"),
        "parser": paper.get("parser"),
        "judge_id": paper.get("judge_id"),
        "judge_config_fingerprint": judge_config.get("fingerprint"),
        "judge_assets_fingerprint": judge_assets.get("fingerprint"),
        "returned_judge_models": tuple(
            sorted(manifest.get("returned_judge_models", []))
        ),
    }
    required = {
        "benchmark_sha256",
        "case_ids_sha256",
        "main_table_filter",
        "aggregation",
        "parser",
        "judge_id",
        "judge_config_fingerprint",
        "judge_assets_fingerprint",
    }
    missing = sorted(key for key in required if not contract.get(key))
    if missing:
        raise ValueError(
            "Run manifest lacks comparison-contract metadata; regenerate its "
            f"report. Missing: {', '.join(missing)}"
        )
    if "conv" in contract["tasks"] and len(contract["returned_judge_models"]) != 1:
        raise ValueError(
            "Conversational comparison runs must record exactly one returned "
            "judge model snapshot; regenerate or rerun the report."
        )
    return contract


def _assert_comparable(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    reference_run_id: str,
    candidate_run_id: str,
) -> None:
    differences = [
        key for key in reference if reference.get(key) != candidate.get(key)
    ]
    if differences:
        raise ValueError(
            "Comparison runs must share the same benchmark cohort, samples, "
            "tasks, scoring parser, judge configuration/snapshot, and assets. "
            f"{candidate_run_id!r} differs from {reference_run_id!r} in: "
            f"{', '.join(differences)}"
        )


def combine_reports(
    *,
    run_ids: list[str],
    results_root: Path | None = None,
    destination: Path | None = None,
) -> Path:
    """Combine any configured models' generated rows without a hardcoded order."""
    if not run_ids:
        raise ValueError("At least one run ID is required")
    root = results_root or project_root() / "results"
    output = destination or root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    table_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    usage_frames: list[pd.DataFrame] = []
    latency_frames: list[pd.DataFrame] = []
    execution_frames: list[pd.DataFrame] = []
    frontier_records: list[dict[str, Any]] = []
    client_latency_profiles: dict[str, tuple[bool, int, str]] = {}
    reference_contract: dict[str, Any] | None = None
    reference_run_id: str | None = None
    for run_id in run_ids:
        report = root / run_id
        table_path = report / "tables/table2.csv"
        metrics_path = report / "tables/metrics_long.csv"
        manifest_path = report / "run_manifest.json"
        if (
            not table_path.exists()
            or not metrics_path.exists()
            or not manifest_path.exists()
        ):
            raise FileNotFoundError(
                f"Report files are missing for {run_id!r}; run `acuitybench report` first"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("report_complete"):
            raise ValueError(
                f"Run {run_id!r} is incomplete and cannot be plotted as a frontier"
            )
        contract = _comparison_contract(manifest)
        if reference_contract is None:
            reference_contract = contract
            reference_run_id = run_id
        else:
            _assert_comparable(
                reference_contract,
                contract,
                reference_run_id=reference_run_id or run_ids[0],
                candidate_run_id=run_id,
            )
        table = pd.read_csv(table_path, dtype=str)
        table.insert(0, "Run ID", run_id)
        table_frames.append(table)
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, "run_id", run_id)
        metric_frames.append(metrics)
        optional_frames: dict[str, tuple[list[pd.DataFrame], pd.DataFrame]] = {}
        for filename, collection in (
            ("usage_and_cost.csv", usage_frames),
            ("latency_summary.csv", latency_frames),
            ("execution_summary.csv", execution_frames),
        ):
            path = report / "tables" / filename
            frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
            if path.exists():
                frame.insert(0, "run_id", run_id)
                collection.append(frame)
            optional_frames[filename] = (collection, frame)

        table_row = table.iloc[0]
        overall_metrics = metrics[
            (metrics["group_type"] == "overall")
            & metrics["task_type"].isin(["qa", "conv"])
        ].copy()
        overall_by_task = overall_metrics.set_index("task_type")
        accuracy_complete = {"qa", "conv"} <= set(overall_by_task.index)
        if accuracy_complete:
            qa_n = int(overall_by_task.loc["qa", "n_evaluable"])
            conv_n = int(overall_by_task.loc["conv", "n_evaluable"])
            qa_total = int(overall_by_task.loc["qa", "n_total"])
            conv_total = int(overall_by_task.loc["conv", "n_total"])
            accuracy_complete = (
                qa_n > 0
                and conv_n > 0
                and qa_n == qa_total
                and conv_n == conv_total
            )
        else:
            qa_n = conv_n = qa_total = conv_total = 0
        qa_exact = (
            float(overall_by_task.loc["qa", "exact"])
            if accuracy_complete
            else None
        )
        conv_exact = (
            float(overall_by_task.loc["conv", "exact"])
            if accuracy_complete
            else None
        )
        average_exact = (
            (qa_exact + conv_exact) / 2
            if qa_exact is not None and conv_exact is not None
            else None
        )
        usage = optional_frames["usage_and_cost.csv"][1]
        latency = optional_frames["latency_summary.csv"][1]
        target_usage = (
            usage[usage["phase"] == "target"] if not usage.empty else usage
        )

        target_calls = (
            float(pd.to_numeric(target_usage["calls"], errors="coerce").sum())
            if not target_usage.empty
            else 0.0
        )
        target_cost = (
            float(
                pd.to_numeric(
                    target_usage["estimated_cost_usd"], errors="coerce"
                ).sum()
            )
            if not target_usage.empty
            else None
        )
        (
            target_reasoning_tokens,
            target_reasoning_coverage,
        ) = _verified_reasoning_total(target_usage)
        inference_contract = manifest.get("inference_contract") or {}
        configured_inference = inference_contract.get("configured", {})
        observed_inference = inference_contract.get("observed", {})
        run_model_config = manifest.get("run", {}).get("model_config", {})
        reasoning_effort = configured_inference.get(
            "reasoning_effort", run_model_config.get("reasoning_effort")
        )
        reasoning_effort_basis = configured_inference.get(
            "reasoning_effort_basis",
            run_model_config.get("reasoning_effort_basis"),
        )
        completion_cap = configured_inference.get(
            "max_completion_tokens", run_model_config.get("max_output_tokens")
        )
        (
            retry_completion_cap,
            retry_completion_cap_basis,
        ) = _effective_retry_completion_cap(
            configured_inference, run_model_config
        )
        latency_profile_streaming = tuple(
            configured_inference.get("streaming_values", [])
        )
        latency_profile_concurrency = tuple(
            configured_inference.get("concurrency_values", [])
        )
        latency_profile_service_tier = configured_inference.get(
            "service_tier", run_model_config.get("service_tier")
        )
        latency_profile_execution_coverage = observed_inference.get(
            "execution_profile_coverage"
        )
        latency_profile_returned_service_tiers = tuple(
            observed_inference.get("returned_service_tiers", [])
        )
        latency_profile_returned_service_tier_coverage = (
            observed_inference.get("returned_service_tier_coverage")
        )
        cost_per_1000_calls = (
            target_cost / target_calls * 1000
            if target_cost is not None and target_calls > 0
            else None
        )
        instrumented_sources = {
            "instrumented_stream",
            "instrumented_nonstream",
        }
        service_latency_p95 = _frontier_latency_macro(
            latency,
            metric="service_latency_ms",
            percentile="p95_ms",
            required_sources=instrumented_sources,
        )
        provider_processing_p95 = _frontier_latency_macro(
            latency,
            metric="server_processing_ms",
            percentile="p95_ms",
        )
        legacy_provider_processing_p95 = _frontier_latency_macro(
            latency,
            metric="server_processing_ms",
            percentile="p95_ms",
            required_sources={"legacy_aggregate"},
        )
        latency_plot_p95 = (
            service_latency_p95
            if service_latency_p95 is not None
            else legacy_provider_processing_p95
        )
        latency_plot_source = (
            "client_service_latency"
            if service_latency_p95 is not None
            else "provider_processing_legacy_proxy"
            if legacy_provider_processing_p95 is not None
            else None
        )
        if latency_plot_source == "client_service_latency":
            client_latency_profiles[run_id] = _validated_client_latency_profile(
                run_id=run_id,
                streaming_values=latency_profile_streaming,
                concurrency_values=latency_profile_concurrency,
                execution_profile_coverage=(
                    latency_profile_execution_coverage
                ),
                configured_service_tier=latency_profile_service_tier,
                returned_service_tiers=(
                    latency_profile_returned_service_tiers
                ),
                returned_service_tier_coverage=(
                    latency_profile_returned_service_tier_coverage
                ),
            )
        frontier_records.append(
            {
                "run_id": run_id,
                "model": table_row.get("Model"),
                "average_exact": average_exact,
                "qa_exact": qa_exact,
                "conv_exact": conv_exact,
                "accuracy_complete": accuracy_complete,
                "qa_n_evaluable": qa_n,
                "conv_n_evaluable": conv_n,
                "target_cost_per_1000_successful_calls_usd": cost_per_1000_calls,
                # Compatibility alias retained for existing analysis notebooks.
                "target_cost_per_1000_tasks_usd": cost_per_1000_calls,
                "target_cost_completeness": (
                    ",".join(
                        sorted(
                            str(value)
                            for value in target_usage.get(
                                "cost_completeness", pd.Series(dtype=object)
                            ).dropna().unique()
                        )
                    )
                    or None
                ),
                "reasoning_effort": reasoning_effort,
                "reasoning_effort_basis": reasoning_effort_basis,
                "max_completion_tokens": completion_cap,
                "max_retry_completion_tokens": retry_completion_cap,
                "max_retry_completion_tokens_basis": (
                    retry_completion_cap_basis
                ),
                "target_reasoning_tokens": target_reasoning_tokens,
                "target_reasoning_tokens_per_successful_call": (
                    target_reasoning_tokens / target_calls
                    if target_reasoning_tokens is not None and target_calls > 0
                    else None
                ),
                "target_reasoning_token_coverage": target_reasoning_coverage,
                "latency_profile_streaming": ",".join(
                    str(value).lower() for value in latency_profile_streaming
                ),
                "latency_profile_concurrency": ",".join(
                    str(value) for value in latency_profile_concurrency
                ),
                "latency_profile_service_tier": latency_profile_service_tier,
                "latency_profile_returned_service_tiers": ",".join(
                    str(value)
                    for value in latency_profile_returned_service_tiers
                ),
                "latency_profile_execution_coverage": (
                    latency_profile_execution_coverage
                ),
                "latency_profile_returned_service_tier_coverage": (
                    latency_profile_returned_service_tier_coverage
                ),
                "service_latency_p50_macro_ms": _frontier_latency_macro(
                    latency,
                    metric="service_latency_ms",
                    percentile="p50_ms",
                    required_sources=instrumented_sources,
                ),
                "service_latency_p95_macro_ms": service_latency_p95,
                "qa_service_latency_p95_ms": _frontier_latency_value(
                    latency,
                    metric="service_latency_ms",
                    percentile="p95_ms",
                    task_type="qa",
                    required_sources=instrumented_sources,
                ),
                "conv_service_latency_p95_ms": _frontier_latency_value(
                    latency,
                    metric="service_latency_ms",
                    percentile="p95_ms",
                    task_type="conv",
                    required_sources=instrumented_sources,
                ),
                "ttft_p50_macro_ms": _frontier_latency_macro(
                    latency,
                    metric="ttft_ms",
                    percentile="p50_ms",
                    required_sources={"instrumented_stream"},
                ),
                "ttft_p95_macro_ms": _frontier_latency_macro(
                    latency,
                    metric="ttft_ms",
                    percentile="p95_ms",
                    required_sources={"instrumented_stream"},
                ),
                "qa_ttft_p95_ms": _frontier_latency_value(
                    latency,
                    metric="ttft_ms",
                    percentile="p95_ms",
                    task_type="qa",
                    required_sources={"instrumented_stream"},
                ),
                "conv_ttft_p95_ms": _frontier_latency_value(
                    latency,
                    metric="ttft_ms",
                    percentile="p95_ms",
                    task_type="conv",
                    required_sources={"instrumented_stream"},
                ),
                "server_processing_p95_macro_ms": provider_processing_p95,
                "legacy_provider_processing_p95_macro_ms": (
                    legacy_provider_processing_p95
                ),
                "latency_plot_p95_ms": latency_plot_p95,
                "latency_plot_source": latency_plot_source,
                "latency_plot_is_proxy": (
                    latency_plot_source == "provider_processing_legacy_proxy"
                ),
            }
        )
    _assert_latency_profiles_comparable(client_latency_profiles)
    combined_table = pd.concat(table_frames, ignore_index=True)
    combined_metrics = pd.concat(metric_frames, ignore_index=True)
    combined_table.to_csv(output / "table2.csv", index=False)
    (output / "table2.md").write_text(
        _markdown(combined_table), encoding="utf-8"
    )
    combined_metrics.to_csv(output / "metrics_long.csv", index=False)
    for filename, frames in (
        ("usage_and_cost.csv", usage_frames),
        ("latency_summary.csv", latency_frames),
        ("execution_summary.csv", execution_frames),
    ):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                output / filename, index=False
            )
    frontier_path = output / "frontier.csv"
    pd.DataFrame(frontier_records).to_csv(frontier_path, index=False)
    from acuitybench.plotting import write_frontier_charts

    write_frontier_charts(frontier_path, output)
    return output
