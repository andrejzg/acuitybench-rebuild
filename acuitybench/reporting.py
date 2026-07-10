"""Audit-friendly exports and paper-compatible AcuityBench tables."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from acuitybench.distributional import evaluate_panel_distribution
from acuitybench.evaluation import default_store_path
from acuitybench.models import ModelConfig, ModelRegistry
from acuitybench.sources import project_root
from acuitybench.store import EvaluationStore


LABELS = ("A", "B", "C", "D")
RANK = {label: index for index, label in enumerate(LABELS)}
PAPER_GPT5_MINI = {
    "qa": {"exact": 0.780, "over": 0.055, "under": 0.165},
    "conv": {"exact": 0.677, "over": 0.036, "under": 0.286},
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
        reasoning_tokens = int(
            pd.to_numeric(usage_rows["reasoning_tokens"], errors="coerce")
            .fillna(0)
            .sum()
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
        records.append(
            {
                "phase": phase,
                "configured_model": config.api_model,
                "returned_models": ",".join(
                    sorted(str(value) for value in returned_models.dropna().unique())
                ),
                "calls": int(
                    frame.get("status", pd.Series(dtype=object)).eq("ok").sum()
                ),
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


def _paper_comparison(run: dict[str, Any], metrics: pd.DataFrame) -> pd.DataFrame:
    if run["model_id"] != "gpt-5-mini":
        return pd.DataFrame()
    overall = metrics[metrics["group_type"] == "overall"].set_index("task_type")
    records = []
    for task, published in PAPER_GPT5_MINI.items():
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
    full_contract = (
        run["selected_cases"] == 914
        and run["samples"] == 5
        and set(run["tasks"]) == {"qa", "conv"}
    )
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
    if not comparison.empty:
        comparison.to_csv(tables / "published_comparison.csv", index=False)
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
        "returned_judge_models": sorted(
            str(value) for value in judgments.get("returned_model", pd.Series(dtype=str)).dropna().unique()
        ),
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
            "parser": r"first ACUITY\s*[:\-]\s*([A-D]) match, case-insensitive",
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
    for run_id in run_ids:
        report = root / run_id
        table_path = report / "tables/table2.csv"
        metrics_path = report / "tables/metrics_long.csv"
        if not table_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(
                f"Report files are missing for {run_id!r}; run `acuitybench report` first"
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
        exact_values = pd.to_numeric(
            pd.Series([table_row.get("QA Exact"), table_row.get("Conv Exact")]),
            errors="coerce",
        ).dropna()
        usage = optional_frames["usage_and_cost.csv"][1]
        latency = optional_frames["latency_summary.csv"][1]
        target_usage = (
            usage[usage["phase"] == "target"] if not usage.empty else usage
        )

        def latency_value(
            metric: str,
            percentile: str,
            task_type: str | None = None,
        ) -> float | None:
            if latency.empty:
                return None
            selected = latency[
                (latency["phase"] == "target")
                & (latency["metric"] == metric)
            ]
            if task_type is not None:
                selected = selected[selected["task_type"] == task_type]
            values = pd.to_numeric(
                selected[percentile],
                errors="coerce",
            ).dropna()
            return float(values.mean()) if not values.empty else None

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
        frontier_records.append(
            {
                "run_id": run_id,
                "model": table_row.get("Model"),
                "average_exact": (
                    float(exact_values.mean()) if not exact_values.empty else None
                ),
                "qa_exact": pd.to_numeric(
                    pd.Series([table_row.get("QA Exact")]), errors="coerce"
                ).iloc[0],
                "conv_exact": pd.to_numeric(
                    pd.Series([table_row.get("Conv Exact")]), errors="coerce"
                ).iloc[0],
                "target_cost_per_1000_tasks_usd": (
                    target_cost / target_calls * 1000
                    if target_cost is not None and target_calls > 0
                    else None
                ),
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
                "service_latency_p50_macro_ms": latency_value(
                    "service_latency_ms", "p50_ms"
                ),
                "service_latency_p95_macro_ms": latency_value(
                    "service_latency_ms", "p95_ms"
                ),
                "qa_service_latency_p95_ms": latency_value(
                    "service_latency_ms", "p95_ms", "qa"
                ),
                "conv_service_latency_p95_ms": latency_value(
                    "service_latency_ms", "p95_ms", "conv"
                ),
                "ttft_p50_macro_ms": latency_value("ttft_ms", "p50_ms"),
                "ttft_p95_macro_ms": latency_value("ttft_ms", "p95_ms"),
                "qa_ttft_p95_ms": latency_value("ttft_ms", "p95_ms", "qa"),
                "conv_ttft_p95_ms": latency_value(
                    "ttft_ms", "p95_ms", "conv"
                ),
                "server_processing_p95_macro_ms": latency_value(
                    "server_processing_ms", "p95_ms"
                ),
            }
        )
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
    pd.DataFrame(frontier_records).to_csv(output / "frontier.csv", index=False)
    return output
