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
    *,
    target: ModelConfig,
    judge: ModelConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for phase, frame, config in (
        ("target", generations, target),
        ("judge", judgments, judge),
    ):
        if frame.empty:
            continue
        input_tokens = int(frame["input_tokens"].fillna(0).sum())
        cached_tokens = int(frame["cached_input_tokens"].fillna(0).sum())
        output_tokens = int(frame["output_tokens"].fillna(0).sum())
        reasoning_tokens = int(frame["reasoning_tokens"].fillna(0).sum())
        uncached_tokens = max(input_tokens - cached_tokens, 0)
        cost = (
            uncached_tokens * config.input_cost_per_million
            + cached_tokens * config.cached_input_cost_per_million
            + output_tokens * config.output_cost_per_million
        ) / 1_000_000
        records.append(
            {
                "phase": phase,
                "configured_model": config.api_model,
                "returned_models": ",".join(
                    sorted(str(value) for value in frame["returned_model"].dropna().unique())
                ),
                "calls": int((frame["status"] == "ok").sum()),
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
    generations = pd.DataFrame(generation_rows)
    judgments = pd.DataFrame(judgment_rows)
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
        generations, judgments, target=target_config, judge=judge_config
    )
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
    case_predictions.to_csv(exports / "case_predictions.csv", index=False)
    case_predictions.to_parquet(exports / "case_predictions.parquet", index=False)
    metrics.to_csv(tables / "metrics_long.csv", index=False)
    table2.to_csv(tables / "table2.csv", index=False)
    (tables / "table2.md").write_text(_markdown(table2), encoding="utf-8")
    boundary.to_csv(tables / "boundary_metrics.csv", index=False)
    usage.to_csv(tables / "usage_and_cost.csv", index=False)
    distributional.to_csv(tables / "distributional_metrics.csv", index=False)
    if not comparison.empty:
        comparison.to_csv(tables / "published_comparison.csv", index=False)
    for task in run["tasks"]:
        _confusion(case_predictions, task).to_csv(
            tables / f"confusion_{task}.csv"
        )

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
        "estimated_total_cost_usd": float(usage["estimated_cost_usd"].sum()),
        "pricing_note": "Estimate from token usage and prices recorded in configs/models.yaml at report time.",
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
        "physician_panel_distributional_metrics": distributional.to_dict(
            orient="records"
        ),
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_parts = [
        f"# AcuityBench report: {run_id}",
        "",
        f"Scope: {scope}. Requested model: `{run['api_model']}`. ",
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
        f"Estimated total: ${manifest['estimated_total_cost_usd']:.4f} USD.",
    ]
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
        distributional[distributional["split"] == "ambiguous"].copy()
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
    combined_table = pd.concat(table_frames, ignore_index=True)
    combined_metrics = pd.concat(metric_frames, ignore_index=True)
    combined_table.to_csv(output / "table2.csv", index=False)
    (output / "table2.md").write_text(
        _markdown(combined_table), encoding="utf-8"
    )
    combined_metrics.to_csv(output / "metrics_long.csv", index=False)
    return output
