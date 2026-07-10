"""Physician-panel distributional metrics for primary and ambiguous cases."""

from __future__ import annotations

import itertools
import math
from typing import Any

import pandas as pd


LABELS = ("A", "B", "C", "D")
RANK = {label: index for index, label in enumerate(LABELS)}
RATER_COLUMNS = tuple(f"anon_label_{index}" for index in range(1, 6))
ORDINAL = {
    "A": 1.0,
    "A|B": 1.5,
    "B": 2.0,
    "B|C": 2.5,
    "C": 3.0,
    "C|D": 3.5,
    "D": 4.0,
}
SOFT = {
    "A": (1.0, 0.0, 0.0, 0.0),
    "A|B": (0.5, 0.5, 0.0, 0.0),
    "B": (0.0, 1.0, 0.0, 0.0),
    "B|C": (0.0, 0.5, 0.5, 0.0),
    "C": (0.0, 0.0, 1.0, 0.0),
    "C|D": (0.0, 0.0, 0.5, 0.5),
    "D": (0.0, 0.0, 0.0, 1.0),
}


def _mode_severe(values: list[str]) -> str | None:
    counts = {label: values.count(label) for label in LABELS}
    maximum = max(counts.values(), default=0)
    if maximum == 0:
        return None
    return max(
        (label for label, count in counts.items() if count == maximum),
        key=RANK.__getitem__,
    )


def _rater_distribution(values: list[Any]) -> list[float] | None:
    total = [0.0] * 4
    n = 0
    for value in values:
        label = value.strip() if isinstance(value, str) else ""
        if label not in SOFT:
            continue
        for index, weight in enumerate(SOFT[label]):
            total[index] += weight
        n += 1
    if not n:
        return None
    return [value / n for value in total]


def _model_distribution(values: list[str]) -> list[float] | None:
    valid = [value for value in values if value in RANK]
    if not valid:
        return None
    return [valid.count(label) / len(valid) for label in LABELS]


def jensen_shannon(p: list[float], q: list[float]) -> float:
    """Natural-log JSD, matching the released analysis (maximum ln(2))."""
    midpoint = [(left + right) / 2 for left, right in zip(p, q)]

    def kl(left: list[float], right: list[float]) -> float:
        return sum(
            value * math.log(value / other)
            for value, other in zip(left, right)
            if value > 0 and other > 0
        )

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def wasserstein_1(p: list[float], q: list[float]) -> float:
    return sum(
        abs(sum(p[: index + 1]) - sum(q[: index + 1]))
        for index in range(4)
    )


def _ordinal_median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    lower, upper = ordered[n // 2 - 1 : n // 2 + 1]
    if lower == upper:
        return lower
    midpoint = (lower + upper) / 2
    return midpoint if midpoint in {1.5, 2.5, 3.5} else upper


def _consensus_loo(
    rater_values: list[Any], mode_prediction: str | None, gold: str
) -> tuple[float | None, float | None]:
    if mode_prediction not in ORDINAL or gold not in ORDINAL:
        return None, None
    ordinals = [
        ORDINAL.get(value.strip()) if isinstance(value, str) else None
        for value in rater_values
    ]
    changes: list[tuple[bool, float]] = []
    for slot, value in enumerate(ordinals):
        if value is None:
            continue
        substituted = [
            ORDINAL[mode_prediction] if index == slot else other
            for index, other in enumerate(ordinals)
            if index == slot or other is not None
        ]
        delta = abs(_ordinal_median(substituted) - ORDINAL[gold])
        changes.append((delta > 0, delta))
    if not changes:
        return None, None
    return (
        sum(changed for changed, _ in changes) / len(changes),
        sum(delta for _, delta in changes) / len(changes),
    )


def _components(value: float) -> tuple[float, ...]:
    return {
        1.5: (1.0, 2.0),
        2.5: (2.0, 3.0),
        3.5: (3.0, 4.0),
    }.get(value, (value,))


def _set_distance(left: float, right: float) -> float:
    return min(
        (a - b) ** 2 for a in _components(left) for b in _components(right)
    )


def _reported_alpha(units: list[list[float | None]]) -> float:
    observed_total = 0.0
    observed_units = 0
    for unit in units:
        values = [value for value in unit if value is not None]
        if len(values) < 2:
            continue
        pairs = list(itertools.combinations(values, 2))
        observed_total += sum(_set_distance(a, b) for a, b in pairs) / len(pairs)
        observed_units += 1
    if not observed_units:
        return math.nan
    observed = observed_total / observed_units
    pooled = [value for unit in units for value in unit if value is not None]
    if len(pooled) < 2:
        return math.nan
    pooled_pairs = list(itertools.combinations(pooled, 2))
    expected = sum(_set_distance(a, b) for a, b in pooled_pairs) / len(pooled_pairs)
    return 1.0 if expected == 0 else 1.0 - observed / expected


def _alpha_loo(
    cases: pd.DataFrame, mode_predictions: dict[tuple[str, str], str | None]
) -> dict[str, Any]:
    units: list[list[float | None]] = []
    for row in cases.itertuples(index=False):
        values = [
            ORDINAL.get(str(getattr(row, column)).strip())
            if pd.notna(getattr(row, column))
            else None
            for column in RATER_COLUMNS
        ]
        if sum(value is not None for value in values) >= 2:
            units.append(values)
    baseline = _reported_alpha(units)
    loo_alphas: list[float] = []
    for slot, column in enumerate(RATER_COLUMNS):
        substituted_units: list[list[float | None]] = []
        for row in cases.itertuples(index=False):
            values = [
                ORDINAL.get(str(getattr(row, current)).strip())
                if pd.notna(getattr(row, current))
                else None
                for current in RATER_COLUMNS
            ]
            prediction = mode_predictions.get((str(row.dataset), str(row.source_id)))
            if values[slot] is not None and prediction in ORDINAL:
                values[slot] = ORDINAL[prediction]
            if sum(value is not None for value in values) >= 2:
                substituted_units.append(values)
        if len(substituted_units) >= 2:
            loo_alphas.append(_reported_alpha(substituted_units))
    mean_loo = sum(loo_alphas) / len(loo_alphas) if loo_alphas else math.nan
    return {
        "baseline_human_alpha": baseline,
        "mean_loo_alpha": mean_loo,
        "alpha_delta": mean_loo - baseline,
        "loo_alphas_per_slot": loo_alphas,
    }


def evaluate_panel_distribution(
    *,
    generations: pd.DataFrame,
    judgments: pd.DataFrame,
    benchmark: pd.DataFrame,
    task_type: str,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if task_type == "qa":
        samples = generations[generations["task_type"] == "qa"].copy()
        samples["sample_label"] = samples["parsed_label"]
    elif task_type == "conv":
        samples = generations[generations["task_type"] == "conv"].copy()
        samples = samples.merge(
            judgments[["run_id", "dataset", "source_id", "sample_idx", "judge_label"]],
            on=["run_id", "dataset", "source_id", "sample_idx"],
            how="left",
        )
        samples["sample_label"] = samples["judge_label"]
    else:
        raise ValueError(task_type)
    samples = samples[samples["split"] == split]
    panel = benchmark[
        ["dataset", "source_id", "normalized_label", *RATER_COLUMNS]
    ].copy()
    panel["source_id"] = panel["source_id"].astype(str)
    sample_groups = {
        (str(dataset), str(source_id)): group
        for (dataset, source_id), group in samples.groupby(["dataset", "source_id"])
    }
    selected_cases = panel[
        panel.apply(
            lambda row: (str(row["dataset"]), str(row["source_id"])) in sample_groups,
            axis=1,
        )
    ].copy()
    records: list[dict[str, Any]] = []
    mode_predictions: dict[tuple[str, str], str | None] = {}
    for row in selected_cases.itertuples(index=False):
        key = (str(row.dataset), str(row.source_id))
        group = sample_groups[key]
        predictions = [
            str(value) for value in group["sample_label"] if value in RANK
        ]
        mode = _mode_severe(predictions)
        mode_predictions[key] = mode
        rater_values = [getattr(row, column) for column in RATER_COLUMNS]
        human = _rater_distribution(rater_values)
        model = _model_distribution(predictions)
        change_rate, mean_delta = _consensus_loo(
            rater_values, mode, str(row.normalized_label)
        )
        records.append(
            {
                "dataset": row.dataset,
                "source_id": row.source_id,
                "normalized_label": row.normalized_label,
                "task_type": task_type,
                "split": split,
                "mode_prediction": mode,
                "n_samples": len(group),
                "n_valid": len(predictions),
                "has_rater_labels": human is not None,
                "human_distribution": human,
                "model_distribution": model,
                "jsd_natural_log": (
                    jensen_shannon(model, human) if model and human else None
                ),
                "wasserstein_1": wasserstein_1(model, human) if model and human else None,
                "rater_probability_of_mode": (
                    human[RANK[mode]] if human and mode in RANK else None
                ),
                "consensus_loo_change_rate": change_rate,
                "consensus_loo_mean_delta": mean_delta,
            }
        )
    result = pd.DataFrame(records)
    alpha = _alpha_loo(selected_cases, mode_predictions)
    summary = {
        "task_type": task_type,
        "split": split,
        "n": len(result),
        "n_evaluable": int(result["mode_prediction"].notna().sum()),
        "n_with_rater_labels": int(result["has_rater_labels"].sum()),
        "jsd_mean": float(result["jsd_natural_log"].mean()),
        "wasserstein_mean": float(result["wasserstein_1"].mean()),
        "rater_probability_mean": float(result["rater_probability_of_mode"].mean()),
        "consensus_loo_change_rate_mean": float(
            result["consensus_loo_change_rate"].mean()
        ),
        "consensus_loo_mean_delta_mean": float(
            result["consensus_loo_mean_delta"].mean()
        ),
        **{key: value for key, value in alpha.items() if key != "loo_alphas_per_slot"},
        "loo_alphas_per_slot": json_list(alpha["loo_alphas_per_slot"]),
    }
    return result, summary


def json_list(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in values) + "]"
