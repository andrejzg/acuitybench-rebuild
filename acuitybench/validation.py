from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_DATASETS = {
    "healthbench": 369,
    "pmr_reddit": 362,
    "pmr_synth": 60,
    "semigran": 45,
    "structured_triage": 78,
}
EXPECTED_METHODS = {
    "direct": 247,
    "median": 450,
    "excluded_disagreement": 217,
}
EXPECTED_SPLITS = {"primary": 697, "ambiguous": 217}
EXPECTED_LABELS = {
    "A": 108,
    "A|B": 48,
    "B": 140,
    "B|C": 74,
    "C": 135,
    "C|D": 123,
    "D": 286,
}
EXPECTED_PRIMARY_LABELS = {
    "A": 77,
    "A|B": 31,
    "B": 110,
    "B|C": 48,
    "C": 78,
    "C|D": 91,
    "D": 262,
}
EXPECTED_AMBIGUOUS_LABELS = {
    "A": 31,
    "A|B": 17,
    "B": 30,
    "B|C": 26,
    "C": 57,
    "C|D": 32,
    "D": 24,
}
VALID_LABELS = set(EXPECTED_LABELS)


class ValidationError(RuntimeError):
    pass


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().items()}


def _assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValidationError(f"{name} mismatch: expected {expected}, got {actual}")


def read_reference_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def validate_frame(df: pd.DataFrame, reference_ids: set[str]) -> dict[str, Any]:
    required = {
        "dataset",
        "source_id",
        "normalized_label",
        "is_edge_case",
        "mapping_method",
    }
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValidationError(f"Missing required columns: {sorted(missing_columns)}")

    _assert_equal("row count", len(df), 914)
    if df["normalized_label"].isna().any():
        raise ValidationError("Final benchmark contains null normalized labels")

    case_ids = df["dataset"].astype(str) + "_" + df["source_id"].astype(str)
    if case_ids.duplicated().any():
        duplicates = case_ids[case_ids.duplicated()].tolist()[:5]
        raise ValidationError(f"Duplicate case IDs: {duplicates}")
    actual_ids = set(case_ids)
    if actual_ids != reference_ids:
        missing = sorted(reference_ids - actual_ids)[:5]
        extra = sorted(actual_ids - reference_ids)[:5]
        raise ValidationError(
            f"Reference case IDs differ: {len(reference_ids - actual_ids)} missing "
            f"({missing}), {len(actual_ids - reference_ids)} extra ({extra})"
        )

    splits = df.get("split")
    if splits is None:
        splits = df["mapping_method"].map(
            lambda value: "primary" if value in {"direct", "median"} else "ambiguous"
        )

    dataset_counts = _counts(df["dataset"])
    method_counts = _counts(df["mapping_method"])
    split_counts = _counts(splits)
    label_counts = _counts(df["normalized_label"])
    primary_labels = _counts(df.loc[splits == "primary", "normalized_label"])
    ambiguous_labels = _counts(df.loc[splits == "ambiguous", "normalized_label"])

    _assert_equal("dataset counts", dataset_counts, EXPECTED_DATASETS)
    _assert_equal("mapping method counts", method_counts, EXPECTED_METHODS)
    _assert_equal("split counts", split_counts, EXPECTED_SPLITS)
    _assert_equal("label counts", label_counts, EXPECTED_LABELS)
    _assert_equal("primary label counts", primary_labels, EXPECTED_PRIMARY_LABELS)
    _assert_equal(
        "ambiguous label counts", ambiguous_labels, EXPECTED_AMBIGUOUS_LABELS
    )

    invalid_labels = set(df["normalized_label"].astype(str)) - VALID_LABELS
    if invalid_labels:
        raise ValidationError(f"Invalid normalized labels: {sorted(invalid_labels)}")

    edge_expected = df["normalized_label"].astype(str).str.contains("|", regex=False)
    edge_actual = df["is_edge_case"].map(
        lambda value: value if isinstance(value, bool) else str(value).lower() == "true"
    )
    if not edge_expected.equals(edge_actual):
        raise ValidationError("is_edge_case is inconsistent with normalized_label")

    rater_columns = [f"anon_label_{index}" for index in range(1, 6)]
    if not set(rater_columns).issubset(df.columns):
        raise ValidationError("Released five-physician rating columns are missing")
    panel_mask = df["mapping_method"].isin({"median", "excluded_disagreement"})
    panel_counts = df.loc[panel_mask, rater_columns].notna().sum(axis=1)
    if len(panel_counts) != 667 or not panel_counts.eq(5).all():
        raise ValidationError("Physician-panel coverage is not five labels for 667 cases")

    clear_primary = int(
        ((splits == "primary") & ~df["normalized_label"].str.contains("|", regex=False)).sum()
    )
    boundary_primary = int(
        ((splits == "primary") & df["normalized_label"].str.contains("|", regex=False)).sum()
    )
    _assert_equal("clear primary cases", clear_primary, 527)
    _assert_equal("boundary primary cases", boundary_primary, 170)

    return {
        "rows": len(df),
        "raw_source_cases": 998,
        "reference_ids": len(actual_ids),
        "dataset_counts": dataset_counts,
        "mapping_method_counts": method_counts,
        "split_counts": split_counts,
        "label_counts": label_counts,
        "primary_label_counts": primary_labels,
        "ambiguous_label_counts": ambiguous_labels,
        "physician_panel_cases": int(panel_mask.sum()),
        "clear_primary_cases": clear_primary,
        "boundary_primary_cases": boundary_primary,
    }
