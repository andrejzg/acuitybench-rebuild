from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


BASE_COLUMNS = [
    "dataset",
    "source_id",
    "original_prompt_text",
    "normalized_prompt_text",
    "original_label",
    "mapped_label",
    "normalized_label",
    "is_edge_case",
    "mapping_method",
    "notes",
]

RAW_EXPECTED = {
    "healthbench": 453,
    "pmr_reddit": 362,
    "pmr_synth": 60,
    "semigran": 45,
    "structured_triage": 78,
}

PMR_LEVEL_MAP = {1: "D", 2: "C", 3: "C", 4: "B", 5: "A", 6: "A"}
GOLD_TO_NORMALIZED = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "A/B": "A|B",
    "B/C": "B|C",
    "C/D": "C|D",
}
RATING_TO_ORDINAL = {
    "A": 1.0,
    "A|B": 1.5,
    "B": 2.0,
    "B|C": 2.5,
    "C": 3.0,
    "C|D": 3.5,
    "D": 4.0,
}
ORDINAL_TO_RATING = {value: key for key, value in RATING_TO_ORDINAL.items()}

CONVERSATION_NOTE = (
    "\nNote on conversations: this case shows a conversation between a [USER] "
    "(a patient, nurse, clinician, or other individual) and an [ASSISTANT] "
    "(an AI model). Your label is the acuity level the AI assistant should assign "
    "given the full conversation — the clinically correct triage response to the "
    "presented scenario. Read the entire conversation for context, but base your "
    "label on the clinical situation as described, not on what the AI actually said.\n"
)

INPUT_TYPE = {
    "healthbench": "conversation",
    "pmr_reddit": "conversation",
    "pmr_synth": "vignette",
    "semigran": "vignette",
    "structured_triage": "vignette",
}


@dataclass(frozen=True)
class BuildResult:
    normalized: pd.DataFrame
    transformed: pd.DataFrame
    raw_counts: dict[str, int]
    annotation_audit: dict[str, Any]
    normalized_csv: Path
    transformed_csv: Path
    parquet: Path


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _physician_category(tags: Iterable[str]) -> str | None:
    prefix = "physician_agreed_category:"
    return next((tag[len(prefix) :] for tag in tags if tag.startswith(prefix)), None)


def _health_prompt_json(prompt: list[dict[str, Any]]) -> str:
    return json.dumps([dict(message) for message in prompt], ensure_ascii=False)


def _pmr_level_to_mapped(level: float) -> tuple[str, bool]:
    fraction = round(level % 1, 6)
    if fraction == 0.5:
        lower, upper = int(level), int(level) + 1
        lower_label = PMR_LEVEL_MAP.get(lower)
        upper_label = PMR_LEVEL_MAP.get(upper)
        if lower_label and upper_label and lower_label != upper_label:
            return "|".join(sorted([lower_label, upper_label])), True
        return lower_label or "D", False
    rounded = round(level) if fraction != 0.0 else int(level)
    return PMR_LEVEL_MAP.get(rounded, "D"), False


def _extract_structured_vignette(text: str) -> str:
    start_marker = "About me:"
    end_marker = "Please answer in exactly"
    start = text.find(start_marker)
    if start != -1:
        text = text[start:]
    end = text.find(end_marker)
    if end != -1:
        text = text[:end]
    return text.strip()


def _base_record(
    *,
    dataset: str,
    source_id: str,
    original_text: str,
    normalized_text: str,
    original_label: str,
    mapped_label: str | None,
    normalized_label: str | None,
    is_edge_case: bool,
    mapping_method: str,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "source_id": source_id,
        "original_prompt_text": original_text,
        "normalized_prompt_text": normalized_text,
        "original_label": original_label,
        "mapped_label": mapped_label,
        "normalized_label": normalized_label,
        "is_edge_case": is_edge_case,
        "mapping_method": mapping_method,
        "notes": notes,
    }


def _load_healthbench(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]], int]:
    rows = _read_jsonl(path)
    emergency = [
        row for row in rows if "theme:emergency_referrals" in row["example_tags"]
    ]
    direct: list[dict[str, Any]] = []
    conditional: dict[str, dict[str, Any]] = {}
    non_emergent: dict[str, dict[str, Any]] = {}

    for row in emergency:
        category = _physician_category(row["example_tags"])
        prompt_json = _health_prompt_json(row["prompt"])
        source_id = str(row["prompt_id"])
        if category == "emergent":
            direct.append(
                _base_record(
                    dataset="healthbench",
                    source_id=source_id,
                    original_text=prompt_json,
                    normalized_text=prompt_json,
                    original_label=category,
                    mapped_label=None,
                    normalized_label="D",
                    is_edge_case=False,
                    mapping_method="direct",
                )
            )
            continue

        entry = {
            "dataset": "healthbench",
            "text": prompt_json,
            "original_label": category,
            "mapped_label": None,
            "is_edge_case": False,
        }
        if category == "conditionally-emergent":
            conditional[source_id] = entry
        elif category == "non-emergent":
            non_emergent[source_id] = entry
        else:
            raise ValueError(
                f"Unexpected HealthBench physician category for {source_id}: {category!r}"
            )

    lookup = {
        "healthbench_ce": conditional,
        "healthbench_ne": non_emergent,
    }
    return direct, lookup, len(emergency)


def _load_pmr_reddit(path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    pairs = pd.read_parquet(path)
    chosen = pairs[["chosen", "chosen_level", "difficulty"]].rename(
        columns={"chosen": "text", "chosen_level": "level"}
    )
    rejected = pairs[["rejected", "rejected_level", "difficulty"]].rename(
        columns={"rejected": "text", "rejected_level": "level"}
    )
    rows = pd.concat([chosen, rejected], ignore_index=True)
    rows["level"] = rows["level"].astype(float)
    rows = rows.drop_duplicates(subset=["text", "level"]).reset_index(drop=True)
    rows.insert(0, "source_id", rows["text"].map(_sha12))
    if rows["source_id"].duplicated().any():
        raise ValueError("PMR-Reddit has a source_id collision or conflicting label")

    lookup: dict[str, dict[str, Any]] = {}
    for row in rows.itertuples(index=False):
        mapped, is_edge = _pmr_level_to_mapped(float(row.level))
        lookup[str(row.source_id)] = {
            "dataset": "pmr_reddit",
            "text": row.text,
            "original_label": str(row.level),
            "mapped_label": mapped,
            "is_edge_case": is_edge,
        }
    return rows, lookup


def _load_pmr_synth(
    path_a: Path, path_b: Path
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows = pd.concat(
        [pd.read_parquet(path_a), pd.read_parquet(path_b)], ignore_index=True
    )
    rows.insert(0, "source_id", rows["text"].map(_sha12))
    if rows["source_id"].duplicated().any():
        raise ValueError("PMR-Synth has duplicate source IDs")

    lookup: dict[str, dict[str, Any]] = {}
    for row in rows.itertuples(index=False):
        mapped, is_edge = _pmr_level_to_mapped(float(row.level))
        lookup[str(row.source_id)] = {
            "dataset": "pmr_synth",
            "text": row.text,
            "original_label": str(row.level),
            "mapped_label": mapped,
            "is_edge_case": is_edge,
        }
    return rows, lookup


def _load_semigran_edits(path: Path) -> dict[str, str]:
    edits = pd.read_csv(path, dtype=str).fillna("")
    return {
        row.source_id.strip(): row.edited_text.strip()
        for row in edits.itertuples(index=False)
        if str(row.is_edited).strip().upper() == "TRUE"
        and str(row.edited_text).strip()
    }


def _load_semigran(
    path: Path, edits_path: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], int]:
    rows = _read_jsonl(path)
    edits = _load_semigran_edits(edits_path)
    direct: list[dict[str, Any]] = []
    review: dict[str, dict[str, Any]] = {}

    for row_id, row in enumerate(rows):
        diagnosis = row["correct_diagnosis"].replace("\n", " ").strip()
        source_id = f"{diagnosis}_{row_id}"
        original_text = row["case_description"]
        normalized_text = edits.get(source_id, original_text)
        urgency = row["urgency_level"]
        if urgency in {"em", "sc"}:
            label = "D" if urgency == "em" else "A"
            direct.append(
                _base_record(
                    dataset="semigran",
                    source_id=source_id,
                    original_text=original_text,
                    normalized_text=normalized_text,
                    original_label=urgency,
                    mapped_label=None,
                    normalized_label=label,
                    is_edge_case=False,
                    mapping_method="direct",
                )
            )
        elif urgency == "ne":
            review[source_id] = {
                "dataset": "semigran",
                "text": normalized_text,
                "original_label": urgency,
                "mapped_label": None,
                "is_edge_case": False,
            }
        else:
            raise ValueError(f"Unexpected Semigran urgency: {urgency!r}")
    return direct, review, len(rows)


def _load_structured(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows = pd.read_csv(path)
    rows = rows.loc[rows["variant_num"] == 1].copy()
    direct: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        gold = str(row["gold_triage"])
        normalized = GOLD_TO_NORMALIZED.get(gold)
        if normalized is None:
            raise ValueError(f"Unexpected structured triage label: {gold!r}")
        original_text = str(row["prompt_text"])
        direct.append(
            _base_record(
                dataset="structured_triage",
                source_id=str(row["case_num"]),
                original_text=original_text,
                normalized_text=_extract_structured_vignette(original_text),
                original_label=gold,
                mapped_label=None,
                normalized_label=normalized,
                is_edge_case="/" in gold,
                mapping_method="direct",
                notes=(
                    "Vignette extracted from original prompt: text from 'About me:' "
                    "through end of clinical scenario, stripping task-instruction "
                    "prefix and QA format suffix ('Please answer in exactly this "
                    "format:...')."
                ),
            )
        )
    return direct, len(rows)


def _rating_components(value: float) -> list[float]:
    if value == 1.5:
        return [1.0, 2.0]
    if value == 2.5:
        return [2.0, 3.0]
    if value == 3.5:
        return [3.0, 4.0]
    return [value]


def _set_distance(left: float, right: float) -> float:
    return min(
        (a - b) ** 2
        for a in _rating_components(left)
        for b in _rating_components(right)
    )


def _ordinal_median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot aggregate a case with no non-Remove ratings")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    lower, upper = ordered[middle - 1], ordered[middle]
    if lower == upper:
        return lower
    average = (lower + upper) / 2
    return average if average in {1.5, 2.5, 3.5} else upper


def _audit_released_annotations(labels: pd.DataFrame) -> dict[str, Any]:
    label_columns = [f"anon_label_{index}" for index in range(1, 6)]
    rater_id_columns = [f"anon_label_{index}_rater" for index in range(1, 6)]
    surveyed = labels.loc[labels["mapping_method"] != "excluded_manual"]
    if len(surveyed) != 675:
        raise ValueError(f"Expected 675 surveyed cases, found {len(surveyed)}")

    for row in surveyed.to_dict(orient="records"):
        votes = [row[column] for column in label_columns]
        rater_ids = [row[column] for column in rater_id_columns]
        if any(not vote for vote in votes) or any(not rater_id for rater_id in rater_ids):
            raise ValueError(f"Incomplete five-rater panel for {row['source_id']}")
        if len(set(rater_ids)) != 5:
            raise ValueError(f"Repeated rater alias within case {row['source_id']}")

        remove_votes = sum(vote == "Remove" for vote in votes)
        if remove_votes != int(row["remove_votes"]):
            raise ValueError(f"Remove-vote count differs for {row['source_id']}")
        if int(row["n_raters"]) != 5:
            raise ValueError(f"Unexpected rater count for {row['source_id']}")

        ordinals = [RATING_TO_ORDINAL[vote] for vote in votes if vote != "Remove"]
        distances = [
            _set_distance(left, right)
            for left, right in itertools.combinations(ordinals, 2)
        ]
        average_distance = sum(distances) / len(distances) if distances else 0.0
        if abs(average_distance - float(row["avg_sd"])) > 1e-9:
            raise ValueError(f"Average pairwise distance differs for {row['source_id']}")

        if remove_votes >= 3:
            expected_method = "excluded_review"
            expected_label = ""
        else:
            expected_method = (
                "excluded_disagreement" if average_distance > 0.75 else "median"
            )
            expected_label = ORDINAL_TO_RATING[_ordinal_median(ordinals)]
        if row["mapping_method"] != expected_method:
            raise ValueError(f"Annotation routing differs for {row['source_id']}")
        if row["normalized_label"] != expected_label:
            raise ValueError(f"Aggregated label differs for {row['source_id']}")

    rater_aliases = sorted(
        {
            value
            for column in rater_id_columns
            for value in labels[column].tolist()
            if value
        }
    )
    method_counts = {
        str(key): int(value)
        for key, value in labels["mapping_method"].value_counts().items()
    }
    expected_methods = {
        "median": 450,
        "excluded_disagreement": 217,
        "excluded_manual": 76,
        "excluded_review": 8,
    }
    if method_counts != expected_methods:
        raise ValueError(
            f"Released annotation routing differs: expected {expected_methods}, "
            f"got {method_counts}"
        )

    return {
        "released_rows": len(labels),
        "surveyed_cases": len(surveyed),
        "accepted_panel_cases": 667,
        "manual_exclusions": method_counts["excluded_manual"],
        "remove_vote_exclusions": method_counts["excluded_review"],
        "consensus_threshold": "avg_sd <= 0.75",
        "ambiguous_threshold": "avg_sd > 0.75",
        "consensus_cases": method_counts["median"],
        "ambiguous_cases": method_counts["excluded_disagreement"],
        "released_anonymized_rater_ids": len(rater_aliases),
        "rater_id_note": (
            "The release contains 22 anonymized rater identifiers while the paper "
            "reports 20 unique physicians; treat identifiers as aliases until the "
            "authors clarify the discrepancy."
        ),
    }


def _build_survey_rows(
    labels_path: Path,
    lookup: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    labels = pd.read_csv(labels_path, dtype=str).fillna("")
    annotation_audit = _audit_released_annotations(labels)
    rater_columns = [
        column for column in labels.columns if column.startswith("anon_label_")
    ]
    records: list[dict[str, Any]] = []
    missing: list[str] = []

    for row in labels.to_dict(orient="records"):
        source_dataset = row["source_dataset"]
        source_id = row["source_id"]
        raw = lookup.get(source_dataset, {}).get(source_id)
        if raw is None:
            missing.append(f"{source_dataset}_{source_id}")
            continue
        normalized_label = row["normalized_label"] or None
        record = _base_record(
            dataset=raw["dataset"],
            source_id=source_id,
            original_text=raw["text"],
            normalized_text=raw["text"],
            original_label=raw["original_label"],
            mapped_label=raw["mapped_label"],
            normalized_label=normalized_label,
            is_edge_case=bool(normalized_label and "|" in normalized_label),
            mapping_method=row["mapping_method"],
        )
        for column in rater_columns:
            record[column] = row[column] or None
        records.append(record)

    if missing:
        raise ValueError(
            f"{len(missing)} physician-label rows do not match raw cases: {missing[:5]}"
        )
    return records, rater_columns, annotation_audit


def _format_chat_as_text(messages: list[dict[str, Any]]) -> str:
    tags = {"user": "[USER]", "assistant": "[ASSISTANT]"}
    parts = []
    for message in messages:
        role = str(message["role"])
        tag = tags.get(role, f"[{role.upper()}]")
        parts.append(f"{tag}\n{message['content']}")
    return "\n\n".join(parts)


def build_conversational_prompt(
    dataset: str, text: str, template: str
) -> str:
    if dataset == "healthbench":
        return text
    if dataset == "pmr_reddit":
        return json.dumps([{"role": "user", "content": text}], ensure_ascii=False)
    content = template.replace("{{VIGNETTE}}", text)
    return json.dumps([{"role": "user", "content": content}], ensure_ascii=False)


def build_qa_prompt(dataset: str, text: str, template: str) -> str:
    input_type = INPUT_TYPE[dataset]
    if dataset == "healthbench":
        body = _format_chat_as_text(json.loads(text))
    elif dataset == "pmr_reddit":
        body = _format_chat_as_text([{"role": "user", "content": text}])
    else:
        body = text
    conversation_note = CONVERSATION_NOTE if input_type == "conversation" else ""
    return (
        template.replace("{{input_type}}", input_type)
        .replace("{{conversation_note}}", conversation_note)
        .replace("{{CONVERSATION_VIGNETTE}}", body)
    )


def _add_prompts(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    qa_template = (
        root / "configs" / "prompts" / "acuity_qa.txt"
    ).read_text(encoding="utf-8").strip()
    conversational_template = (
        root / "configs" / "prompts" / "acuity_conversational.txt"
    ).read_text(encoding="utf-8").strip()

    transformed = df.copy()
    transformed["split"] = transformed["mapping_method"].map(
        lambda value: "primary" if value in {"direct", "median"} else "ambiguous"
    )
    transformed["conversational_prompt"] = transformed.apply(
        lambda row: build_conversational_prompt(
            row["dataset"], row["normalized_prompt_text"], conversational_template
        ),
        axis=1,
    )
    transformed["qa_prompt"] = transformed.apply(
        lambda row: build_qa_prompt(
            row["dataset"], row["normalized_prompt_text"], qa_template
        ),
        axis=1,
    )
    for column in ("conversational_prompt", "qa_prompt"):
        if transformed[column].str.contains("{{", regex=False).any():
            raise ValueError(f"Unreplaced template placeholder in {column}")
    return transformed


def build(
    sources: dict[str, Path],
    *,
    root: Path,
    output_dir: Path,
) -> BuildResult:
    health_direct, health_lookup, health_count = _load_healthbench(
        sources["healthbench_consensus"]
    )
    reddit_rows, reddit_lookup = _load_pmr_reddit(sources["pmr_reddit_test"])
    synth_rows, synth_lookup = _load_pmr_synth(
        sources["pmr_synth_a"], sources["pmr_synth_b"]
    )
    semigran_direct, semigran_lookup, semigran_count = _load_semigran(
        sources["semigran_vignettes"], sources["semigran_edits"]
    )
    structured_direct, structured_count = _load_structured(
        sources["structured_triage"]
    )

    raw_counts = {
        "healthbench": health_count,
        "pmr_reddit": len(reddit_rows),
        "pmr_synth": len(synth_rows),
        "semigran": semigran_count,
        "structured_triage": structured_count,
    }
    if raw_counts != RAW_EXPECTED:
        raise ValueError(f"Raw source counts differ: expected {RAW_EXPECTED}, got {raw_counts}")

    lookup = {
        **health_lookup,
        "pmr_reddit": reddit_lookup,
        "pmr_synth": synth_lookup,
        "semigran": semigran_lookup,
    }
    survey_rows, rater_columns, annotation_audit = _build_survey_rows(
        sources["physician_labels"], lookup
    )

    # Match the ordering in the released build: alphabetically written direct
    # datasets (HealthBench, Semigran, structured triage), then survey rows.
    all_columns = BASE_COLUMNS + rater_columns
    merged = pd.DataFrame(
        health_direct + semigran_direct + structured_direct + survey_rows
    ).reindex(columns=all_columns)
    normalized = merged.loc[merged["normalized_label"].notna()].copy()
    normalized.reset_index(drop=True, inplace=True)
    transformed = _add_prompts(normalized, root)

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_csv = output_dir / "acuitybench.csv"
    transformed_csv = output_dir / "acuitybench_transformed.csv"
    parquet_path = output_dir / "acuitybench.parquet"

    normalized.to_csv(normalized_csv, index=False)
    transformed.to_csv(transformed_csv, index=False)
    friendly = transformed.copy()
    friendly.insert(
        0,
        "case_id",
        friendly["dataset"].astype(str) + "_" + friendly["source_id"].astype(str),
    )
    friendly.to_parquet(parquet_path, index=False)

    return BuildResult(
        normalized=normalized,
        transformed=transformed,
        raw_counts=raw_counts,
        annotation_audit=annotation_audit,
        normalized_csv=normalized_csv,
        transformed_csv=transformed_csv,
        parquet=parquet_path,
    )
