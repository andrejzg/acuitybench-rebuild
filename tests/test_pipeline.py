from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from acuitybench.pipeline import (
    _extract_structured_vignette,
    _pmr_level_to_mapped,
    build_conversational_prompt,
    build_qa_prompt,
)
from acuitybench.sources import project_root
from acuitybench.validation import read_reference_ids, validate_frame


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (1.0, ("D", False)),
        (1.5, ("C|D", True)),
        (3.5, ("B|C", True)),
        (4.5, ("A|B", True)),
        (5.5, ("A", False)),
        (4.333333333333333, ("B", False)),
        (4.666666666666667, ("A", False)),
    ],
)
def test_pmr_mapping(level: float, expected: tuple[str, bool]) -> None:
    assert _pmr_level_to_mapped(level) == expected


def test_structured_vignette_extraction() -> None:
    prompt = (
        "Ignore this prefix\nAbout me: I am 42.\nMy situation: pain.\n"
        "Please answer in exactly this format:\nACUITY: A"
    )
    assert _extract_structured_vignette(prompt) == (
        "About me: I am 42.\nMy situation: pain."
    )


def test_prompt_routing_has_no_placeholders() -> None:
    root = project_root()
    qa_template = (root / "configs/prompts/acuity_qa.txt").read_text().strip()
    conv_template = (
        root / "configs/prompts/acuity_conversational.txt"
    ).read_text().strip()
    health_text = json.dumps([{"content": "Chest pain", "role": "user"}])
    qa = build_qa_prompt("healthbench", health_text, qa_template)
    conversation = build_conversational_prompt(
        "structured_triage", "A vignette", conv_template
    )
    assert "[USER]\nChest pain" in qa
    assert "Note on conversations" in qa
    assert "{{" not in qa
    assert json.loads(conversation)[0]["content"].endswith(
        "What is the urgency of this situation as described?"
    )


def test_generated_dataset_when_present() -> None:
    root = project_root()
    output = root / "data/processed/acuitybench_transformed.csv"
    reference_path = root / "data/cache/sources/reference_case_ids.txt"
    if not output.exists() or not reference_path.exists():
        pytest.skip("Run the benchmark build for the integration assertion")
    frame = pd.read_csv(output)
    result = validate_frame(frame, read_reference_ids(reference_path))
    assert result["rows"] == 914
    assert result["physician_panel_cases"] == 667
