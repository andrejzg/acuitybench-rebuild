from __future__ import annotations

import copy
from pathlib import Path

import pytest

from acuitybench.interactive.schema_validation import (
    SchemaValidationError,
    validate_action_instance,
    validate_case_card,
)
from acuitybench.interactive.seed import load_case_cards


def test_every_committed_case_passes_the_strict_schema() -> None:
    root = Path(__file__).resolve().parents[1]
    cards = load_case_cards(root / "data/interactive/seed_v1/case_cards.jsonl")
    assert len(cards) == 100
    for card in cards:
        validate_case_card(card)


def test_case_schema_rejects_hidden_target_leakage_field() -> None:
    root = Path(__file__).resolve().parents[1]
    card = copy.deepcopy(
        load_case_cards(root / "data/interactive/seed_v1/case_cards.jsonl")[0]
    )
    card["diagnosis"] = "must not enter the policy wire format"
    with pytest.raises(SchemaValidationError, match="additional fields"):
        validate_case_card(card)


def test_action_schema_rejects_cross_branch_fields() -> None:
    action = {
        "schema_version": "interactive-action/v1",
        "type": "ASK",
        "question_id": "symptom.onset",
        "wording": "When did this start?",
        "acuity": "D",
    }
    with pytest.raises(SchemaValidationError, match="oneOf"):
        validate_action_instance(action)
