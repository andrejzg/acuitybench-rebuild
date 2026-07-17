from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from acuitybench.interactive.seed import (
    _positive_red_flag,
    _profile,
    build_seed_set,
    extract_user_text,
    load_case_cards,
    validate_seed_set,
)


def test_healthbench_extraction_keeps_only_user_messages() -> None:
    prompt = json.dumps(
        [
            {"role": "user", "content": "My chest hurts."},
            {"role": "assistant", "content": "You should go to hospital."},
            {"role": "user", "content": "It started ten minutes ago."},
        ]
    )
    result = extract_user_text("healthbench", prompt)
    assert result == "My chest hurts.\n\nIt started ten minutes ago."
    assert "hospital" not in result


def test_seed_build_is_byte_deterministic_and_valid(tmp_path: Path) -> None:
    first = build_seed_set(output_dir=tmp_path / "first")
    second = build_seed_set(output_dir=tmp_path / "second")

    assert first.case_count == second.case_count == 100
    assert first.case_cards_path.read_bytes() == second.case_cards_path.read_bytes()
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    validation = validate_seed_set(
        case_cards_path=first.case_cards_path,
        manifest_path=first.manifest_path,
    )
    assert validation["label_counts"] == {"A": 25, "B": 25, "C": 25, "D": 25}
    assert validation["question_count"] == 33
    assert validation["training_allowed"] is False
    assert validation["clinician_content_reviewed_cases"] == 0


def test_seed_cards_preserve_guardrails_and_provenance(tmp_path: Path) -> None:
    result = build_seed_set(output_dir=tmp_path / "seed")
    cards = load_case_cards(result.case_cards_path)

    assert {card["intended_use"] for card in cards} == {"evaluation_only"}
    assert {card["training_allowed"] for card in cards} == {False}
    assert {card["review_status"]["requires_clinician_review"] for card in cards} == {True}
    assert {card["review_status"]["case_content"] for card in cards} == {
        "automated_checked"
    }
    assert [card["case_id"] for card in cards] == [
        f"acuity-interactive-v1-{index:03d}" for index in range(1, 101)
    ]
    assert all(
        not re.search(r"(?:^|[-_.])[abcd](?:[-_.]|$)", card["case_id"], re.IGNORECASE)
        for card in cards
    )
    assert {card["provenance"]["source_dataset"] for card in cards} == {
        "healthbench",
        "pmr_reddit",
    }
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", card["provenance"]["source_text_sha256"])
        for card in cards
    )
    assert all(
        "gold_acuity" not in fact
        for card in cards
        for fact in card["queryable_facts"]
    )
    visible_text = "\n".join(
        [card["opening_utterance"] for card in cards]
        + [fact["answer"] for card in cards for fact in card["queryable_facts"]]
    )
    assert not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", visible_text)
    assert not re.search(r"https?://|www\.", visible_text, flags=re.IGNORECASE)


@pytest.mark.parametrize(
    ("question_id", "fragment"),
    [
        ("red_flag.breathing", "I don't feel short of breath."),
        ("red_flag.chest_pain", "The exam was negative for chest pain."),
        ("red_flag.chest_pain", "Chest pain is absent."),
        ("red_flag.bleeding", "I don’t see any blood."),
        (
            "red_flag.breathing",
            "They didn't say if there's any shortness of breath.",
        ),
        ("red_flag.consciousness", "Some days it is just a faint throbbing."),
        ("red_flag.bleeding", "My sister had a hemorrhagic stroke."),
        (
            "red_flag.neurologic",
            "From what I've learned online, the symptoms are rather simple: "
            "weakness or paralysis on one side of the body, eyesight problems, "
            "confusion, slurred speech and some degree of mental impairment.",
        ),
        (
            "red_flag.bleeding",
            "There was some bleeding but not my normal heavy bloody nightmare.",
        ),
    ],
)
def test_red_flag_rules_reject_negated_or_contextual_mentions(
    question_id: str, fragment: str
) -> None:
    assert _positive_red_flag(question_id, fragment) is False


@pytest.mark.parametrize(
    ("question_id", "fragment"),
    [
        ("red_flag.breathing", "I am struggling to breathe."),
        ("red_flag.breathing", "No fever, but I am struggling to breathe."),
        (
            "red_flag.chest_pain",
            "No fever or cough, but I have crushing chest pain.",
        ),
        (
            "red_flag.neurologic",
            "No fever, but I have new one-sided weakness.",
        ),
        ("red_flag.neurologic", "No fever. New one-sided weakness."),
        ("red_flag.chest_pain", "No fever. Crushing chest pain."),
        (
            "red_flag.chest_pain",
            "Is this serious? I have crushing chest pain.",
        ),
        (
            "red_flag.breathing",
            "Is this an emergency? I am struggling to breathe.",
        ),
        (
            "red_flag.neurologic",
            "I am worried about this because I have new facial droop.",
        ),
        ("red_flag.bleeding", "The heavy bleeding will not stop."),
        ("red_flag.bleeding", "No pain, but the heavy bleeding will not stop."),
        (
            "red_flag.bleeding",
            "There is a lot of bleeding that seems heavier than usual, with big clots.",
        ),
        ("red_flag.bleeding", "My sister has heavy bleeding that will not stop."),
        ("red_flag.bleeding", "My mother is vomiting blood."),
        ("red_flag.bleeding", "I just puked up blood."),
        (
            "red_flag.bleeding",
            "Could this be serious? The heavy bleeding will not stop.",
        ),
        ("red_flag.consciousness", "I fainted twice this morning."),
        ("red_flag.consciousness", "He can't stay conscious."),
        ("red_flag.consciousness", "She is barely conscious."),
        ("red_flag.consciousness", "I can barely keep my eyes open."),
        ("red_flag.consciousness", "I am basically blacking out."),
        ("red_flag.consciousness", "He keeps slipping in and out of it."),
        ("red_flag.neurologic", "He can't really get words out."),
        ("red_flag.neurologic", "She's slurring her words."),
        ("red_flag.neurologic", "Their speech is slurred."),
        ("red_flag.mental_health", "I attempted to hang myself twice."),
    ],
)
def test_red_flag_rules_keep_positive_safety_signals(
    question_id: str, fragment: str
) -> None:
    assert _positive_red_flag(question_id, fragment) is True


@pytest.mark.parametrize(
    ("opening", "expected_role"),
    [
        ("Patient is my father and he has chest pain.", "caregiver"),
        ("Postpartum bleeding. My sister has soaked three pads.", "caregiver"),
        ("Bf with Crohn's is having new symptoms.", "caregiver"),
        ("Thomas had a sudden severe coughing fit.", "caregiver"),
        ("I have a 57-year-old patient with new weakness.", "clinician"),
        ("I have a roommate who keeps passing out.", "caregiver"),
        ("I have chest pain that began this morning.", "patient"),
        ("What is recommended? Should I go to hospital?", "patient"),
        ("Am I pregnant? I had sex with my boyfriend.", "patient"),
    ],
)
def test_profile_identifies_the_person_speaking(
    opening: str, expected_role: str
) -> None:
    assert _profile(opening, opening)["speaker_role"] == expected_role


def test_committed_seed_artifact_validates_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    case_cards = root / "data/interactive/seed_v1/case_cards.jsonl"
    manifest = root / "data/interactive/seed_v1/manifest.json"
    if not case_cards.exists() or not manifest.exists():
        return
    result = validate_seed_set(case_cards_path=case_cards, manifest_path=manifest)
    assert result["case_count"] == 100
