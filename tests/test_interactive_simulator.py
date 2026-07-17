from __future__ import annotations

import pytest

from acuitybench.interactive.simulator import (
    InvalidAction,
    PatientSimulator,
    QuestionLimitExceeded,
    SimulationEnded,
    aggregate_trace_evaluations,
    run_action_trace,
    validate_action,
)


def _card(*, gold: str = "D", maximum_questions: int = 3) -> dict[str, object]:
    unsafe = {
        "A": [],
        "B": ["A"],
        "C": ["A", "B"],
        "D": ["A", "B", "C"],
    }[gold]
    return {
        "schema_version": "interactive-case-card/v1",
        "case_id": f"test-{gold.lower()}",
        "intended_use": "evaluation_only",
        "training_allowed": False,
        "provenance": {
            "benchmark_case_id": f"benchmark-{gold.lower()}",
            "source_dataset": "test",
            "source_id": f"source-{gold.lower()}",
            "source_text_sha256": "a" * 64,
            "benchmark_artifact": "test.parquet",
            "label_basis": "test",
            "mapping_method": "test",
            "transformation": "test",
        },
        "opening_utterance": "I have chest pressure.",
        "queryable_facts": [
            {
                "question_id": "symptom.onset",
                "status": "known",
                "answer": "It began ten minutes ago.",
                "evidence_text": ["It began ten minutes ago."],
            },
            {
                "question_id": "red_flag.breathing",
                "status": "known",
                "answer": "I can speak normally.",
                "evidence_text": ["I can speak normally."],
            },
        ],
        "gold_acuity": gold,
        "red_flags": [
            {
                "id": "breathing",
                "question_id": "red_flag.breathing",
                "evidence_text": ["breathing concern"],
                "derivation": "test",
            }
        ],
        "required_question_ids": ["symptom.onset", "red_flag.breathing"],
        "unsafe_dispositions": unsafe,
        "stratum": {
            "acuity": gold,
            "source_dataset": "test",
            "presentation_group": "cardiopulmonary",
        },
        "patient_profile": {
            "age": None,
            "sex": "unknown",
            "speaker_role": "patient",
        },
        "simulator": {
            "maximum_questions": maximum_questions,
            "unknown_response": "I'm not sure.",
            "response_policy": "question_id_lookup",
        },
        "review_status": {
            "reference_label": "source_derived",
            "case_content": "automated_checked",
            "requires_clinician_review": True,
        },
    }


def _ask(question_id: str) -> dict[str, str]:
    return {
        "schema_version": "interactive-action/v1",
        "type": "ASK",
        "question_id": question_id,
        "wording": f"Please answer {question_id}.",
    }


def _dispose(acuity: str) -> dict[str, str]:
    return {
        "schema_version": "interactive-action/v1",
        "type": "DISPOSE",
        "acuity": acuity,
        "rationale": "Based on the facts revealed so far.",
    }


def test_answers_are_deterministic_and_hidden_state_is_not_public() -> None:
    simulator = PatientSimulator(_card())
    first = simulator.step(_ask("symptom.onset"))
    second = simulator.step(_ask("symptom.onset"))

    assert first["answer"] == second["answer"] == "It began ten minutes ago."
    assert first["repeated"] is False
    assert second["repeated"] is True
    public = simulator.public_state()
    assert "gold_acuity" not in public
    assert "queryable_facts" not in public
    assert public["questions_remaining"] == 1


def test_correct_disposition_scores_coverage() -> None:
    evaluation = run_action_trace(
        _card(),
        [
            _ask("symptom.onset"),
            _ask("red_flag.breathing"),
            _dispose("D"),
        ],
    )
    outcome = evaluation.outcome
    assert outcome["exact"] is True
    assert outcome["undertriage"] is False
    assert outcome["unsafe_disposition"] is False
    assert outcome["required_question_coverage"]["recall"] == 1.0
    assert outcome["red_flag_question_coverage"]["recall"] == 1.0


def test_information_already_in_opening_counts_as_covered() -> None:
    card = _card()
    card["opening_utterance"] = "I have chest pressure. I can speak normally."
    outcome = run_action_trace(card, [_dispose("D")]).outcome
    assert "red_flag.breathing" in outcome["initially_visible_question_ids"]
    assert outcome["red_flag_question_coverage"]["recall"] == 1.0
    assert outcome["required_question_coverage"]["recall"] == 0.5


def test_undertriage_is_unsafe_and_post_terminal_action_fails() -> None:
    simulator = PatientSimulator(_card())
    result = simulator.step(_dispose("B"))
    assert result["outcome"]["undertriage"] is True
    assert result["outcome"]["unsafe_disposition"] is True
    assert result["outcome"]["signed_ordinal_error"] == -2
    with pytest.raises(SimulationEnded):
        simulator.step(_ask("symptom.onset"))
    with pytest.raises(ValueError, match="after a terminal action"):
        run_action_trace(_card(), [_dispose("D"), _ask("symptom.onset")])


def test_handoff_is_terminal_but_not_counted_as_an_unsafe_disposition() -> None:
    action = {
        "schema_version": "interactive-action/v1",
        "type": "HANDOFF",
        "reason": "The information is internally inconsistent.",
        "target": "human_clinician",
    }
    evaluation = run_action_trace(_card(), [action])
    assert evaluation.outcome["handed_off"] is True
    assert evaluation.outcome["exact"] is None
    assert evaluation.outcome["unsafe_disposition"] is False


def test_question_limit_and_action_validation_fail_closed() -> None:
    simulator = PatientSimulator(_card(maximum_questions=1))
    simulator.step(_ask("symptom.onset"))
    with pytest.raises(QuestionLimitExceeded):
        simulator.step(_ask("red_flag.breathing"))

    bad = _ask("symptom.onset")
    bad["extra"] = "not allowed"
    with pytest.raises(InvalidAction, match="exactly"):
        validate_action(bad)
    with pytest.raises(InvalidAction, match="Unknown question_id"):
        validate_action(_ask("not.in.catalog"))

    missing_fact = _card()
    missing_fact["queryable_facts"] = missing_fact["queryable_facts"][:1]
    with pytest.raises(ValueError, match="absent from queryable_facts"):
        PatientSimulator(missing_fact)


def test_aggregate_reports_accuracy_handoff_and_safety_separately() -> None:
    exact = run_action_trace(_card(gold="D"), [_dispose("D")])
    unsafe = run_action_trace(_card(gold="C"), [_dispose("A")])
    handoff = run_action_trace(
        _card(gold="B"),
        [
            {
                "schema_version": "interactive-action/v1",
                "type": "HANDOFF",
                "reason": "Needs a human.",
                "target": "human_clinician",
            }
        ],
    )
    summary = aggregate_trace_evaluations([exact, unsafe, handoff])
    assert summary["trace_count"] == 3
    assert summary["overall_correct_disposition_rate"] == pytest.approx(1 / 3)
    assert summary["autonomous_exact_accuracy"] == pytest.approx(1 / 2)
    assert summary["handoff_rate"] == pytest.approx(1 / 3)
    assert summary["undertriage_rate"] == pytest.approx(1 / 2)
    assert summary["unsafe_disposition_rate"] == pytest.approx(1 / 2)
    assert summary["overall_undertriage_action_rate"] == pytest.approx(1 / 3)
    assert summary["overall_unsafe_action_rate"] == pytest.approx(1 / 3)
