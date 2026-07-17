"""Deterministic patient simulation and trajectory scoring for case-card v1."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from acuitybench.interactive.seed import (
    ACTION_SCHEMA_VERSION,
    CASE_SCHEMA_VERSION,
    LABELS,
    SEVERITY,
    load_question_catalog,
)
from acuitybench.interactive.schema_validation import (
    validate_action_instance,
    validate_case_card,
)


class InvalidAction(ValueError):
    """Raised when an action does not conform to interactive-action/v1."""


class SimulationEnded(RuntimeError):
    """Raised when an action is submitted after a terminal action."""


class QuestionLimitExceeded(RuntimeError):
    """Raised when ASK exceeds the case-card question budget."""


@dataclass(frozen=True)
class TraceEvaluation:
    case_id: str
    terminal_action: str
    transcript: tuple[dict[str, Any], ...]
    outcome: dict[str, Any]


def _nonempty_string(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidAction(f"{label} must be a non-empty string")
    if len(value) > maximum:
        raise InvalidAction(f"{label} must be at most {maximum} characters")
    return value


def validate_action(
    action: Mapping[str, Any],
    *,
    question_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate and copy one closed action object without extra dependencies."""

    if not isinstance(action, Mapping):
        raise InvalidAction("Action must be an object")
    value = dict(action)
    if value.get("schema_version") != ACTION_SCHEMA_VERSION:
        raise InvalidAction(f"schema_version must be {ACTION_SCHEMA_VERSION!r}")
    action_type = value.get("type")
    if question_ids is None:
        question_ids = {
            str(question["id"])
            for question in load_question_catalog()["questions"]
        }

    if action_type == "ASK":
        expected = {"schema_version", "type", "question_id", "wording"}
        if set(value) != expected:
            raise InvalidAction(f"ASK fields must be exactly {sorted(expected)}")
        question_id = _nonempty_string(value["question_id"], "question_id", maximum=200)
        if question_id not in question_ids:
            raise InvalidAction(f"Unknown question_id: {question_id}")
        _nonempty_string(value["wording"], "wording", maximum=1000)
    elif action_type == "DISPOSE":
        expected = {"schema_version", "type", "acuity", "rationale"}
        if set(value) != expected:
            raise InvalidAction(f"DISPOSE fields must be exactly {sorted(expected)}")
        if value["acuity"] not in LABELS:
            raise InvalidAction("acuity must be one of A, B, C, or D")
        _nonempty_string(value["rationale"], "rationale", maximum=2000)
    elif action_type == "HANDOFF":
        expected = {"schema_version", "type", "reason", "target"}
        if set(value) != expected:
            raise InvalidAction(f"HANDOFF fields must be exactly {sorted(expected)}")
        _nonempty_string(value["reason"], "reason", maximum=1000)
        if value["target"] != "human_clinician":
            raise InvalidAction("HANDOFF target must be 'human_clinician'")
    else:
        raise InvalidAction("type must be ASK, DISPOSE, or HANDOFF")
    try:
        validate_action_instance(value)
    except ValueError as exc:
        raise InvalidAction(str(exc)) from exc
    return copy.deepcopy(value)


def _coverage(required: Sequence[str], asked: set[str]) -> dict[str, Any]:
    required_set = set(required)
    covered = sorted(required_set & asked)
    missed = sorted(required_set - asked)
    return {
        "required_count": len(required_set),
        "covered_count": len(covered),
        "recall": len(covered) / len(required_set) if required_set else 1.0,
        "covered_question_ids": covered,
        "missed_question_ids": missed,
    }


class PatientSimulator:
    """Stateful deterministic simulator backed by a single hidden case card.

    A policy receives :attr:`opening_utterance`, its own actions, and patient
    answers.  Gold fields and other hidden facts are used only by the evaluator.
    """

    def __init__(self, case_card: Mapping[str, Any]):
        self._card = copy.deepcopy(dict(case_card))
        validate_case_card(self._card)
        if self._card.get("schema_version") != CASE_SCHEMA_VERSION:
            raise ValueError(f"Case card must use {CASE_SCHEMA_VERSION!r}")
        facts = self._card.get("queryable_facts")
        if not isinstance(facts, list) or not facts:
            raise ValueError("Case card must contain queryable_facts")
        self._facts: dict[str, dict[str, Any]] = {}
        for fact in facts:
            if not isinstance(fact, dict) or not isinstance(fact.get("question_id"), str):
                raise ValueError("Every queryable fact must contain question_id")
            question_id = str(fact["question_id"])
            if question_id in self._facts:
                raise ValueError(f"Duplicate queryable fact: {question_id}")
            self._facts[question_id] = copy.deepcopy(fact)
        self._question_ids = set(self._facts)
        required_ids = {str(value) for value in self._card["required_question_ids"]}
        red_flag_ids = {
            str(flag["question_id"]) for flag in self._card["red_flags"]
        }
        missing_references = (required_ids | red_flag_ids) - self._question_ids
        if missing_references:
            raise ValueError(
                "Case card references questions absent from queryable_facts: "
                f"{sorted(missing_references)}"
            )
        gold = str(self._card["gold_acuity"])
        expected_unsafe = [
            label for label in LABELS if SEVERITY[label] < SEVERITY[gold]
        ]
        if self._card["unsafe_dispositions"] != expected_unsafe:
            raise ValueError("unsafe_dispositions must be every acuity below gold")
        opening = str(self._card["opening_utterance"])
        self._initially_visible = {
            question_id
            for question_id, fact in self._facts.items()
            if fact.get("status") == "known"
            and any(
                isinstance(fragment, str) and fragment in opening
                for fragment in fact.get("evidence_text", [])
            )
        }
        self._maximum_questions = int(self._card["simulator"]["maximum_questions"])
        self._asked: list[str] = []
        self._terminal = False
        self._terminal_action: dict[str, Any] | None = None
        self._outcome: dict[str, Any] | None = None
        self._transcript: list[dict[str, Any]] = [
            {
                "sequence": 0,
                "role": "patient",
                "kind": "OPENING",
                "content": str(self._card["opening_utterance"]),
            }
        ]

    @property
    def case_id(self) -> str:
        return str(self._card["case_id"])

    @property
    def opening_utterance(self) -> str:
        return str(self._card["opening_utterance"])

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def questions_asked(self) -> int:
        return len(self._asked)

    def public_state(self) -> dict[str, Any]:
        """Return policy-visible state; hidden facts and gold never appear."""

        return {
            "case_id": self.case_id,
            "opening_utterance": self.opening_utterance,
            "questions_asked": len(self._asked),
            "questions_remaining": self._maximum_questions - len(self._asked),
            "terminal": self._terminal,
            "transcript": copy.deepcopy(self._transcript),
        }

    def _base_outcome(self) -> dict[str, Any]:
        required = [str(value) for value in self._card["required_question_ids"]]
        red_flag_questions = [
            str(flag["question_id"]) for flag in self._card["red_flags"]
        ]
        asked_set = set(self._asked)
        observed_set = asked_set | self._initially_visible
        return {
            "case_id": self.case_id,
            "gold_acuity": str(self._card["gold_acuity"]),
            "questions_asked": len(self._asked),
            "unique_questions_asked": len(asked_set),
            "repeated_questions": len(self._asked) - len(asked_set),
            "initially_visible_question_ids": sorted(self._initially_visible),
            "required_question_coverage": _coverage(required, observed_set),
            "red_flag_question_coverage": _coverage(red_flag_questions, observed_set),
        }

    def step(self, action: Mapping[str, Any]) -> dict[str, Any]:
        if self._terminal:
            raise SimulationEnded(f"Case {self.case_id} has already ended")
        validated = validate_action(action, question_ids=self._question_ids)
        action_type = str(validated["type"])

        if action_type == "ASK":
            if len(self._asked) >= self._maximum_questions:
                raise QuestionLimitExceeded(
                    f"Case {self.case_id} allows at most {self._maximum_questions} questions"
                )
            question_id = str(validated["question_id"])
            fact = self._facts[question_id]
            repeated = question_id in self._asked
            self._asked.append(question_id)
            self._transcript.extend(
                [
                    {
                        "sequence": len(self._transcript),
                        "role": "clinician",
                        "kind": "ASK",
                        "question_id": question_id,
                        "content": str(validated["wording"]),
                    },
                    {
                        "sequence": len(self._transcript) + 1,
                        "role": "patient",
                        "kind": "ANSWER",
                        "question_id": question_id,
                        "content": str(fact["answer"]),
                        "fact_status": str(fact["status"]),
                    },
                ]
            )
            return {
                "case_id": self.case_id,
                "type": "ANSWER",
                "question_id": question_id,
                "answer": str(fact["answer"]),
                "fact_status": str(fact["status"]),
                "question_number": len(self._asked),
                "repeated": repeated,
                "terminal": False,
            }

        self._terminal = True
        self._terminal_action = validated
        base = self._base_outcome()
        if action_type == "DISPOSE":
            predicted = str(validated["acuity"])
            gold = str(self._card["gold_acuity"])
            signed_error = SEVERITY[predicted] - SEVERITY[gold]
            base.update(
                {
                    "terminal_action": "DISPOSE",
                    "predicted_acuity": predicted,
                    "rationale": str(validated["rationale"]),
                    "exact": predicted == gold,
                    "signed_ordinal_error": signed_error,
                    "absolute_ordinal_error": abs(signed_error),
                    "undertriage": signed_error < 0,
                    "overtriage": signed_error > 0,
                    "unsafe_disposition": signed_error < 0,
                    "handed_off": False,
                }
            )
        else:
            base.update(
                {
                    "terminal_action": "HANDOFF",
                    "predicted_acuity": None,
                    "rationale": None,
                    "exact": None,
                    "signed_ordinal_error": None,
                    "absolute_ordinal_error": None,
                    "undertriage": False,
                    "overtriage": False,
                    "unsafe_disposition": False,
                    "handed_off": True,
                    "handoff_reason": str(validated["reason"]),
                    "handoff_target": str(validated["target"]),
                }
            )
        self._outcome = base
        self._transcript.append(
            {
                "sequence": len(self._transcript),
                "role": "clinician",
                "kind": action_type,
                "content": (
                    str(validated["rationale"])
                    if action_type == "DISPOSE"
                    else str(validated["reason"])
                ),
                **(
                    {"acuity": str(validated["acuity"])}
                    if action_type == "DISPOSE"
                    else {"target": str(validated["target"])}
                ),
            }
        )
        return {
            "case_id": self.case_id,
            "type": "RESULT",
            "terminal": True,
            "outcome": copy.deepcopy(base),
        }

    def evaluation(self) -> TraceEvaluation:
        if not self._terminal or self._terminal_action is None or self._outcome is None:
            raise RuntimeError(f"Case {self.case_id} has no terminal action")
        return TraceEvaluation(
            case_id=self.case_id,
            terminal_action=str(self._terminal_action["type"]),
            transcript=tuple(copy.deepcopy(self._transcript)),
            outcome=copy.deepcopy(self._outcome),
        )


def run_action_trace(
    case_card: Mapping[str, Any],
    actions: Iterable[Mapping[str, Any]],
) -> TraceEvaluation:
    simulator = PatientSimulator(case_card)
    action_list = list(actions)
    for index, action in enumerate(action_list):
        simulator.step(action)
        if simulator.terminal:
            if index != len(action_list) - 1:
                raise ValueError(
                    f"Trace for {simulator.case_id} contains actions after a terminal action"
                )
            break
    if not simulator.terminal:
        raise ValueError(f"Trace for {simulator.case_id} has no terminal action")
    return simulator.evaluation()


def aggregate_trace_evaluations(
    evaluations: Sequence[TraceEvaluation],
) -> dict[str, Any]:
    if not evaluations:
        raise ValueError("At least one terminal trace evaluation is required")
    outcomes = [evaluation.outcome for evaluation in evaluations]
    disposed = [outcome for outcome in outcomes if not outcome["handed_off"]]
    exact_count = sum(outcome["exact"] is True for outcome in outcomes)
    unsafe_count = sum(bool(outcome["unsafe_disposition"]) for outcome in outcomes)
    undertriage_count = sum(bool(outcome["undertriage"]) for outcome in outcomes)
    return {
        "schema_version": "interactive-evaluation-summary/v1",
        "trace_count": len(outcomes),
        "disposed_count": len(disposed),
        "handoff_count": sum(bool(outcome["handed_off"]) for outcome in outcomes),
        "overall_correct_disposition_rate": exact_count / len(outcomes),
        "autonomous_exact_accuracy": (
            sum(outcome["exact"] is True for outcome in disposed) / len(disposed)
            if disposed
            else None
        ),
        "handoff_rate": sum(bool(outcome["handed_off"]) for outcome in outcomes)
        / len(outcomes),
        "undertriage_rate": undertriage_count / len(disposed) if disposed else None,
        "unsafe_disposition_rate": unsafe_count / len(disposed) if disposed else None,
        "overall_undertriage_action_rate": undertriage_count / len(outcomes),
        "overall_unsafe_action_rate": unsafe_count / len(outcomes),
        "mean_questions_asked": fmean(float(outcome["questions_asked"]) for outcome in outcomes),
        "mean_required_question_recall": fmean(
            float(outcome["required_question_coverage"]["recall"])
            for outcome in outcomes
        ),
        "mean_red_flag_question_recall": fmean(
            float(outcome["red_flag_question_coverage"]["recall"])
            for outcome in outcomes
        ),
    }
