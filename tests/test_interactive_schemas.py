from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

try:  # jsonschema is optional; the repository does not require it at runtime.
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - exercised in the dependency-minimal environment
    Draft202012Validator = None  # type: ignore[assignment,misc]


ROOT = Path(__file__).resolve().parents[1]
CASE_SCHEMA_PATH = ROOT / "schemas" / "interactive-case-card-v1.schema.json"
ACTION_SCHEMA_PATH = ROOT / "schemas" / "interactive-action-v1.schema.json"
ACTION_CATALOG_PATH = ROOT / "configs" / "interactive" / "action_catalog.v1.yaml"


class _FallbackValidationError(AssertionError):
    """Validation failure from the dependency-free schema subset below."""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_local_ref(root_schema: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise _FallbackValidationError(f"Only local references are supported: {reference}")
    node: Any = root_schema
    for segment in reference[2:].split("/"):
        segment = segment.replace("~1", "/").replace("~0", "~")
        node = node[segment]
    if not isinstance(node, dict):
        raise _FallbackValidationError(f"Reference does not resolve to a schema: {reference}")
    return node


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    raise _FallbackValidationError(f"Unsupported JSON Schema type in test helper: {expected}")


def _fallback_validate(
    value: Any,
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any],
    path: str = "$",
) -> None:
    if "$ref" in schema:
        _fallback_validate(
            value,
            _resolve_local_ref(root_schema, schema["$ref"]),
            root_schema=root_schema,
            path=path,
        )

    for subschema in schema.get("allOf", []):
        _fallback_validate(value, subschema, root_schema=root_schema, path=path)

    if "oneOf" in schema:
        matches = 0
        for subschema in schema["oneOf"]:
            try:
                _fallback_validate(value, subschema, root_schema=root_schema, path=path)
            except _FallbackValidationError:
                continue
            matches += 1
        if matches != 1:
            raise _FallbackValidationError(
                f"{path}: expected exactly one oneOf branch, matched {matches}"
            )

    if "if" in schema:
        try:
            _fallback_validate(value, schema["if"], root_schema=root_schema, path=path)
        except _FallbackValidationError:
            selected = schema.get("else")
        else:
            selected = schema.get("then")
        if selected is not None:
            _fallback_validate(value, selected, root_schema=root_schema, path=path)

    if "const" in schema and value != schema["const"]:
        raise _FallbackValidationError(f"{path}: expected const {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise _FallbackValidationError(f"{path}: {value!r} is not in the enum")

    expected_types = schema.get("type")
    if expected_types is not None:
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if not any(_matches_type(value, expected) for expected in expected_types):
            raise _FallbackValidationError(
                f"{path}: expected type {expected_types}, got {type(value).__name__}"
            )

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise _FallbackValidationError(f"{path}: missing required properties {missing}")

        properties = schema.get("properties", {})
        for key, property_schema in properties.items():
            if key in value:
                _fallback_validate(
                    value[key],
                    property_schema,
                    root_schema=root_schema,
                    path=f"{path}.{key}",
                )

        extra_keys = set(value) - set(properties)
        additional = schema.get("additionalProperties", True)
        if additional is False and extra_keys:
            raise _FallbackValidationError(
                f"{path}: additional properties are not allowed: {sorted(extra_keys)}"
            )
        if isinstance(additional, dict):
            for key in extra_keys:
                _fallback_validate(
                    value[key],
                    additional,
                    root_schema=root_schema,
                    path=f"{path}.{key}",
                )

        if len(value) < schema.get("minProperties", 0):
            raise _FallbackValidationError(f"{path}: too few properties")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            raise _FallbackValidationError(f"{path}: too many properties")

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise _FallbackValidationError(f"{path}: too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise _FallbackValidationError(f"{path}: too many items")
        if schema.get("uniqueItems"):
            canonical_items = [
                json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value
            ]
            if len(canonical_items) != len(set(canonical_items)):
                raise _FallbackValidationError(f"{path}: items are not unique")
        if "items" in schema:
            for index, item in enumerate(value):
                _fallback_validate(
                    item,
                    schema["items"],
                    root_schema=root_schema,
                    path=f"{path}[{index}]",
                )

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise _FallbackValidationError(f"{path}: string is too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise _FallbackValidationError(f"{path}: string is too long")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise _FallbackValidationError(
                f"{path}: string does not match {schema['pattern']!r}"
            )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise _FallbackValidationError(f"{path}: number is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise _FallbackValidationError(f"{path}: number is above maximum")


def _is_valid(value: Any, schema: dict[str, Any]) -> bool:
    if Draft202012Validator is not None:
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema).is_valid(value)
    try:
        _fallback_validate(value, schema, root_schema=schema)
    except _FallbackValidationError:
        return False
    return True


def _valid_case_card() -> dict[str, Any]:
    return {
        "schema_version": "interactive-case-card/v1",
        "case_id": "interactive.healthbench.example-001",
        "intended_use": "evaluation_only",
        "training_allowed": False,
        "provenance": {
            "benchmark_case_id": "healthbench.example-001",
            "source_dataset": "healthbench",
            "source_id": "example-001",
            "source_text_sha256": "a" * 64,
            "benchmark_artifact": "data/processed/acuitybench.parquet",
            "label_basis": "released physician/reference label",
            "mapping_method": "direct",
            "transformation": "Opening utterance extracted without adding clinical facts.",
        },
        "opening_utterance": "I have crushing chest pressure and feel sweaty and sick.",
        "queryable_facts": [
            {
                "question_id": "symptom.onset",
                "status": "known",
                "answer": "It started suddenly about 20 minutes ago.",
                "evidence_text": ["started suddenly about 20 minutes ago"],
            },
            {
                "question_id": "red_flag.chest_pain",
                "status": "known",
                "answer": "It is heavy pressure spreading into my left arm.",
                "evidence_text": ["crushing chest pressure", "spreading into my left arm"],
            },
            {
                "question_id": "red_flag.breathing",
                "status": "known",
                "answer": "I am a little breathless but can speak normally.",
                "evidence_text": ["a little breathless"],
            },
        ],
        "gold_acuity": "D",
        "red_flags": [
            {
                "id": "possible-acute-coronary-syndrome",
                "question_id": "red_flag.chest_pain",
                "evidence_text": ["crushing chest pressure", "sweaty and sick"],
                "derivation": "Source presentation is compatible with an acute coronary syndrome.",
            }
        ],
        "required_question_ids": [
            "symptom.onset",
            "red_flag.chest_pain",
            "red_flag.breathing",
        ],
        "unsafe_dispositions": ["A", "B", "C"],
        "stratum": {
            "acuity": "D",
            "source_dataset": "healthbench",
            "presentation_group": "cardiovascular",
        },
        "patient_profile": {
            "age": 54,
            "sex": "male",
            "speaker_role": "patient",
        },
        "simulator": {
            "maximum_questions": 8,
            "unknown_response": "I don't know.",
            "response_policy": "question_id_lookup",
        },
        "review_status": {
            "reference_label": "source_derived",
            "case_content": "automated_checked",
            "requires_clinician_review": True,
        },
    }


def _bad_case_cards() -> list[tuple[str, dict[str, Any]]]:
    variants: list[tuple[str, dict[str, Any]]] = []

    missing_opening = _valid_case_card()
    del missing_opening["opening_utterance"]
    variants.append(("missing opening utterance", missing_opening))

    training_enabled = _valid_case_card()
    training_enabled["training_allowed"] = True
    variants.append(("training use enabled", training_enabled))

    invalid_gold = _valid_case_card()
    invalid_gold["gold_acuity"] = "E"
    variants.append(("invalid gold acuity", invalid_gold))

    unknown_question = _valid_case_card()
    unknown_question["queryable_facts"][0]["question_id"] = "symptom.unversioned"
    variants.append(("unknown question id", unknown_question))

    gold_marked_unsafe = _valid_case_card()
    gold_marked_unsafe["unsafe_dispositions"] = ["D"]
    variants.append(("gold disposition marked unsafe", gold_marked_unsafe))

    mismatched_stratum = _valid_case_card()
    mismatched_stratum["stratum"]["acuity"] = "C"
    variants.append(("stratum acuity mismatch", mismatched_stratum))

    malformed_digest = _valid_case_card()
    malformed_digest["provenance"]["source_text_sha256"] = "not-a-sha256"
    variants.append(("malformed provenance digest", malformed_digest))

    unexpected_field = _valid_case_card()
    unexpected_field["diagnosis"] = "This must remain out of the v1 wire format."
    variants.append(("unexpected top-level field", unexpected_field))

    return variants


def test_schema_documents_are_json_schema_2020_12() -> None:
    schemas = [_load_json(CASE_SCHEMA_PATH), _load_json(ACTION_SCHEMA_PATH)]

    assert {schema["$schema"] for schema in schemas} == {
        "https://json-schema.org/draft/2020-12/schema"
    }
    assert len({schema["$id"] for schema in schemas}) == len(schemas)
    if Draft202012Validator is not None:
        for schema in schemas:
            Draft202012Validator.check_schema(schema)


def test_valid_case_card_is_accepted() -> None:
    assert _is_valid(_valid_case_card(), _load_json(CASE_SCHEMA_PATH))


@pytest.mark.parametrize(
    ("_name", "case_card"),
    _bad_case_cards(),
    ids=[name for name, _ in _bad_case_cards()],
)
def test_invalid_case_cards_are_rejected(_name: str, case_card: dict[str, Any]) -> None:
    assert not _is_valid(case_card, _load_json(CASE_SCHEMA_PATH))


@pytest.mark.parametrize(
    "action",
    [
        {
            "schema_version": "interactive-action/v1",
            "type": "ASK",
            "question_id": "symptom.onset",
            "wording": "When did the pain begin?",
        },
        {
            "schema_version": "interactive-action/v1",
            "type": "DISPOSE",
            "acuity": "D",
            "rationale": "The revealed chest-pain features need immediate emergency assessment.",
        },
        {
            "schema_version": "interactive-action/v1",
            "type": "HANDOFF",
            "reason": "The patient's answers are internally inconsistent and cannot be resolved safely.",
            "target": "human_clinician",
        },
    ],
)
def test_valid_actions_are_accepted(action: dict[str, Any]) -> None:
    assert _is_valid(action, _load_json(ACTION_SCHEMA_PATH))


@pytest.mark.parametrize(
    "action",
    [
        {
            "schema_version": "interactive-action/v1",
            "type": "ASK",
            "question_id": "symptom.onset",
        },
        {
            "schema_version": "interactive-action/v1",
            "type": "ASK",
            "question_id": "symptom.not-in-v1",
            "wording": "Tell me more.",
        },
        {
            "schema_version": "interactive-action/v1",
            "type": "ASK",
            "question_id": "symptom.onset",
            "wording": "   ",
        },
        {
            "schema_version": "interactive-action/v1",
            "type": "DISPOSE",
            "disposition": "D",
        },
        {
            "schema_version": "interactive-action/v1",
            "type": "DISPOSE",
            "acuity": "E",
            "rationale": "Invalid acuity.",
        },
        {
            "schema_version": "interactive-action/v1",
            "type": "HANDOFF",
            "reason": "Needs review.",
            "target": "automated_system",
        },
        {
            "schema_version": "interactive-action/v1",
            "type": "WAIT",
        },
    ],
)
def test_invalid_actions_are_rejected(action: dict[str, Any]) -> None:
    assert not _is_valid(action, _load_json(ACTION_SCHEMA_PATH))


def test_action_catalog_and_schemas_have_identical_closed_vocabularies() -> None:
    catalog = yaml.safe_load(ACTION_CATALOG_PATH.read_text(encoding="utf-8"))
    action_schema = _load_json(ACTION_SCHEMA_PATH)
    case_schema = _load_json(CASE_SCHEMA_PATH)

    question_ids = [question["id"] for question in catalog["questions"]]
    assert len(question_ids) == len(set(question_ids))
    assert set(question_ids) == set(action_schema["$defs"]["questionId"]["enum"])
    assert set(question_ids) == set(case_schema["$defs"]["questionId"]["enum"])
    assert catalog["schema_version"] == "interactive-action/v1"
    assert set(catalog["action_types"]) == {"ASK", "DISPOSE", "HANDOFF"}
    assert set(catalog["acuities"]) == {"A", "B", "C", "D"}

    required_fields = {
        action_type: definition["required_fields"]
        for action_type, definition in catalog["action_types"].items()
    }
    assert required_fields == {
        "ASK": ["question_id", "wording"],
        "DISPOSE": ["acuity", "rationale"],
        "HANDOFF": ["reason", "target"],
    }


def test_valid_case_fixture_has_resolvable_question_references() -> None:
    case_card = copy.deepcopy(_valid_case_card())
    queryable_ids = {fact["question_id"] for fact in case_card["queryable_facts"]}

    assert set(case_card["required_question_ids"]) <= queryable_ids
    assert {red_flag["question_id"] for red_flag in case_card["red_flags"]} <= queryable_ids
