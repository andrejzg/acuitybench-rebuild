"""Dependency-free validation for the JSON Schema subset used by v1.

The schema files remain the source of truth.  This evaluator intentionally
supports only the Draft 2020-12 keywords present in the committed schemas and
fails if an unsupported local construct is encountered.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from acuitybench.sources import project_root


class SchemaValidationError(ValueError):
    """Raised when an instance does not satisfy a committed schema."""


def load_json_schema(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON Schema must be an object: {path}")
    if value.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise ValueError(f"Unsupported JSON Schema dialect: {path}")
    return value


def default_case_schema_path() -> Path:
    return project_root() / "schemas/interactive-case-card-v1.schema.json"


def default_action_schema_path() -> Path:
    return project_root() / "schemas/interactive-action-v1.schema.json"


@lru_cache(maxsize=1)
def _default_case_schema() -> dict[str, Any]:
    return load_json_schema(default_case_schema_path())


@lru_cache(maxsize=1)
def _default_action_schema() -> dict[str, Any]:
    return load_json_schema(default_action_schema_path())


def _resolve(root: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        raise ValueError(f"Only local schema references are supported: {reference}")
    node: Any = root
    for segment in reference[2:].split("/"):
        segment = segment.replace("~1", "/").replace("~0", "~")
        node = node[segment]
    if not isinstance(node, Mapping):
        raise ValueError(f"Schema reference does not resolve to an object: {reference}")
    return node


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
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
    raise ValueError(f"Unsupported JSON Schema type: {expected}")


def _validate(
    value: Any,
    schema: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
    path: str,
) -> None:
    if "$ref" in schema:
        _validate(value, _resolve(root, str(schema["$ref"])), root=root, path=path)

    for subschema in schema.get("allOf", []):
        _validate(value, subschema, root=root, path=path)

    if "oneOf" in schema:
        matches = 0
        failures: list[str] = []
        for subschema in schema["oneOf"]:
            try:
                _validate(value, subschema, root=root, path=path)
            except SchemaValidationError as exc:
                failures.append(str(exc))
            else:
                matches += 1
        if matches != 1:
            detail = failures[0] if failures else "multiple branches matched"
            raise SchemaValidationError(
                f"{path}: expected exactly one oneOf branch, matched {matches} ({detail})"
            )

    if "if" in schema:
        try:
            _validate(value, schema["if"], root=root, path=path)
        except SchemaValidationError:
            selected = schema.get("else")
        else:
            selected = schema.get("then")
        if selected is not None:
            _validate(value, selected, root=root, path=path)

    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path}: {value!r} is not in the allowed enum")

    expected_types = schema.get("type")
    if expected_types is not None:
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if not any(_matches_type(value, str(expected)) for expected in expected_types):
            raise SchemaValidationError(
                f"{path}: expected type {expected_types}, got {type(value).__name__}"
            )

    if isinstance(value, Mapping):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise SchemaValidationError(f"{path}: missing required fields {missing}")
        properties = schema.get("properties", {})
        for key, property_schema in properties.items():
            if key in value:
                _validate(
                    value[key],
                    property_schema,
                    root=root,
                    path=f"{path}.{key}",
                )
        extra = set(value) - set(properties)
        additional = schema.get("additionalProperties", True)
        if additional is False and extra:
            raise SchemaValidationError(
                f"{path}: additional fields are not allowed: {sorted(extra)}"
            )
        if isinstance(additional, Mapping):
            for key in extra:
                _validate(
                    value[key],
                    additional,
                    root=root,
                    path=f"{path}.{key}",
                )

    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            raise SchemaValidationError(f"{path}: too few items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise SchemaValidationError(f"{path}: too many items")
        if schema.get("uniqueItems"):
            encoded = [
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in value
            ]
            if len(encoded) != len(set(encoded)):
                raise SchemaValidationError(f"{path}: items must be unique")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate(
                    item,
                    schema["items"],
                    root=root,
                    path=f"{path}[{index}]",
                )

    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            raise SchemaValidationError(f"{path}: string is too short")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise SchemaValidationError(f"{path}: string is too long")
        if "pattern" in schema and re.search(str(schema["pattern"]), value) is None:
            raise SchemaValidationError(
                f"{path}: value does not match {schema['pattern']!r}"
            )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise SchemaValidationError(f"{path}: number is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise SchemaValidationError(f"{path}: number is above maximum")


def validate_instance(value: Any, schema: Mapping[str, Any]) -> None:
    _validate(value, schema, root=schema, path="$")


def validate_case_card(
    value: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
) -> None:
    validate_instance(value, schema or _default_case_schema())


def validate_action_instance(
    value: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
) -> None:
    validate_instance(value, schema or _default_action_schema())
