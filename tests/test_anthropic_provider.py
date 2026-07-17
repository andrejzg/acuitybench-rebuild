from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import acuitybench.providers.anthropic as provider_module
from acuitybench.models import ModelRegistry
from acuitybench.providers.anthropic import AnthropicProvider, _provider_safe_schema


def test_provider_safe_schema_preserves_shape_and_drops_unsupported_constraints() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "age": {"type": "integer", "minimum": 0, "maximum": 110},
            "label": {"type": "string", "enum": ["A", "B"]},
        },
        "required": ["age", "label"],
    }

    safe = _provider_safe_schema(schema)

    assert "$schema" not in safe
    assert safe["properties"]["age"] == {"type": "integer"}
    assert safe["properties"]["label"]["enum"] == ["A", "B"]
    assert safe["additionalProperties"] is False


def test_anthropic_completion_retains_usage_refusal_and_standard_contract(
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}
    response = SimpleNamespace(
        id="msg_test",
        model="claude-fable-5",
        content=[SimpleNamespace(type="text", text='{"acuity":"A"}')],
        stop_reason="end_turn",
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=120,
            output_tokens=30,
            cache_read_input_tokens=20,
            cache_creation_input_tokens=0,
            output_tokens_details=SimpleNamespace(thinking_tokens=7),
            service_tier="standard",
            inference_geo="us",
        ),
    )

    class FakeCreate:
        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(
                parse=lambda: response,
                headers={"request-id": "req_test"},
                request_id="req_test",
                status_code=200,
                http_version="HTTP/2",
                retries_taken=0,
            )

    fake_client = SimpleNamespace(
        messages=SimpleNamespace(
            with_raw_response=SimpleNamespace(create=FakeCreate().create)
        )
    )
    provider = AnthropicProvider()
    monkeypatch.setattr(provider, "_client", lambda *_: fake_client)
    schema = {
        "type": "object",
        "properties": {"acuity": {"type": "string", "minLength": 1}},
        "required": ["acuity"],
        "additionalProperties": False,
    }

    result = asyncio.run(
        provider.complete(
            config=ModelRegistry().get("claude-fable-5"),
            messages=[{"role": "user", "content": "fictional case"}],
            stream=False,
            output_schema=schema,
        )
    )

    assert captured["service_tier"] == "standard_only"
    assert captured["output_config"]["effort"] == "medium"
    assert "minLength" not in captured["output_config"]["format"]["schema"]["properties"]["acuity"]
    assert result.request_id == "req_test"
    assert result.reasoning_tokens == 7
    assert result.cached_input_tokens == 20
    assert result.provider_metadata["returned_service_tier"] == "standard"
