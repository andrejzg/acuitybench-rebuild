"""Anthropic Messages adapter for auditable, structured synthetic-data calls."""

from __future__ import annotations

import os
from typing import Any, Mapping

import anthropic
from anthropic import AsyncAnthropic

from acuitybench.models import ModelConfig
from acuitybench.providers.base import CompletionResult


_SAFE_HEADERS = (
    "request-id",
    "anthropic-ratelimit-requests-limit",
    "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-requests-reset",
    "anthropic-ratelimit-tokens-limit",
    "anthropic-ratelimit-tokens-remaining",
    "anthropic-ratelimit-tokens-reset",
    "retry-after",
)

_UNSUPPORTED_FORMAT_KEYWORDS = {
    "$id",
    "$schema",
    "maximum",
    "maxItems",
    "maxLength",
    "minimum",
    "minItems",
    "minLength",
    "pattern",
    "title",
    "uniqueItems",
}


def _value(value: object | None, name: str) -> Any:
    return getattr(value, name, None) if value is not None else None


def _provider_safe_schema(value: Any) -> Any:
    """Remove validation-only keywords unsupported by Anthropic's compiler.

    The unmodified schema is still applied after the response, so these
    constraints remain hard acceptance checks in the research artifact.
    """
    if isinstance(value, dict):
        return {
            key: _provider_safe_schema(item)
            for key, item in value.items()
            if key not in _UNSUPPORTED_FORMAT_KEYWORDS
        }
    if isinstance(value, list):
        return [_provider_safe_schema(item) for item in value]
    return value


class AnthropicProvider:
    """Non-streaming Messages API adapter used by the fictional-data pipeline."""

    def __init__(self) -> None:
        self._clients: dict[str, AsyncAnthropic] = {}

    def _client(self, key_env: str, base_url_env: str | None = None) -> AsyncAnthropic:
        api_key = os.getenv(key_env)
        if not api_key:
            raise RuntimeError(
                f"{key_env} is not set. Put it in .env or export it before running."
            )
        # Do not silently inherit a machine-wide ANTHROPIC_BASE_URL. This repo's
        # public model profile targets Anthropic directly unless it explicitly
        # names a base_url_env (as OpenAI-compatible profiles do).
        base_url = "https://api.anthropic.com"
        if base_url_env is not None:
            configured = os.getenv(base_url_env)
            if not configured:
                raise RuntimeError(f"{base_url_env} is not set")
            base_url = configured
        client_key = f"{key_env}@{base_url}"
        if client_key not in self._clients:
            self._clients[client_key] = AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=180.0,
                max_retries=0,
            )
        return self._clients[client_key]

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

    @staticmethod
    def _request_parts(
        messages: list[dict[str, str]],
    ) -> tuple[str | None, list[dict[str, str]]]:
        system_parts = [item["content"] for item in messages if item["role"] == "system"]
        conversation = [
            {"role": item["role"], "content": item["content"]}
            for item in messages
            if item["role"] != "system"
        ]
        return ("\n\n".join(system_parts) or None), conversation

    @staticmethod
    def _output_config(
        config: ModelConfig, output_schema: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        output_config: dict[str, Any] = {}
        if config.reasoning_effort is not None:
            output_config["effort"] = config.reasoning_effort
        if output_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": _provider_safe_schema(dict(output_schema)),
            }
        return output_config or None

    async def count_tokens(
        self,
        *,
        config: ModelConfig,
        messages: list[dict[str, str]],
        output_schema: Mapping[str, Any] | None = None,
    ) -> int:
        system, conversation = self._request_parts(messages)
        kwargs: dict[str, Any] = {
            "model": config.api_model,
            "messages": conversation,
        }
        if system is not None:
            kwargs["system"] = system
        output_config = self._output_config(config, output_schema)
        if output_config is not None:
            kwargs["output_config"] = output_config
        result = await self._client(
            config.api_key_env, config.base_url_env
        ).messages.count_tokens(**kwargs)
        return int(result.input_tokens)

    async def complete(
        self,
        *,
        config: ModelConfig,
        messages: list[dict[str, str]],
        max_output_tokens: int | None = None,
        stream: bool = True,
        output_schema: Mapping[str, Any] | None = None,
        output_schema_name: str | None = None,
    ) -> CompletionResult:
        del output_schema_name  # Anthropic's JSON-schema format has no name field.
        if config.endpoint != "messages":
            raise ValueError("AnthropicProvider requires endpoint: messages")
        if stream:
            raise ValueError(
                "Anthropic structured synthetic calls currently require stream: false"
            )
        system, conversation = self._request_parts(messages)
        kwargs: dict[str, Any] = {
            "model": config.api_model,
            "messages": conversation,
            "max_tokens": max_output_tokens or config.max_output_tokens,
            "service_tier": "standard_only",
        }
        if system is not None:
            kwargs["system"] = system
        if config.send_temperature and config.temperature is not None:
            kwargs["temperature"] = config.temperature
        output_config = self._output_config(config, output_schema)
        if output_config is not None:
            kwargs["output_config"] = output_config

        raw = await self._client(
            config.api_key_env, config.base_url_env
        ).messages.with_raw_response.create(**kwargs)
        response = raw.parse()
        headers = raw.headers
        safe_headers = {
            name: value for name in _SAFE_HEADERS if (value := headers.get(name))
        }
        usage = response.usage
        output_details = _value(usage, "output_tokens_details")
        text = "".join(
            str(block.text)
            for block in response.content
            if _value(block, "type") == "text" and _value(block, "text") is not None
        )
        stop_details = _value(response, "stop_details")
        input_tokens = _value(usage, "input_tokens")
        output_tokens = _value(usage, "output_tokens")
        metadata = {
            "endpoint": config.endpoint,
            "streaming": False,
            "anthropic_sdk_version": anthropic.__version__,
            "http_status": getattr(raw, "status_code", None),
            "http_version": getattr(raw, "http_version", None),
            "sdk_retries": getattr(raw, "retries_taken", None),
            "server_headers": safe_headers,
            "returned_service_tier": _value(usage, "service_tier"),
            "inference_geo": _value(usage, "inference_geo"),
            "stop_details": (
                None
                if stop_details is None
                else {
                    "type": _value(stop_details, "type"),
                    "category": _value(stop_details, "category"),
                    "explanation": _value(stop_details, "explanation"),
                }
            ),
            "output_contains_refusal": response.stop_reason == "refusal",
            "request_contract": {
                "reasoning_effort_requested": config.reasoning_effort,
                "reasoning_effort_basis": config.reasoning_effort_basis,
                "service_tier_requested": "standard_only",
                "structured_output": output_schema is not None,
                "structured_output_schema_constraints_removed": sorted(
                    _UNSUPPORTED_FORMAT_KEYWORDS
                ) if output_schema is not None else [],
                "max_output_tokens": max_output_tokens or config.max_output_tokens,
            },
        }
        return CompletionResult(
            text=text,
            finish_reason=response.stop_reason or "unknown",
            response_id=response.id,
            returned_model=str(response.model),
            request_id=getattr(raw, "request_id", None) or headers.get("request-id"),
            input_tokens=input_tokens,
            cached_input_tokens=_value(usage, "cache_read_input_tokens"),
            cache_write_tokens=_value(usage, "cache_creation_input_tokens"),
            output_tokens=output_tokens,
            reasoning_tokens=_value(output_details, "thinking_tokens"),
            total_tokens=(
                int(input_tokens) + int(output_tokens)
                if input_tokens is not None and output_tokens is not None
                else None
            ),
            rate_limit=safe_headers,
            provider_metadata=metadata,
        )
