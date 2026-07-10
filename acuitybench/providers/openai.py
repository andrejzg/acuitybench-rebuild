"""OpenAI adapter with audit metadata and both text endpoints."""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from acuitybench.models import ModelConfig
from acuitybench.providers.base import CompletionResult


_RATE_HEADERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "retry-after",
)


def _value(obj: object | None, name: str) -> Any:
    return getattr(obj, name, None) if obj is not None else None


class OpenAIProvider:
    def __init__(self) -> None:
        self._clients: dict[str, AsyncOpenAI] = {}

    def _client(self, key_env: str) -> AsyncOpenAI:
        api_key = os.getenv(key_env)
        if not api_key:
            raise RuntimeError(
                f"{key_env} is not set. Put it in .env or export it before running."
            )
        if key_env not in self._clients:
            # Retry accounting lives in the runner so attempts and backoff are
            # visible in SQLite rather than hidden inside the SDK.
            self._clients[key_env] = AsyncOpenAI(
                api_key=api_key, timeout=180.0, max_retries=0
            )
        return self._clients[key_env]

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

    @staticmethod
    def _headers(raw: Any) -> tuple[str | None, dict[str, str], dict[str, Any]]:
        headers = raw.headers
        request_id = headers.get("x-request-id")
        rate_limit = {
            name: value for name in _RATE_HEADERS if (value := headers.get(name))
        }
        metadata: dict[str, Any] = {}
        if processing_ms := headers.get("openai-processing-ms"):
            metadata["openai_processing_ms"] = processing_ms
        return request_id, rate_limit, metadata

    async def complete(
        self,
        *,
        config: ModelConfig,
        messages: list[dict[str, str]],
        max_output_tokens: int | None = None,
    ) -> CompletionResult:
        if config.endpoint == "chat_completions":
            return await self._chat_completion(
                config=config,
                messages=messages,
                max_output_tokens=max_output_tokens,
            )
        if config.endpoint == "responses":
            return await self._response(
                config=config,
                messages=messages,
                max_output_tokens=max_output_tokens,
            )
        raise ValueError(
            f"OpenAI endpoint {config.endpoint!r} is not supported; use "
            "chat_completions or responses"
        )

    async def _chat_completion(
        self,
        *,
        config: ModelConfig,
        messages: list[dict[str, str]],
        max_output_tokens: int | None,
    ) -> CompletionResult:
        kwargs: dict[str, Any] = {
            "model": config.api_model,
            "messages": messages,
            config.token_parameter: max_output_tokens or config.max_output_tokens,
        }
        if config.send_temperature and config.temperature is not None:
            kwargs["temperature"] = config.temperature
        client = self._client(config.api_key_env)
        raw = await client.chat.completions.with_raw_response.create(**kwargs)
        response = raw.parse()
        request_id, rate_limit, metadata = self._headers(raw)
        choice = response.choices[0]
        usage = response.usage
        prompt_details = _value(usage, "prompt_tokens_details")
        completion_details = _value(usage, "completion_tokens_details")
        return CompletionResult(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason or "unknown",
            response_id=response.id,
            returned_model=response.model,
            system_fingerprint=response.system_fingerprint,
            request_id=request_id,
            input_tokens=_value(usage, "prompt_tokens"),
            cached_input_tokens=_value(prompt_details, "cached_tokens"),
            output_tokens=_value(usage, "completion_tokens"),
            reasoning_tokens=_value(completion_details, "reasoning_tokens"),
            total_tokens=_value(usage, "total_tokens"),
            rate_limit=rate_limit,
            provider_metadata=metadata,
        )

    async def _response(
        self,
        *,
        config: ModelConfig,
        messages: list[dict[str, str]],
        max_output_tokens: int | None,
    ) -> CompletionResult:
        kwargs: dict[str, Any] = {
            "model": config.api_model,
            "input": messages,
            "max_output_tokens": max_output_tokens or config.max_output_tokens,
            "store": False,
        }
        if config.send_temperature and config.temperature is not None:
            kwargs["temperature"] = config.temperature
        client = self._client(config.api_key_env)
        raw = await client.responses.with_raw_response.create(**kwargs)
        response = raw.parse()
        request_id, rate_limit, metadata = self._headers(raw)
        usage = response.usage
        input_details = _value(usage, "input_tokens_details")
        output_details = _value(usage, "output_tokens_details")
        incomplete_details = _value(response, "incomplete_details")
        incomplete_reason = _value(incomplete_details, "reason")
        finish_reason = _value(response, "status") or "unknown"
        if finish_reason == "incomplete" and incomplete_reason == "max_output_tokens":
            finish_reason = "length"
        if incomplete_reason:
            metadata["incomplete_reason"] = incomplete_reason
        return CompletionResult(
            text=response.output_text or "",
            finish_reason=finish_reason,
            response_id=response.id,
            returned_model=response.model,
            request_id=request_id,
            input_tokens=_value(usage, "input_tokens"),
            cached_input_tokens=_value(input_details, "cached_tokens"),
            cache_write_tokens=_value(input_details, "cache_write_tokens"),
            output_tokens=_value(usage, "output_tokens"),
            reasoning_tokens=_value(output_details, "reasoning_tokens"),
            total_tokens=_value(usage, "total_tokens"),
            rate_limit=rate_limit,
            provider_metadata=metadata,
        )
