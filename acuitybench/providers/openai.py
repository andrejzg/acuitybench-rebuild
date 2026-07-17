"""OpenAI adapter with streaming TTFT, usage, and audit metadata."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

import openai
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
    "retry-after-ms",
)

# Keep a small, non-sensitive allowlist rather than serializing every response
# header into public result artifacts.
_SERVER_HEADERS = (
    "x-request-id",
    "openai-processing-ms",
    "openai-version",
    "server-timing",
    "x-envoy-upstream-service-time",
    "cf-ray",
)

_UNSUPPORTED_STRICT_SCHEMA_KEYWORDS = {
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


def _value(obj: object | None, name: str) -> Any:
    return getattr(obj, name, None) if obj is not None else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float(value: object | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _provider_safe_schema(value: Any) -> Any:
    if isinstance(value, dict):
        result = {
            key: _provider_safe_schema(item)
            for key, item in value.items()
            if key not in _UNSUPPORTED_STRICT_SCHEMA_KEYWORDS
        }
        if "const" in result and "type" not in result:
            constant = result["const"]
            if isinstance(constant, str):
                result["type"] = "string"
            elif isinstance(constant, bool):
                result["type"] = "boolean"
            elif isinstance(constant, int):
                result["type"] = "integer"
            elif isinstance(constant, float):
                result["type"] = "number"
            elif constant is None:
                result["type"] = "null"
        return result
    if isinstance(value, list):
        return [_provider_safe_schema(item) for item in value]
    return value


def _usage_metadata(
    metadata: dict[str, Any],
    usage: object | None,
    *,
    input_field: str,
    output_field: str,
    input_details_field: str,
) -> None:
    """Make incomplete billing telemetry explicit instead of pricing it as zero."""
    metadata["usage_reported"] = usage is not None
    metadata["usage_complete"] = bool(
        usage is not None
        and _value(usage, input_field) is not None
        and _value(usage, output_field) is not None
    )
    input_details = _value(usage, input_details_field)
    metadata["cache_breakdown_reported"] = bool(
        input_details is not None
        and _value(input_details, "cached_tokens") is not None
    )


def _request_contract(
    config: ModelConfig, max_output_tokens: int | None
) -> dict[str, Any]:
    return {
        "temperature_configured": config.temperature,
        "temperature_parameter_sent": config.send_temperature,
        "reasoning_effort_requested": config.reasoning_effort,
        "reasoning_effort_basis": config.reasoning_effort_basis,
        "service_tier_requested": config.service_tier,
        "max_output_tokens": max_output_tokens or config.max_output_tokens,
        "max_retry_output_tokens": config.max_retry_output_tokens or 8192,
    }


def _chat_visible_text(message: object) -> tuple[str, bool]:
    parts: list[str] = []
    content = _value(message, "content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for item in content:
            text = _value(item, "text")
            if isinstance(text, str):
                parts.append(text)
    refusal = _value(message, "refusal")
    if isinstance(refusal, str) and refusal:
        parts.append(refusal)
    return "".join(parts), bool(refusal)


def _response_visible_text(response: object) -> tuple[str, bool]:
    parts: list[str] = []
    output_text = _value(response, "output_text")
    if isinstance(output_text, str) and output_text:
        parts.append(output_text)
    refusals: list[str] = []
    for item in _value(response, "output") or []:
        for content in _value(item, "content") or []:
            refusal = _value(content, "refusal")
            if isinstance(refusal, str) and refusal:
                refusals.append(refusal)
    for refusal in refusals:
        if refusal not in parts:
            parts.append(refusal)
    return "\n".join(parts), bool(refusals)


_RETRYABLE_STREAM_ERROR_NAMES = {
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "InternalServerError",
    "RateLimitError",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "TimeoutException",
}

_RETRYABLE_RESPONSE_CODES = {
    "rate_limit_exceeded",
    "server_error",
    "temporarily_unavailable",
    "timeout",
    "overloaded",
}


class OpenAIProviderError(RuntimeError):
    """Structured provider failure that retains safe post-header audit data."""

    def __init__(
        self,
        message: str,
        *,
        request_id: str | None,
        rate_limit: dict[str, str],
        provider_metadata: dict[str, Any],
        server_processing_ms: float | None,
        status_code: int | None = None,
        retryable: bool = False,
        error_code: str | None = None,
        first_event_ms: float | None = None,
        ttft_ms: float | None = None,
        time_after_first_token_ms: float | None = None,
        first_token_at: str | None = None,
        finish_reason: str | None = None,
        response_id: str | None = None,
        returned_model: str | None = None,
        input_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        output_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        total_tokens: int | None = None,
        partial_response: str | None = None,
    ) -> None:
        super().__init__(message)
        if error_code:
            provider_metadata.setdefault("error_code", error_code)
        provider_metadata.setdefault("retryable", retryable)
        headers = dict(provider_metadata.get("server_headers", {}))
        headers.update(rate_limit)
        self.response = SimpleNamespace(headers=headers)
        self.request_id = request_id
        self.status_code = status_code
        self.retryable = retryable
        self.error_code = error_code
        self.provider_metadata = provider_metadata
        self.rate_limit = rate_limit
        self.server_processing_ms = server_processing_ms
        self.first_event_ms = first_event_ms
        self.ttft_ms = ttft_ms
        self.time_after_first_token_ms = time_after_first_token_ms
        self.first_token_at = first_token_at
        self.finish_reason = finish_reason
        self.response_id = response_id
        self.returned_model = returned_model
        self.input_tokens = input_tokens
        self.cached_input_tokens = cached_input_tokens
        self.cache_write_tokens = cache_write_tokens
        self.output_tokens = output_tokens
        self.reasoning_tokens = reasoning_tokens
        self.total_tokens = total_tokens
        self.partial_response = partial_response


def _is_retryable_stream_exception(exc: BaseException) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429} or (
        isinstance(status_code, int) and status_code >= 500
    ):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "error_code", None)
    return bool(
        type(exc).__name__ in _RETRYABLE_STREAM_ERROR_NAMES
        or code in _RETRYABLE_RESPONSE_CODES
    )


def _exception_transport_context(
    exc: BaseException,
    *,
    endpoint: str,
) -> tuple[str | None, dict[str, str], dict[str, Any], float | None, int | None]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    request_id = getattr(exc, "request_id", None) or headers.get("x-request-id")
    rate_limit = {
        name: value for name in _RATE_HEADERS if (value := headers.get(name))
    }
    server_headers = {
        name: value for name in _SERVER_HEADERS if (value := headers.get(name))
    }
    status_code = getattr(exc, "status_code", None) or getattr(
        response, "status_code", None
    )
    processing_ms = _float(headers.get("openai-processing-ms"))
    metadata: dict[str, Any] = {
        "endpoint": endpoint,
        "streaming": True,
        "openai_sdk_version": openai.__version__,
        "http_status": status_code,
        "http_version": getattr(response, "http_version", None),
        "sdk_retries": getattr(exc, "retries_taken", None),
        "server_headers": server_headers,
    }
    if processing_ms is not None:
        metadata["openai_processing_ms"] = processing_ms
    return request_id, rate_limit, metadata, processing_ms, status_code


def _response_error_fields(response: object) -> dict[str, Any]:
    usage = _value(response, "usage")
    input_details = _value(usage, "input_tokens_details")
    output_details = _value(usage, "output_tokens_details")
    partial_response, _ = _response_visible_text(response)
    return {
        "finish_reason": _value(response, "status"),
        "response_id": _value(response, "id"),
        "returned_model": _value(response, "model"),
        "input_tokens": _value(usage, "input_tokens"),
        "cached_input_tokens": _value(input_details, "cached_tokens"),
        "cache_write_tokens": _value(input_details, "cache_write_tokens"),
        "output_tokens": _value(usage, "output_tokens"),
        "reasoning_tokens": _value(output_details, "reasoning_tokens"),
        "total_tokens": _value(usage, "total_tokens"),
        "partial_response": partial_response or None,
    }


class OpenAIProvider:
    def __init__(self) -> None:
        self._clients: dict[str, AsyncOpenAI] = {}

    def _client(
        self, key_env: str, base_url_env: str | None = None
    ) -> AsyncOpenAI:
        api_key = os.getenv(key_env)
        if not api_key:
            raise RuntimeError(
                f"{key_env} is not set. Put it in .env or export it before running."
            )
        base_url = None
        if base_url_env is not None:
            base_url = os.getenv(base_url_env)
            if not base_url:
                raise RuntimeError(
                    f"{base_url_env} is not set. Put the OpenAI-compatible "
                    "server URL in .env or export it before running."
                )
        client_key = key_env if base_url is None else f"{key_env}@{base_url}"
        if client_key not in self._clients:
            # Retry accounting lives in the runner so attempts and backoff are
            # visible in SQLite rather than hidden inside the SDK.
            self._clients[client_key] = AsyncOpenAI(
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
    def _headers(
        raw: Any,
        *,
        endpoint: str,
        streaming: bool,
    ) -> tuple[str | None, dict[str, str], dict[str, Any], float | None]:
        headers = raw.headers
        request_id = getattr(raw, "request_id", None) or headers.get("x-request-id")
        rate_limit = {
            name: value for name in _RATE_HEADERS if (value := headers.get(name))
        }
        server_headers = {
            name: value for name in _SERVER_HEADERS if (value := headers.get(name))
        }
        processing_ms = _float(headers.get("openai-processing-ms"))
        metadata: dict[str, Any] = {
            "endpoint": endpoint,
            "streaming": streaming,
            "openai_sdk_version": openai.__version__,
            "http_status": getattr(raw, "status_code", None),
            "http_version": getattr(raw, "http_version", None),
            "sdk_retries": getattr(raw, "retries_taken", None),
            "server_headers": server_headers,
        }
        # Retain the historical flat key for old analysis scripts.
        if processing_ms is not None:
            metadata["openai_processing_ms"] = processing_ms
        return request_id, rate_limit, metadata, processing_ms

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
        if config.endpoint == "chat_completions":
            return await self._chat_completion(
                config=config,
                messages=messages,
                max_output_tokens=max_output_tokens,
                stream=stream,
                output_schema=output_schema,
                output_schema_name=output_schema_name,
            )
        if config.endpoint == "responses":
            return await self._response(
                config=config,
                messages=messages,
                max_output_tokens=max_output_tokens,
                stream=stream,
                output_schema=output_schema,
                output_schema_name=output_schema_name,
            )
        raise ValueError(
            f"OpenAI endpoint {config.endpoint!r} is not supported; use "
            "chat_completions or responses"
        )

    @staticmethod
    def _chat_kwargs(
        config: ModelConfig,
        messages: list[dict[str, str]],
        max_output_tokens: int | None,
        output_schema: Mapping[str, Any] | None = None,
        output_schema_name: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": config.api_model,
            "messages": messages,
            config.token_parameter: max_output_tokens or config.max_output_tokens,
        }
        if config.send_temperature and config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = config.reasoning_effort
        if config.service_tier is not None:
            kwargs["service_tier"] = config.service_tier
        if output_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_schema_name or "structured_output",
                    "strict": True,
                    "schema": _provider_safe_schema(dict(output_schema)),
                },
            }
        return kwargs

    async def _chat_completion(
        self,
        *,
        config: ModelConfig,
        messages: list[dict[str, str]],
        max_output_tokens: int | None,
        stream: bool,
        output_schema: Mapping[str, Any] | None = None,
        output_schema_name: str | None = None,
    ) -> CompletionResult:
        kwargs = self._chat_kwargs(
            config,
            messages,
            max_output_tokens,
            output_schema,
            output_schema_name,
        )
        request_contract = _request_contract(config, max_output_tokens)
        request_contract["structured_output"] = output_schema is not None
        request_contract["structured_output_schema_constraints_removed"] = (
            sorted(_UNSUPPORTED_STRICT_SCHEMA_KEYWORDS)
            if output_schema is not None
            else []
        )
        client = self._client(config.api_key_env, config.base_url_env)
        if not stream:
            raw = await client.chat.completions.with_raw_response.create(**kwargs)
            response = raw.parse()
            request_id, rate_limit, metadata, processing_ms = self._headers(
                raw, endpoint=config.endpoint, streaming=False
            )
            metadata["request_contract"] = request_contract
            metadata["returned_service_tier"] = _value(
                response, "service_tier"
            )
            if not response.choices or response.choices[0].finish_reason is None:
                raise OpenAIProviderError(
                    "Chat Completions response had no terminal finish reason",
                    request_id=request_id,
                    rate_limit=rate_limit,
                    provider_metadata=metadata,
                    server_processing_ms=processing_ms,
                    status_code=getattr(raw, "status_code", None),
                    retryable=True,
                )
            choice = response.choices[0]
            usage = response.usage
            _usage_metadata(
                metadata,
                usage,
                input_field="prompt_tokens",
                output_field="completion_tokens",
                input_details_field="prompt_tokens_details",
            )
            prompt_details = _value(usage, "prompt_tokens_details")
            completion_details = _value(usage, "completion_tokens_details")
            text, contains_refusal = _chat_visible_text(choice.message)
            metadata["output_contains_refusal"] = contains_refusal
            return CompletionResult(
                text=text,
                finish_reason=choice.finish_reason,
                response_id=response.id,
                returned_model=response.model,
                system_fingerprint=response.system_fingerprint,
                request_id=request_id,
                input_tokens=_value(usage, "prompt_tokens"),
                cached_input_tokens=_value(prompt_details, "cached_tokens"),
                cache_write_tokens=_value(prompt_details, "cache_write_tokens"),
                output_tokens=_value(usage, "completion_tokens"),
                reasoning_tokens=_value(completion_details, "reasoning_tokens"),
                total_tokens=_value(usage, "total_tokens"),
                server_processing_ms=processing_ms,
                rate_limit=rate_limit,
                provider_metadata=metadata,
            )

        started = time.perf_counter()
        first_event_ms: float | None = None
        ttft_ms: float | None = None
        first_token_at: str | None = None
        parts: list[str] = []
        usage: Any = None
        finish_reason: str | None = None
        response_id: str | None = None
        returned_model: str | None = None
        system_fingerprint: str | None = None
        returned_service_tier: str | None = None
        event_count = 0
        text_chunk_count = 0
        contains_refusal = False
        request_id: str | None = None
        rate_limit: dict[str, str] = {}
        metadata: dict[str, Any] = {}
        processing_ms: float | None = None
        headers_received_ms: float | None = None
        status_code: int | None = None

        try:
            async with client.chat.completions.with_streaming_response.create(
                **kwargs,
                stream=True,
                stream_options={"include_usage": True},
            ) as raw:
                headers_received_ms = (time.perf_counter() - started) * 1000
                status_code = getattr(raw, "status_code", None)
                request_id, rate_limit, metadata, processing_ms = self._headers(
                    raw, endpoint=config.endpoint, streaming=True
                )
                chunks = await raw.parse()
                async for chunk in chunks:
                    event_count += 1
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    if first_event_ms is None:
                        first_event_ms = elapsed_ms
                    response_id = response_id or _value(chunk, "id")
                    returned_model = returned_model or _value(chunk, "model")
                    returned_service_tier = returned_service_tier or _value(
                        chunk, "service_tier"
                    )
                    system_fingerprint = system_fingerprint or _value(
                        chunk, "system_fingerprint"
                    )
                    if _value(chunk, "usage") is not None:
                        usage = chunk.usage
                    # The final usage-only chunk deliberately has choices=[].
                    for choice in chunk.choices:
                        if choice.finish_reason is not None:
                            finish_reason = choice.finish_reason
                        content = _value(choice.delta, "content")
                        refusal = _value(choice.delta, "refusal")
                        for visible, is_refusal in (
                            (content, False),
                            (refusal, True),
                        ):
                            if not isinstance(visible, str) or not visible:
                                continue
                            if visible.strip() and ttft_ms is None:
                                ttft_ms = elapsed_ms
                                first_token_at = _utc_now()
                            contains_refusal = contains_refusal or is_refusal
                            text_chunk_count += 1
                            parts.append(visible)
        except asyncio.CancelledError:
            raise
        except OpenAIProviderError:
            raise
        except Exception as exc:
            (
                exc_request_id,
                exc_rate_limit,
                exc_metadata,
                exc_processing_ms,
                exc_status_code,
            ) = _exception_transport_context(exc, endpoint=config.endpoint)
            request_id = request_id or exc_request_id
            rate_limit = {**exc_rate_limit, **rate_limit}
            if not metadata:
                metadata = exc_metadata
            processing_ms = (
                processing_ms
                if processing_ms is not None
                else exc_processing_ms
            )
            status_code = status_code or exc_status_code
            ended_ms = (time.perf_counter() - started) * 1000
            metadata.update(
                {
                    "response_headers_ms": headers_received_ms,
                    "stream_event_count": event_count,
                    "text_chunk_count": text_chunk_count,
                    "stream_interrupted": True,
                    "request_contract": request_contract,
                    "returned_service_tier": returned_service_tier,
                }
            )
            raise OpenAIProviderError(
                f"Chat Completions stream interrupted: {type(exc).__name__}: {exc}",
                request_id=request_id,
                rate_limit=rate_limit,
                provider_metadata=metadata,
                server_processing_ms=processing_ms,
                status_code=getattr(exc, "status_code", None) or status_code,
                retryable=_is_retryable_stream_exception(exc),
                error_code=getattr(exc, "code", None),
                first_event_ms=first_event_ms,
                ttft_ms=ttft_ms,
                time_after_first_token_ms=(
                    max(ended_ms - ttft_ms, 0.0) if ttft_ms is not None else None
                ),
                first_token_at=first_token_at,
                finish_reason=finish_reason,
                response_id=response_id,
                returned_model=returned_model,
                partial_response="".join(parts) or None,
            ) from exc

        ended_ms = (time.perf_counter() - started) * 1000
        stream_tail_ms = (
            max(ended_ms - ttft_ms, 0.0) if ttft_ms is not None else None
        )
        _usage_metadata(
            metadata,
            usage,
            input_field="prompt_tokens",
            output_field="completion_tokens",
            input_details_field="prompt_tokens_details",
        )
        prompt_details = _value(usage, "prompt_tokens_details")
        completion_details = _value(usage, "completion_tokens_details")
        metadata.update(
            {
                "response_headers_ms": headers_received_ms,
                "stream_event_count": event_count,
                "text_chunk_count": text_chunk_count,
                "output_contains_refusal": contains_refusal,
                "request_contract": request_contract,
                "returned_service_tier": returned_service_tier,
            }
        )
        if finish_reason is None:
            metadata["stream_incomplete"] = True
            raise OpenAIProviderError(
                "Chat Completions stream ended without a terminal finish reason",
                request_id=request_id,
                rate_limit=rate_limit,
                provider_metadata=metadata,
                server_processing_ms=processing_ms,
                status_code=status_code,
                retryable=True,
                error_code="incomplete_stream",
                first_event_ms=first_event_ms,
                ttft_ms=ttft_ms,
                time_after_first_token_ms=stream_tail_ms,
                first_token_at=first_token_at,
                response_id=response_id,
                returned_model=returned_model,
                partial_response="".join(parts) or None,
            )
        return CompletionResult(
            text="".join(parts),
            finish_reason=finish_reason,
            response_id=response_id,
            returned_model=returned_model,
            system_fingerprint=system_fingerprint,
            request_id=request_id,
            input_tokens=_value(usage, "prompt_tokens"),
            cached_input_tokens=_value(prompt_details, "cached_tokens"),
            cache_write_tokens=_value(prompt_details, "cache_write_tokens"),
            output_tokens=_value(usage, "completion_tokens"),
            reasoning_tokens=_value(completion_details, "reasoning_tokens"),
            total_tokens=_value(usage, "total_tokens"),
            first_event_ms=first_event_ms,
            ttft_ms=ttft_ms,
            time_after_first_token_ms=stream_tail_ms,
            first_token_at=first_token_at,
            server_processing_ms=processing_ms,
            rate_limit=rate_limit,
            provider_metadata=metadata,
        )

    @staticmethod
    def _response_kwargs(
        config: ModelConfig,
        messages: list[dict[str, str]],
        max_output_tokens: int | None,
        output_schema: Mapping[str, Any] | None = None,
        output_schema_name: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": config.api_model,
            "input": messages,
            "max_output_tokens": max_output_tokens or config.max_output_tokens,
            "store": False,
        }
        if config.send_temperature and config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": config.reasoning_effort}
        if config.service_tier is not None:
            kwargs["service_tier"] = config.service_tier
        if output_schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": output_schema_name or "structured_output",
                    "strict": True,
                    "schema": _provider_safe_schema(dict(output_schema)),
                }
            }
        return kwargs

    @staticmethod
    def _response_result(
        response: Any,
        *,
        request_id: str | None,
        rate_limit: dict[str, str],
        metadata: dict[str, Any],
        processing_ms: float | None,
        first_event_ms: float | None = None,
        ttft_ms: float | None = None,
        time_after_first_token_ms: float | None = None,
        first_token_at: str | None = None,
    ) -> CompletionResult:
        usage = response.usage
        _usage_metadata(
            metadata,
            usage,
            input_field="input_tokens",
            output_field="output_tokens",
            input_details_field="input_tokens_details",
        )
        input_details = _value(usage, "input_tokens_details")
        output_details = _value(usage, "output_tokens_details")
        incomplete_details = _value(response, "incomplete_details")
        incomplete_reason = _value(incomplete_details, "reason")
        finish_reason = _value(response, "status") or "unknown"
        if finish_reason == "incomplete" and incomplete_reason == "max_output_tokens":
            finish_reason = "length"
        if incomplete_reason:
            metadata["incomplete_reason"] = incomplete_reason
        text, contains_refusal = _response_visible_text(response)
        metadata["output_contains_refusal"] = contains_refusal
        return CompletionResult(
            text=text,
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
            first_event_ms=first_event_ms,
            ttft_ms=ttft_ms,
            time_after_first_token_ms=time_after_first_token_ms,
            first_token_at=first_token_at,
            server_processing_ms=processing_ms,
            rate_limit=rate_limit,
            provider_metadata=metadata,
        )

    async def _response(
        self,
        *,
        config: ModelConfig,
        messages: list[dict[str, str]],
        max_output_tokens: int | None,
        stream: bool,
        output_schema: Mapping[str, Any] | None = None,
        output_schema_name: str | None = None,
    ) -> CompletionResult:
        kwargs = self._response_kwargs(
            config,
            messages,
            max_output_tokens,
            output_schema,
            output_schema_name,
        )
        request_contract = _request_contract(config, max_output_tokens)
        request_contract["structured_output"] = output_schema is not None
        request_contract["structured_output_schema_constraints_removed"] = (
            sorted(_UNSUPPORTED_STRICT_SCHEMA_KEYWORDS)
            if output_schema is not None
            else []
        )
        client = self._client(config.api_key_env, config.base_url_env)
        if not stream:
            raw = await client.responses.with_raw_response.create(**kwargs)
            response = raw.parse()
            request_id, rate_limit, metadata, processing_ms = self._headers(
                raw, endpoint=config.endpoint, streaming=False
            )
            metadata["request_contract"] = request_contract
            metadata["returned_service_tier"] = _value(
                response, "service_tier"
            )
            status = _value(response, "status")
            incomplete_reason = _value(
                _value(response, "incomplete_details"), "reason"
            )
            if status == "failed" or (
                status == "incomplete"
                and incomplete_reason != "max_output_tokens"
            ):
                _usage_metadata(
                    metadata,
                    _value(response, "usage"),
                    input_field="input_tokens",
                    output_field="output_tokens",
                    input_details_field="input_tokens_details",
                )
                _, contains_refusal = _response_visible_text(response)
                metadata["output_contains_refusal"] = contains_refusal
            response_error_fields = _response_error_fields(response)
            if status == "failed":
                error = _value(response, "error")
                code = _value(error, "code")
                raise OpenAIProviderError(
                    f"Responses request failed: {_value(error, 'message') or error}",
                    request_id=request_id,
                    rate_limit=rate_limit,
                    provider_metadata=metadata,
                    server_processing_ms=processing_ms,
                    status_code=getattr(raw, "status_code", None),
                    retryable=code in _RETRYABLE_RESPONSE_CODES,
                    error_code=code,
                    **response_error_fields,
                )
            if status == "incomplete" and incomplete_reason != "max_output_tokens":
                metadata["incomplete_reason"] = incomplete_reason
                raise OpenAIProviderError(
                    f"Responses request incomplete: {incomplete_reason or 'unknown'}",
                    request_id=request_id,
                    rate_limit=rate_limit,
                    provider_metadata=metadata,
                    server_processing_ms=processing_ms,
                    status_code=getattr(raw, "status_code", None),
                    retryable=False,
                    error_code=incomplete_reason or "incomplete_response",
                    **response_error_fields,
                )
            return self._response_result(
                response,
                request_id=request_id,
                rate_limit=rate_limit,
                metadata=metadata,
                processing_ms=processing_ms,
            )

        started = time.perf_counter()
        first_event_ms: float | None = None
        ttft_ms: float | None = None
        first_token_at: str | None = None
        terminal: Any = None
        event_count = 0
        text_chunk_count = 0
        request_id: str | None = None
        rate_limit: dict[str, str] = {}
        metadata: dict[str, Any] = {}
        processing_ms: float | None = None
        headers_received_ms: float | None = None
        status_code: int | None = None

        try:
            async with client.responses.with_streaming_response.create(
                **kwargs, stream=True
            ) as raw:
                headers_received_ms = (time.perf_counter() - started) * 1000
                status_code = getattr(raw, "status_code", None)
                request_id, rate_limit, metadata, processing_ms = self._headers(
                    raw, endpoint=config.endpoint, streaming=True
                )
                events = await raw.parse()
                async for event in events:
                    event_count += 1
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    if first_event_ms is None:
                        first_event_ms = elapsed_ms
                    if event.type in {
                        "response.output_text.delta",
                        "response.refusal.delta",
                    } and _value(event, "delta"):
                        delta = str(event.delta)
                        if delta.strip() and ttft_ms is None:
                            ttft_ms = elapsed_ms
                            first_token_at = _utc_now()
                        text_chunk_count += 1
                    if event.type in {
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                    }:
                        terminal = event.response
        except asyncio.CancelledError:
            raise
        except OpenAIProviderError:
            raise
        except Exception as exc:
            (
                exc_request_id,
                exc_rate_limit,
                exc_metadata,
                exc_processing_ms,
                exc_status_code,
            ) = _exception_transport_context(exc, endpoint=config.endpoint)
            request_id = request_id or exc_request_id
            rate_limit = {**exc_rate_limit, **rate_limit}
            if not metadata:
                metadata = exc_metadata
            processing_ms = (
                processing_ms
                if processing_ms is not None
                else exc_processing_ms
            )
            status_code = status_code or exc_status_code
            ended_ms = (time.perf_counter() - started) * 1000
            metadata.update(
                {
                    "response_headers_ms": headers_received_ms,
                    "stream_event_count": event_count,
                    "text_chunk_count": text_chunk_count,
                    "stream_interrupted": True,
                    "request_contract": request_contract,
                }
            )
            raise OpenAIProviderError(
                f"Responses stream interrupted: {type(exc).__name__}: {exc}",
                request_id=request_id,
                rate_limit=rate_limit,
                provider_metadata=metadata,
                server_processing_ms=processing_ms,
                status_code=getattr(exc, "status_code", None) or status_code,
                retryable=_is_retryable_stream_exception(exc),
                error_code=getattr(exc, "code", None),
                first_event_ms=first_event_ms,
                ttft_ms=ttft_ms,
                time_after_first_token_ms=(
                    max(ended_ms - ttft_ms, 0.0) if ttft_ms is not None else None
                ),
                first_token_at=first_token_at,
            ) from exc

        ended_ms = (time.perf_counter() - started) * 1000
        stream_tail_ms = (
            max(ended_ms - ttft_ms, 0.0) if ttft_ms is not None else None
        )
        metadata.update(
            {
                "response_headers_ms": headers_received_ms,
                "stream_event_count": event_count,
                "text_chunk_count": text_chunk_count,
                "request_contract": request_contract,
                "returned_service_tier": _value(
                    terminal, "service_tier"
                ),
            }
        )
        if terminal is None:
            metadata["stream_incomplete"] = True
            raise OpenAIProviderError(
                "Responses stream ended without a terminal event",
                request_id=request_id,
                rate_limit=rate_limit,
                provider_metadata=metadata,
                server_processing_ms=processing_ms,
                status_code=status_code,
                retryable=True,
                error_code="incomplete_stream",
                first_event_ms=first_event_ms,
                ttft_ms=ttft_ms,
                time_after_first_token_ms=stream_tail_ms,
                first_token_at=first_token_at,
            )
        terminal_status = _value(terminal, "status")
        response_error_fields = _response_error_fields(terminal)
        if terminal_status == "failed" or terminal_status == "incomplete":
            _usage_metadata(
                metadata,
                _value(terminal, "usage"),
                input_field="input_tokens",
                output_field="output_tokens",
                input_details_field="input_tokens_details",
            )
            _, contains_refusal = _response_visible_text(terminal)
            metadata["output_contains_refusal"] = contains_refusal
        if terminal_status == "failed":
            error = _value(terminal, "error")
            code = _value(error, "code")
            metadata["response_error_code"] = code
            raise OpenAIProviderError(
                f"Responses stream failed: {_value(error, 'message') or error}",
                request_id=request_id,
                rate_limit=rate_limit,
                provider_metadata=metadata,
                server_processing_ms=processing_ms,
                status_code=status_code,
                retryable=code in _RETRYABLE_RESPONSE_CODES,
                error_code=code,
                first_event_ms=first_event_ms,
                ttft_ms=ttft_ms,
                time_after_first_token_ms=stream_tail_ms,
                first_token_at=first_token_at,
                **response_error_fields,
            )
        incomplete_reason = _value(
            _value(terminal, "incomplete_details"), "reason"
        )
        if terminal_status == "incomplete" and incomplete_reason != "max_output_tokens":
            metadata["incomplete_reason"] = incomplete_reason
            raise OpenAIProviderError(
                f"Responses stream incomplete: {incomplete_reason or 'unknown'}",
                request_id=request_id,
                rate_limit=rate_limit,
                provider_metadata=metadata,
                server_processing_ms=processing_ms,
                status_code=status_code,
                retryable=False,
                error_code=incomplete_reason or "incomplete_response",
                first_event_ms=first_event_ms,
                ttft_ms=ttft_ms,
                time_after_first_token_ms=stream_tail_ms,
                first_token_at=first_token_at,
                **response_error_fields,
            )
        return self._response_result(
            terminal,
            request_id=request_id,
            rate_limit=rate_limit,
            metadata=metadata,
            processing_ms=processing_ms,
            first_event_ms=first_event_ms,
            ttft_ms=ttft_ms,
            time_after_first_token_ms=stream_tail_ms,
            first_token_at=first_token_at,
        )
