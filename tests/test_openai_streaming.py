from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import acuitybench.providers.openai as openai_provider_module
from acuitybench.models import ModelRegistry
from acuitybench.providers.openai import OpenAIProvider, OpenAIProviderError


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_paper_openai_profiles_send_resolved_reasoning_and_service_tier() -> None:
    registry = ModelRegistry()
    mini = OpenAIProvider._chat_kwargs(
        registry.get("gpt-5-mini"),
        [{"role": "user", "content": "case"}],
        None,
    )
    frontier = OpenAIProvider._chat_kwargs(
        registry.get("gpt-5.4"),
        [{"role": "user", "content": "case"}],
        None,
    )
    assert mini["reasoning_effort"] == "medium"
    assert mini["service_tier"] == "default"
    assert mini["max_completion_tokens"] == 4096
    assert "temperature" not in mini
    assert frontier["reasoning_effort"] == "none"
    assert frontier["temperature"] == 1
    assert frontier["service_tier"] == "default"
    assert frontier["max_completion_tokens"] == 4096

    responses = OpenAIProvider._response_kwargs(
        registry.get("gpt-5.4"),
        [{"role": "user", "content": "case"}],
        None,
    )
    assert responses["reasoning"] == {"effort": "none"}
    assert responses["service_tier"] == "default"


class FakeAsyncStream:
    def __init__(self, clock: FakeClock, events: list[tuple[float, Any]]) -> None:
        self.clock = clock
        self.events = iter(events)

    def __aiter__(self) -> "FakeAsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            delay, event = next(self.events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        self.clock.advance(delay)
        if isinstance(event, BaseException):
            raise event
        return event


class FakeRawResponse:
    def __init__(self, stream: FakeAsyncStream, headers: dict[str, str]) -> None:
        self._stream = stream
        self.headers = headers
        self.request_id = headers.get("x-request-id")
        self.status_code = 200
        self.http_version = "HTTP/2"
        self.retries_taken = 0

    async def parse(self) -> FakeAsyncStream:
        return self._stream


class FakeResponseContext:
    def __init__(self, clock: FakeClock, raw: FakeRawResponse) -> None:
        self.clock = clock
        self.raw = raw

    async def __aenter__(self) -> FakeRawResponse:
        self.clock.advance(0.005)
        return self.raw

    async def __aexit__(self, *_: object) -> None:
        return None


class FailingResponseContext:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def __aenter__(self) -> FakeRawResponse:
        raise self.error

    async def __aexit__(self, *_: object) -> None:
        return None


class FakeCreate:
    def __init__(self, context: FakeResponseContext) -> None:
        self.context = context
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> FakeResponseContext:
        self.kwargs = kwargs
        return self.context


def _choice(content: str | None, finish: str | None = None) -> Any:
    return SimpleNamespace(
        finish_reason=finish,
        delta=SimpleNamespace(content=content, refusal=None),
    )


def _refusal_choice(refusal: str, finish: str | None = None) -> Any:
    return SimpleNamespace(
        finish_reason=finish,
        delta=SimpleNamespace(content=None, refusal=refusal),
    )


def _chat_chunk(
    *,
    choices: list[Any],
    usage: Any = None,
) -> Any:
    return SimpleNamespace(
        id="chat-response",
        model="gpt-test-snapshot",
        system_fingerprint="fingerprint",
        choices=choices,
        usage=usage,
    )


def test_chat_stream_collects_ttft_headers_and_usage_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(openai_provider_module.time, "perf_counter", clock)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    usage = SimpleNamespace(
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
        prompt_tokens_details=SimpleNamespace(cached_tokens=3, cache_write_tokens=2),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=4),
    )
    stream = FakeAsyncStream(
        clock,
        [
            (0.010, _chat_chunk(choices=[_choice(None)])),
            (0.020, _chat_chunk(choices=[_choice("Hel")])),
            (0.010, _chat_chunk(choices=[_choice("lo")])),
            (0.005, _chat_chunk(choices=[_choice(None, "stop")])),
            # The final usage chunk has no choices and must still be consumed.
            (0.005, _chat_chunk(choices=[], usage=usage)),
        ],
    )
    raw = FakeRawResponse(
        stream,
        {
            "x-request-id": "request-123",
            "openai-processing-ms": "12.5",
            "openai-version": "2026-01-01",
            "x-ratelimit-remaining-requests": "99",
        },
    )
    create = FakeCreate(FakeResponseContext(clock, raw))
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(with_streaming_response=create)
        )
    )
    provider = OpenAIProvider()
    provider._clients["OPENAI_API_KEY"] = client  # type: ignore[assignment]

    result = asyncio.run(
        provider.complete(
            config=ModelRegistry().get("gpt-5-mini"),
            messages=[{"role": "user", "content": "hello"}],
            max_output_tokens=20,
            stream=True,
        )
    )

    assert result.text == "Hello"
    assert result.finish_reason == "stop"
    assert result.first_event_ms == pytest.approx(15)
    assert result.ttft_ms == pytest.approx(35)
    assert result.server_processing_ms == pytest.approx(12.5)
    assert result.request_id == "request-123"
    assert result.input_tokens == 11
    assert result.cached_input_tokens == 3
    assert result.cache_write_tokens == 2
    assert result.output_tokens == 7
    assert result.reasoning_tokens == 4
    assert result.total_tokens == 18
    assert result.provider_metadata["response_headers_ms"] == pytest.approx(5)
    assert result.provider_metadata["server_headers"]["openai-version"] == "2026-01-01"
    assert result.provider_metadata["stream_event_count"] == 5
    assert result.provider_metadata["usage_complete"] is True
    assert create.kwargs is not None
    assert create.kwargs["stream"] is True
    assert create.kwargs["stream_options"] == {"include_usage": True}


def test_responses_stream_uses_visible_text_not_first_sse_for_ttft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(openai_provider_module.time, "perf_counter", clock)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    usage = SimpleNamespace(
        input_tokens=9,
        output_tokens=5,
        total_tokens=14,
        input_tokens_details=SimpleNamespace(cached_tokens=1, cache_write_tokens=0),
        output_tokens_details=SimpleNamespace(reasoning_tokens=2),
    )
    terminal = SimpleNamespace(
        output_text="ACUITY: D",
        usage=usage,
        id="response-123",
        model="responses-test-snapshot",
        status="completed",
        incomplete_details=None,
    )
    events = FakeAsyncStream(
        clock,
        [
            (0.010, SimpleNamespace(type="response.created")),
            (
                0.020,
                SimpleNamespace(
                    type="response.output_text.delta", delta="ACUITY: D"
                ),
            ),
            (
                0.030,
                SimpleNamespace(type="response.completed", response=terminal),
            ),
        ],
    )
    raw = FakeRawResponse(
        events,
        {"x-request-id": "responses-request", "openai-processing-ms": "8"},
    )
    create = FakeCreate(FakeResponseContext(clock, raw))
    client = SimpleNamespace(
        responses=SimpleNamespace(with_streaming_response=create)
    )
    provider = OpenAIProvider()
    provider._clients["OPENAI_API_KEY"] = client  # type: ignore[assignment]
    config = ModelRegistry().get("gpt-5-mini")
    config = type(config)(**{**config.as_dict(), "endpoint": "responses"})

    result = asyncio.run(
        provider.complete(
            config=config,
            messages=[{"role": "user", "content": "case"}],
            max_output_tokens=20,
            stream=True,
        )
    )

    assert result.text == "ACUITY: D"
    assert result.first_event_ms == pytest.approx(15)
    assert result.ttft_ms == pytest.approx(35)
    assert result.request_id == "responses-request"
    assert result.server_processing_ms == 8
    assert result.input_tokens == 9
    assert result.output_tokens == 5
    assert result.reasoning_tokens == 2
    assert result.provider_metadata["stream_event_count"] == 3


def test_chat_stream_without_terminal_finish_is_retryable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(openai_provider_module.time, "perf_counter", clock)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    stream = FakeAsyncStream(
        clock,
        [(0.010, _chat_chunk(choices=[_choice("partial")]))],
    )
    raw = FakeRawResponse(stream, {"x-request-id": "incomplete-request"})
    create = FakeCreate(FakeResponseContext(clock, raw))
    provider = OpenAIProvider()
    provider._clients["OPENAI_API_KEY"] = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(
            completions=SimpleNamespace(with_streaming_response=create)
        )
    )

    with pytest.raises(OpenAIProviderError) as caught:
        asyncio.run(
            provider.complete(
                config=ModelRegistry().get("gpt-5-mini"),
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            )
        )

    assert caught.value.retryable is True
    assert caught.value.request_id == "incomplete-request"
    assert caught.value.first_event_ms == pytest.approx(15)
    assert caught.value.provider_metadata["stream_incomplete"] is True


def test_chat_stream_missing_usage_is_explicitly_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(openai_provider_module.time, "perf_counter", clock)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    stream = FakeAsyncStream(
        clock,
        [
            (0.010, _chat_chunk(choices=[_choice("answer")])),
            (0.010, _chat_chunk(choices=[_choice(None, "stop")])),
        ],
    )
    create = FakeCreate(
        FakeResponseContext(clock, FakeRawResponse(stream, {"x-request-id": "no-usage"}))
    )
    provider = OpenAIProvider()
    provider._clients["OPENAI_API_KEY"] = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(
            completions=SimpleNamespace(with_streaming_response=create)
        )
    )

    result = asyncio.run(
        provider.complete(
            config=ModelRegistry().get("gpt-5-mini"),
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )

    assert result.text == "answer"
    assert result.input_tokens is None
    assert result.provider_metadata["usage_reported"] is False
    assert result.provider_metadata["usage_complete"] is False


def test_chat_refusal_is_preserved_and_whitespace_does_not_start_ttft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(openai_provider_module.time, "perf_counter", clock)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    usage = SimpleNamespace(
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=5,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    stream = FakeAsyncStream(
        clock,
        [
            (0.010, _chat_chunk(choices=[_choice("\n")])),
            (0.020, _chat_chunk(choices=[_refusal_choice("Cannot comply")])),
            (0.010, _chat_chunk(choices=[_choice(None, "content_filter")])),
            (0.005, _chat_chunk(choices=[], usage=usage)),
        ],
    )
    create = FakeCreate(
        FakeResponseContext(clock, FakeRawResponse(stream, {"x-request-id": "refusal"}))
    )
    provider = OpenAIProvider()
    provider._clients["OPENAI_API_KEY"] = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(
            completions=SimpleNamespace(with_streaming_response=create)
        )
    )

    result = asyncio.run(
        provider.complete(
            config=ModelRegistry().get("gpt-5-mini"),
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    )

    assert result.text == "\nCannot comply"
    assert result.ttft_ms == pytest.approx(35)
    assert result.time_after_first_token_ms == pytest.approx(15)
    assert result.provider_metadata["output_contains_refusal"] is True


def test_midstream_transport_error_keeps_headers_and_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReadError(Exception):
        pass

    clock = FakeClock()
    monkeypatch.setattr(openai_provider_module.time, "perf_counter", clock)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    stream = FakeAsyncStream(
        clock,
        [
            (0.010, _chat_chunk(choices=[_choice("partial")])),
            (0.020, ReadError("socket closed")),
        ],
    )
    create = FakeCreate(
        FakeResponseContext(
            clock,
            FakeRawResponse(stream, {"x-request-id": "broken-request"}),
        )
    )
    provider = OpenAIProvider()
    provider._clients["OPENAI_API_KEY"] = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(
            completions=SimpleNamespace(with_streaming_response=create)
        )
    )

    with pytest.raises(OpenAIProviderError) as caught:
        asyncio.run(
            provider.complete(
                config=ModelRegistry().get("gpt-5-mini"),
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            )
        )

    assert caught.value.retryable is True
    assert caught.value.request_id == "broken-request"
    assert caught.value.ttft_ms == pytest.approx(15)
    assert caught.value.time_after_first_token_ms == pytest.approx(20)
    assert caught.value.provider_metadata["stream_interrupted"] is True


def test_responses_failed_event_preserves_error_code_and_retryability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(openai_provider_module.time, "perf_counter", clock)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    usage = SimpleNamespace(
        input_tokens=6,
        output_tokens=2,
        total_tokens=8,
        input_tokens_details=SimpleNamespace(
            cached_tokens=1, cache_write_tokens=0
        ),
        output_tokens_details=SimpleNamespace(reasoning_tokens=1),
    )
    terminal = SimpleNamespace(
        status="failed",
        error=SimpleNamespace(code="server_error", message="try again"),
        incomplete_details=None,
        id="failed-response-object",
        model="failed-model-snapshot",
        usage=usage,
        output_text="",
        output=[
            SimpleNamespace(
                content=[SimpleNamespace(refusal="Partial refusal")]
            )
        ],
    )
    stream = FakeAsyncStream(
        clock,
        [(0.010, SimpleNamespace(type="response.failed", response=terminal))],
    )
    create = FakeCreate(
        FakeResponseContext(
            clock,
            FakeRawResponse(stream, {"x-request-id": "failed-response"}),
        )
    )
    provider = OpenAIProvider()
    provider._clients["OPENAI_API_KEY"] = SimpleNamespace(  # type: ignore[assignment]
        responses=SimpleNamespace(with_streaming_response=create)
    )
    config = ModelRegistry().get("gpt-5-mini")
    config = type(config)(**{**config.as_dict(), "endpoint": "responses"})

    with pytest.raises(OpenAIProviderError) as caught:
        asyncio.run(
            provider.complete(
                config=config,
                messages=[{"role": "user", "content": "case"}],
                stream=True,
            )
        )

    assert caught.value.retryable is True
    assert caught.value.error_code == "server_error"
    assert caught.value.request_id == "failed-response"
    assert caught.value.response_id == "failed-response-object"
    assert caught.value.returned_model == "failed-model-snapshot"
    assert caught.value.input_tokens == 6
    assert caught.value.cached_input_tokens == 1
    assert caught.value.output_tokens == 2
    assert caught.value.partial_response == "Partial refusal"


def test_stream_context_entry_error_preserves_retry_after_and_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RateLimitAtEnter(Exception):
        status_code = 429
        request_id = "entry-request"

        def __init__(self) -> None:
            self.response = SimpleNamespace(
                status_code=429,
                headers={
                    "x-request-id": "entry-request",
                    "retry-after": "0.25",
                    "x-ratelimit-remaining-requests": "0",
                },
            )
            super().__init__("rate limited before stream context")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    create = FakeCreate(FailingResponseContext(RateLimitAtEnter()))  # type: ignore[arg-type]
    provider = OpenAIProvider()
    provider._clients["OPENAI_API_KEY"] = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(
            completions=SimpleNamespace(with_streaming_response=create)
        )
    )

    with pytest.raises(OpenAIProviderError) as caught:
        asyncio.run(
            provider.complete(
                config=ModelRegistry().get("gpt-5-mini"),
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            )
        )

    assert caught.value.retryable is True
    assert caught.value.request_id == "entry-request"
    assert caught.value.response.headers["retry-after"] == "0.25"
    assert caught.value.rate_limit["x-ratelimit-remaining-requests"] == "0"
