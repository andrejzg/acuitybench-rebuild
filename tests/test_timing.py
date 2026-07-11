from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from acuitybench.evaluation import (
    AttemptContext,
    _complete_with_retries,
)
from acuitybench.models import ModelRegistry
from acuitybench.providers.base import CompletionResult


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.wall_tick = 0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def utcnow(self) -> str:
        self.wall_tick += 1
        return f"2026-01-01T00:00:{self.wall_tick:02d}+00:00"


class FakeStore:
    def __init__(self) -> None:
        self.attempts: list[dict[str, Any]] = []
        self.backoffs: list[dict[str, Any]] = []

    def upsert_attempt(self, row: dict[str, Any]) -> None:
        self.attempts.append(dict(row))

    def update_attempt_backoff(self, **row: Any) -> None:
        self.backoffs.append(dict(row))


class ScriptedProvider:
    def __init__(self, clock: FakeClock, script: list[tuple[float, Any]]) -> None:
        self.clock = clock
        self.script = list(script)
        self.calls = 0
        self.max_output_tokens: list[int | None] = []

    async def complete(self, **kwargs: Any) -> CompletionResult:
        self.max_output_tokens.append(kwargs.get("max_output_tokens"))
        duration, value = self.script[self.calls]
        self.calls += 1
        self.clock.advance(duration)
        if isinstance(value, BaseException):
            raise value
        return value


def _context() -> AttemptContext:
    return AttemptContext(
        execution_id="execution",
        request_key="request",
        dataset="synthetic",
        source_id="1",
        task_type="qa",
        sample_idx=0,
    )


def _result(*, ttft_ms: float | None = 5.0) -> CompletionResult:
    return CompletionResult(
        text="ACUITY: A",
        finish_reason="stop",
        ttft_ms=ttft_ms,
        first_event_ms=2.0 if ttft_ms is not None else None,
        first_token_at=(
            "2026-01-01T00:00:03+00:00" if ttft_ms is not None else None
        ),
        time_after_first_token_ms=7.0 if ttft_ms is not None else None,
        server_processing_ms=4.0,
    )


def test_semaphore_wait_is_separate_from_request_latency() -> None:
    async def scenario() -> tuple[Any, FakeStore]:
        clock = FakeClock()
        store = FakeStore()
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        provider = ScriptedProvider(clock, [(0.010, _result())])
        task = asyncio.create_task(
            _complete_with_retries(
                store=store,  # type: ignore[arg-type]
                context=_context(),
                provider=provider,
                config=ModelRegistry().get("gpt-5-mini"),
                messages=[{"role": "user", "content": "case"}],
                semaphore=semaphore,
                stream=True,
                monotonic=clock,
                utcnow=clock.utcnow,
            )
        )
        await asyncio.sleep(0)
        clock.advance(0.050)
        semaphore.release()
        return await task, store

    outcome, store = asyncio.run(scenario())
    assert outcome.timing.queue_wait_ms == pytest.approx(50)
    assert outcome.timing.request_wall_ms == pytest.approx(10)
    assert outcome.timing.request_wall_total_ms == pytest.approx(10)
    assert outcome.timing.retry_backoff_ms == 0
    assert outcome.timing.service_latency_ms == pytest.approx(10)
    assert outcome.timing.total_duration_ms == pytest.approx(60)
    assert outcome.timing.ttft_ms == 5
    assert outcome.timing.time_after_first_token_ms == 7
    assert store.attempts[0]["queue_wait_ms"] == pytest.approx(50)
    assert store.attempts[0]["request_wall_ms"] == pytest.approx(10)


def test_retry_backoff_and_attempt_wall_times_are_disjoint() -> None:
    class RateLimitLikeError(Exception):
        status_code = 429

        def __init__(self) -> None:
            self.response = SimpleNamespace(
                headers={"retry-after": "0.25", "x-request-id": "failed-request"}
            )
            super().__init__("slow down")

    async def scenario() -> tuple[Any, FakeStore]:
        clock = FakeClock()
        store = FakeStore()
        provider = ScriptedProvider(
            clock,
            [(0.020, RateLimitLikeError()), (0.030, _result(ttft_ms=10.0))],
        )

        async def fake_sleep(seconds: float) -> None:
            clock.advance(seconds)

        outcome = await _complete_with_retries(
            store=store,  # type: ignore[arg-type]
            context=_context(),
            provider=provider,
            config=ModelRegistry().get("gpt-5-mini"),
            messages=[{"role": "user", "content": "case"}],
            semaphore=asyncio.Semaphore(1),
            stream=True,
            monotonic=clock,
            utcnow=clock.utcnow,
            sleep=fake_sleep,
            jitter=lambda: 0.0,
        )
        return outcome, store

    outcome, store = asyncio.run(scenario())
    assert outcome.attempts == 2
    assert outcome.timing.request_wall_ms == pytest.approx(30)
    assert outcome.timing.request_wall_total_ms == pytest.approx(50)
    assert outcome.timing.retry_backoff_ms == pytest.approx(250)
    assert outcome.timing.service_latency_ms == pytest.approx(300)
    assert outcome.timing.total_duration_ms == pytest.approx(300)
    assert outcome.timing.ttft_ms == 10
    assert [row["outcome"] for row in store.attempts] == [
        "retry_error",
        "success",
    ]
    assert store.backoffs[0]["planned_ms"] == pytest.approx(250)
    assert store.backoffs[0]["actual_ms"] == pytest.approx(250)
    assert store.backoffs[0]["source"] == "retry-after"


def test_nonstreaming_call_has_no_ttft() -> None:
    async def scenario() -> Any:
        clock = FakeClock()
        provider = ScriptedProvider(clock, [(0.012, _result(ttft_ms=None))])
        return await _complete_with_retries(
            store=FakeStore(),  # type: ignore[arg-type]
            context=_context(),
            provider=provider,
            config=ModelRegistry().get("gpt-5-mini"),
            messages=[{"role": "user", "content": "case"}],
            semaphore=asyncio.Semaphore(1),
            stream=False,
            monotonic=clock,
            utcnow=clock.utcnow,
        )

    outcome = asyncio.run(scenario())
    assert outcome.timing.timing_source == "instrumented_nonstream"
    assert outcome.timing.ttft_ms is None
    assert outcome.timing.first_token_at is None
    assert outcome.timing.request_wall_ms == pytest.approx(12)


def test_empty_truncated_response_at_hard_cap_is_terminal_failure() -> None:
    async def scenario() -> tuple[Any, FakeStore, ScriptedProvider]:
        clock = FakeClock()
        store = FakeStore()
        truncated = CompletionResult(
            text="",
            finish_reason="length",
            first_event_ms=2.0,
            server_processing_ms=4.0,
        )
        provider = ScriptedProvider(clock, [(0.010, truncated)])
        outcome = await _complete_with_retries(
            store=store,  # type: ignore[arg-type]
            context=_context(),
            provider=provider,
            config=ModelRegistry().get("gpt-5-mini"),
            messages=[{"role": "user", "content": "case"}],
            semaphore=asyncio.Semaphore(1),
            stream=True,
            monotonic=clock,
            utcnow=clock.utcnow,
        )
        return outcome, store, provider

    outcome, store, provider = asyncio.run(scenario())
    assert outcome.result is None
    assert outcome.error == (
        "Provider returned a length-truncated response after 1 attempt(s)"
    )
    assert outcome.timing.request_wall_ms == pytest.approx(10)
    assert outcome.timing.server_processing_ms == pytest.approx(4)
    assert provider.calls == 1
    assert provider.max_output_tokens == [4096]
    assert store.attempts[0]["outcome"] == "terminal_error"
    assert store.attempts[0]["error_type"] == "LengthTruncatedResponse"


@pytest.mark.parametrize("model_id", ["gpt-5-mini", "gpt-5.4"])
def test_paper_profile_never_expands_past_4096_tokens(model_id: str) -> None:
    async def scenario() -> tuple[Any, FakeStore, ScriptedProvider]:
        clock = FakeClock()
        store = FakeStore()
        provider = ScriptedProvider(
            clock,
            [
                (
                    0.010,
                    CompletionResult(
                        text="REASONING: capped but usable\nACUITY: A",
                        finish_reason="length",
                    ),
                ),
                (0.020, _result()),
            ],
        )
        outcome = await _complete_with_retries(
            store=store,  # type: ignore[arg-type]
            context=_context(),
            provider=provider,
            config=ModelRegistry().get(model_id),
            messages=[{"role": "user", "content": "case"}],
            semaphore=asyncio.Semaphore(1),
            stream=True,
            monotonic=clock,
            utcnow=clock.utcnow,
        )
        return outcome, store, provider

    outcome, store, provider = asyncio.run(scenario())
    assert outcome.result is not None
    assert outcome.result.text.endswith("ACUITY: A")
    assert outcome.result.finish_reason == "length"
    assert outcome.error is None
    assert outcome.attempts == 1
    assert provider.calls == 1
    assert provider.max_output_tokens == [4096]
    assert store.attempts[0]["max_output_tokens"] == 4096
    assert store.attempts[0]["outcome"] == "success"
    assert store.attempts[0]["error_type"] is None


def test_partial_length_response_is_retried_and_not_returned() -> None:
    async def scenario() -> tuple[Any, FakeStore]:
        clock = FakeClock()
        store = FakeStore()
        partial = CompletionResult(text="partial", finish_reason="length")
        complete = _result()
        outcome = await _complete_with_retries(
            store=store,  # type: ignore[arg-type]
            context=_context(),
            provider=ScriptedProvider(
                clock,
                [(0.010, partial), (0.020, complete)],
            ),
            config=replace(
                ModelRegistry().get("gpt-5-mini"),
                max_retry_output_tokens=8192,
            ),
            messages=[{"role": "user", "content": "case"}],
            semaphore=asyncio.Semaphore(1),
            stream=True,
            monotonic=clock,
            utcnow=clock.utcnow,
        )
        return outcome, store

    outcome, store = asyncio.run(scenario())
    assert outcome.result is not None
    assert outcome.result.text == "ACUITY: A"
    assert outcome.attempts == 2
    assert [row["outcome"] for row in store.attempts] == [
        "retry_length",
        "success",
    ]


def test_length_retry_then_permanent_error_does_not_return_stale_result() -> None:
    async def scenario() -> tuple[Any, FakeStore]:
        clock = FakeClock()
        store = FakeStore()
        outcome = await _complete_with_retries(
            store=store,  # type: ignore[arg-type]
            context=_context(),
            provider=ScriptedProvider(
                clock,
                [
                    (0.010, CompletionResult(text="partial", finish_reason="length")),
                    (0.020, ValueError("permanent")),
                ],
            ),
            config=replace(
                ModelRegistry().get("gpt-5-mini"),
                max_retry_output_tokens=8192,
            ),
            messages=[{"role": "user", "content": "case"}],
            semaphore=asyncio.Semaphore(1),
            stream=True,
            monotonic=clock,
            utcnow=clock.utcnow,
        )
        return outcome, store

    outcome, store = asyncio.run(scenario())
    assert outcome.result is None
    assert "permanent" in (outcome.error or "")
    assert outcome.timing.request_wall_ms == pytest.approx(20)
    assert outcome.timing.ttft_ms is None
    assert [row["outcome"] for row in store.attempts] == [
        "retry_length",
        "terminal_error",
    ]
