"""Resumable, configuration-driven AcuityBench inference and judging."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

from acuitybench.models import JudgeConfig, ModelConfig, ModelRegistry
from acuitybench.providers import CompletionResult, get_provider
from acuitybench.sources import project_root, sha256_file
from acuitybench.store import EvaluationStore, utc_now


ACUITY_RE = re.compile(r"ACUITY\s*[:\-]\s*([A-D])", re.IGNORECASE)
REASONING_RE = re.compile(
    r"REASONING\s*[:\-]\s*(.*?)(?=\nACUITY|$)", re.IGNORECASE | re.DOTALL
)
TASK_TYPES = ("qa", "conv")
MAX_ATTEMPTS = 6
TIMING_VERSION = 2


@dataclass(frozen=True)
class GenerationTask:
    run_id: str
    case_id: str
    dataset: str
    source_id: str
    task_type: str
    sample_idx: int
    normalized_label: str
    split: str
    mapping_method: str
    is_edge_case: bool
    prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class JudgeTask:
    generation: dict[str, Any]
    prompt: str
    prompt_sha256: str


@dataclass(frozen=True)
class AttemptContext:
    execution_id: str
    request_key: str
    dataset: str
    source_id: str
    task_type: str
    sample_idx: int
    judge_id: str | None = None


@dataclass(frozen=True)
class CallTiming:
    queued_at: str
    request_started_at: str | None
    first_token_at: str | None
    response_completed_at: str
    queue_wait_ms: float
    request_wall_ms: float | None
    request_wall_total_ms: float
    service_latency_ms: float
    retry_backoff_ms: float
    first_event_ms: float | None
    ttft_ms: float | None
    time_after_first_token_ms: float | None
    total_duration_ms: float
    server_processing_ms: float | None
    timing_source: str


@dataclass(frozen=True)
class CallOutcome:
    result: CompletionResult | None
    attempts: int
    error: str | None
    timing: CallTiming


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_label(response: str | None) -> str | None:
    if not response:
        return None
    match = ACUITY_RE.search(response)
    return match.group(1).upper() if match else None


def label_parser_contract() -> dict[str, Any]:
    """Return the explicit scoring-parser contract recorded in reports."""
    return {
        "version": 1,
        "pattern": ACUITY_RE.pattern,
        "flags": ACUITY_RE.flags,
        "selection": "first match; capture group uppercased",
    }


def extract_reasoning(response: str | None) -> str:
    if not response:
        return ""
    match = REASONING_RE.search(response)
    return match.group(1).strip() if match else ""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _redact_error(exc: BaseException, key_env: str) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if secret := os.getenv(key_env):
        message = message.replace(secret, "[REDACTED]")
    return message[:4000]


def _retry_after(exc: BaseException) -> tuple[float | None, str | None]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None, None
    try:
        milliseconds = headers.get("retry-after-ms")
        if milliseconds is not None:
            return float(milliseconds) / 1000.0, "retry-after-ms"
        seconds = headers.get("retry-after")
        if seconds is not None:
            return float(seconds), "retry-after"
    except (TypeError, ValueError):
        pass
    return None, None


def _is_retryable(exc: BaseException) -> bool:
    if getattr(exc, "retryable", False):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429} or (status_code and status_code >= 500):
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def _messages(task_type: str, prompt: str) -> list[dict[str, str]]:
    if task_type == "qa":
        return [{"role": "user", "content": prompt}]
    try:
        parsed = json.loads(prompt)
    except json.JSONDecodeError as exc:
        raise ValueError("Conversational prompt is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValueError("Conversational prompt must be a JSON message list")
    return [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in parsed
    ]


def _select_cases(
    frame: pd.DataFrame,
    *,
    datasets: tuple[str, ...] | None,
    limit: int | None,
) -> pd.DataFrame:
    selected = frame
    if datasets:
        unknown = set(datasets) - set(frame["dataset"].unique())
        if unknown:
            raise ValueError(f"Unknown datasets: {sorted(unknown)}")
        selected = selected[selected["dataset"].isin(datasets)]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        selected = selected.head(limit)
    if selected.empty:
        raise ValueError("The requested selection has no benchmark cases")
    return selected.copy()


def prepare_run(
    *,
    store: EvaluationStore,
    model: ModelConfig,
    benchmark_path: Path,
    tasks: tuple[str, ...],
    samples: int,
    datasets: tuple[str, ...] | None,
    limit: int | None,
    run_id: str | None,
    stream: bool = True,
) -> tuple[str, list[GenerationTask]]:
    if samples < 1:
        raise ValueError("--samples must be at least 1")
    if not tasks or set(tasks) - set(TASK_TYPES):
        raise ValueError("Tasks must contain qa and/or conv")
    if len(set(tasks)) != len(tasks):
        raise ValueError("Tasks must not contain duplicates")
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {benchmark_path}. Run `acuitybench build` first."
        )
    frame = pd.read_csv(benchmark_path, dtype={"source_id": str})
    required = {
        "dataset", "source_id", "normalized_label", "split", "mapping_method",
        "is_edge_case", "qa_prompt", "conversational_prompt",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Benchmark is missing required columns: {sorted(missing)}")
    selected = _select_cases(frame, datasets=datasets, limit=limit)
    selection = {
        "datasets": list(datasets) if datasets else None,
        "limit": limit,
        "case_ids_sha256": sha256_text(
            "\n".join(f"{r.dataset}:{r.source_id}" for r in selected.itertuples())
        ),
    }
    identity = {
        "model_config": model.as_dict(),
        "benchmark_sha256": sha256_file(benchmark_path),
        "tasks": list(tasks),
        "samples": samples,
        "selection": selection,
    }
    fingerprint = sha256_text(_canonical_json(identity))
    actual_run_id = run_id or f"{model.id}-{fingerprint[:12]}"
    manifest = {
        "run_id": actual_run_id,
        "manifest_fingerprint": fingerprint,
        "model_id": model.id,
        "provider": model.provider,
        "api_model": model.api_model,
        "model_config": model.as_dict(),
        "benchmark_path": str(benchmark_path.resolve()),
        "benchmark_sha256": identity["benchmark_sha256"],
        "tasks": list(tasks),
        "samples": samples,
        "selected_cases": len(selected),
        "expected_generations": len(selected) * len(tasks) * samples,
        "selection": selection,
        # Transport is execution provenance, not output-cache identity. Every
        # actual invocation records its own stream mode in run_executions.
        "execution_config": {"initial_stream": stream},
    }
    store.ensure_run(manifest)

    generation_tasks: list[GenerationTask] = []
    for row in selected.itertuples(index=False):
        case_id = f"{row.dataset}:{row.source_id}"
        for task_type in tasks:
            prompt = row.qa_prompt if task_type == "qa" else row.conversational_prompt
            for sample_idx in range(samples):
                generation_tasks.append(
                    GenerationTask(
                        run_id=actual_run_id,
                        case_id=case_id,
                        dataset=str(row.dataset),
                        source_id=str(row.source_id),
                        task_type=task_type,
                        sample_idx=sample_idx,
                        normalized_label=str(row.normalized_label),
                        split=str(row.split),
                        mapping_method=str(row.mapping_method),
                        is_edge_case=bool(row.is_edge_case),
                        prompt=str(prompt),
                        prompt_sha256=sha256_text(str(prompt)),
                    )
                )
    return actual_run_id, generation_tasks


async def _complete_with_retries(
    *,
    store: EvaluationStore,
    context: AttemptContext,
    provider: Any,
    config: ModelConfig,
    messages: list[dict[str, str]],
    semaphore: asyncio.Semaphore,
    stream: bool,
    max_attempts: int = MAX_ATTEMPTS,
    monotonic: Callable[[], float] = time.perf_counter,
    utcnow: Callable[[], str] = utc_now,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[], float] = random.random,
) -> CallOutcome:
    operation_started = monotonic()
    queued_at = utcnow()
    max_output_tokens = config.max_output_tokens
    last_error: str | None = None
    attempts = 0
    queue_wait_total_ms = 0.0
    request_wall_total_ms = 0.0
    retry_backoff_total_ms = 0.0
    terminal_result: CompletionResult | None = None
    successful_result: CompletionResult | None = None
    terminal_exception: BaseException | None = None
    terminal_request_ms: float | None = None
    terminal_request_started_at: str | None = None
    terminal_completed_at = queued_at

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        eligible_at = utcnow()
        wait_started = monotonic()
        await semaphore.acquire()
        request_started_mono = monotonic()
        request_started_at = utcnow()
        queue_wait_ms = (request_started_mono - wait_started) * 1000
        queue_wait_total_ms += queue_wait_ms
        result: CompletionResult | None = None
        try:
            result = await provider.complete(
                config=config,
                messages=messages,
                max_output_tokens=max_output_tokens,
                stream=stream,
            )
        except asyncio.CancelledError:
            request_finished_mono = monotonic()
            request_finished_at = utcnow()
            request_wall_ms = (request_finished_mono - request_started_mono) * 1000
            request_wall_total_ms += request_wall_ms
            semaphore.release()
            store.upsert_attempt(
                {
                    **_attempt_identity(context),
                    "attempt_index": attempt,
                    "max_output_tokens": max_output_tokens,
                    "eligible_at": eligible_at,
                    "request_started_at": request_started_at,
                    "request_finished_at": request_finished_at,
                    "queue_wait_ms": queue_wait_ms,
                    "request_wall_ms": request_wall_ms,
                    "attempt_wall_ms": queue_wait_ms + request_wall_ms,
                    "outcome": "cancelled",
                    "retryable": 0,
                    "error_type": "CancelledError",
                    "error": "Request cancelled",
                }
            )
            raise
        except Exception as exc:  # provider exceptions intentionally normalized here
            request_finished_mono = monotonic()
            request_finished_at = utcnow()
            request_wall_ms = (request_finished_mono - request_started_mono) * 1000
            request_wall_total_ms += request_wall_ms
            semaphore.release()
            terminal_result = None
            terminal_exception = exc
            last_error = _redact_error(exc, config.api_key_env)
            retryable = _is_retryable(exc)
            will_retry = retryable and attempt < max_attempts
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", {}) or {}
            error_metadata = getattr(exc, "provider_metadata", {}) or {}
            error_rate_limit = getattr(exc, "rate_limit", {}) or {}
            store.upsert_attempt(
                {
                    **_attempt_identity(context),
                    "attempt_index": attempt,
                    "max_output_tokens": max_output_tokens,
                    "eligible_at": eligible_at,
                    "request_started_at": request_started_at,
                    "request_finished_at": request_finished_at,
                    "queue_wait_ms": queue_wait_ms,
                    "request_wall_ms": request_wall_ms,
                    "attempt_wall_ms": queue_wait_ms + request_wall_ms,
                    "outcome": "retry_error" if will_retry else "terminal_error",
                    "retryable": int(retryable),
                    "http_status": getattr(exc, "status_code", None),
                    "error_type": type(exc).__name__,
                    "error": last_error,
                    "request_id": (
                        getattr(exc, "request_id", None)
                        or headers.get("x-request-id")
                    ),
                    "first_token_at": getattr(exc, "first_token_at", None),
                    "first_event_ms": getattr(exc, "first_event_ms", None),
                    "ttft_ms": getattr(exc, "ttft_ms", None),
                    "time_after_first_token_ms": getattr(
                        exc, "time_after_first_token_ms", None
                    ),
                    "server_processing_ms": getattr(
                        exc, "server_processing_ms", None
                    ),
                    "rate_limit_json": json.dumps(
                        error_rate_limit, sort_keys=True
                    ),
                    "provider_metadata_json": json.dumps(
                        error_metadata, sort_keys=True
                    ),
                    "partial_response": getattr(
                        exc, "partial_response", None
                    ),
                    "finish_reason": getattr(exc, "finish_reason", None),
                    "response_id": getattr(exc, "response_id", None),
                    "returned_model": getattr(exc, "returned_model", None),
                    "input_tokens": getattr(exc, "input_tokens", None),
                    "cached_input_tokens": getattr(
                        exc, "cached_input_tokens", None
                    ),
                    "cache_write_tokens": getattr(
                        exc, "cache_write_tokens", None
                    ),
                    "output_tokens": getattr(exc, "output_tokens", None),
                    "reasoning_tokens": getattr(
                        exc, "reasoning_tokens", None
                    ),
                    "total_tokens": getattr(exc, "total_tokens", None),
                }
            )
            terminal_request_ms = request_wall_ms
            terminal_request_started_at = request_started_at
            terminal_completed_at = request_finished_at
            if not will_retry:
                break
            delay, backoff_source = _retry_after(exc)
            if delay is None:
                delay = min(60.0, 2.0 ** (attempt - 1)) + jitter()
                backoff_source = "exponential_jitter"
            backoff_started = monotonic()
            try:
                await sleep(delay)
            finally:
                actual_backoff_ms = (monotonic() - backoff_started) * 1000
                retry_backoff_total_ms += actual_backoff_ms
                store.update_attempt_backoff(
                    execution_id=context.execution_id,
                    request_key=context.request_key,
                    attempt_index=attempt,
                    planned_ms=delay * 1000,
                    actual_ms=actual_backoff_ms,
                    source=backoff_source or "unknown",
                )
            continue

        request_finished_mono = monotonic()
        request_finished_at = utcnow()
        request_wall_ms = (request_finished_mono - request_started_mono) * 1000
        request_wall_total_ms += request_wall_ms
        semaphore.release()
        needs_length_retry = result.finish_reason == "length"
        will_retry_length = (
            needs_length_retry
            and attempt < max_attempts
            and max_output_tokens < 8192
        )
        terminal_length_error = needs_length_retry and not will_retry_length
        if terminal_length_error:
            last_error = (
                "Provider returned a length-truncated response after "
                f"{attempt} attempt(s)"
            )
        store.upsert_attempt(
            {
                **_attempt_identity(context),
                "attempt_index": attempt,
                "max_output_tokens": max_output_tokens,
                "eligible_at": eligible_at,
                "request_started_at": request_started_at,
                "first_token_at": result.first_token_at,
                "request_finished_at": request_finished_at,
                "queue_wait_ms": queue_wait_ms,
                "request_wall_ms": request_wall_ms,
                "attempt_wall_ms": queue_wait_ms + request_wall_ms,
                "first_event_ms": result.first_event_ms,
                "ttft_ms": result.ttft_ms,
                "time_after_first_token_ms": result.time_after_first_token_ms,
                "server_processing_ms": result.server_processing_ms,
                "outcome": (
                    "retry_length"
                    if will_retry_length
                    else "terminal_error"
                    if terminal_length_error
                    else "success"
                ),
                "retryable": int(will_retry_length),
                "error_type": (
                    "LengthTruncatedResponse" if terminal_length_error else None
                ),
                "error": last_error if terminal_length_error else None,
                **_attempt_result_fields(result),
            }
        )
        terminal_result = result
        terminal_exception = None
        terminal_request_ms = request_wall_ms
        terminal_request_started_at = request_started_at
        terminal_completed_at = request_finished_at
        if will_retry_length:
            max_output_tokens = min(max_output_tokens * 2, 8192)
            continue
        if terminal_length_error:
            break
        successful_result = result
        last_error = None
        break

    total_duration_ms = (monotonic() - operation_started) * 1000
    ttft_ms = (
        terminal_result.ttft_ms
        if terminal_result
        else getattr(terminal_exception, "ttft_ms", None)
    )
    after_first_token = (
        terminal_result.time_after_first_token_ms
        if terminal_result
        else getattr(terminal_exception, "time_after_first_token_ms", None)
    )
    timing = CallTiming(
        queued_at=queued_at,
        request_started_at=terminal_request_started_at,
        first_token_at=(
            terminal_result.first_token_at
            if terminal_result
            else getattr(terminal_exception, "first_token_at", None)
        ),
        response_completed_at=terminal_completed_at,
        queue_wait_ms=queue_wait_total_ms,
        request_wall_ms=terminal_request_ms,
        request_wall_total_ms=request_wall_total_ms,
        service_latency_ms=request_wall_total_ms + retry_backoff_total_ms,
        retry_backoff_ms=retry_backoff_total_ms,
        first_event_ms=(
            terminal_result.first_event_ms
            if terminal_result
            else getattr(terminal_exception, "first_event_ms", None)
        ),
        ttft_ms=ttft_ms,
        time_after_first_token_ms=after_first_token,
        total_duration_ms=total_duration_ms,
        server_processing_ms=(
            terminal_result.server_processing_ms
            if terminal_result
            else getattr(terminal_exception, "server_processing_ms", None)
        ),
        timing_source="instrumented_stream" if stream else "instrumented_nonstream",
    )
    return CallOutcome(successful_result, attempts, last_error, timing)


def _attempt_identity(context: AttemptContext) -> dict[str, Any]:
    return {
        "execution_id": context.execution_id,
        "request_key": context.request_key,
        "dataset": context.dataset,
        "source_id": context.source_id,
        "task_type": context.task_type,
        "sample_idx": context.sample_idx,
        "judge_id": context.judge_id,
    }


def _attempt_result_fields(result: CompletionResult) -> dict[str, Any]:
    return {
        "finish_reason": result.finish_reason,
        "request_id": result.request_id,
        "response_id": result.response_id,
        "returned_model": result.returned_model,
        "input_tokens": result.input_tokens,
        "cached_input_tokens": result.cached_input_tokens,
        "cache_write_tokens": result.cache_write_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "total_tokens": result.total_tokens,
        "rate_limit_json": json.dumps(result.rate_limit, sort_keys=True),
        "provider_metadata_json": json.dumps(result.provider_metadata, sort_keys=True),
    }


def _parent_timing_fields(timing: CallTiming, execution_id: str) -> dict[str, Any]:
    return {
        "latest_execution_id": execution_id,
        "timing_version": TIMING_VERSION,
        "timing_source": timing.timing_source,
        "queued_at": timing.queued_at,
        "request_started_at": timing.request_started_at,
        "first_token_at": timing.first_token_at,
        "response_completed_at": timing.response_completed_at,
        "queue_wait_ms": timing.queue_wait_ms,
        "request_wall_ms": timing.request_wall_ms,
        "request_wall_total_ms": timing.request_wall_total_ms,
        "service_latency_ms": timing.service_latency_ms,
        "retry_backoff_ms": timing.retry_backoff_ms,
        "first_event_ms": timing.first_event_ms,
        "ttft_ms": timing.ttft_ms,
        "time_after_first_token_ms": timing.time_after_first_token_ms,
        "total_duration_ms": timing.total_duration_ms,
        "server_processing_ms": timing.server_processing_ms,
    }


def _result_fields(result: CompletionResult | None) -> dict[str, Any]:
    if result is None:
        return {
            "finish_reason": None, "response_id": None, "returned_model": None,
            "system_fingerprint": None, "request_id": None, "input_tokens": None,
            "cached_input_tokens": None, "cache_write_tokens": None,
            "output_tokens": None, "reasoning_tokens": None, "total_tokens": None,
            "rate_limit_json": None, "provider_metadata_json": None,
        }
    return {
        "finish_reason": result.finish_reason,
        "response_id": result.response_id,
        "returned_model": result.returned_model,
        "system_fingerprint": result.system_fingerprint,
        "request_id": result.request_id,
        "input_tokens": result.input_tokens,
        "cached_input_tokens": result.cached_input_tokens,
        "cache_write_tokens": result.cache_write_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "total_tokens": result.total_tokens,
        "rate_limit_json": json.dumps(result.rate_limit, sort_keys=True),
        "provider_metadata_json": json.dumps(result.provider_metadata, sort_keys=True),
    }


def _request_key(
    *,
    phase: str,
    dataset: str,
    source_id: str,
    task_type: str,
    sample_idx: int,
    judge_id: str | None = None,
) -> str:
    return sha256_text(
        _canonical_json(
            {
                "phase": phase,
                "dataset": dataset,
                "source_id": source_id,
                "task_type": task_type,
                "sample_idx": sample_idx,
                "judge_id": judge_id,
            }
        )
    )


def _runner_metadata() -> dict[str, Any]:
    try:
        package_version = importlib.metadata.version("acuitybench-rebuild")
    except importlib.metadata.PackageNotFoundError:
        package_version = "source-tree"
    try:
        openai_version = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        openai_version = None
    return {
        "acuitybench_version": package_version,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": sys.platform,
        "openai_sdk_version": openai_version,
        "limiter": "asyncio.Semaphore",
        "clock": "time.perf_counter",
    }


async def _run_generation_task(
    task: GenerationTask,
    *,
    store: EvaluationStore,
    execution_id: str,
    model: ModelConfig,
    provider: Any,
    semaphore: asyncio.Semaphore,
    stream: bool,
) -> dict[str, Any]:
    context = AttemptContext(
        execution_id=execution_id,
        request_key=_request_key(
            phase="generation",
            dataset=task.dataset,
            source_id=task.source_id,
            task_type=task.task_type,
            sample_idx=task.sample_idx,
        ),
        dataset=task.dataset,
        source_id=task.source_id,
        task_type=task.task_type,
        sample_idx=task.sample_idx,
    )
    outcome = await _complete_with_retries(
        store=store,
        context=context,
        provider=provider,
        config=model,
        messages=_messages(task.task_type, task.prompt),
        semaphore=semaphore,
        stream=stream,
    )
    result = outcome.result
    response = result.text if result else None
    parsed_label = extract_label(response) if task.task_type == "qa" else None
    return {
        "run_id": task.run_id,
        "case_id": task.case_id,
        "dataset": task.dataset,
        "source_id": task.source_id,
        "task_type": task.task_type,
        "sample_idx": task.sample_idx,
        "normalized_label": task.normalized_label,
        "split": task.split,
        "mapping_method": task.mapping_method,
        "is_edge_case": int(task.is_edge_case),
        "prompt": task.prompt,
        "prompt_sha256": task.prompt_sha256,
        "model_config_sha256": model.fingerprint,
        "response": response,
        "response_sha256": sha256_text(response) if response is not None else None,
        "parsed_label": parsed_label,
        "parse_ok": int(parsed_label is not None) if task.task_type == "qa" else 1,
        "status": "ok" if result else "failed",
        "error": outcome.error,
        "attempts": outcome.attempts,
        # Compatibility alias: historical latency_ms also meant total logical
        # task residence time, including local queue and retry backoff.
        "latency_ms": outcome.timing.total_duration_ms,
        **_result_fields(result),
        **_parent_timing_fields(outcome.timing, execution_id),
    }


async def run_inference_async(
    *,
    store: EvaluationStore,
    run_id: str,
    tasks: list[GenerationTask],
    model: ModelConfig,
    concurrency: int,
    stream: bool = True,
) -> dict[str, int]:
    if concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    completed = store.successful_generation_keys(run_id)
    pending = [
        task
        for task in tasks
        if (task.dataset, task.source_id, task.task_type, task.sample_idx)
        not in completed
    ]
    if not pending:
        if len(completed) == store.get_run(run_id)["expected_generations"]:
            store.set_run_status(run_id, "generated")
        return {"pending": 0, "completed": len(completed), "failed": 0}
    load_dotenv(project_root() / ".env", override=False)
    provider = get_provider(model.provider)
    semaphore = asyncio.Semaphore(concurrency)
    execution_id = store.start_execution(
        run_id=run_id,
        phase="generation",
        profile_id=model.id,
        judge_id=None,
        provider=model.provider,
        api_model=model.api_model,
        endpoint=model.endpoint,
        config_sha256=model.fingerprint,
        concurrency=concurrency,
        streaming=stream,
        max_attempts=MAX_ATTEMPTS,
        retry_policy={
            "retryable_statuses": [408, 409, 429, "5xx"],
            "retry_after_headers": ["retry-after-ms", "retry-after"],
            "fallback": "exponential_jitter",
            "maximum_seconds": 60,
        },
        runner_metadata=_runner_metadata(),
        task_count=len(tasks),
        cache_hit_count=len(completed),
        pending_count=len(pending),
    )
    futures: list[asyncio.Task[dict[str, Any]]] = []
    store.set_run_status(run_id, "running")
    started = time.monotonic()
    successful = failed = 0
    execution_status = "failed"
    try:
        futures = [
            asyncio.create_task(
                _run_generation_task(
                    task,
                    store=store,
                    execution_id=execution_id,
                    model=model,
                    provider=provider,
                    semaphore=semaphore,
                    stream=stream,
                )
            )
            for task in pending
        ]
        for index, future in enumerate(asyncio.as_completed(futures), start=1):
            row = await future
            store.upsert_generation(row)
            if row["status"] == "ok":
                successful += 1
            else:
                failed += 1
            if index == 1 or index % 50 == 0 or index == len(futures):
                elapsed = max(time.monotonic() - started, 0.001)
                rate = index / elapsed
                remaining = (len(futures) - index) / rate if rate else 0
                print(
                    f"  generation {index:,}/{len(futures):,} "
                    f"({successful:,} ok, {failed:,} failed; ~{remaining/60:.1f} min left)",
                    flush=True,
                )
        execution_status = "complete" if failed == 0 else "incomplete"
    except asyncio.CancelledError:
        execution_status = "cancelled"
        raise
    finally:
        for future in futures:
            if not future.done():
                future.cancel()
        if futures:
            await asyncio.gather(*futures, return_exceptions=True)
        cancelled = sum(future.cancelled() for future in futures)
        unpersisted = max(
            len(futures) - successful - failed - cancelled,
            0,
        )
        try:
            await provider.close()
        finally:
            store.finish_execution(
                execution_id,
                status=execution_status,
                success_count=successful,
                failure_count=failed,
                cancelled_count=cancelled,
                unpersisted_count=unpersisted,
            )
            global_statuses = store.count_statuses("generations", run_id)
            expected_generations = store.get_run(run_id)["expected_generations"]
            globally_complete = (
                global_statuses.get("ok", 0) == expected_generations
            )
            store.set_run_status(
                run_id,
                {
                    "complete": (
                        "generated"
                        if globally_complete
                        else "generation_incomplete"
                    ),
                    "incomplete": "generation_incomplete",
                    "cancelled": "generation_cancelled",
                    "failed": "generation_failed",
                }[execution_status],
            )
    statuses = store.count_statuses("generations", run_id)
    run = store.get_run(run_id)
    if statuses.get("ok", 0) == run["expected_generations"]:
        store.set_run_status(run_id, "generated")
    elif failed:
        store.set_run_status(run_id, "generation_incomplete")
    return {"pending": len(pending), "completed": successful, "failed": failed}


def load_judge_assets() -> tuple[dict[str, Any], str]:
    root = project_root()
    rubric = yaml.safe_load(
        (root / "configs/rubric.yaml").read_text(encoding="utf-8")
    )["rubric"]
    template = (root / "configs/prompts/judge_rubric.txt").read_text(
        encoding="utf-8"
    ).strip()
    return rubric, template


def _format_conversation(prompt_json: str) -> str:
    roles = {"user": "[USER]", "assistant": "[ASSISTANT]"}
    messages = json.loads(prompt_json)
    return "\n\n".join(
        f"{roles.get(str(item.get('role', '')), '[' + str(item.get('role', 'unknown')).upper() + ']')}\n"
        f"{item.get('content', '')}"
        for item in messages
    )


def build_judge_prompt(
    generation: dict[str, Any], rubric: dict[str, Any], template: str
) -> str:
    replacements = {
        "{{CONVERSATION}}": _format_conversation(generation["prompt"]),
        "{{RESPONSE}}": str(generation["response"] or ""),
        "{{RUBRIC_A}}": rubric["A"]["criterion"].strip(),
        "{{RUBRIC_B}}": rubric["B"]["criterion"].strip(),
        "{{RUBRIC_C}}": rubric["C"]["criterion"].strip(),
        "{{RUBRIC_D}}": rubric["D"]["criterion"].strip(),
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    if "{{" in prompt:
        raise ValueError("Unresolved placeholder in judge prompt")
    return prompt


async def _run_judge_task(
    task: JudgeTask,
    *,
    store: EvaluationStore,
    execution_id: str,
    run_id: str,
    judge: JudgeConfig,
    provider: Any,
    semaphore: asyncio.Semaphore,
    stream: bool,
) -> dict[str, Any]:
    generation = task.generation
    context = AttemptContext(
        execution_id=execution_id,
        request_key=_request_key(
            phase="judge",
            dataset=generation["dataset"],
            source_id=generation["source_id"],
            task_type="conv",
            sample_idx=generation["sample_idx"],
            judge_id=judge.id,
        ),
        dataset=generation["dataset"],
        source_id=generation["source_id"],
        task_type="conv",
        sample_idx=generation["sample_idx"],
        judge_id=judge.id,
    )
    outcome = await _complete_with_retries(
        store=store,
        context=context,
        provider=provider,
        config=judge.model,
        messages=[{"role": "user", "content": task.prompt}],
        semaphore=semaphore,
        stream=stream,
    )
    result = outcome.result
    response = result.text if result else None
    label = extract_label(response)
    return {
        "run_id": run_id,
        "case_id": generation["case_id"],
        "dataset": generation["dataset"],
        "source_id": generation["source_id"],
        "sample_idx": generation["sample_idx"],
        "judge_id": judge.id,
        "judge_config_sha256": judge.fingerprint,
        "generation_response_sha256": generation["response_sha256"],
        "judge_prompt": task.prompt,
        "judge_prompt_sha256": task.prompt_sha256,
        "response": response,
        "response_sha256": sha256_text(response) if response is not None else None,
        "judge_label": label,
        "judge_reasoning": extract_reasoning(response),
        "parse_ok": int(label is not None),
        "status": "ok" if result else "failed",
        "error": outcome.error,
        "attempts": outcome.attempts,
        "latency_ms": outcome.timing.total_duration_ms,
        **_result_fields(result),
        **_parent_timing_fields(outcome.timing, execution_id),
    }


async def run_judge_async(
    *,
    store: EvaluationStore,
    run_id: str,
    judge: JudgeConfig,
    concurrency: int,
    stream: bool = True,
) -> dict[str, int]:
    if concurrency < 1:
        raise ValueError("--judge-concurrency must be at least 1")
    rubric, template = load_judge_assets()
    generations = store.completed_conv_generations(run_id)
    existing = store.judgment_records(run_id, judge.id)
    pending: list[JudgeTask] = []
    for generation in generations:
        prompt = build_judge_prompt(generation, rubric, template)
        prompt_hash = sha256_text(prompt)
        key = (generation["dataset"], generation["source_id"], generation["sample_idx"])
        cached = existing.get(key)
        if (
            cached
            and cached["status"] == "ok"
            and cached["judge_config_sha256"] == judge.fingerprint
            and cached["generation_response_sha256"] == generation["response_sha256"]
            and cached["judge_prompt_sha256"] == prompt_hash
        ):
            continue
        pending.append(JudgeTask(generation, prompt, prompt_hash))
    if not pending:
        run = store.get_run(run_id)
        expected = (
            run["selected_cases"] * run["samples"]
            if "conv" in run["tasks"]
            else 0
        )
        generations_complete = (
            store.count_statuses("generations", run_id).get("ok", 0)
            == run["expected_generations"]
        )
        if (
            generations_complete
            and sum(1 for row in existing.values() if row["status"] == "ok")
            == expected
        ):
            store.set_run_status(run_id, "complete")
        else:
            store.set_run_status(run_id, "judge_incomplete")
        return {"pending": 0, "completed": len(existing), "failed": 0}
    load_dotenv(project_root() / ".env", override=False)
    provider = get_provider(judge.model.provider)
    semaphore = asyncio.Semaphore(concurrency)
    execution_id = store.start_execution(
        run_id=run_id,
        phase="judge",
        profile_id=judge.id,
        judge_id=judge.id,
        provider=judge.model.provider,
        api_model=judge.model.api_model,
        endpoint=judge.model.endpoint,
        config_sha256=judge.fingerprint,
        concurrency=concurrency,
        streaming=stream,
        max_attempts=MAX_ATTEMPTS,
        retry_policy={
            "retryable_statuses": [408, 409, 429, "5xx"],
            "retry_after_headers": ["retry-after-ms", "retry-after"],
            "fallback": "exponential_jitter",
            "maximum_seconds": 60,
        },
        runner_metadata=_runner_metadata(),
        task_count=len(generations),
        cache_hit_count=len(generations) - len(pending),
        pending_count=len(pending),
    )
    futures: list[asyncio.Task[dict[str, Any]]] = []
    store.set_run_status(run_id, "judging")
    started = time.monotonic()
    successful = failed = 0
    execution_status = "failed"
    try:
        futures = [
            asyncio.create_task(
                _run_judge_task(
                    task,
                    store=store,
                    execution_id=execution_id,
                    run_id=run_id,
                    judge=judge,
                    provider=provider,
                    semaphore=semaphore,
                    stream=stream,
                )
            )
            for task in pending
        ]
        for index, future in enumerate(asyncio.as_completed(futures), start=1):
            row = await future
            store.upsert_judgment(row)
            if row["status"] == "ok":
                successful += 1
            else:
                failed += 1
            if index == 1 or index % 50 == 0 or index == len(futures):
                elapsed = max(time.monotonic() - started, 0.001)
                rate = index / elapsed
                remaining = (len(futures) - index) / rate if rate else 0
                print(
                    f"  judge {index:,}/{len(futures):,} "
                    f"({successful:,} ok, {failed:,} failed; ~{remaining/60:.1f} min left)",
                    flush=True,
                )
        execution_status = "complete" if failed == 0 else "incomplete"
    except asyncio.CancelledError:
        execution_status = "cancelled"
        raise
    finally:
        for future in futures:
            if not future.done():
                future.cancel()
        if futures:
            await asyncio.gather(*futures, return_exceptions=True)
        cancelled = sum(future.cancelled() for future in futures)
        unpersisted = max(
            len(futures) - successful - failed - cancelled,
            0,
        )
        try:
            await provider.close()
        finally:
            store.finish_execution(
                execution_id,
                status=execution_status,
                success_count=successful,
                failure_count=failed,
                cancelled_count=cancelled,
                unpersisted_count=unpersisted,
            )
            run = store.get_run(run_id)
            expected_judgments = (
                run["selected_cases"] * run["samples"]
                if "conv" in run["tasks"]
                else 0
            )
            judgment_rows = store.judgment_records(run_id, judge.id)
            judgments_complete = (
                sum(
                    1
                    for row in judgment_rows.values()
                    if row["status"] == "ok"
                )
                == expected_judgments
            )
            generations_complete = (
                store.count_statuses("generations", run_id).get("ok", 0)
                == run["expected_generations"]
            )
            store.set_run_status(
                run_id,
                {
                    "complete": (
                        "complete"
                        if generations_complete and judgments_complete
                        else "judge_incomplete"
                    ),
                    "incomplete": "judge_incomplete",
                    "cancelled": "judge_cancelled",
                    "failed": "judge_failed",
                }[execution_status],
            )
    run = store.get_run(run_id)
    expected_judgments = (
        run["selected_cases"] * run["samples"]
        if "conv" in run["tasks"]
        else 0
    )
    judgment_rows = store.judgment_records(run_id, judge.id)
    generations_complete = (
        store.count_statuses("generations", run_id).get("ok", 0)
        == run["expected_generations"]
    )
    if (
        generations_complete
        and sum(
            1 for row in judgment_rows.values() if row["status"] == "ok"
        )
        == expected_judgments
    ):
        store.set_run_status(run_id, "complete")
    else:
        store.set_run_status(run_id, "judge_incomplete")
    return {"pending": len(pending), "completed": successful, "failed": failed}


def default_store_path() -> Path:
    return project_root() / "results/evaluations.sqlite3"


def default_benchmark_path() -> Path:
    return project_root() / "data/processed/acuitybench_transformed.csv"


def run_evaluation(
    *,
    model_id: str,
    tasks: tuple[str, ...] = TASK_TYPES,
    samples: int = 5,
    datasets: tuple[str, ...] | None = None,
    limit: int | None = None,
    run_id: str | None = None,
    concurrency: int = 20,
    judge_id: str = "paper-gpt-4.1",
    judge_concurrency: int = 20,
    stream: bool = True,
    include_judge: bool = True,
    store_path: Path | None = None,
    benchmark_path: Path | None = None,
) -> str:
    registry = ModelRegistry()
    model = registry.get(model_id)
    with EvaluationStore(store_path or default_store_path()) as store:
        actual_run_id, generation_tasks = prepare_run(
            store=store,
            model=model,
            benchmark_path=benchmark_path or default_benchmark_path(),
            tasks=tasks,
            samples=samples,
            datasets=datasets,
            limit=limit,
            run_id=run_id,
            stream=stream,
        )
        print(
            f"Run {actual_run_id}: {len(generation_tasks):,} expected target calls "
            f"({len(set(task.case_id for task in generation_tasks)):,} cases, "
            f"{samples} samples, {', '.join(tasks)})"
        )
        result = asyncio.run(
            run_inference_async(
                store=store,
                run_id=actual_run_id,
                tasks=generation_tasks,
                model=model,
                concurrency=concurrency,
                stream=stream,
            )
        )
        if result["pending"] == 0:
            print("  generation cache is complete")
        if result["failed"]:
            raise RuntimeError(
                f"{result['failed']} generation calls failed; rerun the same command to retry"
            )
        if include_judge and "conv" in tasks:
            judge = registry.get_judge(judge_id)
            judge_result = asyncio.run(
                run_judge_async(
                    store=store,
                    run_id=actual_run_id,
                    judge=judge,
                    concurrency=judge_concurrency,
                    stream=stream,
                )
            )
            if judge_result["pending"] == 0:
                print("  judge cache is complete")
            if judge_result["failed"]:
                raise RuntimeError(
                    f"{judge_result['failed']} judge calls failed; rerun to retry"
                )
        return actual_run_id
