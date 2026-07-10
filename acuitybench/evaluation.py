"""Resumable, configuration-driven AcuityBench inference and judging."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

from acuitybench.models import JudgeConfig, ModelConfig, ModelRegistry
from acuitybench.providers import CompletionResult, get_provider
from acuitybench.sources import project_root, sha256_file
from acuitybench.store import EvaluationStore


ACUITY_RE = re.compile(r"ACUITY\s*[:\-]\s*([A-D])", re.IGNORECASE)
REASONING_RE = re.compile(
    r"REASONING\s*[:\-]\s*(.*?)(?=\nACUITY|$)", re.IGNORECASE | re.DOTALL
)
TASK_TYPES = ("qa", "conv")


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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_label(response: str | None) -> str | None:
    if not response:
        return None
    match = ACUITY_RE.search(response)
    return match.group(1).upper() if match else None


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


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: BaseException) -> bool:
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
    provider: Any,
    config: ModelConfig,
    messages: list[dict[str, str]],
    semaphore: asyncio.Semaphore,
    max_retries: int = 6,
) -> tuple[CompletionResult | None, int, str | None, float]:
    started = time.perf_counter()
    max_output_tokens = config.max_output_tokens
    last_error: str | None = None
    attempts = 0
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            async with semaphore:
                result = await provider.complete(
                    config=config,
                    messages=messages,
                    max_output_tokens=max_output_tokens,
                )
            if (
                result.finish_reason == "length"
                and not result.text
                and max_output_tokens < 8192
            ):
                max_output_tokens = min(max_output_tokens * 2, 8192)
                continue
            latency_ms = (time.perf_counter() - started) * 1000
            return result, attempt, None, latency_ms
        except Exception as exc:  # provider exceptions intentionally normalized here
            last_error = _redact_error(exc, config.api_key_env)
            if not _is_retryable(exc) or attempt == max_retries:
                break
            delay = _retry_after(exc)
            if delay is None:
                delay = min(60.0, 2.0 ** (attempt - 1)) + random.random()
            await asyncio.sleep(delay)
    latency_ms = (time.perf_counter() - started) * 1000
    return None, attempts, last_error, latency_ms


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


async def _run_generation_task(
    task: GenerationTask,
    *,
    model: ModelConfig,
    provider: Any,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    result, attempts, error, latency_ms = await _complete_with_retries(
        provider=provider,
        config=model,
        messages=_messages(task.task_type, task.prompt),
        semaphore=semaphore,
    )
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
        "error": error,
        "attempts": attempts,
        "latency_ms": latency_ms,
        **_result_fields(result),
    }


async def run_inference_async(
    *,
    store: EvaluationStore,
    run_id: str,
    tasks: list[GenerationTask],
    model: ModelConfig,
    concurrency: int,
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
    futures = [
        asyncio.create_task(
            _run_generation_task(
                task, model=model, provider=provider, semaphore=semaphore
            )
        )
        for task in pending
    ]
    store.set_run_status(run_id, "running")
    started = time.monotonic()
    successful = failed = 0
    try:
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
    finally:
        for future in futures:
            if not future.done():
                future.cancel()
        await provider.close()
    statuses = store.count_statuses("generations", run_id)
    run = store.get_run(run_id)
    if statuses.get("ok", 0) == run["expected_generations"]:
        store.set_run_status(run_id, "generated")
    elif failed:
        store.set_run_status(run_id, "generation_incomplete")
    return {"pending": len(pending), "completed": successful, "failed": failed}


def _load_judge_assets() -> tuple[dict[str, Any], str]:
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


def _judge_prompt(generation: dict[str, Any], rubric: dict[str, Any], template: str) -> str:
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
    run_id: str,
    judge: JudgeConfig,
    provider: Any,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    result, attempts, error, latency_ms = await _complete_with_retries(
        provider=provider,
        config=judge.model,
        messages=[{"role": "user", "content": task.prompt}],
        semaphore=semaphore,
    )
    response = result.text if result else None
    label = extract_label(response)
    generation = task.generation
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
        "error": error,
        "attempts": attempts,
        "latency_ms": latency_ms,
        **_result_fields(result),
    }


async def run_judge_async(
    *,
    store: EvaluationStore,
    run_id: str,
    judge: JudgeConfig,
    concurrency: int,
) -> dict[str, int]:
    if concurrency < 1:
        raise ValueError("--judge-concurrency must be at least 1")
    rubric, template = _load_judge_assets()
    generations = store.completed_conv_generations(run_id)
    existing = store.judgment_records(run_id, judge.id)
    pending: list[JudgeTask] = []
    for generation in generations:
        prompt = _judge_prompt(generation, rubric, template)
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
        if sum(1 for row in existing.values() if row["status"] == "ok") == expected:
            store.set_run_status(run_id, "complete")
        return {"pending": 0, "completed": len(existing), "failed": 0}
    load_dotenv(project_root() / ".env", override=False)
    provider = get_provider(judge.model.provider)
    semaphore = asyncio.Semaphore(concurrency)
    futures = [
        asyncio.create_task(
            _run_judge_task(
                task,
                run_id=run_id,
                judge=judge,
                provider=provider,
                semaphore=semaphore,
            )
        )
        for task in pending
    ]
    store.set_run_status(run_id, "judging")
    started = time.monotonic()
    successful = failed = 0
    try:
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
    finally:
        for future in futures:
            if not future.done():
                future.cancel()
        await provider.close()
    expected_judgments = sum(
        1 for task in store.get_run(run_id)["tasks"] if task == "conv"
    ) * store.get_run(run_id)["selected_cases"] * store.get_run(run_id)["samples"]
    statuses = store.count_statuses("judgments", run_id)
    if statuses.get("ok", 0) == expected_judgments:
        store.set_run_status(run_id, "complete")
    elif failed:
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
                )
            )
            if judge_result["pending"] == 0:
                print("  judge cache is complete")
            if judge_result["failed"]:
                raise RuntimeError(
                    f"{judge_result['failed']} judge calls failed; rerun to retry"
                )
        return actual_run_id
