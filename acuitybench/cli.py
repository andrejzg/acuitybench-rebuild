from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

from acuitybench import __version__
from acuitybench.evaluation import (
    TASK_TYPES,
    default_benchmark_path,
    default_store_path,
    run_evaluation,
    run_judge_async,
)
from acuitybench.models import ModelRegistry
from acuitybench.providers import get_provider
from acuitybench.interactive.costing import write_cost_report
from acuitybench.interactive.seed import (
    build_seed_set,
    default_seed_output_dir,
    load_case_cards,
    validate_seed_set,
)
from acuitybench.interactive.simulator import run_action_trace
from acuitybench.pipeline import build
from acuitybench.reporting import combine_reports, generate_report
from acuitybench.sources import (
    fetch_sources,
    project_root,
    sha256_file,
    source_report,
)
from acuitybench.validation import read_reference_ids, validate_frame
from acuitybench.store import EvaluationStore
from acuitybench.static_student import (
    inspect_static_plan,
    load_static_plan,
    static_evaluation_contract,
    validate_static_examples,
)
from acuitybench.synthetic import (
    generate_synthetic_cases,
    initialize_synthetic_pilot,
    inspect_synthetic_plan,
    label_synthetic_cases,
    validate_synthetic_pilot,
)


def _data_dir(value: str | None) -> Path:
    if value is None:
        return project_root() / "data"
    return Path(value).expanduser().resolve()


def _add_source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        help="Cache and output directory (default: <project>/data)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--refresh",
        action="store_true",
        help="Download every source again even if its cache is valid",
    )
    mode.add_argument(
        "--offline",
        action="store_true",
        help="Use only checksum-verified files already in the local cache",
    )


def _write_report(
    *,
    data_dir: Path,
    sources: dict[str, Path],
    raw_counts: dict[str, int],
    annotation_audit: dict[str, object],
    validation: dict[str, object],
    output_paths: list[Path],
) -> Path:
    report = {
        "builder_version": __version__,
        "raw_counts": raw_counts,
        "annotation_audit": annotation_audit,
        "validation": validation,
        "sources": source_report(sources),
        "outputs": [
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_paths
        ],
    }
    destination = data_dir / "processed" / "build_report.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def _run_fetch(args: argparse.Namespace) -> int:
    data_dir = _data_dir(args.data_dir)
    paths = fetch_sources(
        data_dir, refresh=args.refresh, offline=args.offline
    )
    total = sum(path.stat().st_size for path in paths.values())
    print(f"\nVerified {len(paths)} source files ({total / 1024 / 1024:.1f} MiB).")
    print(f"Cache: {data_dir / 'cache' / 'sources'}")
    return 0


def _run_build(args: argparse.Namespace) -> int:
    root = project_root()
    data_dir = _data_dir(args.data_dir)
    paths = fetch_sources(
        data_dir, root=root, refresh=args.refresh, offline=args.offline
    )
    print("\n[build] reconstructing benchmark")
    result = build(paths, root=root, output_dir=data_dir / "processed")
    references = read_reference_ids(paths["reference_case_ids"])
    validation = validate_frame(result.transformed, references)
    report_path = _write_report(
        data_dir=data_dir,
        sources=paths,
        raw_counts=result.raw_counts,
        annotation_audit=result.annotation_audit,
        validation=validation,
        output_paths=[
            result.normalized_csv,
            result.transformed_csv,
            result.parquet,
        ],
    )

    print("\n[validate] all published invariants passed")
    print(
        "  998 raw cases -> 914 final "
        "(697 primary, 217 ambiguous; 667 physician-panel cases)"
    )
    print("\nOutputs:")
    for path in (
        result.normalized_csv,
        result.transformed_csv,
        result.parquet,
        report_path,
    ):
        print(f"  {path}")
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    data_dir = _data_dir(args.data_dir)
    paths = fetch_sources(data_dir, offline=True)
    transformed_path = data_dir / "processed" / "acuitybench_transformed.csv"
    if not transformed_path.exists():
        raise FileNotFoundError(
            f"Built benchmark not found: {transformed_path}. Run the build command first."
        )
    frame = pd.read_csv(transformed_path)
    references = read_reference_ids(paths["reference_case_ids"])
    validation = validate_frame(frame, references)
    print(json.dumps(validation, indent=2, sort_keys=True))
    print("\nValidation passed.")
    return 0


def _path(value: str | None, default: Path) -> Path:
    return Path(value).expanduser().resolve() if value else default


def _add_evaluation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Model ID from configs/models.yaml")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_TYPES,
        default=list(TASK_TYPES),
        help="Evaluation formats (default: qa conv)",
    )
    parser.add_argument("--samples", type=int, default=5, help="Calls per case/task")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional dataset subset (default: every dataset)",
    )
    parser.add_argument("--limit", type=int, help="Deterministic case limit for smoke runs")
    parser.add_argument("--run-id", help="Explicit resumable run ID")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Disable streaming (TTFT will be unavailable)",
    )
    parser.add_argument(
        "--judge", default="paper-gpt-4.1", help="Judge ID from configs/models.yaml"
    )
    parser.add_argument("--judge-concurrency", type=int, default=20)
    parser.add_argument("--store", help="SQLite result store")
    parser.add_argument("--benchmark", help="Transformed benchmark CSV")


def _run_models(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    print("Models:")
    for model in registry.models():
        temperature = model.temperature if model.send_temperature else "provider default"
        reasoning = model.reasoning_effort or "not applicable/provider default"
        endpoint_detail = (
            f", base_url_env={model.base_url_env}, deployment={model.deployment}"
            if model.base_url_env
            else ""
        )
        print(
            f"  {model.id:16} {model.provider:8} {model.api_model:24} "
            f"{model.endpoint} (temperature={temperature}, reasoning={reasoning}, "
            f"service_tier={model.service_tier or 'provider default'}{endpoint_detail})"
        )
    print("\nJudges:")
    for judge in registry.judges():
        print(
            f"  {judge.id:16} {judge.model.provider:8} {judge.model.api_model:24} "
            f"temperature={judge.model.temperature}"
        )
    return 0


def _run_runs(args: argparse.Namespace) -> int:
    with EvaluationStore(_path(args.store, default_store_path())) as store:
        runs = store.list_runs()
    if not runs:
        print("No evaluation runs yet.")
        return 0
    for run in runs:
        print(
            f"{run['run_id']}  {run['status']:22} {run['model_id']:14} "
            f"{run['selected_cases']} cases / {run['expected_generations']} calls"
        )
    return 0


def _evaluation_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_id": args.model,
        "tasks": tuple(dict.fromkeys(args.tasks)),
        "samples": args.samples,
        "datasets": tuple(args.datasets) if args.datasets else None,
        "limit": args.limit,
        "run_id": args.run_id,
        "concurrency": args.concurrency,
        "stream": args.stream,
        "judge_id": args.judge,
        "judge_concurrency": args.judge_concurrency,
        "store_path": _path(args.store, default_store_path()),
        "benchmark_path": _path(args.benchmark, default_benchmark_path()),
    }


def _run_infer(args: argparse.Namespace) -> int:
    kwargs = _evaluation_kwargs(args)
    kwargs["include_judge"] = False
    run_id = run_evaluation(**kwargs)
    print(f"\nGeneration run ready: {run_id}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    kwargs = _evaluation_kwargs(args)
    run_id = run_evaluation(**kwargs)
    output = generate_report(
        run_id=run_id,
        store_path=kwargs["store_path"],
        judge_id=args.judge,
    )
    print(f"\nEvaluation complete: {run_id}")
    print(f"Report: {output}")
    return 0


def _run_judge(args: argparse.Namespace) -> int:
    registry = ModelRegistry()
    judge = registry.get_judge(args.judge)
    with EvaluationStore(_path(args.store, default_store_path())) as store:
        result = asyncio.run(
            run_judge_async(
                store=store,
                run_id=args.run_id,
                judge=judge,
                concurrency=args.concurrency,
                stream=args.stream,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["failed"]:
        raise RuntimeError("Some judge calls failed; rerun this command to retry")
    return 0


def _run_report(args: argparse.Namespace) -> int:
    output = generate_report(
        run_id=args.run_id,
        store_path=_path(args.store, default_store_path()),
        output_root=Path(args.output_root).expanduser().resolve() if args.output_root else None,
        judge_id=args.judge,
        allow_incomplete=args.allow_incomplete,
    )
    print(f"Report: {output}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    output = combine_reports(
        run_ids=args.run_ids,
        results_root=(
            Path(args.results_root).expanduser().resolve()
            if args.results_root
            else None
        ),
        destination=(
            Path(args.output).expanduser().resolve() if args.output else None
        ),
    )
    print(f"Combined table: {output}")
    return 0


def _run_static_plan(args: argparse.Namespace) -> int:
    report = inspect_static_plan(
        plan_path=Path(args.config).expanduser().resolve() if args.config else None,
        benchmark_path=(
            Path(args.benchmark).expanduser().resolve() if args.benchmark else None
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _run_static_data_validate(args: argparse.Namespace) -> int:
    report = validate_static_examples(
        Path(args.input).expanduser().resolve(),
        benchmark_path=(
            Path(args.benchmark).expanduser().resolve() if args.benchmark else None
        ),
        schema_path=(
            Path(args.schema).expanduser().resolve() if args.schema else None
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _add_static_evaluation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Model ID from configs/models.yaml")
    parser.add_argument("--config", help="Static-student plan YAML")
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help="Skip the secondary one-shot conversational task and GPT-4.1 judge",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Calls per case/task (default: plan value, currently 5)",
    )
    parser.add_argument("--datasets", nargs="+", help="Optional dataset subset")
    parser.add_argument("--limit", type=int, help="Deterministic smoke-run case limit")
    parser.add_argument("--run-id", help="Explicit resumable run ID")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Disable streaming (TTFT will be unavailable)",
    )
    parser.add_argument("--judge", default="paper-gpt-4.1")
    parser.add_argument("--judge-concurrency", type=int, default=20)
    parser.add_argument("--store", help="SQLite result store")
    parser.add_argument("--benchmark", help="Transformed held-out benchmark CSV")


def _run_static_evaluate(args: argparse.Namespace) -> int:
    plan_path = Path(args.config).expanduser().resolve() if args.config else None
    benchmark_path = _path(args.benchmark, default_benchmark_path())
    inspect_static_plan(plan_path=plan_path, benchmark_path=benchmark_path)
    plan = load_static_plan(plan_path)
    include_conversation = not args.qa_only
    tasks = ("qa", "conv") if include_conversation else ("qa",)
    samples = (
        int(plan["evaluation"]["samples_per_case"])
        if args.samples is None
        else args.samples
    )
    store_path = _path(args.store, default_store_path())
    run_id = run_evaluation(
        model_id=args.model,
        tasks=tasks,
        samples=samples,
        datasets=tuple(args.datasets) if args.datasets else None,
        limit=args.limit,
        run_id=args.run_id,
        concurrency=args.concurrency,
        judge_id=args.judge,
        judge_concurrency=args.judge_concurrency,
        stream=args.stream,
        include_judge=include_conversation,
        store_path=store_path,
        benchmark_path=benchmark_path,
        experiment_contract=static_evaluation_contract(
            plan, include_conversation=include_conversation
        ),
    )
    output = generate_report(
        run_id=run_id,
        store_path=store_path,
        judge_id=args.judge,
    )
    print(f"\nStatic student evaluation complete: {run_id}")
    print(f"Report: {output}")
    return 0


def _run_synthetic_plan(args: argparse.Namespace) -> int:
    report = inspect_synthetic_plan(
        Path(args.config).expanduser().resolve() if args.config else None
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _run_synthetic_init(args: argparse.Namespace) -> int:
    paths = initialize_synthetic_pilot(
        config_path=(
            Path(args.config).expanduser().resolve() if args.config else None
        ),
        output_dir=(
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else None
        ),
    )
    report = validate_synthetic_pilot(
        config_path=(
            Path(args.config).expanduser().resolve() if args.config else None
        ),
        output_dir=paths.output_dir,
        allow_incomplete=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nRequests: {paths.generation_requests}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _require_paid_synthetic_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_spend:
        raise ValueError(
            "Paid synthetic calls require --confirm-spend after reviewing the "
            "synthetic-plan call count and expected cost"
        )
    if not args.terms_reviewed:
        raise ValueError(
            "Paid synthetic calls require --terms-reviewed after reviewing "
            "provider data-handling and output-use terms"
        )


async def _run_synthetic_provider_phase(
    args: argparse.Namespace, *, phase: str
) -> dict[str, object]:
    registry = ModelRegistry()
    model = registry.get(args.model)
    provider = get_provider(model.provider)
    kwargs = {
        "provider": provider,
        "model": model,
        "config_path": (
            Path(args.config).expanduser().resolve() if args.config else None
        ),
        "output_dir": (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else None
        ),
    }
    try:
        if phase == "generate":
            return await generate_synthetic_cases(**kwargs)
        if phase == "label":
            return await label_synthetic_cases(**kwargs)
        raise ValueError(f"Unknown synthetic phase: {phase}")
    finally:
        await provider.close()


def _run_synthetic_generate(args: argparse.Namespace) -> int:
    _require_paid_synthetic_confirmation(args)
    result = asyncio.run(_run_synthetic_provider_phase(args, phase="generate"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_synthetic_label(args: argparse.Namespace) -> int:
    _require_paid_synthetic_confirmation(args)
    result = asyncio.run(_run_synthetic_provider_phase(args, phase="label"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_synthetic_validate(args: argparse.Namespace) -> int:
    result = validate_synthetic_pilot(
        config_path=(
            Path(args.config).expanduser().resolve() if args.config else None
        ),
        output_dir=(
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else None
        ),
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_interactive_build(args: argparse.Namespace) -> int:
    result = build_seed_set(
        config_path=Path(args.config).expanduser().resolve() if args.config else None,
        benchmark_path=(
            Path(args.benchmark).expanduser().resolve() if args.benchmark else None
        ),
        output_dir=(
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else default_seed_output_dir()
        ),
    )
    validation = validate_seed_set(
        case_cards_path=result.case_cards_path,
        manifest_path=result.manifest_path,
        benchmark_path=(
            Path(args.benchmark).expanduser().resolve() if args.benchmark else None
        ),
        config_path=Path(args.config).expanduser().resolve() if args.config else None,
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    print(f"\nCase cards: {result.case_cards_path}")
    print(f"Manifest:   {result.manifest_path}")
    return 0


def _run_interactive_validate(args: argparse.Namespace) -> int:
    seed_dir = (
        Path(args.seed_dir).expanduser().resolve()
        if args.seed_dir
        else default_seed_output_dir()
    )
    result = validate_seed_set(
        case_cards_path=seed_dir / "case_cards.jsonl",
        manifest_path=seed_dir / "manifest.json",
        benchmark_path=(
            Path(args.benchmark).expanduser().resolve() if args.benchmark else None
        ),
        config_path=Path(args.config).expanduser().resolve() if args.config else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _read_actions(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("actions"), list):
            payload = payload["actions"]
        if not isinstance(payload, list) or not all(
            isinstance(value, dict) for value in payload
        ):
            raise ValueError("Action JSON must be a list or an object with an actions list")
        return [dict(value) for value in payload]
    actions: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Action at line {line_number} is not an object")
        actions.append(dict(value))
    return actions


def _run_interactive_simulate(args: argparse.Namespace) -> int:
    seed_dir = (
        Path(args.seed_dir).expanduser().resolve()
        if args.seed_dir
        else default_seed_output_dir()
    )
    cards = {
        str(card["case_id"]): card
        for card in load_case_cards(seed_dir / "case_cards.jsonl")
    }
    if args.case_id not in cards:
        raise ValueError(f"Unknown interactive case ID: {args.case_id}")
    actions = _read_actions(Path(args.actions).expanduser().resolve())
    evaluation = run_action_trace(cards[args.case_id], actions)
    payload = {
        "case_id": evaluation.case_id,
        "terminal_action": evaluation.terminal_action,
        "outcome": evaluation.outcome,
        "transcript": list(evaluation.transcript),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _run_interactive_cost(args: argparse.Namespace) -> int:
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else project_root() / "results/interactive-pilot-v1"
    )
    paths = write_cost_report(
        output_dir,
        assumptions_path=(
            Path(args.assumptions).expanduser().resolve()
            if args.assumptions
            else None
        ),
    )
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    print(json.dumps(report["totals"], indent=2, sort_keys=True))
    print(f"\nJSON:     {paths[0]}")
    print(f"Markdown: {paths[1]}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acuitybench",
        description="Checksum-pinned AcuityBench reconstruction",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="Fetch, reconstruct, transform, and validate AcuityBench"
    )
    _add_source_options(build_parser)
    build_parser.set_defaults(handler=_run_build)

    fetch_parser = subparsers.add_parser(
        "fetch", help="Download and checksum all pinned source files"
    )
    _add_source_options(fetch_parser)
    fetch_parser.set_defaults(handler=_run_fetch)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate an existing processed benchmark"
    )
    validate_parser.add_argument(
        "--data-dir",
        help="Cache and output directory (default: <project>/data)",
    )
    validate_parser.set_defaults(handler=_run_validate)

    models_parser = subparsers.add_parser(
        "models", help="List configured target models and judges"
    )
    models_parser.set_defaults(handler=_run_models)

    runs_parser = subparsers.add_parser("runs", help="List resumable evaluation runs")
    runs_parser.add_argument("--store", help="SQLite result store")
    runs_parser.set_defaults(handler=_run_runs)

    infer_parser = subparsers.add_parser(
        "infer", help="Run or resume target-model generations without judging"
    )
    _add_evaluation_options(infer_parser)
    infer_parser.set_defaults(handler=_run_infer)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run/resume generation, judge conversation samples, and report"
    )
    _add_evaluation_options(evaluate_parser)
    evaluate_parser.set_defaults(handler=_run_evaluate)

    judge_parser = subparsers.add_parser(
        "judge", help="Run or resume the conversational rubric judge for a run"
    )
    judge_parser.add_argument("--run-id", required=True)
    judge_parser.add_argument("--judge", default="paper-gpt-4.1")
    judge_parser.add_argument("--concurrency", type=int, default=20)
    judge_parser.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Disable streaming (TTFT will be unavailable)",
    )
    judge_parser.add_argument("--store", help="SQLite result store")
    judge_parser.set_defaults(handler=_run_judge)

    report_parser = subparsers.add_parser(
        "report", help="Export raw results, metrics, and paper-style tables"
    )
    report_parser.add_argument("--run-id", required=True)
    report_parser.add_argument("--judge", default="paper-gpt-4.1")
    report_parser.add_argument("--store", help="SQLite result store")
    report_parser.add_argument("--output-root", help="Report directory root")
    report_parser.add_argument("--allow-incomplete", action="store_true")
    report_parser.set_defaults(handler=_run_report)

    compare_parser = subparsers.add_parser(
        "compare", help="Combine completed model reports into one table"
    )
    compare_parser.add_argument("--run-ids", nargs="+", required=True)
    compare_parser.add_argument("--results-root")
    compare_parser.add_argument("--output")
    compare_parser.set_defaults(handler=_run_compare)

    static_plan_parser = subparsers.add_parser(
        "static-plan",
        help="Validate and summarize the versioned static-first student plan",
    )
    static_plan_parser.add_argument("--config")
    static_plan_parser.add_argument("--benchmark")
    static_plan_parser.set_defaults(handler=_run_static_plan)

    static_data_parser = subparsers.add_parser(
        "static-data-validate",
        help="Validate a separate static training/development JSONL pool",
    )
    static_data_parser.add_argument("--input", required=True)
    static_data_parser.add_argument("--schema")
    static_data_parser.add_argument("--benchmark")
    static_data_parser.set_defaults(handler=_run_static_data_validate)

    static_evaluate_parser = subparsers.add_parser(
        "static-evaluate",
        help="Run the static AcuityBench-style student contract",
    )
    _add_static_evaluation_options(static_evaluate_parser)
    static_evaluate_parser.set_defaults(handler=_run_static_evaluate)

    synthetic_plan_parser = subparsers.add_parser(
        "synthetic-plan",
        help="Inspect the free, fictional 20-case pilot plan and paid call count",
    )
    synthetic_plan_parser.add_argument("--config")
    synthetic_plan_parser.set_defaults(handler=_run_synthetic_plan)

    synthetic_init_parser = subparsers.add_parser(
        "synthetic-init",
        help="Write the deterministic fictional-pilot scaffold without API calls",
    )
    synthetic_init_parser.add_argument("--config")
    synthetic_init_parser.add_argument("--output-dir")
    synthetic_init_parser.set_defaults(handler=_run_synthetic_init)

    for command, help_text, handler in (
        (
            "synthetic-generate",
            "Run or resume paid fictional-vignette generation",
            _run_synthetic_generate,
        ),
        (
            "synthetic-label",
            "Run or resume paid blinded labeling and machine screening",
            _run_synthetic_label,
        ),
    ):
        synthetic_paid_parser = subparsers.add_parser(command, help=help_text)
        synthetic_paid_parser.add_argument(
            "--model", required=True, help="Model ID from configs/models.yaml"
        )
        synthetic_paid_parser.add_argument("--config")
        synthetic_paid_parser.add_argument("--output-dir")
        synthetic_paid_parser.add_argument(
            "--confirm-spend",
            action="store_true",
            help="Confirm the paid call count and estimated cost were reviewed",
        )
        synthetic_paid_parser.add_argument(
            "--terms-reviewed",
            action="store_true",
            help="Confirm provider data-handling/output terms were reviewed",
        )
        synthetic_paid_parser.set_defaults(handler=handler)

    synthetic_validate_parser = subparsers.add_parser(
        "synthetic-validate",
        help="Validate the fictional-pilot scaffold or completed artifacts",
    )
    synthetic_validate_parser.add_argument("--config")
    synthetic_validate_parser.add_argument("--output-dir")
    synthetic_validate_parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Validate the free scaffold before paid generation is complete",
    )
    synthetic_validate_parser.set_defaults(handler=_run_synthetic_validate)

    interactive_build_parser = subparsers.add_parser(
        "interactive-build",
        help="Build the deterministic, evaluation-only 100-case interactive seed",
    )
    interactive_build_parser.add_argument("--config")
    interactive_build_parser.add_argument("--benchmark")
    interactive_build_parser.add_argument("--output-dir")
    interactive_build_parser.set_defaults(handler=_run_interactive_build)

    interactive_validate_parser = subparsers.add_parser(
        "interactive-validate",
        help="Validate provenance and invariants for the interactive seed",
    )
    interactive_validate_parser.add_argument("--seed-dir")
    interactive_validate_parser.add_argument("--config")
    interactive_validate_parser.add_argument("--benchmark")
    interactive_validate_parser.set_defaults(handler=_run_interactive_validate)

    interactive_simulate_parser = subparsers.add_parser(
        "interactive-simulate",
        help="Replay a JSON or JSONL action trace against one deterministic case",
    )
    interactive_simulate_parser.add_argument("--case-id", required=True)
    interactive_simulate_parser.add_argument("--actions", required=True)
    interactive_simulate_parser.add_argument("--seed-dir")
    interactive_simulate_parser.set_defaults(handler=_run_interactive_simulate)

    interactive_cost_parser = subparsers.add_parser(
        "interactive-cost",
        help="Regenerate the versioned interactive pilot cost estimate",
    )
    interactive_cost_parser.add_argument("--assumptions")
    interactive_cost_parser.add_argument("--output-dir")
    interactive_cost_parser.set_defaults(handler=_run_interactive_cost)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        code = args.handler(args)
    except Exception as exc:
        print(f"acuitybench: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
