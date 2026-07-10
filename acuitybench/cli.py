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
        print(
            f"  {model.id:16} {model.provider:8} {model.api_model:24} "
            f"{model.endpoint} (temperature={temperature})"
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
