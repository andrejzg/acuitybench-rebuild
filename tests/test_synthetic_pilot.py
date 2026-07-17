from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from acuitybench.cli import make_parser
from acuitybench.models import ModelRegistry
from acuitybench.providers.base import CompletionResult
from acuitybench.synthetic import (
    build_contamination_report,
    generate_synthetic_cases,
    initialize_synthetic_pilot,
    inspect_synthetic_plan,
    label_synthetic_cases,
    load_synthetic_plan,
    validate_synthetic_pilot,
)


class FakeSyntheticProvider:
    def __init__(self) -> None:
        self.intended: dict[str, str] = {}
        self.calls = 0
        self.closed = False

    async def complete(
        self,
        *,
        config: Any,
        messages: list[dict[str, str]],
        max_output_tokens: int | None = None,
        stream: bool = True,
    ) -> CompletionResult:
        self.calls += 1
        prompt = messages[0]["content"]
        if "Generation slot:" in prompt:
            slot_text = prompt.split("Generation slot:", 1)[1].split(
                "Required JSON shape:", 1
            )[0]
            slot = json.loads(slot_text)
            case_id = str(slot["case_id"])
            intended = str(slot["intended_acuity"])
            self.intended[case_id] = intended
            output = {
                "schema_version": "synthetic-acuity-generation/v0",
                "vignette": (
                    f"Entirely invented scenario {case_id}. A person describes a "
                    f"unique {slot['presentation_group']} concern with invented "
                    "timing, severity, functional context, and safety details. "
                    "This fixture contains no source patient or benchmark wording."
                ),
                "intended_acuity": intended,
                "presentation_group": slot["presentation_group"],
                "age_years": 40,
                "sex_context": "not clinically relevant in this fictional fixture",
                "relevant_facts": ["invented timing and severity information"],
                "distractor_facts": ["invented non-decisive context"],
                "intended_rationale": "The fictional facts support the requested test disposition.",
                "fictional_attestation": (
                    "Entirely fictional; no real person or source case was used."
                ),
            }
        else:
            case_id = next(case for case in self.intended if case in prompt)
            output = {
                "schema_version": "synthetic-acuity-label/v0",
                "acuity": self.intended[case_id],
                "rationale": "The visible fictional timing and severity support this label.",
                "confidence": "high",
                "ambiguity_flags": [],
            }
        return CompletionResult(
            text=json.dumps(output),
            finish_reason="stop",
            response_id=f"fake-{self.calls}",
            returned_model="fake-synthetic-snapshot",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

    async def close(self) -> None:
        self.closed = True


def _test_plan(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "configs/static/synthetic_pilot.v0.yaml"
    plan = yaml.safe_load(source.read_text(encoding="utf-8"))
    # Disable fuzzy blocking for repeated fixture prose while leaving exact
    # duplicate detection active.
    plan["leakage"]["sequence_similarity_threshold"] = 1.1
    plan["leakage"]["token_trigram_containment_threshold"] = 1.1
    destination = tmp_path / "synthetic-test.yaml"
    destination.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return destination


def test_synthetic_plan_is_balanced_and_explicitly_blocked() -> None:
    report = inspect_synthetic_plan()

    assert report["planned_cases"] == 20
    assert report["label_counts"] == {"A": 5, "B": 5, "C": 5, "D": 5}
    assert report["split_counts"] == {"development": 4, "train": 16}
    assert report["planned_generation_calls"] == 20
    assert report["planned_label_calls"] == 40
    assert report["planned_total_provider_calls"] == 60
    assert report["ready_for_paid_generation"] is False
    assert report["acuitybench_content_in_generation"] is False


def test_synthetic_init_is_deterministic_and_free(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    first = initialize_synthetic_pilot(output_dir=output)
    request_bytes = first.generation_requests.read_bytes()
    manifest = json.loads(first.manifest.read_text(encoding="utf-8"))

    second = initialize_synthetic_pilot(output_dir=output)

    assert second.generation_requests.read_bytes() == request_bytes
    assert manifest["paid_provider_calls_recorded"] == 0
    assert manifest["training_ready"] is False
    validation = validate_synthetic_pilot(
        output_dir=output, allow_incomplete=True
    )
    assert validation["scaffold_valid"] is True
    assert validation["pipeline_complete"] is False


def test_fake_provider_runs_complete_resumable_pipeline(tmp_path: Path) -> None:
    config_path = _test_plan(tmp_path)
    output = tmp_path / "pilot"
    provider = FakeSyntheticProvider()
    model = ModelRegistry().get("gpt-5-mini")

    generation = asyncio.run(
        generate_synthetic_cases(
            provider=provider,
            model=model,
            config_path=config_path,
            output_dir=output,
        )
    )
    labeling = asyncio.run(
        label_synthetic_cases(
            provider=provider,
            model=model,
            config_path=config_path,
            output_dir=output,
        )
    )
    resumed = asyncio.run(
        generate_synthetic_cases(
            provider=provider,
            model=model,
            config_path=config_path,
            output_dir=output,
        )
    )

    assert generation["new_successes"] == 20
    assert labeling["new_successes"] == 40
    assert labeling["finalize"]["accepted_examples"] == 20
    assert resumed["new_successes"] == 0
    assert provider.calls == 60
    validation = validate_synthetic_pilot(
        config_path=config_path,
        output_dir=output,
    )
    assert validation["pipeline_complete"] is True
    assert validation["accepted_examples"] == 20
    assert validation["training_ready"] is False
    assert validation["candidate_validation"]["training_allowed"] is False
    candidates = [
        json.loads(line)
        for line in (output / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {candidate["training_allowed"] for candidate in candidates} == {False}
    assert "manual review" in validation["training_blockers"][0]


def test_contamination_report_blocks_exact_benchmark_copy() -> None:
    plan = load_synthetic_plan()
    benchmark_path = Path(__file__).resolve().parents[1] / str(
        plan["leakage"]["benchmark"]
    )
    benchmark = pd.read_csv(benchmark_path, nrows=1)
    source = json.loads(benchmark.iloc[0]["normalized_prompt_text"])
    copied = "\n".join(item["content"] for item in source)

    report = build_contamination_report(
        {"copied-case": copied}, plan=plan, benchmark_path=benchmark_path
    )

    assert report["blocked_cases"] == 1
    result = report["results"][0]
    assert result["blocked"] is True
    assert "exact_benchmark_match" in result["blocked_reasons"]
    assert report["checks"]["semantic_embedding_similarity"].startswith(
        "not_implemented"
    )


def test_paid_cli_requires_both_explicit_confirmations() -> None:
    parser = make_parser()
    args = parser.parse_args(
        ["synthetic-generate", "--model", "gpt-5-mini"]
    )
    with pytest.raises(ValueError, match="--confirm-spend"):
        args.handler(args)

    args = parser.parse_args(
        [
            "synthetic-generate",
            "--model",
            "gpt-5-mini",
            "--confirm-spend",
        ]
    )
    with pytest.raises(ValueError, match="--terms-reviewed"):
        args.handler(args)
