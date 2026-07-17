from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from acuitybench.interactive.costing import (
    default_cost_assumptions_path,
    estimate_interactive_pilot,
    load_cost_assumptions,
    render_cost_report_markdown,
    write_cost_report,
)
from acuitybench.sources import project_root


def _component(report: dict[str, object], component_id: str) -> dict[str, object]:
    components = report["components"]
    assert isinstance(components, list)
    return next(row for row in components if row["id"] == component_id)


def _default_payload() -> dict[str, object]:
    raw = yaml.safe_load(default_cost_assumptions_path().read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _write_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "assumptions.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_default_report_has_stable_counts_formulas_and_totals() -> None:
    assumptions = load_cost_assumptions()
    first = estimate_interactive_pilot(assumptions)
    second = estimate_interactive_pilot(assumptions)

    assert first == second
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    assert first["formula_version"] == 1
    assert first["estimate_type"] == "planning_assumptions_not_observed_billing"
    assert len(first["assumptions"]["fingerprint_sha256"]) == 64
    assert first["scenario"] == {
        "unique_cases": 500,
        "teacher_rollouts": 2000,
        "mean_gp_actions_per_rollout": 6,
        "supervised_action_targets": 12000,
        "evaluation_cases": 500,
        "on_policy_rollouts": 2000,
    }

    assert _component(first, "teacher_generation")["cost_usd"] == 36
    assert _component(first, "sft")["cost_usd"] == 6.633
    assert _component(first, "sweep")["cost_usd"] == 3.3165
    assert _component(first, "evaluation")["cost_usd"] == 1.026
    assert _component(first, "on_policy")["cost_usd"] == 15.315
    assert _component(first, "clinician_review")["cost_usd"] == 36000
    seed = first["standalone_estimates"]["seed_evaluation_preparation"]
    assert seed["version"] == 1
    assert seed["included_in_pilot_totals"] is False
    assert seed["cases"] == 100
    assert seed["quantities"] == {
        "content_review_minutes_per_case": 10,
        "independent_labels_per_case": 2,
        "minutes_per_label": 5,
        "adjudication_fraction": 0.2,
        "minutes_per_adjudication": 5,
        "content_review_hours": 16.666667,
        "independent_label_hours": 16.666667,
        "adjudication_hours": 1.666667,
        "total_clinician_hours": 35,
        "clinician_hourly_rate_usd": 180,
    }
    assert seed["costs"] == {
        "clinician_review_usd": 6300,
        "local_build_and_provider_compute_usd": 0,
        "total_usd": 6300,
    }
    assert seed["engineering_cost"] == "excluded"
    assert first["totals"] == {
        "tinker_compute_usd": 62.2905,
        "clinician_review_usd": 36000,
        "overall_usd": 36062.2905,
    }


def test_all_cost_formulas_respond_to_explicit_local_assumptions(
    tmp_path: Path,
) -> None:
    payload = _default_payload()
    payload["pricing_per_million_tokens_usd"] = {
        "teacher": {
            "model_id": "teacher",
            "prefill": 2,
            "cached_prefill": 1,
            "sample": 3,
        },
        "student": {
            "model_id": "student",
            "prefill": 4,
            "cached_prefill": 2,
            "sample": 5,
            "train": 7,
        },
    }
    payload["pilot"] = {
        "unique_cases": 1,
        "teacher_rollouts_per_case": 1,
        "mean_gp_actions_per_rollout": 1,
    }
    payload["teacher_generation"] = {
        "cumulative_prefill_tokens_per_rollout": 1_000_000,
        "cached_prefill_fraction": 0.25,
        "sampled_tokens_per_rollout": 1_000_000,
    }
    payload["sft"] = {"sequence_tokens_per_rollout": 1_000_000, "epochs": 1}
    payload["sweep"] = {
        "learning_rates": [0.001],
        "lora_ranks": [1],
        "rollout_fraction_per_configuration": 1,
        "epochs_per_configuration": 1,
    }
    payload["evaluation"] = {
        "cases": 1,
        "rollouts_per_case": 1,
        "cumulative_prefill_tokens_per_rollout": 1_000_000,
        "cached_prefill_fraction": 0.5,
        "sampled_tokens_per_rollout": 1_000_000,
    }
    payload["on_policy"] = {
        "cases": 1,
        "rollouts_per_case": 1,
        "rounds": 1,
        "student_cumulative_prefill_tokens_per_rollout": 1_000_000,
        "student_cached_prefill_fraction": 0.5,
        "student_sampled_tokens_per_rollout": 1_000_000,
        "teacher_scored_tokens_per_rollout": 1_000_000,
        "student_train_tokens_per_rollout": 1_000_000,
    }
    payload["clinician_review"] = {
        "hourly_rate_usd": 60,
        "training_cases_reviewed_fraction": 1,
        "minutes_per_training_case": 60,
        "generated_rollouts_reviewed_fraction": 1,
        "minutes_per_reviewed_rollout": 60,
        "evaluation_labels_per_case": 1,
        "minutes_per_evaluation_label": 60,
        "evaluation_adjudication_fraction": 1,
        "minutes_per_adjudication": 60,
    }

    report = estimate_interactive_pilot(
        assumptions_path=_write_payload(tmp_path, payload)
    )

    assert _component(report, "teacher_generation")["cost_usd"] == 4.75
    assert _component(report, "sft")["cost_usd"] == 7
    assert _component(report, "sweep")["cost_usd"] == 7
    assert _component(report, "evaluation")["cost_usd"] == 8
    assert _component(report, "on_policy")["cost_usd"] == 17
    assert _component(report, "clinician_review")["cost_usd"] == 240
    assert report["totals"] == {
        "tinker_compute_usd": 43.75,
        "clinician_review_usd": 240,
        "overall_usd": 283.75,
    }


def test_semantic_assumption_fingerprint_is_independent_of_yaml_formatting(
    tmp_path: Path,
) -> None:
    original = load_cost_assumptions()
    copied = load_cost_assumptions(_write_payload(tmp_path, _default_payload()))

    assert copied.fingerprint_sha256 == original.fingerprint_sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("schema_version", 2), "schema version"),
        (
            lambda payload: payload["teacher_generation"].__setitem__(
                "cached_prefill_fraction", 1.01
            ),
            "between 0 and 1",
        ),
        (
            lambda payload: payload["sweep"].__setitem__(
                "learning_rates", [0.001, 0.001]
            ),
            "must not contain duplicates",
        ),
        (
            lambda payload: payload["seed_evaluation_preparation"].__setitem__(
                "version", 2
            ),
            "seed evaluation preparation version",
        ),
    ],
)
def test_invalid_or_unsupported_assumptions_fail_closed(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    payload = _default_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        load_cost_assumptions(_write_payload(tmp_path, payload))


def test_estimator_rejects_ambiguous_provenance(tmp_path: Path) -> None:
    assumptions = load_cost_assumptions()

    with pytest.raises(ValueError, match="not both"):
        estimate_interactive_pilot(
            assumptions,
            assumptions_path=tmp_path / "unused.yaml",
        )


def test_markdown_renderer_is_deterministic_and_exposes_provenance() -> None:
    report = estimate_interactive_pilot()

    first = render_cost_report_markdown(report)
    second = render_cost_report_markdown(report)

    assert first == second
    assert first.endswith("\n")
    assert first.startswith("# Interactive pilot cost estimate\n")
    assert "not observed billing or a provider quote" in first
    assert report["assumptions"]["fingerprint_sha256"] in first
    assert "Qwen/Qwen3.5-397B-A17B" in first
    assert "Tinker compute: **$62.2905**" in first
    assert "Clinician review: **$36,000.0000**" in first
    assert "Overall estimate: **$36,062.2905**" in first
    assert "actual 100-case seed evaluation set" in first
    assert "Standalone seed preparation total: **$6,300.0000**" in first
    assert "Not included in the 500-case pilot totals" in first
    assert "Deterministic local build and provider compute: **$0.0000**" in first
    assert "Engineering cost: **excluded**" in first
    assert "## Formula details" in first


def test_write_cost_report_is_byte_stable_and_preserves_totals(
    tmp_path: Path,
) -> None:
    first_paths = write_cost_report(tmp_path / "first")
    first_bytes = [path.read_bytes() for path in first_paths]

    repeated_paths = write_cost_report(tmp_path / "first")
    second_paths = write_cost_report(tmp_path / "second")

    assert [path.read_bytes() for path in repeated_paths] == first_bytes
    assert [path.read_bytes() for path in second_paths] == first_bytes
    assert [path.name for path in first_paths] == [
        "cost_estimate.json",
        "cost_estimate.md",
    ]
    payload = json.loads(first_paths[0].read_text(encoding="utf-8"))
    assert payload["totals"] == {
        "tinker_compute_usd": 62.2905,
        "clinician_review_usd": 36000,
        "overall_usd": 36062.2905,
    }
    assert payload["standalone_estimates"]["seed_evaluation_preparation"][
        "costs"
    ]["total_usd"] == 6300
    assert first_paths[1].read_text(encoding="utf-8") == (
        render_cost_report_markdown(payload)
    )


def test_committed_default_cost_artifacts_match_the_renderer(tmp_path: Path) -> None:
    expected_paths = write_cost_report(tmp_path / "expected")
    committed_dir = project_root() / "results/interactive-pilot-v1"
    committed_paths = [
        committed_dir / "cost_estimate.json",
        committed_dir / "cost_estimate.md",
    ]

    assert [path.read_bytes() for path in committed_paths] == [
        path.read_bytes() for path in expected_paths
    ]
