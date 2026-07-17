"""Deterministic cost planning for the interactive-acuity pilot.

This module deliberately performs no provider or network calls.  It turns a
versioned local assumptions file into auditable, JSON-serializable report data.
The result is a planning estimate, not observed billing telemetry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping

import yaml

from acuitybench.sources import project_root


ASSUMPTIONS_SCHEMA_VERSION = 1
FORMULA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
SEED_EVALUATION_PREPARATION_VERSION = 1
_MILLION = Decimal("1000000")
_REPORT_PRECISION = Decimal("0.000001")
_COMPONENT_TITLES = {
    "teacher_generation": "Teacher generation",
    "sft": "SFT",
    "sweep": "Sweep",
    "evaluation": "Evaluation",
    "on_policy": "On-policy",
    "clinician_review": "Clinician review",
}
_CATEGORY_TITLES = {
    "tinker_compute": "Tinker compute",
    "human_review": "Human review",
}


@dataclass(frozen=True)
class AssumptionMetadata:
    id: str
    description: str
    currency: str
    pricing_source_url: str
    pricing_effective_date: str
    pricing_accessed_date: str


@dataclass(frozen=True)
class ModelPricing:
    model_id: str
    prefill_per_million: Decimal
    cached_prefill_per_million: Decimal
    sample_per_million: Decimal
    train_per_million: Decimal | None = None


@dataclass(frozen=True)
class PilotScale:
    unique_cases: int
    teacher_rollouts_per_case: int
    mean_gp_actions_per_rollout: int


@dataclass(frozen=True)
class GenerationWorkload:
    cumulative_prefill_tokens_per_rollout: int
    cached_prefill_fraction: Decimal
    sampled_tokens_per_rollout: int


@dataclass(frozen=True)
class SFTWorkload:
    sequence_tokens_per_rollout: int
    epochs: int


@dataclass(frozen=True)
class SweepWorkload:
    learning_rates: tuple[Decimal, ...]
    lora_ranks: tuple[int, ...]
    rollout_fraction_per_configuration: Decimal
    epochs_per_configuration: int


@dataclass(frozen=True)
class EvaluationWorkload:
    cases: int
    rollouts_per_case: int
    cumulative_prefill_tokens_per_rollout: int
    cached_prefill_fraction: Decimal
    sampled_tokens_per_rollout: int


@dataclass(frozen=True)
class OnPolicyWorkload:
    cases: int
    rollouts_per_case: int
    rounds: int
    student_cumulative_prefill_tokens_per_rollout: int
    student_cached_prefill_fraction: Decimal
    student_sampled_tokens_per_rollout: int
    teacher_scored_tokens_per_rollout: int
    student_train_tokens_per_rollout: int


@dataclass(frozen=True)
class ClinicianReviewWorkload:
    hourly_rate_usd: Decimal
    training_cases_reviewed_fraction: Decimal
    minutes_per_training_case: Decimal
    generated_rollouts_reviewed_fraction: Decimal
    minutes_per_reviewed_rollout: Decimal
    evaluation_labels_per_case: int
    minutes_per_evaluation_label: Decimal
    evaluation_adjudication_fraction: Decimal
    minutes_per_adjudication: Decimal


@dataclass(frozen=True)
class SeedEvaluationPreparation:
    version: int
    cases: int


@dataclass(frozen=True)
class CostAssumptions:
    """Validated, immutable inputs to the v1 cost formulas."""

    schema_version: int
    formula_version: int
    fingerprint_sha256: str
    metadata: AssumptionMetadata
    teacher_pricing: ModelPricing
    student_pricing: ModelPricing
    pilot: PilotScale
    teacher_generation: GenerationWorkload
    sft: SFTWorkload
    sweep: SweepWorkload
    evaluation: EvaluationWorkload
    on_policy: OnPolicyWorkload
    clinician_review: ClinicianReviewWorkload
    seed_evaluation_preparation: SeedEvaluationPreparation


def default_cost_assumptions_path() -> Path:
    return project_root() / "configs/interactive/cost_assumptions.v1.yaml"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _check_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    label: str,
) -> None:
    optional = optional or set()
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise ValueError(f"{label} is missing: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {sorted(unknown)}")


def _text(value: Any, label: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result != value or result < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return result


def _decimal(value: Any, label: str, *, minimum: Decimal = Decimal("0")) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not result.is_finite() or result < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _fraction(value: Any, label: str) -> Decimal:
    result = _decimal(value, label)
    if result > 1:
        raise ValueError(f"{label} must be between 0 and 1")
    return result


def _semantic_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_metadata(raw: Any) -> AssumptionMetadata:
    values = _mapping(raw, "assumption_set")
    fields = {
        "id",
        "description",
        "currency",
        "pricing_source_url",
        "pricing_effective_date",
        "pricing_accessed_date",
    }
    _check_keys(values, required=fields, label="assumption_set")
    currency = _text(values["currency"], "assumption_set.currency")
    if currency != "USD":
        raise ValueError("assumption_set.currency must be 'USD' for formula v1")
    return AssumptionMetadata(
        id=_text(values["id"], "assumption_set.id"),
        description=_text(values["description"], "assumption_set.description"),
        currency=currency,
        pricing_source_url=_text(
            values["pricing_source_url"], "assumption_set.pricing_source_url"
        ),
        pricing_effective_date=_text(
            values["pricing_effective_date"],
            "assumption_set.pricing_effective_date",
        ),
        pricing_accessed_date=_text(
            values["pricing_accessed_date"],
            "assumption_set.pricing_accessed_date",
        ),
    )


def _parse_pricing(raw: Any, label: str, *, require_train: bool) -> ModelPricing:
    values = _mapping(raw, label)
    required = {"model_id", "prefill", "cached_prefill", "sample"}
    optional = {"train"}
    if require_train:
        required.add("train")
        optional.clear()
    _check_keys(values, required=required, optional=optional, label=label)
    train = (
        _decimal(values["train"], f"{label}.train")
        if "train" in values
        else None
    )
    return ModelPricing(
        model_id=_text(values["model_id"], f"{label}.model_id"),
        prefill_per_million=_decimal(values["prefill"], f"{label}.prefill"),
        cached_prefill_per_million=_decimal(
            values["cached_prefill"], f"{label}.cached_prefill"
        ),
        sample_per_million=_decimal(values["sample"], f"{label}.sample"),
        train_per_million=train,
    )


def _parse_pilot(raw: Any) -> PilotScale:
    values = _mapping(raw, "pilot")
    fields = {
        "unique_cases",
        "teacher_rollouts_per_case",
        "mean_gp_actions_per_rollout",
    }
    _check_keys(values, required=fields, label="pilot")
    return PilotScale(
        unique_cases=_integer(values["unique_cases"], "pilot.unique_cases"),
        teacher_rollouts_per_case=_integer(
            values["teacher_rollouts_per_case"],
            "pilot.teacher_rollouts_per_case",
        ),
        mean_gp_actions_per_rollout=_integer(
            values["mean_gp_actions_per_rollout"],
            "pilot.mean_gp_actions_per_rollout",
        ),
    )


def _parse_generation(raw: Any, label: str) -> GenerationWorkload:
    values = _mapping(raw, label)
    fields = {
        "cumulative_prefill_tokens_per_rollout",
        "cached_prefill_fraction",
        "sampled_tokens_per_rollout",
    }
    _check_keys(values, required=fields, label=label)
    return GenerationWorkload(
        cumulative_prefill_tokens_per_rollout=_integer(
            values["cumulative_prefill_tokens_per_rollout"],
            f"{label}.cumulative_prefill_tokens_per_rollout",
            minimum=0,
        ),
        cached_prefill_fraction=_fraction(
            values["cached_prefill_fraction"],
            f"{label}.cached_prefill_fraction",
        ),
        sampled_tokens_per_rollout=_integer(
            values["sampled_tokens_per_rollout"],
            f"{label}.sampled_tokens_per_rollout",
            minimum=0,
        ),
    )


def _parse_sft(raw: Any) -> SFTWorkload:
    values = _mapping(raw, "sft")
    fields = {"sequence_tokens_per_rollout", "epochs"}
    _check_keys(values, required=fields, label="sft")
    return SFTWorkload(
        sequence_tokens_per_rollout=_integer(
            values["sequence_tokens_per_rollout"],
            "sft.sequence_tokens_per_rollout",
        ),
        epochs=_integer(values["epochs"], "sft.epochs"),
    )


def _parse_sweep(raw: Any) -> SweepWorkload:
    values = _mapping(raw, "sweep")
    fields = {
        "learning_rates",
        "lora_ranks",
        "rollout_fraction_per_configuration",
        "epochs_per_configuration",
    }
    _check_keys(values, required=fields, label="sweep")
    raw_learning_rates = values["learning_rates"]
    raw_lora_ranks = values["lora_ranks"]
    if not isinstance(raw_learning_rates, list) or not raw_learning_rates:
        raise ValueError("sweep.learning_rates must be a non-empty list")
    if not isinstance(raw_lora_ranks, list) or not raw_lora_ranks:
        raise ValueError("sweep.lora_ranks must be a non-empty list")
    learning_rates = tuple(
        _decimal(value, "sweep.learning_rates[]", minimum=Decimal("0.000000000001"))
        for value in raw_learning_rates
    )
    lora_ranks = tuple(
        _integer(value, "sweep.lora_ranks[]") for value in raw_lora_ranks
    )
    if len(set(learning_rates)) != len(learning_rates):
        raise ValueError("sweep.learning_rates must not contain duplicates")
    if len(set(lora_ranks)) != len(lora_ranks):
        raise ValueError("sweep.lora_ranks must not contain duplicates")
    return SweepWorkload(
        learning_rates=learning_rates,
        lora_ranks=lora_ranks,
        rollout_fraction_per_configuration=_fraction(
            values["rollout_fraction_per_configuration"],
            "sweep.rollout_fraction_per_configuration",
        ),
        epochs_per_configuration=_integer(
            values["epochs_per_configuration"],
            "sweep.epochs_per_configuration",
        ),
    )


def _parse_evaluation(raw: Any) -> EvaluationWorkload:
    values = _mapping(raw, "evaluation")
    fields = {
        "cases",
        "rollouts_per_case",
        "cumulative_prefill_tokens_per_rollout",
        "cached_prefill_fraction",
        "sampled_tokens_per_rollout",
    }
    _check_keys(values, required=fields, label="evaluation")
    return EvaluationWorkload(
        cases=_integer(values["cases"], "evaluation.cases"),
        rollouts_per_case=_integer(
            values["rollouts_per_case"], "evaluation.rollouts_per_case"
        ),
        cumulative_prefill_tokens_per_rollout=_integer(
            values["cumulative_prefill_tokens_per_rollout"],
            "evaluation.cumulative_prefill_tokens_per_rollout",
            minimum=0,
        ),
        cached_prefill_fraction=_fraction(
            values["cached_prefill_fraction"],
            "evaluation.cached_prefill_fraction",
        ),
        sampled_tokens_per_rollout=_integer(
            values["sampled_tokens_per_rollout"],
            "evaluation.sampled_tokens_per_rollout",
            minimum=0,
        ),
    )


def _parse_on_policy(raw: Any) -> OnPolicyWorkload:
    values = _mapping(raw, "on_policy")
    fields = {
        "cases",
        "rollouts_per_case",
        "rounds",
        "student_cumulative_prefill_tokens_per_rollout",
        "student_cached_prefill_fraction",
        "student_sampled_tokens_per_rollout",
        "teacher_scored_tokens_per_rollout",
        "student_train_tokens_per_rollout",
    }
    _check_keys(values, required=fields, label="on_policy")
    return OnPolicyWorkload(
        cases=_integer(values["cases"], "on_policy.cases"),
        rollouts_per_case=_integer(
            values["rollouts_per_case"], "on_policy.rollouts_per_case"
        ),
        rounds=_integer(values["rounds"], "on_policy.rounds"),
        student_cumulative_prefill_tokens_per_rollout=_integer(
            values["student_cumulative_prefill_tokens_per_rollout"],
            "on_policy.student_cumulative_prefill_tokens_per_rollout",
            minimum=0,
        ),
        student_cached_prefill_fraction=_fraction(
            values["student_cached_prefill_fraction"],
            "on_policy.student_cached_prefill_fraction",
        ),
        student_sampled_tokens_per_rollout=_integer(
            values["student_sampled_tokens_per_rollout"],
            "on_policy.student_sampled_tokens_per_rollout",
            minimum=0,
        ),
        teacher_scored_tokens_per_rollout=_integer(
            values["teacher_scored_tokens_per_rollout"],
            "on_policy.teacher_scored_tokens_per_rollout",
            minimum=0,
        ),
        student_train_tokens_per_rollout=_integer(
            values["student_train_tokens_per_rollout"],
            "on_policy.student_train_tokens_per_rollout",
            minimum=0,
        ),
    )


def _parse_clinician_review(raw: Any) -> ClinicianReviewWorkload:
    values = _mapping(raw, "clinician_review")
    fields = {
        "hourly_rate_usd",
        "training_cases_reviewed_fraction",
        "minutes_per_training_case",
        "generated_rollouts_reviewed_fraction",
        "minutes_per_reviewed_rollout",
        "evaluation_labels_per_case",
        "minutes_per_evaluation_label",
        "evaluation_adjudication_fraction",
        "minutes_per_adjudication",
    }
    _check_keys(values, required=fields, label="clinician_review")
    return ClinicianReviewWorkload(
        hourly_rate_usd=_decimal(
            values["hourly_rate_usd"], "clinician_review.hourly_rate_usd"
        ),
        training_cases_reviewed_fraction=_fraction(
            values["training_cases_reviewed_fraction"],
            "clinician_review.training_cases_reviewed_fraction",
        ),
        minutes_per_training_case=_decimal(
            values["minutes_per_training_case"],
            "clinician_review.minutes_per_training_case",
        ),
        generated_rollouts_reviewed_fraction=_fraction(
            values["generated_rollouts_reviewed_fraction"],
            "clinician_review.generated_rollouts_reviewed_fraction",
        ),
        minutes_per_reviewed_rollout=_decimal(
            values["minutes_per_reviewed_rollout"],
            "clinician_review.minutes_per_reviewed_rollout",
        ),
        evaluation_labels_per_case=_integer(
            values["evaluation_labels_per_case"],
            "clinician_review.evaluation_labels_per_case",
        ),
        minutes_per_evaluation_label=_decimal(
            values["minutes_per_evaluation_label"],
            "clinician_review.minutes_per_evaluation_label",
        ),
        evaluation_adjudication_fraction=_fraction(
            values["evaluation_adjudication_fraction"],
            "clinician_review.evaluation_adjudication_fraction",
        ),
        minutes_per_adjudication=_decimal(
            values["minutes_per_adjudication"],
            "clinician_review.minutes_per_adjudication",
        ),
    )


def _parse_seed_evaluation_preparation(
    raw: Any,
) -> SeedEvaluationPreparation:
    values = _mapping(raw, "seed_evaluation_preparation")
    _check_keys(
        values,
        required={"version", "cases"},
        label="seed_evaluation_preparation",
    )
    version = _integer(
        values["version"],
        "seed_evaluation_preparation.version",
    )
    if version != SEED_EVALUATION_PREPARATION_VERSION:
        raise ValueError(
            "Unsupported seed evaluation preparation version "
            f"{version}; expected {SEED_EVALUATION_PREPARATION_VERSION}"
        )
    return SeedEvaluationPreparation(
        version=version,
        cases=_integer(
            values["cases"],
            "seed_evaluation_preparation.cases",
        ),
    )


def load_cost_assumptions(path: Path | None = None) -> CostAssumptions:
    """Load and strictly validate a local v1 assumptions document."""

    path = path or default_cost_assumptions_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values = _mapping(raw, str(path))
    top_level_fields = {
        "schema_version",
        "formula_version",
        "assumption_set",
        "pricing_per_million_tokens_usd",
        "pilot",
        "teacher_generation",
        "sft",
        "sweep",
        "evaluation",
        "on_policy",
        "clinician_review",
        "seed_evaluation_preparation",
    }
    _check_keys(values, required=top_level_fields, label=str(path))
    schema_version = _integer(
        values["schema_version"], "schema_version", minimum=1
    )
    formula_version = _integer(
        values["formula_version"], "formula_version", minimum=1
    )
    if schema_version != ASSUMPTIONS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported cost assumptions schema version {schema_version}; "
            f"expected {ASSUMPTIONS_SCHEMA_VERSION}"
        )
    if formula_version != FORMULA_VERSION:
        raise ValueError(
            f"Unsupported cost formula version {formula_version}; "
            f"expected {FORMULA_VERSION}"
        )

    pricing = _mapping(
        values["pricing_per_million_tokens_usd"],
        "pricing_per_million_tokens_usd",
    )
    _check_keys(
        pricing,
        required={"teacher", "student"},
        label="pricing_per_million_tokens_usd",
    )
    return CostAssumptions(
        schema_version=schema_version,
        formula_version=formula_version,
        fingerprint_sha256=_semantic_fingerprint(values),
        metadata=_parse_metadata(values["assumption_set"]),
        teacher_pricing=_parse_pricing(
            pricing["teacher"],
            "pricing_per_million_tokens_usd.teacher",
            require_train=False,
        ),
        student_pricing=_parse_pricing(
            pricing["student"],
            "pricing_per_million_tokens_usd.student",
            require_train=True,
        ),
        pilot=_parse_pilot(values["pilot"]),
        teacher_generation=_parse_generation(
            values["teacher_generation"], "teacher_generation"
        ),
        sft=_parse_sft(values["sft"]),
        sweep=_parse_sweep(values["sweep"]),
        evaluation=_parse_evaluation(values["evaluation"]),
        on_policy=_parse_on_policy(values["on_policy"]),
        clinician_review=_parse_clinician_review(values["clinician_review"]),
        seed_evaluation_preparation=_parse_seed_evaluation_preparation(
            values["seed_evaluation_preparation"]
        ),
    )


def _report_number(value: Decimal | int) -> int | float:
    if isinstance(value, int):
        return value
    rounded = value.quantize(_REPORT_PRECISION, rounding=ROUND_HALF_UP)
    if rounded == rounded.to_integral_value():
        return int(rounded)
    return float(rounded)


def _prefill_split(
    total_prefill_tokens: Decimal, cached_fraction: Decimal
) -> tuple[Decimal, Decimal]:
    cached = total_prefill_tokens * cached_fraction
    return total_prefill_tokens - cached, cached


def _generation_cost(
    *,
    rollouts: Decimal,
    prefill_tokens_per_rollout: int,
    cached_fraction: Decimal,
    sampled_tokens_per_rollout: int,
    pricing: ModelPricing,
) -> tuple[Decimal, dict[str, Decimal]]:
    total_prefill = rollouts * prefill_tokens_per_rollout
    uncached_prefill, cached_prefill = _prefill_split(
        total_prefill, cached_fraction
    )
    sampled = rollouts * sampled_tokens_per_rollout
    prefill_cost = uncached_prefill * pricing.prefill_per_million / _MILLION
    cached_cost = (
        cached_prefill * pricing.cached_prefill_per_million / _MILLION
    )
    sample_cost = sampled * pricing.sample_per_million / _MILLION
    return prefill_cost + cached_cost + sample_cost, {
        "rollouts": rollouts,
        "uncached_prefill_tokens": uncached_prefill,
        "cached_prefill_tokens": cached_prefill,
        "sampled_tokens": sampled,
        "uncached_prefill_cost_usd": prefill_cost,
        "cached_prefill_cost_usd": cached_cost,
        "sample_cost_usd": sample_cost,
    }


def _json_numbers(values: Mapping[str, Decimal | int]) -> dict[str, int | float]:
    return {key: _report_number(value) for key, value in values.items()}


def _component(
    *,
    component_id: str,
    category: str,
    formula: str,
    quantities: Mapping[str, Decimal | int],
    cost: Decimal,
) -> dict[str, Any]:
    return {
        "id": component_id,
        "category": category,
        "formula": formula,
        "quantities": _json_numbers(quantities),
        "cost_usd": _report_number(cost),
    }


def estimate_interactive_pilot(
    assumptions: CostAssumptions | None = None,
    *,
    assumptions_path: Path | None = None,
) -> dict[str, Any]:
    """Return deterministic v1 planning data for one interactive pilot.

    Passing ``assumptions_path`` is a convenience for local scenario files.
    Supplying both a parsed object and a path is rejected to keep provenance
    unambiguous.
    """

    if assumptions is not None and assumptions_path is not None:
        raise ValueError("Pass assumptions or assumptions_path, not both")
    assumptions = assumptions or load_cost_assumptions(assumptions_path)
    if assumptions.student_pricing.train_per_million is None:
        raise ValueError("Student pricing must include a train rate")

    teacher_rollouts = Decimal(
        assumptions.pilot.unique_cases
        * assumptions.pilot.teacher_rollouts_per_case
    )
    action_targets = int(
        teacher_rollouts * assumptions.pilot.mean_gp_actions_per_rollout
    )

    teacher_cost, teacher_quantities = _generation_cost(
        rollouts=teacher_rollouts,
        prefill_tokens_per_rollout=(
            assumptions.teacher_generation.cumulative_prefill_tokens_per_rollout
        ),
        cached_fraction=assumptions.teacher_generation.cached_prefill_fraction,
        sampled_tokens_per_rollout=(
            assumptions.teacher_generation.sampled_tokens_per_rollout
        ),
        pricing=assumptions.teacher_pricing,
    )
    teacher_component = _component(
        component_id="teacher_generation",
        category="tinker_compute",
        formula=(
            "(uncached_prefill_tokens * teacher_prefill_rate + "
            "cached_prefill_tokens * teacher_cached_prefill_rate + "
            "sampled_tokens * teacher_sample_rate) / 1_000_000"
        ),
        quantities=teacher_quantities,
        cost=teacher_cost,
    )

    sft_tokens = (
        teacher_rollouts
        * assumptions.sft.sequence_tokens_per_rollout
        * assumptions.sft.epochs
    )
    sft_cost = (
        sft_tokens * assumptions.student_pricing.train_per_million / _MILLION
    )
    sft_component = _component(
        component_id="sft",
        category="tinker_compute",
        formula=(
            "teacher_rollouts * sequence_tokens_per_rollout * epochs * "
            "student_train_rate / 1_000_000"
        ),
        quantities={
            "teacher_rollouts": teacher_rollouts,
            "sequence_tokens_per_rollout": assumptions.sft.sequence_tokens_per_rollout,
            "epochs": assumptions.sft.epochs,
            "train_tokens": sft_tokens,
        },
        cost=sft_cost,
    )

    sweep_configurations = len(assumptions.sweep.learning_rates) * len(
        assumptions.sweep.lora_ranks
    )
    sweep_rollouts_per_configuration = (
        teacher_rollouts * assumptions.sweep.rollout_fraction_per_configuration
    )
    sweep_tokens = (
        sweep_rollouts_per_configuration
        * sweep_configurations
        * assumptions.sft.sequence_tokens_per_rollout
        * assumptions.sweep.epochs_per_configuration
    )
    sweep_cost = (
        sweep_tokens * assumptions.student_pricing.train_per_million / _MILLION
    )
    sweep_component = _component(
        component_id="sweep",
        category="tinker_compute",
        formula=(
            "teacher_rollouts * rollout_fraction * learning_rate_count * "
            "lora_rank_count * sequence_tokens_per_rollout * sweep_epochs * "
            "student_train_rate / 1_000_000"
        ),
        quantities={
            "configurations": sweep_configurations,
            "learning_rate_count": len(assumptions.sweep.learning_rates),
            "lora_rank_count": len(assumptions.sweep.lora_ranks),
            "rollouts_per_configuration": sweep_rollouts_per_configuration,
            "epochs_per_configuration": assumptions.sweep.epochs_per_configuration,
            "train_tokens": sweep_tokens,
        },
        cost=sweep_cost,
    )

    evaluation_rollouts = Decimal(
        assumptions.evaluation.cases * assumptions.evaluation.rollouts_per_case
    )
    evaluation_cost, evaluation_quantities = _generation_cost(
        rollouts=evaluation_rollouts,
        prefill_tokens_per_rollout=(
            assumptions.evaluation.cumulative_prefill_tokens_per_rollout
        ),
        cached_fraction=assumptions.evaluation.cached_prefill_fraction,
        sampled_tokens_per_rollout=assumptions.evaluation.sampled_tokens_per_rollout,
        pricing=assumptions.student_pricing,
    )
    evaluation_component = _component(
        component_id="evaluation",
        category="tinker_compute",
        formula=(
            "(uncached_prefill_tokens * student_prefill_rate + "
            "cached_prefill_tokens * student_cached_prefill_rate + "
            "sampled_tokens * student_sample_rate) / 1_000_000"
        ),
        quantities=evaluation_quantities,
        cost=evaluation_cost,
    )

    on_policy_rollouts = Decimal(
        assumptions.on_policy.cases
        * assumptions.on_policy.rollouts_per_case
        * assumptions.on_policy.rounds
    )
    on_policy_generation_cost, on_policy_quantities = _generation_cost(
        rollouts=on_policy_rollouts,
        prefill_tokens_per_rollout=(
            assumptions.on_policy.student_cumulative_prefill_tokens_per_rollout
        ),
        cached_fraction=assumptions.on_policy.student_cached_prefill_fraction,
        sampled_tokens_per_rollout=(
            assumptions.on_policy.student_sampled_tokens_per_rollout
        ),
        pricing=assumptions.student_pricing,
    )
    teacher_scored_tokens = (
        on_policy_rollouts
        * assumptions.on_policy.teacher_scored_tokens_per_rollout
    )
    teacher_scoring_cost = (
        teacher_scored_tokens
        * assumptions.teacher_pricing.prefill_per_million
        / _MILLION
    )
    student_train_tokens = (
        on_policy_rollouts
        * assumptions.on_policy.student_train_tokens_per_rollout
    )
    student_training_cost = (
        student_train_tokens
        * assumptions.student_pricing.train_per_million
        / _MILLION
    )
    on_policy_quantities.update(
        {
            "rounds": Decimal(assumptions.on_policy.rounds),
            "teacher_scored_tokens": teacher_scored_tokens,
            "teacher_scoring_cost_usd": teacher_scoring_cost,
            "student_train_tokens": student_train_tokens,
            "student_training_cost_usd": student_training_cost,
        }
    )
    on_policy_cost = (
        on_policy_generation_cost + teacher_scoring_cost + student_training_cost
    )
    on_policy_component = _component(
        component_id="on_policy",
        category="tinker_compute",
        formula=(
            "student_generation_cost + teacher_scored_tokens * "
            "teacher_prefill_rate / 1_000_000 + student_train_tokens * "
            "student_train_rate / 1_000_000"
        ),
        quantities=on_policy_quantities,
        cost=on_policy_cost,
    )

    review = assumptions.clinician_review
    training_case_hours = (
        assumptions.pilot.unique_cases
        * review.training_cases_reviewed_fraction
        * review.minutes_per_training_case
        / Decimal(60)
    )
    rollout_review_hours = (
        teacher_rollouts
        * review.generated_rollouts_reviewed_fraction
        * review.minutes_per_reviewed_rollout
        / Decimal(60)
    )
    evaluation_label_hours = (
        assumptions.evaluation.cases
        * review.evaluation_labels_per_case
        * review.minutes_per_evaluation_label
        / Decimal(60)
    )
    adjudication_hours = (
        assumptions.evaluation.cases
        * review.evaluation_adjudication_fraction
        * review.minutes_per_adjudication
        / Decimal(60)
    )
    clinician_hours = (
        training_case_hours
        + rollout_review_hours
        + evaluation_label_hours
        + adjudication_hours
    )
    clinician_cost = clinician_hours * review.hourly_rate_usd
    clinician_component = _component(
        component_id="clinician_review",
        category="human_review",
        formula=(
            "(training_case_review_hours + rollout_review_hours + "
            "evaluation_label_hours + adjudication_hours) * "
            "clinician_hourly_rate"
        ),
        quantities={
            "training_case_review_hours": training_case_hours,
            "rollout_review_hours": rollout_review_hours,
            "evaluation_label_hours": evaluation_label_hours,
            "adjudication_hours": adjudication_hours,
            "total_hours": clinician_hours,
            "hourly_rate_usd": review.hourly_rate_usd,
        },
        cost=clinician_cost,
    )

    seed = assumptions.seed_evaluation_preparation
    seed_content_review_hours = (
        seed.cases
        * review.minutes_per_training_case
        / Decimal(60)
    )
    seed_independent_label_hours = (
        seed.cases
        * review.evaluation_labels_per_case
        * review.minutes_per_evaluation_label
        / Decimal(60)
    )
    seed_adjudication_hours = (
        seed.cases
        * review.evaluation_adjudication_fraction
        * review.minutes_per_adjudication
        / Decimal(60)
    )
    seed_clinician_hours = (
        seed_content_review_hours
        + seed_independent_label_hours
        + seed_adjudication_hours
    )
    seed_clinician_cost = seed_clinician_hours * review.hourly_rate_usd

    components = [
        teacher_component,
        sft_component,
        sweep_component,
        evaluation_component,
        on_policy_component,
        clinician_component,
    ]
    tinker_compute_cost = (
        teacher_cost
        + sft_cost
        + sweep_cost
        + evaluation_cost
        + on_policy_cost
    )
    overall_cost = tinker_compute_cost + clinician_cost

    def pricing_row(pricing: ModelPricing) -> dict[str, Any]:
        row: dict[str, Any] = {
            "model_id": pricing.model_id,
            "prefill_usd_per_million": _report_number(
                pricing.prefill_per_million
            ),
            "cached_prefill_usd_per_million": _report_number(
                pricing.cached_prefill_per_million
            ),
            "sample_usd_per_million": _report_number(pricing.sample_per_million),
        }
        if pricing.train_per_million is not None:
            row["train_usd_per_million"] = _report_number(
                pricing.train_per_million
            )
        return row

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "formula_version": assumptions.formula_version,
        "estimate_type": "planning_assumptions_not_observed_billing",
        "assumptions": {
            "id": assumptions.metadata.id,
            "schema_version": assumptions.schema_version,
            "fingerprint_sha256": assumptions.fingerprint_sha256,
            "description": assumptions.metadata.description,
            "currency": assumptions.metadata.currency,
            "pricing_source_url": assumptions.metadata.pricing_source_url,
            "pricing_effective_date": assumptions.metadata.pricing_effective_date,
            "pricing_accessed_date": assumptions.metadata.pricing_accessed_date,
        },
        "models": {
            "teacher": pricing_row(assumptions.teacher_pricing),
            "student": pricing_row(assumptions.student_pricing),
        },
        "scenario": {
            "unique_cases": assumptions.pilot.unique_cases,
            "teacher_rollouts": int(teacher_rollouts),
            "mean_gp_actions_per_rollout": (
                assumptions.pilot.mean_gp_actions_per_rollout
            ),
            "supervised_action_targets": action_targets,
            "evaluation_cases": assumptions.evaluation.cases,
            "on_policy_rollouts": int(on_policy_rollouts),
        },
        "components": components,
        "standalone_estimates": {
            "seed_evaluation_preparation": {
                "version": seed.version,
                "included_in_pilot_totals": False,
                "cases": seed.cases,
                "formula": (
                    "cases * (content_review_minutes_per_case + "
                    "independent_labels_per_case * minutes_per_label + "
                    "adjudication_fraction * minutes_per_adjudication) / 60 * "
                    "clinician_hourly_rate"
                ),
                "quantities": {
                    "content_review_minutes_per_case": _report_number(
                        review.minutes_per_training_case
                    ),
                    "independent_labels_per_case": (
                        review.evaluation_labels_per_case
                    ),
                    "minutes_per_label": _report_number(
                        review.minutes_per_evaluation_label
                    ),
                    "adjudication_fraction": _report_number(
                        review.evaluation_adjudication_fraction
                    ),
                    "minutes_per_adjudication": _report_number(
                        review.minutes_per_adjudication
                    ),
                    "content_review_hours": _report_number(
                        seed_content_review_hours
                    ),
                    "independent_label_hours": _report_number(
                        seed_independent_label_hours
                    ),
                    "adjudication_hours": _report_number(
                        seed_adjudication_hours
                    ),
                    "total_clinician_hours": _report_number(
                        seed_clinician_hours
                    ),
                    "clinician_hourly_rate_usd": _report_number(
                        review.hourly_rate_usd
                    ),
                },
                "costs": {
                    "clinician_review_usd": _report_number(
                        seed_clinician_cost
                    ),
                    "local_build_and_provider_compute_usd": 0,
                    "total_usd": _report_number(seed_clinician_cost),
                },
                "engineering_cost": "excluded",
            }
        },
        "totals": {
            "tinker_compute_usd": _report_number(tinker_compute_cost),
            "clinician_review_usd": _report_number(clinician_cost),
            "overall_usd": _report_number(overall_cost),
        },
    }


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_report_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    return _markdown_escape(value)


def _format_usd(value: Any) -> str:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric USD value, got {value!r}") from exc
    if not amount.is_finite():
        raise ValueError(f"Expected a finite USD value, got {value!r}")
    return f"${amount:,.4f}"


def _format_rate(value: Any) -> str:
    if value is None:
        return "—"
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric token rate, got {value!r}") from exc
    if not amount.is_finite():
        raise ValueError(f"Expected a finite token rate, got {value!r}")
    return f"${amount:,.3f}"


def _report_mapping(
    value: Any,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _display_title(value: Any, configured: Mapping[str, str]) -> str:
    raw = str(value)
    return configured.get(raw, raw.replace("_", " ").capitalize())


def render_cost_report_markdown(report: Mapping[str, Any]) -> str:
    """Render deterministic Markdown from ``estimate_interactive_pilot`` data."""

    assumptions = _report_mapping(report.get("assumptions"), "report.assumptions")
    models = _report_mapping(report.get("models"), "report.models")
    teacher = _report_mapping(models.get("teacher"), "report.models.teacher")
    student = _report_mapping(models.get("student"), "report.models.student")
    scenario = _report_mapping(report.get("scenario"), "report.scenario")
    totals = _report_mapping(report.get("totals"), "report.totals")
    standalone_estimates = _report_mapping(
        report.get("standalone_estimates"),
        "report.standalone_estimates",
    )
    seed_preparation = _report_mapping(
        standalone_estimates.get("seed_evaluation_preparation"),
        "report.standalone_estimates.seed_evaluation_preparation",
    )
    seed_quantities = _report_mapping(
        seed_preparation.get("quantities"),
        "report.standalone_estimates.seed_evaluation_preparation.quantities",
    )
    seed_costs = _report_mapping(
        seed_preparation.get("costs"),
        "report.standalone_estimates.seed_evaluation_preparation.costs",
    )
    components = report.get("components")
    if not isinstance(components, list):
        raise ValueError("report.components must be a list")

    lines = [
        "# Interactive pilot cost estimate",
        "",
        "> **Planning estimate:** generated from versioned assumptions; this is "
        "not observed billing or a provider quote.",
        "",
        "## Assumptions and provenance",
        "",
        f"- Assumption set: `{_markdown_escape(assumptions.get('id', ''))}`",
        f"- Description: {_markdown_escape(assumptions.get('description', ''))}",
        f"- Assumptions schema: v{_markdown_escape(assumptions.get('schema_version', ''))}",
        f"- Formula version: v{_markdown_escape(report.get('formula_version', ''))}",
        f"- Assumptions fingerprint: `{_markdown_escape(assumptions.get('fingerprint_sha256', ''))}`",
        (
            "- Pricing source: "
            f"[{_markdown_escape(assumptions.get('pricing_source_url', ''))}]"
            f"({_markdown_escape(assumptions.get('pricing_source_url', ''))})"
        ),
        f"- Pricing effective: {_markdown_escape(assumptions.get('pricing_effective_date', ''))}",
        f"- Pricing accessed: {_markdown_escape(assumptions.get('pricing_accessed_date', ''))}",
        "",
        "## Scenario",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
    ]
    scenario_labels = (
        ("unique_cases", "Unique clinical cases"),
        ("teacher_rollouts", "Accepted teacher consultations"),
        ("mean_gp_actions_per_rollout", "Mean GP actions per consultation"),
        ("supervised_action_targets", "Supervised action targets"),
        ("evaluation_cases", "Evaluation cases"),
        ("on_policy_rollouts", "On-policy consultations"),
    )
    for key, label in scenario_labels:
        lines.append(
            f"| {label} | {_format_report_number(scenario.get(key, ''))} |"
        )

    lines.extend(
        [
            "",
            "## Standalone seed evaluation preparation",
            "",
            "> **Not included in the 500-case pilot totals below.** This "
            "separately estimates preparation of the actual 100-case seed "
            "evaluation set.",
            "",
            f"- Estimate version: v{_format_report_number(seed_preparation.get('version', ''))}",
            f"- Seed cases: **{_format_report_number(seed_preparation.get('cases', ''))}**",
            f"- Content review: {_format_report_number(seed_quantities.get('content_review_minutes_per_case', ''))} minutes per case",
            (
                "- Independent labels: "
                f"{_format_report_number(seed_quantities.get('independent_labels_per_case', ''))} "
                "per case × "
                f"{_format_report_number(seed_quantities.get('minutes_per_label', ''))} minutes"
            ),
            (
                "- Adjudication: "
                f"{_format_report_number(seed_quantities.get('adjudication_fraction', ''))} "
                "of cases × "
                f"{_format_report_number(seed_quantities.get('minutes_per_adjudication', ''))} minutes"
            ),
            f"- Clinician time: {_format_report_number(seed_quantities.get('total_clinician_hours', ''))} hours at {_format_usd(seed_quantities.get('clinician_hourly_rate_usd'))}/hour",
            f"- Clinician review: **{_format_usd(seed_costs.get('clinician_review_usd'))}**",
            f"- Deterministic local build and provider compute: **{_format_usd(seed_costs.get('local_build_and_provider_compute_usd'))}**",
            f"- Standalone seed preparation total: **{_format_usd(seed_costs.get('total_usd'))}**",
            f"- Engineering cost: **{_markdown_escape(seed_preparation.get('engineering_cost', ''))}**",
            "",
            f"Formula: `{_markdown_escape(seed_preparation.get('formula', ''))}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Models and unit prices",
            "",
            "Prices are USD per million tokens.",
            "",
            "| Role | Model | Prefill | Cached prefill | Sample | Train |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for role, values in (("Teacher", teacher), ("Student", student)):
        lines.append(
            "| "
            + " | ".join(
                (
                    role,
                    f"`{_markdown_escape(values.get('model_id', ''))}`",
                    _format_rate(values.get("prefill_usd_per_million")),
                    _format_rate(values.get("cached_prefill_usd_per_million")),
                    _format_rate(values.get("sample_usd_per_million")),
                    _format_rate(values.get("train_usd_per_million")),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Cost summary",
            "",
            "| Component | Category | Estimated cost |",
            "| --- | --- | ---: |",
        ]
    )
    parsed_components: list[Mapping[str, Any]] = []
    for index, raw_component in enumerate(components):
        component = _report_mapping(
            raw_component, f"report.components[{index}]"
        )
        parsed_components.append(component)
        label = _display_title(component.get("id", ""), _COMPONENT_TITLES)
        category = _display_title(
            component.get("category", ""), _CATEGORY_TITLES
        )
        lines.append(
            f"| {_markdown_escape(label)} | {_markdown_escape(category)} | "
            f"{_format_usd(component.get('cost_usd'))} |"
        )

    lines.extend(
        [
            "",
            f"- Tinker compute: **{_format_usd(totals.get('tinker_compute_usd'))}**",
            f"- Clinician review: **{_format_usd(totals.get('clinician_review_usd'))}**",
            f"- Overall estimate: **{_format_usd(totals.get('overall_usd'))}**",
            "",
            "## Formula details",
        ]
    )
    for component in parsed_components:
        component_id = str(component.get("id", ""))
        title = _display_title(component_id, _COMPONENT_TITLES)
        formula = _markdown_escape(component.get("formula", ""))
        quantities = _report_mapping(
            component.get("quantities"),
            f"report.components[{component_id}].quantities",
        )
        lines.extend(
            [
                "",
                f"### {_markdown_escape(title)}",
                "",
                f"Formula: `{formula}`",
                "",
                "| Quantity | Value |",
                "| --- | ---: |",
            ]
        )
        for key in sorted(quantities):
            value = quantities[key]
            rendered_value = (
                _format_usd(value)
                if key.endswith("_usd")
                else _format_report_number(value)
            )
            lines.append(
                f"| `{_markdown_escape(key)}` | {rendered_value} |"
            )

    lines.extend(
        [
            "",
            "## Scope note",
            "",
            "The estimate includes only the itemized Tinker token operations and "
            "clinician-review workload. It excludes unmodeled engineering, data "
            "acquisition, taxes, and production serving costs.",
            "",
        ]
    )
    return "\n".join(lines)


def write_cost_report(
    output_dir: Path,
    assumptions_path: Path | None = None,
) -> list[Path]:
    """Write byte-stable JSON and Markdown planning artifacts."""

    report = estimate_interactive_pilot(assumptions_path=assumptions_path)
    json_text = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    markdown_text = render_cost_report_markdown(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cost_estimate.json"
    markdown_path = output_dir / "cost_estimate.md"
    json_path.write_text(json_text, encoding="utf-8", newline="\n")
    markdown_path.write_text(markdown_text, encoding="utf-8", newline="\n")
    return [json_path, markdown_path]
