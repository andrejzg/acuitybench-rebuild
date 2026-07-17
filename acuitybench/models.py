"""Configuration-driven model and judge registry."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from acuitybench.sources import project_root


_GPT_5_4_MODEL = re.compile(r"gpt-5\.4(?:-\d{4}-\d{2}-\d{2})?")


@dataclass(frozen=True)
class ModelConfig:
    id: str
    display_name: str
    provider: str
    api_model: str
    endpoint: str
    api_key_env: str
    temperature: float | None
    send_temperature: bool
    max_output_tokens: int
    token_parameter: str
    input_cost_per_million: float
    cached_input_cost_per_million: float
    output_cost_per_million: float
    reasoning_effort: str | None = None
    reasoning_effort_basis: str | None = None
    service_tier: str | None = None
    max_retry_output_tokens: int | None = None
    base_url_env: str | None = None
    deployment: str | None = None

    def as_dict(self) -> dict[str, Any]:
        # Keep historical fingerprints stable when newly supported request
        # controls are absent from an older/non-reasoning model profile.
        payload = asdict(self)
        for key in (
            "reasoning_effort",
            "reasoning_effort_basis",
            "service_tier",
            "max_retry_output_tokens",
            "base_url_env",
            "deployment",
        ):
            if payload[key] is None:
                payload.pop(key)
        return payload

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class JudgeConfig:
    id: str
    display_name: str
    model: ModelConfig

    @property
    def fingerprint(self) -> str:
        payload = {
            "id": self.id,
            "display_name": self.display_name,
            "model": self.model.as_dict(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class ModelRegistry:
    """Validated models.yaml reader.

    Model IDs are user-facing aliases; api_model is the provider's request name.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or project_root() / "configs/models.yaml"
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        if raw.get("version") != 1:
            raise ValueError(f"Unsupported model registry version in {self.path}")
        self._raw = raw
        self._models = self._parse_models(raw.get("models", {}))
        self._judges = self._parse_judges(raw.get("judges", {}))

    @staticmethod
    def _parse_models(raw_models: dict[str, Any]) -> dict[str, ModelConfig]:
        if not isinstance(raw_models, dict) or not raw_models:
            raise ValueError("models.yaml must define at least one model")
        models: dict[str, ModelConfig] = {}
        required = {
            "display_name",
            "provider",
            "api_model",
            "endpoint",
            "api_key_env",
            "max_output_tokens",
            "token_parameter",
            "input_cost_per_million",
            "cached_input_cost_per_million",
            "output_cost_per_million",
        }
        for model_id, values in raw_models.items():
            missing = required - set(values or {})
            if missing:
                raise ValueError(f"Model {model_id!r} is missing: {sorted(missing)}")
            reasoning_effort = values.get("reasoning_effort")
            if reasoning_effort is not None:
                reasoning_effort = str(reasoning_effort)
                allowed_efforts = {
                    "none",
                    "minimal",
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                }
                if reasoning_effort not in allowed_efforts:
                    raise ValueError(
                        f"Model {model_id!r} has unsupported reasoning_effort "
                        f"{reasoning_effort!r}; expected one of "
                        f"{sorted(allowed_efforts)}"
                    )
            api_model = str(values["api_model"])
            provider = str(values["provider"])
            base_url_env = (
                None
                if values.get("base_url_env") is None
                else str(values["base_url_env"])
            )
            deployment = (
                None
                if values.get("deployment") is None
                else str(values["deployment"])
            )
            if provider == "openai_compatible" and not base_url_env:
                raise ValueError(
                    f"Model {model_id!r} uses openai_compatible and must set "
                    "base_url_env"
                )
            if provider == "openai_compatible" and not deployment:
                raise ValueError(
                    f"Model {model_id!r} uses openai_compatible and must set "
                    "a stable deployment description"
                )
            send_temperature = bool(values.get("send_temperature", True))
            if (
                _GPT_5_4_MODEL.fullmatch(api_model)
                and send_temperature
                and reasoning_effort != "none"
            ):
                raise ValueError(
                    f"Model {model_id!r} sends temperature to {api_model!r}, "
                    "so reasoning_effort must be 'none' for current OpenAI "
                    "API compatibility"
                )
            max_output_tokens = int(values["max_output_tokens"])
            max_retry_output_tokens = (
                None
                if values.get("max_retry_output_tokens") is None
                else int(values["max_retry_output_tokens"])
            )
            if (
                max_retry_output_tokens is not None
                and max_retry_output_tokens < max_output_tokens
            ):
                raise ValueError(
                    f"Model {model_id!r} max_retry_output_tokens cannot be "
                    "lower than max_output_tokens"
                )
            models[model_id] = ModelConfig(
                id=model_id,
                display_name=str(values["display_name"]),
                provider=provider,
                api_model=api_model,
                endpoint=str(values["endpoint"]),
                api_key_env=str(values["api_key_env"]),
                temperature=(
                    None
                    if values.get("temperature") is None
                    else float(values["temperature"])
                ),
                send_temperature=send_temperature,
                max_output_tokens=max_output_tokens,
                token_parameter=str(values["token_parameter"]),
                input_cost_per_million=float(values["input_cost_per_million"]),
                cached_input_cost_per_million=float(
                    values["cached_input_cost_per_million"]
                ),
                output_cost_per_million=float(values["output_cost_per_million"]),
                reasoning_effort=reasoning_effort,
                reasoning_effort_basis=(
                    None
                    if values.get("reasoning_effort_basis") is None
                    else str(values["reasoning_effort_basis"])
                ),
                service_tier=(
                    None
                    if values.get("service_tier") is None
                    else str(values["service_tier"])
                ),
                max_retry_output_tokens=max_retry_output_tokens,
                base_url_env=base_url_env,
                deployment=deployment,
            )
        return models

    def _parse_judges(self, raw_judges: dict[str, Any]) -> dict[str, JudgeConfig]:
        judges: dict[str, JudgeConfig] = {}
        for judge_id, values in raw_judges.items():
            base = self.get(str(values["model"]))
            configured = replace(
                base,
                temperature=float(values.get("temperature", 0)),
                send_temperature=True,
                max_output_tokens=int(values.get("max_output_tokens", 1024)),
            )
            judges[judge_id] = JudgeConfig(
                id=judge_id,
                display_name=str(values.get("display_name", judge_id)),
                model=configured,
            )
        return judges

    def get(self, model_id: str) -> ModelConfig:
        try:
            return self._models[model_id]
        except KeyError as exc:
            choices = ", ".join(sorted(self._models))
            raise ValueError(f"Unknown model {model_id!r}. Configured: {choices}") from exc

    def get_judge(self, judge_id: str) -> JudgeConfig:
        try:
            return self._judges[judge_id]
        except KeyError as exc:
            choices = ", ".join(sorted(self._judges))
            raise ValueError(f"Unknown judge {judge_id!r}. Configured: {choices}") from exc

    def models(self) -> tuple[ModelConfig, ...]:
        return tuple(self._models[key] for key in sorted(self._models))

    def judges(self) -> tuple[JudgeConfig, ...]:
        return tuple(self._judges[key] for key in sorted(self._judges))
