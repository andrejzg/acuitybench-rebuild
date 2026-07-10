"""Configuration-driven model and judge registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from acuitybench.sources import project_root


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

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
            models[model_id] = ModelConfig(
                id=model_id,
                display_name=str(values["display_name"]),
                provider=str(values["provider"]),
                api_model=str(values["api_model"]),
                endpoint=str(values["endpoint"]),
                api_key_env=str(values["api_key_env"]),
                temperature=(
                    None
                    if values.get("temperature") is None
                    else float(values["temperature"])
                ),
                send_temperature=bool(values.get("send_temperature", True)),
                max_output_tokens=int(values["max_output_tokens"]),
                token_parameter=str(values["token_parameter"]),
                input_cost_per_million=float(values["input_cost_per_million"]),
                cached_input_cost_per_million=float(
                    values["cached_input_cost_per_million"]
                ),
                output_cost_per_million=float(values["output_cost_per_million"]),
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
