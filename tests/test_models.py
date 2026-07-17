from __future__ import annotations

from pathlib import Path

import pytest

from acuitybench.models import ModelConfig, ModelRegistry
from acuitybench.providers import get_provider
from acuitybench.providers.openai import OpenAIProvider


def _write_registry(
    path: Path,
    *,
    api_model: str = "gpt-5.4",
    reasoning_effort: str | None,
    send_temperature: bool = True,
) -> Path:
    reasoning_line = (
        "" if reasoning_effort is None else f"    reasoning_effort: {reasoning_effort}\n"
    )
    path.write_text(
        "version: 1\n"
        "models:\n"
        "  candidate:\n"
        "    display_name: Candidate\n"
        "    provider: openai\n"
        f"    api_model: {api_model}\n"
        "    endpoint: chat_completions\n"
        "    api_key_env: OPENAI_API_KEY\n"
        "    temperature: 1.0\n"
        f"    send_temperature: {str(send_temperature).lower()}\n"
        "    max_output_tokens: 4096\n"
        f"{reasoning_line}"
        "    token_parameter: max_completion_tokens\n"
        "    input_cost_per_million: 2.5\n"
        "    cached_input_cost_per_million: 0.25\n"
        "    output_cost_per_million: 15.0\n"
        "judges: {}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("api_model", ["gpt-5.4", "gpt-5.4-2026-03-05"])
@pytest.mark.parametrize("reasoning_effort", [None, "low", "medium", "high"])
def test_gpt_5_4_rejects_temperature_with_non_none_reasoning(
    tmp_path: Path,
    api_model: str,
    reasoning_effort: str | None,
) -> None:
    path = _write_registry(
        tmp_path / "models.yaml",
        api_model=api_model,
        reasoning_effort=reasoning_effort,
    )

    with pytest.raises(ValueError, match="reasoning_effort must be 'none'"):
        ModelRegistry(path)


def test_gpt_5_4_accepts_temperature_with_none_reasoning(tmp_path: Path) -> None:
    path = _write_registry(
        tmp_path / "models.yaml",
        reasoning_effort="none",
    )

    model = ModelRegistry(path).get("candidate")

    assert model.send_temperature is True
    assert model.reasoning_effort == "none"


def test_gpt_5_4_allows_other_effort_when_temperature_is_omitted(
    tmp_path: Path,
) -> None:
    path = _write_registry(
        tmp_path / "models.yaml",
        reasoning_effort="medium",
        send_temperature=False,
    )

    model = ModelRegistry(path).get("candidate")

    assert model.send_temperature is False
    assert model.reasoning_effort == "medium"


def test_legacy_model_config_payload_and_fingerprint_remain_stable() -> None:
    model = ModelConfig(
        id="gpt-4.1",
        display_name="GPT-4.1",
        provider="openai",
        api_model="gpt-4.1",
        endpoint="chat_completions",
        api_key_env="OPENAI_API_KEY",
        temperature=1.0,
        send_temperature=True,
        max_output_tokens=4096,
        token_parameter="max_tokens",
        input_cost_per_million=2.0,
        cached_input_cost_per_million=0.5,
        output_cost_per_million=8.0,
    )

    assert model.as_dict() == {
        "id": "gpt-4.1",
        "display_name": "GPT-4.1",
        "provider": "openai",
        "api_model": "gpt-4.1",
        "endpoint": "chat_completions",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 1.0,
        "send_temperature": True,
        "max_output_tokens": 4096,
        "token_parameter": "max_tokens",
        "input_cost_per_million": 2.0,
        "cached_input_cost_per_million": 0.5,
        "output_cost_per_million": 8.0,
    }
    assert (
        model.fingerprint
        == "7cee1618dbb0ece71704bce7d69f95eb6ba0d91dbd4c27ae1045860346d33b25"
    )


def test_openai_compatible_student_profile_requires_endpoint_provenance(
    tmp_path: Path,
) -> None:
    path = _write_registry(
        tmp_path / "models.yaml",
        api_model="student-checkpoint",
        reasoning_effort=None,
        send_temperature=True,
    )
    text = path.read_text(encoding="utf-8").replace(
        "    provider: openai\n",
        "    provider: openai_compatible\n"
        "    base_url_env: STUDENT_BASE_URL\n"
        "    deployment: vllm-a100-fixture\n",
    )
    path.write_text(text, encoding="utf-8")

    model = ModelRegistry(path).get("candidate")

    assert model.base_url_env == "STUDENT_BASE_URL"
    assert model.deployment == "vllm-a100-fixture"
    assert isinstance(get_provider(model.provider), OpenAIProvider)


def test_openai_compatible_student_profile_rejects_missing_base_url(
    tmp_path: Path,
) -> None:
    path = _write_registry(
        tmp_path / "models.yaml",
        api_model="student-checkpoint",
        reasoning_effort=None,
    )
    text = path.read_text(encoding="utf-8").replace(
        "    provider: openai\n",
        "    provider: openai_compatible\n"
        "    deployment: vllm-a100-fixture\n",
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="must set base_url_env"):
        ModelRegistry(path)
