"""Small provider-neutral completion contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from acuitybench.models import ModelConfig


@dataclass(frozen=True)
class CompletionResult:
    text: str
    finish_reason: str
    response_id: str | None = None
    returned_model: str | None = None
    system_fingerprint: str | None = None
    request_id: str | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    rate_limit: dict[str, str] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    async def complete(
        self,
        *,
        config: ModelConfig,
        messages: list[dict[str, str]],
        max_output_tokens: int | None = None,
    ) -> CompletionResult: ...

    async def close(self) -> None: ...
