"""Provider adapter registry."""

from __future__ import annotations

from acuitybench.providers.base import CompletionResult, Provider


def get_provider(name: str) -> Provider:
    if name == "openai":
        from acuitybench.providers.openai import OpenAIProvider

        return OpenAIProvider()
    raise ValueError(
        f"No provider adapter for {name!r}. Add one in acuitybench/providers/ "
        "and register it in get_provider()."
    )


__all__ = ["CompletionResult", "Provider", "get_provider"]
