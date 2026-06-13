"""Provider registry and factory."""

from __future__ import annotations

from .base import LLMProvider
from .heuristic import HeuristicProvider


def get_provider(name: str) -> LLMProvider:
    """Return a provider by name. Unknown names fall back to heuristic."""
    name = (name or "heuristic").lower()
    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider()
    if name == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider()
    if name in ("compat", "openai_compatible", "custom"):
        from .openai_compatible import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider()
        # Fall back to heuristic if no endpoint is configured.
        return provider if provider.available() else HeuristicProvider()
    return HeuristicProvider()


__all__ = ["LLMProvider", "HeuristicProvider", "get_provider"]
