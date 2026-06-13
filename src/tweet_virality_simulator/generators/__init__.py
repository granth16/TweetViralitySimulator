"""Rewrite-generator registry and factory."""

from __future__ import annotations

from .base import TweetGenerator
from .heuristic import HeuristicGenerator

_LLM_PROVIDERS = {"openai", "ollama", "compat", "openai_compatible", "custom"}


def get_generator(provider: str = "heuristic") -> TweetGenerator:
    """Pick a generator for the given provider.

    LLM providers generate real rewrites when configured; otherwise (or on any
    failure) we fall back to the zero-setup heuristic generator so ``improve``
    always works.
    """
    name = (provider or "heuristic").lower()
    if name in _LLM_PROVIDERS:
        from .llm import LLMGenerator

        gen = LLMGenerator(name)
        if gen.available():
            return gen
    return HeuristicGenerator()


__all__ = ["TweetGenerator", "HeuristicGenerator", "get_generator"]
