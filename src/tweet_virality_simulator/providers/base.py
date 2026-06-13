"""LLM provider interface — the firewall.

The engine only ever asks a provider to score a tweet's semantic dimensions.
Everything else (structural features, the simulation) is provider-independent.
A future trained model is just another provider behind this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def score_tweet(self, text: str) -> Optional[Dict[str, float]]:
        """Return a dict of semantic dimension -> score in 0..1.

        Return ``None`` if scoring is unavailable (e.g. no key / network error);
        the engine will fall back to the heuristic provider.
        """
        raise NotImplementedError

    def available(self) -> bool:
        return True
