"""Tweet generator interface — produces candidate rewrites to be simulated.

A generator only *proposes* text; it never decides what's good. The engine
scores every candidate, so generation quality is judged by the (calibrated)
simulator, not by the generator's own confidence. A trained rewrite model is
just another generator behind this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class TweetGenerator(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, text: str, n: int) -> List[str]:
        """Return up to ``n`` distinct rewrite candidates (excluding the original)."""
        raise NotImplementedError

    def available(self) -> bool:
        return True
