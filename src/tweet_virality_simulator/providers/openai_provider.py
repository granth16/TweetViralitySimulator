"""Optional OpenAI-backed tweet scorer.

Returns ``None`` on any failure so the engine falls back to the heuristic
provider. Requires ``pip install tweet-virality-simulator[llm]`` and
``OPENAI_API_KEY``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from ..models import SEMANTIC_DIMS
from .base import LLMProvider

_PROMPT = (
    "You score a tweet for its viral potential on X. Return ONLY a JSON object "
    "with these keys, each a float in 0..1: "
    + ", ".join(SEMANTIC_DIMS)
    + ". No markdown, no explanation.\n\nTweet:\n"
)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.getenv("TVS_OPENAI_MODEL", "gpt-4o-mini")
        self._client = None

    def available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # imported lazily

            self._client = OpenAI()
        return self._client

    def score_tweet(self, text: str) -> Optional[Dict[str, float]]:
        if not self.available():
            return None
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": _PROMPT + text}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            return {d: float(max(0.0, min(1.0, data.get(d, 0.0)))) for d in SEMANTIC_DIMS}
        except Exception:
            return None
