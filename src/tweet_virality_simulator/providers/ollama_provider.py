"""Optional Ollama (local model) tweet scorer.

Uses only the standard library (urllib) so there is no extra dependency. Returns
``None`` on any failure so the engine falls back to the heuristic provider.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, Optional

from ..models import SEMANTIC_DIMS
from .base import LLMProvider

_PROMPT = (
    "You score a tweet for its viral potential on X. Return ONLY a JSON object "
    "with these keys, each a float in 0..1: "
    + ", ".join(SEMANTIC_DIMS)
    + ". No markdown, no explanation.\n\nTweet:\n"
)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None) -> None:
        self.model = model or os.getenv("TVS_OLLAMA_MODEL", "llama3.1")
        self.host = (host or os.getenv("TVS_OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def score_tweet(self, text: str) -> Optional[Dict[str, float]]:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": _PROMPT + text,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.host + "/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            data = json.loads(body.get("response", "{}"))
            return {d: float(max(0.0, min(1.0, data.get(d, 0.0)))) for d in SEMANTIC_DIMS}
        except (urllib.error.URLError, ValueError, KeyError, TimeoutError):
            return None
