"""OpenAI-compatible provider — the trained-model attach point.

Points at *any* endpoint that speaks the OpenAI Chat Completions API: a local
vLLM/Ollama-compat server, a hosted gateway, or the private backend's trained
reaction/persona model. Flip ``config.provider = "compat"`` and set the env
vars below — the engine is unchanged and never learns whether it's a generic
LLM or our custom model.

    TVS_COMPAT_BASE_URL   e.g. https://api.internal/v1   (required)
    TVS_COMPAT_API_KEY    bearer token                   (optional)
    TVS_COMPAT_MODEL      model name                      (default: tvs-reaction)

Uses only the standard library so it adds no dependency. Returns ``None`` on any
failure so the engine falls back to the heuristic provider.
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


class OpenAICompatibleProvider(LLMProvider):
    name = "compat"

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("TVS_COMPAT_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("TVS_COMPAT_API_KEY", "")
        self.model = model or os.getenv("TVS_COMPAT_MODEL", "tvs-reaction")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.base_url)

    def score_tweet(self, text: str) -> Optional[Dict[str, float]]:
        if not self.available():
            return None
        url = f"{self.base_url}/chat/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": _PROMPT + text}],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"] or "{}"
            data = json.loads(content)
            return {d: float(max(0.0, min(1.0, data.get(d, 0.0)))) for d in SEMANTIC_DIMS}
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError):
            return None
