"""LLM-backed rewrite generator (optional).

Asks an LLM for several viral rewrites of a tweet. Works with the same backends
as the scorers — ``openai``, ``ollama``, or any ``compat`` (OpenAI-compatible)
endpoint, including a private trained rewrite model. Returns ``[]`` on any
failure so ``tvs improve`` falls back to the heuristic generator.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional

_SYS = (
    "You rewrite tweets to maximize organic spread on X without changing the "
    "author's core message or inventing facts. Keep their voice. Prefer a strong "
    "first-line hook, emotional or contrarian framing, and a reason to reply. "
    "Move outbound links to a reply. Keep each under 280 characters."
)


def _user_prompt(text: str, n: int) -> str:
    return (
        f"Rewrite this tweet into {n} distinct higher-spread versions. "
        f'Return ONLY a JSON array of {n} strings, no markdown.\n\nTweet:\n{text}'
    )


def _parse_array(raw: str, n: int) -> List[str]:
    raw = raw.strip()
    # Tolerate models that wrap the array in an object or code fences.
    raw = re.sub(r"^```(?:json)?|```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        return []
    out = [str(x).strip() for x in data if str(x).strip()]
    return out[:n]


class LLMGenerator:
    def __init__(self, provider: str) -> None:
        self.provider = provider.lower()
        self.name = self.provider

    # --- availability ---
    def available(self) -> bool:
        if self.provider == "openai":
            return bool(os.getenv("OPENAI_API_KEY"))
        if self.provider in ("compat", "openai_compatible", "custom"):
            return bool(os.getenv("TVS_COMPAT_BASE_URL"))
        if self.provider == "ollama":
            return True  # assume a local daemon; call will fail-soft otherwise
        return False

    # --- generation ---
    def generate(self, text: str, n: int) -> List[str]:
        try:
            if self.provider == "openai":
                return self._openai(text, n)
            if self.provider == "ollama":
                return self._ollama(text, n)
            return self._compat(text, n)
        except Exception:
            return []

    def _openai(self, text: str, n: int) -> List[str]:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("TVS_OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user", "content": _user_prompt(text, n)},
            ],
            temperature=0.9,
        )
        return _parse_array(resp.choices[0].message.content or "[]", n)

    def _compat(self, text: str, n: int) -> List[str]:
        base = os.getenv("TVS_COMPAT_BASE_URL", "").rstrip("/")
        if not base:
            return []
        headers = {"Content-Type": "application/json"}
        key = os.getenv("TVS_COMPAT_API_KEY", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = json.dumps(
            {
                "model": os.getenv("TVS_COMPAT_MODEL", "tvs-rewrite"),
                "messages": [
                    {"role": "system", "content": _SYS},
                    {"role": "user", "content": _user_prompt(text, n)},
                ],
                "temperature": 0.9,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/chat/completions", data=payload, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return _parse_array(body["choices"][0]["message"]["content"] or "[]", n)

    def _ollama(self, text: str, n: int) -> List[str]:
        host = os.getenv("TVS_OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        payload = json.dumps(
            {
                "model": os.getenv("TVS_OLLAMA_MODEL", "llama3.1"),
                "prompt": _SYS + "\n\n" + _user_prompt(text, n),
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.9},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            host + "/api/generate", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return _parse_array(body.get("response", "[]"), n)
