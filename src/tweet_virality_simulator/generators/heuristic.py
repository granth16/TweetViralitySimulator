"""Heuristic rewrite generator — the zero-setup default.

Applies well-known viral *formats* to the user's own message (contrarian hook,
curiosity gap, question CTA, link-fix, tightening) rather than inventing new
content. No API key, no model — so ``tvs improve`` works out of the box. The
simulator then decides which transformation actually helps for this tweet.

This is a *prior*, not a writer: it reframes, it doesn't fabricate claims.
"""

from __future__ import annotations

import re
from typing import Callable, List

_URL_RE = re.compile(r"\s*(https?://\S+|www\.\S+)")
_TRAILING_HASHTAGS_RE = re.compile(r"(\s+#\w+)+\s*$")
_WS_RE = re.compile(r"\s+")

# Openers we shouldn't stack on top of (already a hook).
_EXISTING_HOOKS = (
    "unpopular opinion", "hot take", "nobody talks about", "stop scrolling",
    "here's why", "heres why", "here's how", "heres how", "the truth about",
)


def _core(text: str) -> str:
    """The user's message with links and trailing hashtags stripped, normalized."""
    t = _URL_RE.sub("", text)
    t = _TRAILING_HASHTAGS_RE.sub("", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


def _strip_trailing_punct(s: str) -> str:
    return s.rstrip(" .!?,;:")


class HeuristicGenerator:
    name = "heuristic"

    def generate(self, text: str, n: int) -> List[str]:
        core = _core(text)
        if not core:
            return []
        had_link = bool(_URL_RE.search(text))
        low = core.lower()
        already_hooked = any(h in low for h in _EXISTING_HOOKS)
        stem = _strip_trailing_punct(core)

        # Each transform is a (condition, builder) pair; applied in priority order.
        transforms: List[Callable[[], str]] = []

        if not already_hooked:
            transforms.append(lambda: f"Unpopular opinion: {_lower_first(core)}")
            transforms.append(lambda: f"Nobody talks about this, but {_lower_first(core)}")
            transforms.append(lambda: f"Hot take: {_lower_first(stem)}. Change my mind.")

        # Question / engagement CTA (replies are the highest-weighted signal).
        if "?" not in core:
            transforms.append(lambda: f"{stem}. Agree or disagree?")

        # Link-fix: move the URL out of the post (X downranks outbound links).
        if had_link:
            transforms.append(lambda: f"{stem}. (link in the replies 👇)")

        # Curiosity gap.
        if not already_hooked:
            transforms.append(lambda: f"Here's something nobody tells you: {_lower_first(core)}")

        # Thread / story opener for longer messages.
        if len(stem) >= 60:
            transforms.append(lambda: f"{stem}. Here's what I learned 🧵")

        # Tighten: keep just the first sentence if there are several.
        first = re.split(r"(?<=[.!?])\s+", stem)
        if len(first) > 1 and len(first[0]) >= 20:
            transforms.append(lambda: f"{_strip_trailing_punct(first[0])}.")

        # Pattern-interrupt opener.
        if not already_hooked:
            transforms.append(lambda: f"Stop scrolling. {core}")

        out: List[str] = []
        seen = {text.strip().lower(), core.lower()}
        for build in transforms:
            cand = _WS_RE.sub(" ", build()).strip()
            key = cand.lower()
            if cand and key not in seen:
                seen.add(key)
                out.append(cand)
            if len(out) >= n:
                break
        return out
