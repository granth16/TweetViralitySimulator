"""Heuristic (no-LLM) tweet scorer — the zero-setup default.

Deterministic lexicon + rule based scoring of the semantic dimensions. Lets the
whole tool run in seconds with no API key. Grounded in well-known findings
(e.g. moral-emotional language and questions/hooks drive engagement); it is a
*prior*, not a calibrated model.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Optional, Set

from ..embedding import tokens
from ..models import SEMANTIC_DIMS
from .base import LLMProvider

_EMOTION: Set[str] = {
    "amazing", "incredible", "shocking", "shocked", "insane", "unbelievable",
    "terrifying", "beautiful", "disgusting", "furious", "heartbreaking",
    "outrageous", "mindblowing", "scary", "wow", "omg", "crying", "obsessed",
    "devastating", "thrilled", "love", "hate", "angry", "fear", "joy", "happy",
    "sad", "stunning", "epic", "horrible", "wonderful", "hilarious", "cringe",
    "heartwarming", "rage", "tears", "goosebumps", "speechless",
}

_CONTROVERSY: Set[str] = {
    "wrong", "debate", "unpopular", "controversial", "overrated", "underrated",
    "stop", "never", "always", "fake", "lie", "lies", "scam", "truth", "banned",
    "cancel", "woke", "politics", "actually", "hot", "take", "agree", "disagree",
    "nobody", "everyone", "honestly", "problem", "myth", "exposed",
}

_HOOK: Set[str] = {
    "how", "why", "nobody", "secret", "secrets", "stop", "heres", "truth",
    "mistake", "mistakes", "before", "after", "what", "things", "reasons",
    "ways", "hack", "hacks", "trick", "tricks", "this", "you", "your",
    "warning", "finally", "guide", "steps",
}

_NOVELTY: Set[str] = {
    "new", "first", "never", "breakthrough", "launched", "introducing",
    "finally", "announcing", "revealed", "discovered", "just", "released",
    "unveiled", "debut", "world", "ever",
}

_RELATABLE: Set[str] = {
    "you", "your", "youre", "we", "us", "i", "me", "my", "everyone", "nobody",
    "weve", "us", "our", "relatable", "same", "literally",
}

_STOP: Set[str] = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "is",
    "it", "this", "that", "with", "as", "at", "by", "be", "are", "was", "im",
}

_URL_RE = re.compile(r"https?://|www\.")
_NUM_RE = re.compile(r"\d")

# High-signal viral formats / openers (matched as substrings on lowercased text).
_VIRAL_PHRASES = [
    "unpopular opinion", "hot take", "fight me", "change my mind", "stop scrolling",
    "this is why", "here's why", "heres why", "here's how", "heres how",
    "nobody talks about", "no one talks about", "nobody tells you", "what nobody",
    "the truth about", "i was wrong", "thread", "🧵", "a story", "let me explain",
    "you're doing", "youre doing", "you've been", "youve been",
]


def _sig(x: float, k: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-k * x))


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


class HeuristicProvider(LLMProvider):
    name = "heuristic"

    def score_tweet(self, text: str) -> Optional[Dict[str, float]]:
        toks = tokens(text)
        n = max(len(toks), 1)
        tset = set(toks)

        def frac(words: Set[str]) -> float:
            return sum(1 for t in toks if t in words) / n

        emotion = frac(_EMOTION)
        controversy = frac(_CONTROVERSY)
        novelty = frac(_NOVELTY)
        relatable = frac(_RELATABLE)
        hook_words = frac(_HOOK)

        low = text.lower()
        phrase_hits = sum(1 for p in _VIRAL_PHRASES if p in low)

        exclaim = text.count("!")
        question = 1.0 if "?" in text else 0.0
        starts_number = 1.0 if toks and _NUM_RE.search(toks[0]) else 0.0
        caps_words = sum(1 for w in text.split() if len(w) > 2 and w.isupper())
        has_link = bool(_URL_RE.search(text))
        n_hash = text.count("#")

        length = len(text)

        # hook: punchy openers, viral formats, questions, hook lexicon, caps
        hook = _clip01(
            0.9 * hook_words
            + 0.30 * min(phrase_hits, 2)
            + 0.25 * question
            + 0.2 * starts_number
            + 0.12 * min(caps_words, 3)
            + (0.15 if length <= 120 else 0.0)
        )

        emotional = _clip01(2.2 * emotion + 0.12 * min(exclaim, 3) + 0.12 * phrase_hits)
        contro = _clip01(2.4 * controversy + 0.1 * question + 0.18 * phrase_hits)
        nov = _clip01(2.6 * novelty)
        rel = _clip01(1.8 * relatable)

        # clarity: shorter, fewer hashtags, fewer links = clearer
        clarity = _clip01(
            1.0
            - max(0.0, (length - 100) / 220.0)
            - 0.12 * n_hash
            - (0.15 if has_link else 0.0)
        )

        content_tokens = sum(1 for t in toks if t not in _STOP)
        info = _clip01(
            0.55 * (content_tokens / n)
            + (0.25 if _NUM_RE.search(text) else 0.0)
        )

        return {
            "hook_strength": hook,
            "emotional_intensity": emotional,
            "controversy": contro,
            "novelty": nov,
            "clarity": clarity,
            "relatability": rel,
            "information_density": info,
        }


# sanity: keep the dim list and scorer in lockstep
assert set(HeuristicProvider().score_tweet("test").keys()) == set(SEMANTIC_DIMS)
