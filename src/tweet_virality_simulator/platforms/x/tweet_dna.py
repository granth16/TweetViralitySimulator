"""Extract Tweet DNA: structural features (always) + semantic scores (provider)."""

from __future__ import annotations

import re

from ...embedding import embed_text
from ...models import SEMANTIC_DIMS, TweetDNA
from ...providers import HeuristicProvider
from ...providers.base import LLMProvider

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"(?<!\w)@\w+")
_HASHTAG_RE = re.compile(r"(?<!\w)#\w+")

_HEURISTIC = HeuristicProvider()


def extract(text: str, provider: LLMProvider, has_media: bool = False) -> TweetDNA:
    """Build a TweetDNA from raw tweet text.

    Semantic dims come from ``provider``; if it returns ``None`` (unavailable or
    error) we fall back to the heuristic scorer so extraction never fails.
    """
    scores = None
    if provider is not None and provider.name != "heuristic":
        scores = provider.score_tweet(text)
        scored_by = provider.name if scores is not None else "heuristic"
    else:
        scored_by = "heuristic"

    if scores is None:
        scores = _HEURISTIC.score_tweet(text)

    # Guard against a malformed provider response.
    scores = {d: float(max(0.0, min(1.0, scores.get(d, 0.0)))) for d in SEMANTIC_DIMS}

    return TweetDNA(
        text=text,
        length_chars=len(text),
        num_hashtags=len(_HASHTAG_RE.findall(text)),
        num_mentions=len(_MENTION_RE.findall(text)),
        has_link=bool(_URL_RE.search(text)),
        has_media=has_media,
        is_question="?" in text,
        topic_vector=embed_text(text).tolist(),
        scored_by=scored_by,
        **scores,
    )
