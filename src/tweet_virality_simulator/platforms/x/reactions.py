"""Per-persona reaction probabilities + the heavy-ranker engagement score.

The LLM (or heuristic) sets the *content appeal* via Tweet DNA; persona traits
and topic affinity modulate it into per-action probabilities. The algorithm's
view of a batch is the heavy-ranker-weighted sum of realized actions.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ...models import TweetDNA
from ...population.audience import Audience
from .weights import HEAVY_RANKER_WEIGHTS


def content_appeal(dna: TweetDNA) -> float:
    """Overall 0..1 appeal of the content, before persona/affinity effects."""
    link_penalty = 0.30 if dna.has_link else 0.0  # X downranks external links
    media_boost = 0.08 if dna.has_media else 0.0
    base = (
        0.26 * dna.hook_strength
        + 0.22 * dna.emotional_intensity
        + 0.18 * dna.novelty
        + 0.16 * dna.controversy
        + 0.10 * dna.relatability
        + 0.08 * dna.clarity
    )
    return float(np.clip(base * 1.7 + media_boost - link_penalty, 0.02, 1.0))


def action_probs(
    dna: TweetDNA,
    audience: Audience,
    idx: np.ndarray,
    tweet_topic: np.ndarray,
    appeal: float,
) -> Dict[str, np.ndarray]:
    """Probabilities of like/retweet/reply/quote for the given exposed nodes."""
    topic = audience.topic[idx]
    cos = topic @ tweet_topic
    affinity = 0.5 + 0.5 * cos  # 0..1

    drive = audience.base_engage[idx] * affinity * appeal
    hook_nov = 0.6 + 0.4 * ((dna.hook_strength + dna.novelty) / 2.0)
    contro = dna.controversy

    p_like = np.clip(drive * audience.like_rate[idx] * 6.0, 0.0, 0.9)
    p_rt = np.clip(drive * audience.share_propensity[idx] * 7.0 * hook_nov, 0.0, 0.6)
    p_reply = np.clip(drive * audience.reply_rate[idx] * 6.0 * (0.35 + 1.3 * contro), 0.0, 0.5)
    p_quote = np.clip(p_rt * (0.15 + 0.9 * contro), 0.0, 0.4)

    return {"like": p_like, "retweet": p_rt, "reply": p_reply, "quote": p_quote}


def engagement_rate(
    n_like: int, n_rt: int, n_reply: int, n_quote: int, exposed: int
) -> float:
    """Heavy-ranker-weighted engagement per exposed account."""
    if exposed <= 0:
        return 0.0
    w = HEAVY_RANKER_WEIGHTS
    score = (
        w["like"] * n_like
        + w["retweet"] * n_rt
        + w["reply"] * n_reply
        + w["quote"] * n_quote
    )
    return float(score / exposed)
