"""Per-persona reaction probabilities + the heavy-ranker engagement score.

The LLM (or heuristic) sets the *content appeal* via Tweet DNA; persona traits
and topic affinity modulate it into per-action probabilities. The algorithm's
view of a batch is the heavy-ranker-weighted sum of realized actions.

All magnitude constants are read from the calibration :class:`Profile`, so the
fitted numbers can ship as a private data artifact without touching this code.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ...models import TweetDNA
from ...profile import Profile
from ...population.audience import Audience


def content_appeal(dna: TweetDNA, profile: Profile) -> float:
    """Overall 0..1 appeal of the content, before persona/affinity effects."""
    link_penalty = profile.link_penalty if dna.has_link else 0.0  # X downranks links
    media_boost = profile.media_boost if dna.has_media else 0.0
    w = profile.appeal_weights
    base = (
        w.get("hook_strength", 0.0) * dna.hook_strength
        + w.get("emotional_intensity", 0.0) * dna.emotional_intensity
        + w.get("novelty", 0.0) * dna.novelty
        + w.get("controversy", 0.0) * dna.controversy
        + w.get("relatability", 0.0) * dna.relatability
        + w.get("clarity", 0.0) * dna.clarity
    )
    return float(np.clip(base * profile.appeal_scale + media_boost - link_penalty, 0.02, 1.0))


def action_probs(
    dna: TweetDNA,
    audience: Audience,
    idx: np.ndarray,
    tweet_topic: np.ndarray,
    appeal: float,
    profile: Profile,
) -> Dict[str, np.ndarray]:
    """Probabilities of like/retweet/reply/quote for the given exposed nodes."""
    topic = audience.topic[idx]
    cos = topic @ tweet_topic
    affinity = 0.5 + 0.5 * cos  # 0..1

    drive = audience.base_engage[idx] * affinity * appeal
    hook_nov = 0.6 + 0.4 * ((dna.hook_strength + dna.novelty) / 2.0)
    contro = dna.controversy

    p_like = np.clip(
        drive * audience.like_rate[idx] * profile.like_scale, 0.0, profile.p_like_cap
    )
    p_rt = np.clip(
        drive * audience.share_propensity[idx] * profile.retweet_scale * hook_nov,
        0.0,
        profile.p_retweet_cap,
    )
    p_reply = np.clip(
        drive * audience.reply_rate[idx] * profile.reply_scale * (0.35 + profile.reply_controversy * contro),
        0.0,
        profile.p_reply_cap,
    )
    p_quote = np.clip(p_rt * (0.15 + profile.quote_controversy * contro), 0.0, profile.p_quote_cap)

    return {"like": p_like, "retweet": p_rt, "reply": p_reply, "quote": p_quote}


def engagement_rate(
    n_like: int, n_rt: int, n_reply: int, n_quote: int, exposed: int, profile: Profile
) -> float:
    """Heavy-ranker-weighted engagement per exposed account."""
    if exposed <= 0:
        return 0.0
    w = profile.heavy_ranker
    score = (
        w.get("like", 0.0) * n_like
        + w.get("retweet", 0.0) * n_rt
        + w.get("reply", 0.0) * n_reply
        + w.get("quote", 0.0) * n_quote
    )
    return float(score / exposed)
