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

from ...models import SEMANTIC_DIMS, TweetDNA
from ...profile import Profile
from ...population.audience import Audience
from ...population.niches import niche_membership


def _global_appeal_base(dna: TweetDNA, profile: Profile) -> float:
    """The original niche-blind DNA->appeal sum (generic prior)."""
    w = profile.appeal_weights
    return (
        w.get("hook_strength", 0.0) * dna.hook_strength
        + w.get("emotional_intensity", 0.0) * dna.emotional_intensity
        + w.get("novelty", 0.0) * dna.novelty
        + w.get("controversy", 0.0) * dna.controversy
        + w.get("relatability", 0.0) * dna.relatability
        + w.get("clarity", 0.0) * dna.clarity
    )


def _niche_appeal_base(dna: TweetDNA, profile: Profile, weight_matrix) -> float:
    """Membership-weighted DNA->appeal sum across niches.

    ``base = sum_k membership_k * (sum_d W[k,d] * dna_d)``, i.e. the tweet's
    appeal is computed under each niche's own weights and blended by how much the
    tweet belongs to that niche. Reduces to a single niche's weights when the
    membership concentrates on one anchor.
    """
    feats = np.array([getattr(dna, d, 0.0) for d in SEMANTIC_DIMS], dtype=np.float64)
    W = np.asarray(weight_matrix, dtype=np.float64)          # (niche_count, dims)
    member = niche_membership(dna.topic_vector, profile)     # (niche_count,)
    per_niche_base = W @ feats                               # (niche_count,)
    return float(member @ per_niche_base)


def content_appeal(dna: TweetDNA, profile: Profile) -> float:
    """Overall 0..1 appeal of the content, before persona/affinity effects.

    Uses per-niche weights when the profile carries them (fitted on outcomes),
    otherwise the global ``appeal_weights`` prior. Link/media adjustments and the
    global ``appeal_scale`` apply identically in both paths.
    """
    link_penalty = profile.link_penalty if dna.has_link else 0.0  # X downranks links
    media_boost = profile.media_boost if dna.has_media else 0.0

    weight_matrix = profile.appeal_weight_matrix(SEMANTIC_DIMS)
    if weight_matrix is None:
        base = _global_appeal_base(dna, profile)
    else:
        base = _niche_appeal_base(dna, profile, weight_matrix)

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
