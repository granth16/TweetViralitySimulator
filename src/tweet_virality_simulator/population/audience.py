"""Synthetic audience: a population of personas as numpy arrays.

Each persona has an interest embedding (clustered into communities), a latent
popularity (power-law, → hubs), and engagement traits. Stored as columnar
arrays so the cascade can vectorize over thousands of nodes.

The sampling parameters come from Profile (global priors). When the private
backend has fitted community-level engagement offsets (from the CMA-ES
calibrator), they are attached to the profile object as ``_community_offsets``
and applied here as per-persona additive corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import EMB_DIM, Config
from ..profile import Profile


@dataclass
class Audience:
    n: int
    topic: np.ndarray             # (n, EMB_DIM) unit-norm interest vectors
    popularity: np.ndarray        # (n,) latent reach 0..1 (power-law)
    base_engage: np.ndarray       # (n,) general willingness to engage
    share_propensity: np.ndarray  # (n,) tendency to retweet/quote
    like_rate: np.ndarray         # (n,)
    reply_rate: np.ndarray        # (n,)
    community: np.ndarray         # (n,) community id


def _apply_community_offsets(
    community: np.ndarray,
    like_rate: np.ndarray,
    reply_rate: np.ndarray,
    share_propensity: np.ndarray,
    offsets: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply per-community engagement offsets (clipped to valid probability range)."""
    like_off   = offsets.get("like_offsets",  [])
    reply_off  = offsets.get("reply_offsets", [])
    share_off  = offsets.get("share_offsets", [])

    n_comm = len(like_off)
    if n_comm == 0:
        return like_rate, reply_rate, share_propensity

    # Build per-persona offset vectors from community membership.
    persona_like_off  = np.array([like_off[min(c, n_comm - 1)]  for c in community])
    persona_reply_off = np.array([reply_off[min(c, n_comm - 1)] for c in community]) if reply_off else np.zeros(len(community))
    persona_share_off = np.array([share_off[min(c, n_comm - 1)] for c in community]) if share_off else np.zeros(len(community))

    return (
        np.clip(like_rate  + persona_like_off,  0.0, 1.0),
        np.clip(reply_rate + persona_reply_off, 0.0, 1.0),
        np.clip(share_propensity + persona_share_off, 0.0, 1.0),
    )


def generate(cfg: Config, profile: Profile, rng: np.random.Generator) -> Audience:
    n = cfg.audience_size
    c = max(1, profile.communities)

    # community centroids in the shared embedding space
    centroids = rng.standard_normal((c, EMB_DIM))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9

    community = rng.integers(0, c, size=n)
    noise = profile.community_noise * rng.standard_normal((n, EMB_DIM))
    topic = centroids[community] + noise
    topic /= np.linalg.norm(topic, axis=1, keepdims=True) + 1e-9

    # popularity: power-law → a few hubs, long tail.  Map to 0..1 by log-rank.
    raw = rng.pareto(profile.popularity_pareto_a, size=n) + 1.0
    pop = np.log1p(raw)
    popularity = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)

    # engagement traits — sampled from birth priors (all in Profile).
    base_engage = np.clip(
        rng.lognormal(mean=profile.base_engage_log_mean, sigma=profile.base_engage_log_sigma, size=n),
        0.0,
        1.5,
    )
    share_propensity = rng.beta(profile.share_beta[0],  profile.share_beta[1],  size=n)
    like_rate        = rng.beta(profile.like_beta[0],   profile.like_beta[1],   size=n)
    reply_rate       = rng.beta(profile.reply_beta[0],  profile.reply_beta[1],  size=n)

    # Apply community-level engagement offsets if fitted by the private backend.
    community_offsets: Optional[dict] = getattr(profile, "_community_offsets", None)
    if community_offsets:
        like_rate, reply_rate, share_propensity = _apply_community_offsets(
            community, like_rate, reply_rate, share_propensity, community_offsets
        )

    return Audience(
        n=n,
        topic=topic,
        popularity=popularity,
        base_engage=base_engage,
        share_propensity=share_propensity,
        like_rate=like_rate,
        reply_rate=reply_rate,
        community=community,
    )
