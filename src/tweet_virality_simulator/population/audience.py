"""Synthetic audience: a population of personas as numpy arrays.

Each persona has an interest embedding (clustered into communities), a latent
popularity (power-law, → hubs), and engagement traits. Stored as columnar
arrays so the cascade can vectorize over thousands of nodes.

This is the part later "trained" by calibrating its parameters so aggregate
behavior matches real engagement data (see validation harness). For v0.1 the
traits are sampled from sensible priors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EMB_DIM, Config


@dataclass
class Audience:
    n: int
    topic: np.ndarray            # (n, EMB_DIM) unit-norm interest vectors
    popularity: np.ndarray       # (n,) latent reach 0..1 (power-law)
    base_engage: np.ndarray      # (n,) general willingness to engage
    share_propensity: np.ndarray  # (n,) tendency to retweet/quote
    like_rate: np.ndarray        # (n,)
    reply_rate: np.ndarray       # (n,)
    community: np.ndarray        # (n,) community id


def generate(cfg: Config, rng: np.random.Generator) -> Audience:
    n = cfg.audience_size
    c = max(1, cfg.communities)

    # community centroids in the shared embedding space
    centroids = rng.standard_normal((c, EMB_DIM))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9

    community = rng.integers(0, c, size=n)
    noise = 0.6 * rng.standard_normal((n, EMB_DIM))
    topic = centroids[community] + noise
    topic /= np.linalg.norm(topic, axis=1, keepdims=True) + 1e-9

    # popularity: power-law -> a few hubs, long tail. Map to 0..1 by log-rank.
    raw = rng.pareto(1.3, size=n) + 1.0
    pop = np.log1p(raw)
    popularity = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)

    # engagement traits (most people engage rarely; a few are very active)
    base_engage = np.clip(rng.lognormal(mean=-1.6, sigma=0.7, size=n), 0.0, 1.5)
    share_propensity = rng.beta(1.5, 9.0, size=n)
    like_rate = rng.beta(2.0, 5.0, size=n)
    reply_rate = rng.beta(1.3, 12.0, size=n)

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
