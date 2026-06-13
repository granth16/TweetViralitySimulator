"""For You algorithmic injection: threshold-gated, similarity-targeted pools.

Models the out-of-network exposure channel. When a batch clears the engagement
bar, the platform releases a larger pool, sampled toward accounts similar to the
ones already engaging (SimClusters/embedding-style targeting) with some
exploration. Social-proof is approximated via interest similarity.
"""

from __future__ import annotations

import numpy as np

from ...population.audience import Audience


def inject(
    rng: np.random.Generator,
    audience: Audience,
    engaged_centroid: np.ndarray,
    pool_size: int,
    seen: np.ndarray,
    exploration: float,
) -> np.ndarray:
    """Sample a new out-of-network pool by similarity to the engaged centroid."""
    available = ~seen
    n_available = int(available.sum())
    if n_available <= 0 or pool_size <= 0:
        return np.empty(0, dtype=np.int64)

    norm = np.linalg.norm(engaged_centroid)
    if norm == 0:
        return np.empty(0, dtype=np.int64)
    centroid = engaged_centroid / norm

    sim = audience.topic @ centroid          # (n,) in [-1, 1]
    score = (sim + 1.0) / 2.0                 # 0..1 similarity preference
    score = (1.0 - exploration) * score + exploration * 0.5  # epsilon exploration
    score = score * available                 # zero out already-seen
    total = score.sum()
    if total <= 0:
        return np.empty(0, dtype=np.int64)

    probs = score / total
    k = int(min(pool_size, n_available))
    return rng.choice(audience.n, size=k, replace=False, p=probs)
