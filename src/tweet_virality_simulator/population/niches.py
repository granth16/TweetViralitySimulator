"""Stable niche taxonomy in the shared embedding space.

A *niche* is a fixed region of interest-space (tech, sports, politics, ...). The
audience's communities are regenerated per tweet and so have no stable identity;
niches do not — their anchor vectors are derived from one fixed seed, so "niche
3" means the same thing for every tweet. That stability is what makes per-niche
calibration (e.g. niche-specific appeal weights) learnable: a weight fitted for
niche 3 on yesterday's tweets still applies to niche 3 tomorrow.

A tweet gets a *soft* membership over niches (softmax of its similarity to each
anchor), never a hard label — that keeps appeal continuous across niche
boundaries and lets a cross-over tweet draw on more than one niche's weights.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import EMB_DIM
from ..profile import Profile

# Fixed forever: the niche anchors must not move when other seeds change, or a
# calibrated per-niche weight would silently bind to a different region.
NICHE_SEED = 20240601


@lru_cache(maxsize=16)
def niche_anchors(niche_count: int, dim: int = EMB_DIM) -> np.ndarray:
    """``(niche_count, dim)`` unit-norm anchor vectors, deterministic and cached.

    Independent of the per-tweet RNG: same anchors every call, every process.
    """
    rng = np.random.default_rng(NICHE_SEED)
    a = rng.standard_normal((max(1, niche_count), dim))
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    return a


def niche_membership(topic_vector: np.ndarray, profile: Profile) -> np.ndarray:
    """Soft membership ``(niche_count,)`` summing to 1 for one tweet topic.

    Softmax over cosine similarity to each anchor; ``niche_temperature`` controls
    sharpness (higher → closer to a hard pick of the nearest niche).
    """
    anchors = niche_anchors(profile.niche_count)
    v = np.asarray(topic_vector, dtype=np.float64)
    nv = np.linalg.norm(v)
    if nv <= 0:
        # Degenerate topic: spread mass uniformly rather than divide by zero.
        return np.full(anchors.shape[0], 1.0 / anchors.shape[0])
    sims = (anchors @ v) / nv  # anchors already unit-norm
    z = profile.niche_temperature * sims
    z -= z.max()  # stabilize
    e = np.exp(z)
    return e / e.sum()
