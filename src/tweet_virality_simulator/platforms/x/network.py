"""Synthetic directed follow graph.

Edges form by preferential attachment (popular accounts gain more followers,
→ power-law in-degree) plus homophily (you follow accounts similar to you).
We materialize, for each node, the list of its followers — i.e. who sees its
posts when it shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ...config import Config
from ..x import weights  # noqa: F401  (kept for discoverability of the X model)
from ...population.audience import Audience

_HOMOPHILY = 3.0   # gamma: weight on interest similarity
_PREF_ATTACH = 1.4  # beta: weight on target popularity


@dataclass
class Network:
    followers: List[np.ndarray]  # followers[j] = node indices that follow j
    in_degree: np.ndarray        # (n,)


def build(audience: Audience, cfg: Config, rng: np.random.Generator) -> Network:
    n = audience.n
    sim = audience.topic @ audience.topic.T  # (n, n) cosine (unit vectors)

    base_pref = _PREF_ATTACH * np.log1p(audience.popularity * 9.0)  # (n,)
    logit = _HOMOPHILY * sim + base_pref[None, :]
    np.fill_diagonal(logit, -1e9)

    # row-softmax -> per-node distribution over who it follows
    logit -= logit.max(axis=1, keepdims=True)
    w = np.exp(logit)
    w /= w.sum(axis=1, keepdims=True)

    follower_lists: List[List[int]] = [[] for _ in range(n)]
    # out-degree varies (some people follow many accounts)
    out_deg = np.clip(
        rng.poisson(cfg.avg_following, size=n), 1, max(1, n - 1)
    )
    for i in range(n):
        k = int(min(out_deg[i], n - 1))
        followees = rng.choice(n, size=k, replace=False, p=w[i])
        for j in followees:
            follower_lists[j].append(i)

    followers = [np.asarray(fl, dtype=np.int64) for fl in follower_lists]
    in_degree = np.array([len(fl) for fl in follower_lists], dtype=np.int64)
    return Network(followers=followers, in_degree=in_degree)


def followers_of(network: Network, sharers: np.ndarray) -> np.ndarray:
    """Union of followers of all sharer nodes (deduplicated)."""
    if sharers.size == 0:
        return np.empty(0, dtype=np.int64)
    parts = [network.followers[s] for s in sharers if network.followers[s].size > 0]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(parts))
