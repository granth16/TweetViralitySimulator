"""Monte Carlo contagion simulation: social cascade + algorithmic injection.

Pure, seeded numpy — thousands of stochastic runs are cheap and reproducible.
Each run spreads a tweet through the synthetic universe round by round and the
aggregate over runs yields a distribution of outcomes (reach, R, viral odds).
"""

from __future__ import annotations

from typing import List

import numpy as np

from .config import Config
from .models import CascadeStats, TweetDNA
from .platforms.x import algo, reactions
from .platforms.x.network import Network, followers_of
from .population.audience import Audience
from .profile import Profile


def _run_once(
    dna: TweetDNA,
    audience: Audience,
    network: Network,
    cfg: Config,
    profile: Profile,
    appeal: float,
    tweet_topic: np.ndarray,
    rng: np.random.Generator,
):
    n = audience.n
    seen = np.zeros(n, dtype=bool)

    # Seed: the author's in-network followers, biased toward topical accounts.
    topical = (audience.topic @ tweet_topic + 1.0) / 2.0
    seed_weight = topical * (0.3 + audience.popularity)
    seed_weight = seed_weight / seed_weight.sum()
    seed_k = int(min(max(cfg.author_followers, 5), int(profile.seed_fraction * n)))
    exposed = rng.choice(n, size=seed_k, replace=False, p=seed_weight)
    seen[exposed] = True

    reach_curve: List[int] = []
    shares_per_round: List[int] = []
    engaged_sum = np.zeros(audience.topic.shape[1])
    in_network = 0
    out_network = seed_k  # the seed is the author's own audience
    total_shares = 0
    depth = 0
    pool = float(seed_k)

    for r in range(cfg.max_rounds):
        m = exposed.size
        if m == 0:
            break
        reach_curve.append(m)

        probs = reactions.action_probs(dna, audience, exposed, tweet_topic, appeal, profile)
        draws = rng.random((m, 4))
        like = draws[:, 0] < probs["like"]
        rt = draws[:, 1] < probs["retweet"]
        reply = draws[:, 2] < probs["reply"]
        quote = draws[:, 3] < probs["quote"]

        share_mask = rt | quote
        sharers = exposed[share_mask]
        n_sharers = int(sharers.size)
        shares_per_round.append(n_sharers)
        total_shares += n_sharers
        if n_sharers > 0:
            engaged_sum += audience.topic[sharers].sum(axis=0)
            depth = r + 1

        eng = reactions.engagement_rate(
            int(like.sum()), int(rt.sum()), int(reply.sum()), int(quote.sum()), m, profile
        )

        # Social (in-network) channel.
        social_next = followers_of(network, sharers)
        social_new = social_next[~seen[social_next]] if social_next.size else social_next
        seen[social_new] = True
        in_network += int(social_new.size)

        # Algorithmic (out-of-network) channel: gated by engagement.
        algo_new = np.empty(0, dtype=np.int64)
        if eng >= profile.promotion_threshold and n_sharers > 0:
            pool *= profile.pool_growth
            algo_cand = algo.inject(
                rng, audience, engaged_sum, int(pool), seen, profile.exploration
            )
            if algo_cand.size:
                algo_new = algo_cand[~seen[algo_cand]]
                seen[algo_new] = True
                out_network += int(algo_new.size)

        if social_new.size + algo_new.size == 0:
            break
        exposed = np.concatenate([social_new, algo_new])
        if seen.sum() >= n:
            break

    reach = int(seen.sum())
    # Reproduction number: average round-over-round growth of new shares
    # (organic branching factor). >1 means the cascade is accelerating.
    ratios = [
        shares_per_round[t + 1] / shares_per_round[t]
        for t in range(len(shares_per_round) - 1)
        if shares_per_round[t] > 0
    ]
    r0 = float(np.mean(ratios)) if ratios else 0.0

    return {
        "reach": reach,
        "seed": seed_k,
        "shares": total_shares,
        "depth": depth,
        "r0": r0,
        "reach_curve": reach_curve,
        "in_network": in_network,
        "out_network": out_network,
        "peak_round": int(np.argmax(reach_curve)) if reach_curve else 0,
    }


def simulate(
    dna: TweetDNA,
    audience: Audience,
    network: Network,
    cfg: Config,
    profile: Profile,
    rng: np.random.Generator,
) -> CascadeStats:
    appeal = reactions.content_appeal(dna, profile)
    tweet_topic = np.asarray(dna.topic_vector, dtype=np.float64)

    reaches: List[int] = []
    shares: List[int] = []
    depths: List[int] = []
    r0s: List[float] = []
    peaks: List[int] = []
    in_net: List[int] = []
    out_net: List[int] = []
    curves: List[List[int]] = []
    seed_size = 0

    for _ in range(cfg.runs):
        out = _run_once(dna, audience, network, cfg, profile, appeal, tweet_topic, rng)
        seed_size = out["seed"]
        reaches.append(out["reach"])
        shares.append(out["shares"])
        depths.append(out["depth"])
        r0s.append(out["r0"])
        peaks.append(out["peak_round"])
        in_net.append(out["in_network"])
        out_net.append(out["out_network"])
        curves.append(out["reach_curve"])

    reaches_arr = np.array(reaches)
    n = audience.n
    viral_threshold = int(profile.viral_reach_fraction * n)

    max_len = max((len(c) for c in curves), default=0)
    avg_curve = []
    if max_len:
        padded = np.zeros((len(curves), max_len))
        for i, c in enumerate(curves):
            padded[i, : len(c)] = c
        avg_curve = padded.mean(axis=0).round(1).tolist()

    total_in = float(np.sum(in_net))
    total_out = float(np.sum(out_net))
    denom = total_in + total_out
    in_share = total_in / denom if denom > 0 else 0.0
    out_share = total_out / denom if denom > 0 else 0.0

    return CascadeStats(
        audience_size=n,
        runs=cfg.runs,
        seed_size=seed_size,
        reach_median=int(np.median(reaches_arr)),
        reach_p10=int(np.percentile(reaches_arr, 10)),
        reach_p90=int(np.percentile(reaches_arr, 90)),
        reach_max=int(reaches_arr.max()),
        reach_fraction_median=float(np.median(reaches_arr) / n),
        p_viral=float(np.mean(reaches_arr >= viral_threshold)),
        viral_threshold=viral_threshold,
        reproduction_number=float(np.mean(r0s)),
        avg_shares=float(np.mean(shares)),
        avg_depth=float(np.mean(depths)),
        peak_round=int(np.round(np.mean(peaks))),
        in_network_share=in_share,
        out_network_share=out_share,
        reach_curve=avg_curve,
    )
