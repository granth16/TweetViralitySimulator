"""Search Profile parameters to maximize benchmark agreement.

Random search over the calibratable knobs, scoring each candidate profile with
the harness. This is the *open method* of calibration; the private backend runs
the identical loop against real outcome data (via the ``storage`` seam) to
produce a fitted profile that ships closed.

    best = tune(n_samples=40)
    best.profile  # -> a Profile you can save and load with TVS_PROFILE_PATH

It optimizes a transparent objective (ranking + invariants - saturation), never
"100% accuracy" — virality is stochastic and the benchmark is priors, not truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import Config
from ..profile import Profile
from .harness import ValidationResult, evaluate


def objective(r: ValidationResult) -> float:
    """Transparent scalar: reward correct ordering and good dynamic range,
    punish saturation. We also want strong tweets to actually score high
    (use the 0..100 range) — not just be ranked correctly."""
    tiers = sorted(r.tier_means)
    spread = (r.tier_means[tiers[-1]] - r.tier_means[tiers[0]]) / 100.0 if tiers else 0.0
    top = r.tier_means[tiers[-1]] / 100.0 if tiers else 0.0
    return (
        1.0 * r.rank_corr
        + 0.5 * r.pair_accuracy
        + 0.4 * r.invariants
        + 0.3 * spread        # reward separation across tiers
        + 0.2 * top           # strong tweets should land high, not be compressed
        - 0.4 * r.saturation
    )


# (attribute, low, high) — the knobs the search is allowed to move.
_SEARCH = [
    ("appeal_scale", 1.2, 2.4),
    ("like_scale", 3.0, 9.0),
    ("retweet_scale", 4.0, 10.0),
    ("reply_scale", 3.0, 9.0),
    ("promotion_threshold", 0.6, 1.1),
    ("pool_growth", 1.3, 2.0),
    ("homophily", 2.0, 4.0),
    ("pref_attach", 0.8, 2.0),
    ("exploration", 0.05, 0.25),
]


def _sample(base: Profile, rng: np.random.Generator) -> Profile:
    overrides = {attr: float(rng.uniform(lo, hi)) for attr, lo, hi in _SEARCH}
    return base.model_copy(update=overrides)


@dataclass
class TuneResult:
    profile: Profile
    result: ValidationResult
    score: float
    baseline: ValidationResult
    baseline_score: float

    def summary(self) -> str:
        return (
            f"baseline obj={self.baseline_score:+.3f}  ({self.baseline.summary()})\n"
            f"tuned    obj={self.score:+.3f}  ({self.result.summary()})"
        )


def tune(
    n_samples: int = 40,
    seed: int = 0,
    base: Optional[Profile] = None,
    config: Optional[Config] = None,
) -> TuneResult:
    rng = np.random.default_rng(seed)
    base = base or Profile()
    cfg = config or Config(audience_size=500, runs=50)

    base_res = evaluate(base, cfg)
    base_obj = objective(base_res)

    best_profile, best_res, best_obj = base, base_res, base_obj
    for _ in range(n_samples):
        cand = _sample(base, rng)
        res = evaluate(cand, cfg)
        obj = objective(res)
        if obj > best_obj:
            best_profile, best_res, best_obj = cand, res, obj

    return TuneResult(
        profile=best_profile,
        result=best_res,
        score=best_obj,
        baseline=base_res,
        baseline_score=base_obj,
    )
