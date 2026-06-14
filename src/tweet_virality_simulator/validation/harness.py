"""Measure a profile against the face-validity benchmark.

Outputs a few honest numbers:

* ``rank_corr``      Spearman correlation between predicted score and tier (-1..1)
* ``pair_accuracy``  fraction of cross-tier pairs ordered correctly (0..1)
* ``invariants``     fraction of directional invariants satisfied (0..1)
* ``saturation``     fraction of scores stuck at the extremes (lower is better)
* ``tier_means``     mean predicted score per tier (should increase, separated)

The same harness scores the open ``default`` profile and any private fitted
profile, so "the fitted profile beats default by X" is a number, not a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import Config
from ..engine import analyze
from ..profile import Profile
from .dataset import INVARIANTS, LABELED


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (1..n), ties shared — for Spearman."""
    order = x.argsort()
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average tied ranks
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


@dataclass
class ValidationResult:
    rank_corr: float
    pair_accuracy: float
    invariants: float
    saturation: float
    tier_means: Dict[int, float]
    n: int
    scores: List[int] = field(default_factory=list)
    invariant_failures: List[Tuple[str, str, int, int]] = field(default_factory=list)

    def summary(self) -> str:
        tiers = " ".join(f"T{t}:{self.tier_means[t]:.0f}" for t in sorted(self.tier_means))
        return (
            f"rank_corr={self.rank_corr:+.3f}  pair_acc={self.pair_accuracy:.2f}  "
            f"invariants={self.invariants:.2f}  saturation={self.saturation:.2f}  "
            f"| tier means [{tiers}]  (n={self.n})"
        )


def _fast_config(profile_name: str, profile_path: Optional[str]) -> Config:
    # Smaller than defaults so a full benchmark runs in seconds.
    return Config(audience_size=600, runs=60, profile=profile_name, profile_path=profile_path)


def evaluate(
    profile: Optional[Profile] = None,
    config: Optional[Config] = None,
) -> ValidationResult:
    """Score the benchmark with the given profile (default if None)."""
    cfg = config or _fast_config(
        profile.name if profile else "default", None
    )

    texts = [t for t, _ in LABELED]
    tiers = np.array([tier for _, tier in LABELED], dtype=float)
    scores = np.array(
        [analyze(t, cfg, profile=profile).virality_score for t in texts], dtype=float
    )

    rank_corr = _spearman(scores, tiers)

    # Cross-tier pairwise ordering accuracy.
    correct = total = 0
    for i in range(len(texts)):
        for j in range(len(texts)):
            if tiers[i] < tiers[j]:
                total += 1
                if scores[i] < scores[j]:
                    correct += 1
                elif scores[i] == scores[j]:
                    correct += 0.5
    pair_accuracy = (correct / total) if total else 0.0

    saturation = float(np.mean((scores >= 98) | (scores <= 2)))
    tier_means = {
        int(t): float(scores[tiers == t].mean()) for t in sorted(set(tiers.tolist()))
    }

    # Directional invariants.
    inv_pass = 0
    failures: List[Tuple[str, str, int, int]] = []
    for worse, better in INVARIANTS:
        sw = analyze(worse, cfg, profile=profile).virality_score
        sb = analyze(better, cfg, profile=profile).virality_score
        if sb > sw:
            inv_pass += 1
        else:
            failures.append((worse, better, sw, sb))
    invariants = inv_pass / len(INVARIANTS) if INVARIANTS else 1.0

    return ValidationResult(
        rank_corr=rank_corr,
        pair_accuracy=pair_accuracy,
        invariants=invariants,
        saturation=saturation,
        tier_means=tier_means,
        n=len(texts),
        scores=[int(s) for s in scores],
        invariant_failures=failures,
    )
