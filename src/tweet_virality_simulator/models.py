"""Stable public schema (Pydantic v2).

These types are the contract between the engine, providers, and any future
trained model. Keep them stable.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

# The semantic content dimensions a provider scores (each in 0..1).
SEMANTIC_DIMS = [
    "hook_strength",
    "emotional_intensity",
    "controversy",
    "novelty",
    "clarity",
    "relatability",
    "information_density",
]


class TweetDNA(BaseModel):
    """Measurable features extracted from a tweet."""

    text: str

    # Semantic signals (0..1), from the heuristic or LLM provider.
    hook_strength: float = 0.0
    emotional_intensity: float = 0.0
    controversy: float = 0.0
    novelty: float = 0.0
    clarity: float = 0.0
    relatability: float = 0.0
    information_density: float = 0.0

    # Structural signals.
    length_chars: int = 0
    num_hashtags: int = 0
    num_mentions: int = 0
    has_link: bool = False
    has_media: bool = False
    is_question: bool = False

    # Topic/interest embedding (unit-norm), used for homophily.
    topic_vector: List[float] = Field(default_factory=list)

    scored_by: str = "heuristic"


class CascadeStats(BaseModel):
    """Aggregate outcome across all Monte Carlo runs."""

    audience_size: int
    runs: int
    seed_size: int

    reach_median: int
    reach_p10: int
    reach_p90: int
    reach_max: int
    reach_fraction_median: float

    p_viral: float
    viral_threshold: int

    reproduction_number: float
    avg_shares: float
    avg_depth: float
    peak_round: int

    in_network_share: float
    out_network_share: float

    reach_curve: List[float] = Field(default_factory=list)


class Driver(BaseModel):
    """A factor that helped or hurt the tweet's spread."""

    label: str
    impact: float  # signed, roughly -1..1
    detail: str


class Report(BaseModel):
    """The shareable artifact: verdict + numbers + the 'why'."""

    tweet: str
    virality_score: int  # 0..100
    verdict: str
    roast: str

    dna: TweetDNA
    stats: CascadeStats

    drivers: List[Driver] = Field(default_factory=list)
    weaknesses: List[Driver] = Field(default_factory=list)

    config: Dict[str, Any] = Field(default_factory=dict)


class Improvement(BaseModel):
    """Result of rewriting a tweet for spread: original + ranked candidates.

    Every candidate is simulated on the *same* synthetic population as the
    original, so score differences are a fair head-to-head lift, not noise.
    """

    original: Report
    variants: List[Report] = Field(default_factory=list)  # ranked best-first
    generated_by: str = "heuristic"

    def best(self) -> Report:
        """Highest-scoring tweet overall (may be the original if nothing beat it)."""
        pool = [self.original, *self.variants]
        return max(pool, key=lambda r: (r.virality_score, r.stats.reach_median))

    def lift(self) -> int:
        """Score points gained by the best rewrite over the original (>= 0)."""
        if not self.variants:
            return 0
        return max(0, self.variants[0].virality_score - self.original.virality_score)
