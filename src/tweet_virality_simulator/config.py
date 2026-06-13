"""Runtime configuration for a simulation.

Defaults are tuned so that on the synthetic universe a weak tweet fizzles and a
strong one can cascade — i.e. outcomes are *relative* indicators, not absolute
view-count predictions. See README on positioning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

# Dimensionality of the shared text/interest embedding space.
EMB_DIM = 24


@dataclass
class Config:
    # --- LLM provider for tweet DNA scoring (engine falls back to heuristic) ---
    provider: str = "heuristic"  # "heuristic" | "openai" | "ollama"

    # --- Population / network ---
    audience_size: int = 1000
    communities: int = 12
    avg_following: int = 18

    # --- Author / seeding (the creator prior) ---
    author_followers: int = 60  # in-network reach the author starts with

    # --- Monte Carlo ---
    runs: int = 120
    max_rounds: int = 12
    seed: Optional[int] = None

    # --- Algorithmic (For You) injection ---
    # ~50/50 in-network vs out-of-network split is documented for X's For You feed.
    promotion_threshold: float = 0.85  # engagement bar to unlock a bigger pool
    pool_growth: float = 1.6           # geometric growth of each promoted pool
    exploration: float = 0.15          # epsilon: occasional out-of-niche sampling

    # --- Virality definition (relative to the simulated universe, not raw views) ---
    viral_reach_fraction: float = 0.30

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
