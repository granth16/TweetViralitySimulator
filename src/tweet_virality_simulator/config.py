"""Runtime configuration: *how* to run a simulation.

The *what* — every calibratable number (reaction scales, network shape, algo
thresholds, trait priors) — lives in :class:`~tweet_virality_simulator.profile.Profile`,
loaded at runtime. That split is the open-core boundary: the engine and these
execution knobs are open; the fitted Profile can stay private.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

# Dimensionality of the shared text/interest embedding space.
EMB_DIM = 24


@dataclass
class Config:
    # --- LLM provider for tweet DNA scoring (engine falls back to heuristic) ---
    provider: str = "heuristic"  # "heuristic" | "openai" | "ollama" | "compat"

    # --- Calibration profile selection (the moat attaches here) ---
    # Name of a built-in profile, or set ``profile_path`` / TVS_PROFILE_PATH to
    # load a fitted profile artifact shipped by the private backend.
    profile: str = "default"
    profile_path: Optional[str] = None

    # --- Population scale ---
    audience_size: int = 1000

    # --- Author / seeding (the creator prior) ---
    author_followers: int = 60  # in-network reach the author starts with

    # --- Monte Carlo ---
    runs: int = 120
    max_rounds: int = 12
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
