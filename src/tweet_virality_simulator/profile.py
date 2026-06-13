"""Calibration profile: every tunable model parameter, in one place.

This is the open-core boundary made real. The *engine code* is open, but the
*numbers* live in a Profile that is loaded at runtime. The open repo ships only
the generic ``default`` profile; the private backend ships a profile fitted
against real cascade data as a data artifact — so calibration stays closed
without forking the engine.

    profile = load_profile()                       # generic, public
    profile = load_profile(path="fitted_x.json")   # the moat, private
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def _heavy_ranker_defaults() -> Dict[str, float]:
    # X "heavy ranker" engagement weights (documented concepts; see weights.py).
    return {
        "like": 0.5,
        "retweet": 1.0,
        "reply": 13.5,
        "quote": 1.0,
    }


def _appeal_weights_defaults() -> Dict[str, float]:
    return {
        "hook_strength": 0.26,
        "emotional_intensity": 0.22,
        "novelty": 0.18,
        "controversy": 0.16,
        "relatability": 0.10,
        "clarity": 0.08,
    }


class Profile(BaseModel):
    """All calibratable parameters of the simulation."""

    name: str = "default"

    # --- content appeal ---
    appeal_weights: Dict[str, float] = Field(default_factory=_appeal_weights_defaults)
    appeal_scale: float = 1.7
    link_penalty: float = 0.30
    media_boost: float = 0.08

    # --- reaction propensity scales ---
    like_scale: float = 6.0
    retweet_scale: float = 7.0
    reply_scale: float = 6.0
    reply_controversy: float = 1.3
    quote_controversy: float = 0.9
    p_like_cap: float = 0.9
    p_retweet_cap: float = 0.6
    p_reply_cap: float = 0.5
    p_quote_cap: float = 0.4

    # --- heavy-ranker engagement weights ---
    heavy_ranker: Dict[str, float] = Field(default_factory=_heavy_ranker_defaults)

    # --- network structure ---
    communities: int = 12
    avg_following: int = 18
    homophily: float = 3.0
    pref_attach: float = 1.4
    community_noise: float = 0.6

    # --- audience trait priors ---
    popularity_pareto_a: float = 1.3
    base_engage_log_mean: float = -1.6
    base_engage_log_sigma: float = 0.7
    share_beta: List[float] = Field(default_factory=lambda: [1.5, 9.0])
    like_beta: List[float] = Field(default_factory=lambda: [2.0, 5.0])
    reply_beta: List[float] = Field(default_factory=lambda: [1.3, 12.0])

    # --- algorithmic (For You) injection ---
    promotion_threshold: float = 0.85
    pool_growth: float = 1.6
    exploration: float = 0.15
    seed_fraction: float = 0.15  # cap on seed as fraction of audience

    # --- virality definition (relative to the simulated universe) ---
    viral_reach_fraction: float = 0.30


DEFAULT_PROFILE = Profile()


def load_profile(name: str = "default", path: Optional[str] = None) -> Profile:
    """Load a calibration profile.

    Resolution order: explicit ``path`` arg > ``TVS_PROFILE_PATH`` env var >
    named built-in (only ``default`` ships in the open repo). A private backend
    points ``path``/env at its fitted profile artifact — no engine change.
    """
    path = path or os.getenv("TVS_PROFILE_PATH")
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            return Profile.model_validate(json.load(fh))
    return Profile(name=name)
