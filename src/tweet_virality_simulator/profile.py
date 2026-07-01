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
    # NOTE: these defaults are tuned to the open face-validity benchmark
    # (validation/), i.e. generic priors. A profile fitted on real outcome data
    # ships separately and stays private.
    appeal_weights: Dict[str, float] = Field(default_factory=_appeal_weights_defaults)
    appeal_scale: float = 1.84
    link_penalty: float = 0.30
    media_boost: float = 0.08

    # --- niche-conditional appeal (learned, optional) ---
    # A tweet's DNA->appeal mapping is not the same in every community: novelty
    # carries tech, emotion carries sports, clarity carries how-to. ``niche_*``
    # lets the backend fit one weight vector per niche directly from outcomes
    # (see learning/appeal_fit). When ``niche_appeal_weights`` is None the engine
    # falls back to the single global ``appeal_weights`` for every niche, so an
    # un-fitted profile behaves exactly as before.
    niche_count: int = 8
    niche_temperature: float = 6.0
    # Optional: one {dim: weight} mapping per niche, length == niche_count.
    niche_appeal_weights: Optional[List[Dict[str, float]]] = None

    def appeal_weight_matrix(self, dims: List[str]) -> Optional[List[List[float]]]:
        """Per-niche weights as a ``(niche_count, len(dims))`` row list.

        Returns None when no niche weights are fitted (caller uses the global
        ``appeal_weights`` path). Rows are padded/truncated to ``niche_count`` and
        missing dims default to 0.0, so a partially-specified artifact is safe.
        """
        rows = self.niche_appeal_weights
        if not rows:
            return None
        out: List[List[float]] = []
        for k in range(self.niche_count):
            row = rows[k] if k < len(rows) else {}
            out.append([float(row.get(d, 0.0)) for d in dims])
        return out

    # --- reaction propensity scales ---
    like_scale: float = 6.56
    retweet_scale: float = 8.16
    reply_scale: float = 7.03
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
    homophily: float = 3.84
    pref_attach: float = 1.60
    community_noise: float = 0.6

    # --- audience trait priors ---
    popularity_pareto_a: float = 1.3
    base_engage_log_mean: float = -1.6
    base_engage_log_sigma: float = 0.7
    share_beta: List[float] = Field(default_factory=lambda: [1.5, 9.0])
    like_beta: List[float] = Field(default_factory=lambda: [2.0, 5.0])
    reply_beta: List[float] = Field(default_factory=lambda: [1.3, 12.0])

    # --- algorithmic (For You) injection ---
    promotion_threshold: float = 0.62
    pool_growth: float = 1.75
    exploration: float = 0.25
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
