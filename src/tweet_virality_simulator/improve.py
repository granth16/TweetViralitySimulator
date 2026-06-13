"""`improve`: rewrite a tweet for spread, then let the simulator pick the winner.

Generation proposes candidates; the (calibrated) engine is the judge. Every
candidate is scored on the *same* synthetic population as the original, so the
reported lift is a fair head-to-head — not the generator marking its own work.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Optional

from .config import Config
from .engine import analyze
from .generators import get_generator
from .generators.heuristic import HeuristicGenerator
from .models import Improvement
from .profile import Profile, load_profile
from .tracing import get_tracer


def _shared_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)


def improve(
    text: str,
    config: Optional[Config] = None,
    variants: int = 6,
    has_media: bool = False,
    profile: Optional[Profile] = None,
) -> Improvement:
    """Generate rewrites of ``text`` and rank them by simulated spread."""
    if not text or not text.strip():
        raise ValueError("tweet text is empty")

    cfg = config or Config()
    # Pin the population so the original and every candidate are judged fairly.
    if cfg.seed is None:
        cfg = replace(cfg, seed=_shared_seed(text))

    profile = profile or load_profile(name=cfg.profile, path=cfg.profile_path)

    original = analyze(text, cfg, has_media=has_media, profile=profile)

    gen = get_generator(cfg.provider)
    candidates = gen.generate(text, variants)
    generated_by = gen.name
    # If an LLM generator produced nothing, fall back so improve always delivers.
    if not candidates and not isinstance(gen, HeuristicGenerator):
        fallback = HeuristicGenerator()
        candidates = fallback.generate(text, variants)
        generated_by = fallback.name

    reports = [analyze(c, cfg, has_media=has_media, profile=profile) for c in candidates]
    reports.sort(key=lambda r: (r.virality_score, r.stats.reach_median), reverse=True)

    imp = Improvement(original=original, variants=reports, generated_by=generated_by)

    # Flywheel seam: the (original, variants, winner, predicted lift) tuple is
    # exactly the supervision a rewrite model wants. No-op unless a sink is set.
    get_tracer().emit(
        {
            "type": "improvement",
            "original": text,
            "generated_by": generated_by,
            "candidates": [
                {"tweet": r.tweet, "virality_score": r.virality_score} for r in reports
            ],
            "original_score": original.virality_score,
            "lift": imp.lift(),
        }
    )

    return imp
