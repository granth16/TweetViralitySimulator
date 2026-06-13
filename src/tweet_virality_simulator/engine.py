"""Top-level orchestration: text in, Report out."""

from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np

from .cascade import simulate
from .config import Config
from .models import Driver, Report, TweetDNA
from .platforms.x import network as net
from .platforms.x import tweet_dna
from .population import audience as pop
from .providers import get_provider


def _seed_from(text: str, cfg: Config) -> int:
    if cfg.seed is not None:
        return int(cfg.seed)
    # Deterministic per tweet: same tweet -> same population -> same answer.
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)


def _score(stats, cfg: Config) -> int:
    # Amplification: how far past the author's own seed audience it traveled.
    seed = max(stats.seed_size, 1)
    amplification = stats.reach_median / seed
    amp_component = min(max(amplification - 1.0, 0.0) / 4.0, 1.0)  # 5x seed -> full
    r_component = min(stats.reproduction_number / 1.2, 1.0)  # R>=1.2 -> full
    score = 0.5 * amp_component + 0.3 * r_component + 0.2 * stats.p_viral
    return int(round(100 * float(np.clip(score, 0.0, 1.0))))


def _verdict(score: int) -> str:
    if score >= 75:
        return "Viral potential"
    if score >= 50:
        return "Has legs"
    if score >= 25:
        return "Limited spread"
    return "Likely to fizzle"


def _drivers(dna: TweetDNA) -> List[Driver]:
    out: List[Driver] = []
    if dna.hook_strength >= 0.5:
        out.append(Driver(label="Strong hook", impact=0.9 * dna.hook_strength,
                           detail="The opening grabs attention and earns the first second."))
    if dna.emotional_intensity >= 0.45:
        out.append(Driver(label="Emotionally charged", impact=0.8 * dna.emotional_intensity,
                          detail="Moral-emotional language reliably lifts retweets."))
    if dna.novelty >= 0.45:
        out.append(Driver(label="Feels novel", impact=0.7 * dna.novelty,
                          detail="Novelty makes people want to be first to share."))
    if dna.controversy >= 0.4:
        out.append(Driver(label="Debate-bait", impact=0.6 * dna.controversy,
                          detail="Sparks replies — and replies are the highest-weighted signal."))
    if dna.relatability >= 0.5:
        out.append(Driver(label="Highly relatable", impact=0.5 * dna.relatability,
                          detail="'That's so me' content gets quote-tweeted."))
    if dna.has_media:
        out.append(Driver(label="Has media", impact=0.3,
                          detail="Media boosts dwell time and reach."))
    out.sort(key=lambda d: d.impact, reverse=True)
    return out[:5]


def _weaknesses(dna: TweetDNA) -> List[Driver]:
    out: List[Driver] = []
    if dna.has_link:
        out.append(Driver(label="External link", impact=-0.6,
                          detail="X downranks posts with outbound links — consider replying with the link."))
    if dna.hook_strength < 0.3:
        out.append(Driver(label="Weak hook", impact=-0.5,
                          detail="No clear reason to stop scrolling in the first second."))
    if dna.emotional_intensity < 0.2:
        out.append(Driver(label="Emotionally flat", impact=-0.4,
                          detail="Little emotional charge — emotion is what gets shared."))
    if dna.length_chars > 220:
        out.append(Driver(label="Too long", impact=-0.3,
                          detail="Long posts lose readers before the payoff."))
    if dna.num_hashtags > 3:
        out.append(Driver(label="Hashtag stuffing", impact=-0.25,
                          detail="Too many hashtags reads as spammy and can suppress reach."))
    if dna.clarity < 0.3:
        out.append(Driver(label="Hard to parse", impact=-0.3,
                          detail="If it takes effort to understand, it won't be shared."))
    out.sort(key=lambda d: d.impact)
    return out[:5]


def _roast(dna: TweetDNA, score: int) -> str:
    if dna.has_link:
        return "You buried an external link in the post — the algorithm just yawned and scrolled past."
    if dna.hook_strength < 0.3:
        return "The hook is so soft it apologized for existing. Nobody's stopping for this."
    if dna.emotional_intensity < 0.2 and dna.controversy < 0.2:
        return "Emotionally this reads like a tax form. Give people a reason to feel something."
    if score >= 75:
        return "Annoyingly good. This has the shape of something that actually moves."
    if score >= 50:
        return "Decent — it has legs, but it's not going to keep anyone up at night."
    return "It's fine. 'Fine' does not trend."


def analyze(
    text: str,
    config: Optional[Config] = None,
    has_media: bool = False,
) -> Report:
    """Run the full pipeline on a tweet and return a Report."""
    cfg = config or Config()
    if not text or not text.strip():
        raise ValueError("tweet text is empty")

    rng = np.random.default_rng(_seed_from(text, cfg))

    provider = get_provider(cfg.provider)
    dna = tweet_dna.extract(text, provider, has_media=has_media)

    # numpy 2.0 on macOS/Accelerate emits spurious FP warnings from matmul on
    # perfectly finite data; silence them around the numeric core.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        audience = pop.generate(cfg, rng)
        network = net.build(audience, cfg, rng)
        stats = simulate(dna, audience, network, cfg, rng)

    score = _score(stats, cfg)
    return Report(
        tweet=text,
        virality_score=score,
        verdict=_verdict(score),
        roast=_roast(dna, score),
        dna=dna,
        stats=stats,
        drivers=_drivers(dna),
        weaknesses=_weaknesses(dna),
        config=cfg.to_dict(),
    )
