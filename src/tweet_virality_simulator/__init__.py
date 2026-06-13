"""TweetViralitySimulator — simulate whether a tweet spreads on X, before you post.

Open-source virality simulator. The engine is fully local and runs without any
API keys (heuristic provider); LLM-backed tweet scoring is optional.

Quick start (Python):

    from tweet_virality_simulator import analyze
    report = analyze("Nobody talks about this AI trick...")
    print(report.virality_score, report.verdict)
"""

from .config import Config
from .engine import analyze
from .models import CascadeStats, Driver, Report, TweetDNA

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "Config",
    "Report",
    "TweetDNA",
    "CascadeStats",
    "Driver",
    "__version__",
]
