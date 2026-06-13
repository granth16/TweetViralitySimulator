"""Minimal Python-API demo.

    python examples/demo.py
"""

from tweet_virality_simulator import Config, analyze
from tweet_virality_simulator.report import render_report

TWEETS = [
    "Nobody talks about this AI trick that saves me 10 hours a week. Here's exactly how it works:",
    "Just had lunch. The sandwich was okay I guess.",
    "Unpopular opinion: remote work made most people worse at their jobs, not better. Fight me.",
    "Read my new 4,000-word blog post about productivity systems https://example.com/blog/productivity",
]

for tweet in TWEETS:
    report = analyze(tweet, Config(runs=80))
    render_report(report)
