"""X "heavy ranker" engagement weights.

These are the *concepts and published weights* from X's open-sourced
recommendation algorithm, re-implemented from scratch (no code copied — that
repo is AGPL-3.0). The heavy ranker scores a candidate post as a weighted sum
of predicted-engagement probabilities; the weights are highly asymmetric, which
is the whole point: a reply the author engages back with is worth far more than
a like, and negative feedback is catastrophic.

Source (concepts only): https://github.com/twitter/the-algorithm and the X
engineering blog on the recommendation algorithm (2023). Treat the exact numbers
as documented defaults; verify against the repo's scored-tweets weights if you
need the canonical constants.
"""

from __future__ import annotations

# action -> weight in the engagement score
HEAVY_RANKER_WEIGHTS = {
    "like": 0.5,
    "retweet": 1.0,
    "reply": 13.5,
    "quote": 1.0,
    # Modeled but not directly emitted by v0.1's reaction sampler:
    "reply_engaged_by_author": 75.0,
    "profile_click_engaged": 12.0,
    "video_watch_half": 0.005,
    "negative_feedback": -74.0,
    "report": -369.0,
}

# Documented ~50/50 split between in-network (follow graph) and out-of-network
# (algorithmically recommended) candidate sources in the For You timeline.
IN_NETWORK_CANDIDATE_SHARE = 0.5
OUT_NETWORK_CANDIDATE_SHARE = 0.5
