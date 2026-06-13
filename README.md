# TweetViralitySimulator

**Simulate whether your tweet spreads on X — before you post.**

Paste a tweet. It builds a synthetic X audience, models how the For-You
algorithm and the follower graph would carry it, runs thousands of simulations,
and tells you how far it spreads — and *why*.

Runs **fully locally, zero setup, no API key.**

<p align="center">
  <img src="assets/card.svg" alt="TweetViralitySimulator report card" width="760">
</p>

<details>
<summary>plain-text version</summary>

```text
$ tvs x "Unpopular opinion: remote work made most people worse at their jobs. Fight me."

╭─  100/100   Viral potential ───────────────────────────────────────────────╮
│   Reach (median)            947 / 1000 accounts                             │
│   Viral odds                98%  (≥300 reached)                             │
│   Reproduction number R     2.13  (grows)                                   │
│   In- vs out-of-network     54% follow graph · 46% For You                  │
│   spread per round          ▄█▅▂▂▁▁                                         │
│   ✓ Strong hook   ✓ Debate-bait                                             │
╰─────────────────────────────────────────────────────────────────────────────╯
```

</details>

> ⚠️ **Simulation, not a crystal ball.** Treat the numbers as *relative*
> indicators (will A out-spread B? where's the weak point?), not as guaranteed
> view counts. Reach is "accounts reached in the simulated audience," not real
> impressions — even X can't perfectly predict virality.

---

## Quick start

```bash
pip install -e .
tvs x "Nobody talks about this AI trick that saves me 10 hours a week. Here's how:"
```

No keys, no accounts. You get the report card above in your terminal.

## A/B test two drafts

The most useful mode — relative comparison is far more reliable than absolute
prediction. Both versions run on the **same** synthetic population.

```text
$ tvs compare "Read my new blog post about productivity https://example.com" \
              "Unpopular opinion: most productivity advice is procrastination in disguise."

╭─ A/B comparison ────────────────────────────────────────────────╮
│                              Version A            Version B       │
│   Virality score                 0/100              100/100       │
│   Verdict             Likely to fizzle      Viral potential       │
│   Reach (median)                    60                  947       │
│   R                               0.00                 1.41       │
│                                                                  │
│   Version B is 15.8× more likely to spread than Version A        │
╰──────────────────────────────────────────────────────────────────╯
```

## More options

```bash
tvs x "your tweet"                 # heuristic scorer, zero setup
tvs x "your tweet" -p ollama       # score the tweet with a local model
tvs x "your tweet" -p openai       # score with OpenAI (needs OPENAI_API_KEY)
tvs x "your tweet" -a 2000 -r 200  # bigger audience, more runs
tvs x "your tweet" --json          # machine-readable output
tvs x "your tweet" --save card.svg # export the report card as an image
```

### Python API

```python
from tweet_virality_simulator import analyze

report = analyze("Unpopular opinion: remote work made people worse at their jobs.")
print(report.virality_score, report.verdict)
for w in report.weaknesses:
    print("-", w.label, w.detail)
```

---

## How it works

```
tweet ──► Tweet DNA (hook, emotion, novelty, controversy, link penalty, ...)   [provider]
              │
        synthetic audience (interest embeddings + popularity + traits)
              │
        follower graph (power-law in-degree + homophily)
              │
   Monte Carlo cascade ×N, seeded & deterministic:
       • in-network channel   (retweets propagate to followers)
       • out-of-network channel (For-You injection: engagement-gated,
         similarity-targeted pools)
              │
        report: score · reach distribution · R · drivers / weaknesses
```

The LLM (optional) only *scores the tweet*. The simulation itself is pure,
seeded numpy — so the same tweet gives the same answer every run.

### Grounding

The For-You injection and engagement weighting are re-implemented **from the
publicly documented concepts** in X's open-sourced recommendation algorithm —
the ~50/50 in-network vs out-of-network candidate split and the heavy ranker's
asymmetric engagement weights (a reply the author engages back with is worth far
more than a like). No code is copied; that repo is AGPL-3.0 and this project is
Apache-2.0. Source of concepts: <https://github.com/twitter/the-algorithm>.

---

## Status & honest roadmap

- **v0.1 (now):** a working, mechanistic simulator with sensible *priors*. It
  reproduces known effects (emotion/hooks lift spread, external links suppress
  it) but is **not yet calibrated** against real outcomes — so don't read the
  numbers as accurate predictions, read them as relative signals.
- **v0.2:** a calibration/validation harness that fits the audience + algorithm
  parameters against real public cascade datasets and ships a reproducible
  benchmark. That's what turns "plausible" into "grounded."

## License

Apache-2.0.
