# TweetViralitySimulator

**Simulate whether your tweet spreads on X — before you post.**

TweetViralitySimulator builds a synthetic X audience, extracts your tweet's
"DNA," and runs thousands of Monte-Carlo contagion simulations across a follower
graph *and* a model of the For-You algorithm. You get a virality score, a
predicted reach distribution, the reproduction number **R**, and — most useful —
*why* it will or won't spread.

It runs **fully locally with zero setup and no API key** (a deterministic
heuristic scorer). Plug in an LLM (OpenAI or a local Ollama model) for sharper
tweet scoring if you want.

> ⚠️ This is a **simulation, not a crystal ball.** Treat outputs as *relative*
> indicators (is variant A likely to out-spread B? where's the weak point?), not
> as absolute view-count predictions. Even X can't perfectly predict virality.

---

## Quick start

```bash
pip install -e .
tvs x "Nobody talks about this AI trick that saves me 10 hours a week. Here's how:"
```

That's it — no keys, no accounts. You'll get a report card in your terminal.

### Python API

```python
from tweet_virality_simulator import analyze

report = analyze("Unpopular opinion: remote work made most people worse at their jobs.")
print(report.virality_score, report.verdict)
for w in report.weaknesses:
    print("-", w.label, w.detail)
```

### Options

```bash
tvs x "your tweet"                 # heuristic, zero setup
tvs x "your tweet" -p ollama       # score DNA with a local model
tvs x "your tweet" -p openai       # score DNA with OpenAI (needs OPENAI_API_KEY)
tvs x "your tweet" -a 2000 -r 200  # bigger audience, more runs
tvs x "your tweet" --json          # machine-readable output
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
       • in-network channel  (retweets propagate to followers)
       • out-of-network channel (For-You injection: engagement-gated,
         similarity-targeted pools)
              │
        report: score · reach distribution · R · drivers / weaknesses
```

The LLM only scores the tweet (cheap, optional). The simulation itself is pure,
seeded numpy — so the same tweet gives the same answer every run.

### Grounding

The For-You injection and engagement weighting are re-implemented **from the
publicly documented concepts** in X's open-sourced recommendation algorithm —
e.g. the ~50/50 in-network vs out-of-network candidate split and the heavy
ranker's asymmetric engagement weights (a reply the author engages back with is
worth far more than a like). No code is copied; that repository is AGPL-3.0 and
this project is Apache-2.0. Source of concepts:
<https://github.com/twitter/the-algorithm>.

---

## Status

v0.1 ships with sensible priors. A calibration/validation harness (fitting the
audience + algorithm parameters against real cascade datasets) is on the
roadmap — that's what turns "plausible" into "grounded."

## License

Apache-2.0.
