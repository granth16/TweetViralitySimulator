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


## Quick start

```bash
pip install -e .
tvs x "Nobody talks about this AI trick that saves me 10 hours a week. Here's how:"
```

No keys, no accounts. You get the report card above in your terminal.

## Rewrite a tweet for more spread

Don't just score it — *fix* it. `tvs improve` generates rewrite candidates,
simulates **every one on the same synthetic audience**, and hands you the
winner with the predicted lift. The simulator is the judge, not the generator.

```text
$ tvs improve "Read my new blog post about productivity. https://example.com #tips"

╭─  ↑ +100 points   0 → 100 ──────────────────────────────────────────────────╮
│   ORIGINAL                                                                   │
│   "Read my new blog post about productivity. https://example.com #tips"      │
│   score 0/100 · Likely to fizzle · reach 60/500                              │
│                                                                              │
│  ╭─ BEST REWRITE · 100/100 · Viral potential ─────────────────────────────╮ │
│  │ Unpopular opinion: most productivity advice is procrastination in       │ │
│  │ disguise.                                                                │ │
│  ╰──────────────────────────────────────────────────────────────────────╯ │
│   reach 500/500 · viral odds 98% · R 1.57                                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Works with **zero setup** (a heuristic rewriter applies known viral formats to
*your* message). Point it at a model for real rewrites: `-p ollama` (local),
`-p openai`, or `-p compat` (any OpenAI-compatible endpoint).

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
tvs x "your tweet"                  # score a tweet, zero setup
tvs improve "your tweet"            # rewrite for spread, ranked by the simulator
tvs improve "your tweet" -n 10      # try more rewrite candidates
tvs x "your tweet" -p ollama        # score the tweet with a local model
tvs x "your tweet" -p openai        # score with OpenAI (needs OPENAI_API_KEY)
tvs x "your tweet" -p compat        # score with any OpenAI-compatible endpoint
tvs x "your tweet" -a 2000 -r 200   # bigger audience, more runs
tvs x "your tweet" --profile p.json # load a calibration profile artifact
tvs x "your tweet" --json           # machine-readable output
tvs x "your tweet" --save card.svg  # export the report card as an image
```

### Python API

```python
from tweet_virality_simulator import analyze, improve

report = analyze("Unpopular opinion: remote work made people worse at their jobs.")
print(report.virality_score, report.verdict)
for w in report.weaknesses:
    print("-", w.label, w.detail)

# Rewrite for spread — every candidate simulated on the same population.
result = improve("Read my new blog post about productivity. https://example.com")
print(f"+{result.lift()} points")
print(result.best().tweet)
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

## Extending it

The engine is pluggable by design — every part you'd want to swap is behind a
clean seam, so you can bring your own model, calibration, and infrastructure
**without editing the engine**.

| Seam | What it's for | How to attach |
| --- | --- | --- |
| **Provider** (`providers/`) | the tweet scorer | built-in `heuristic` / `openai` / `ollama`, or `compat` for any OpenAI-compatible endpoint (incl. your own fine-tuned model) |
| **Profile** (`profile.py`) | every tunable number — reaction scales, network shape, algo thresholds, trait priors | ship a fitted `Profile` JSON and load it at runtime; the engine code never changes |
| **Tracer** (`tracing/`) | emit one `(tweet, dna, params, prediction)` record per run | no-op by default; set `TVS_TRACE_PATH` for local JSONL, or register a custom sink |
| **Store** (`storage/`) | persist reports / datasets / artifacts | `LocalStore` (JSON on disk) ships by default; implement `Store` for Postgres/S3/etc. |

```python
from tweet_virality_simulator import analyze, Config, Profile, load_profile

# Calibration lives in data, not code — load a fitted profile artifact.
report = analyze("your tweet", Config(profile_path="fitted_x.json"))

# Or build/override one programmatically.
profile = Profile(name="my_fit", promotion_threshold=0.7, appeal_scale=1.9)
report = analyze("your tweet", profile=profile)
```

```python
# Point the engine at any OpenAI-compatible model server.
import os
os.environ["TVS_COMPAT_BASE_URL"] = "https://your-endpoint/v1"
os.environ["TVS_COMPAT_API_KEY"]  = "..."          # optional
report = analyze("your tweet", Config(provider="compat"))
```

```python
# Wire your own outcome sink (the data flywheel) without touching the engine.
from tweet_virality_simulator import set_tracer, Tracer

class MySink(Tracer):
    def emit(self, record: dict) -> None:
        ...  # ship to your warehouse / queue

set_tracer(MySink())
```

Config resolution is explicit-arg → env var → built-in default, so nothing
proprietary is required to run: the open package ships only the generic
`default` profile and a local store, and works fully standalone.

---

## Status & honest roadmap

- **v0.1 (now):** a working, mechanistic simulator with sensible *priors*. It
  reproduces known effects (emotion/hooks lift spread, external links suppress
  it) but is **not yet calibrated** against real outcomes — so don't read the
  numbers as accurate predictions, read them as relative signals. All tunable
  parameters now live in a loadable [`Profile`](#extending-it), and the model,
  tracing, and storage seams are in place.
- **v0.2 — calibration & benchmark:** a validation harness that fits the
  `Profile` against real public cascade datasets, then reports accuracy on a
  **held-out** set (ranking correlation + calibration error) so improvements are
  measurable, not vibes. Ships a reproducible benchmark you can run yourself —
  that's what turns "plausible" into "grounded."
- **v0.3 — the learning loop:** the [`tracing`](#extending-it) hook closes into
  an outcome-ingestion loop — predictions are compared against observed spread
  and fed back to re-fit the `Profile` / train the reaction model, so accuracy
  compounds over time instead of staying static.
- **Managed option (later):** a hosted API + web UI for people who'd rather not
  self-host or self-calibrate — same open engine underneath, loaded with a
  fitted profile. The engine stays open and fully usable standalone forever.

## License

Apache-2.0.
