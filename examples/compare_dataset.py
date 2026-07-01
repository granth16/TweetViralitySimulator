"""Score many tweets from a local CSV and compare to actual engagement.

No X API. No TVS cloud API. Just the open engine + a dataset file.

Default dataset: Kaggle mmmarchetti/tweets-dataset (~52k tweets, all famous accounts).

By default personal celebrities are excluded; only brand/news handles remain
(Twitter, YouTube, cnnbrk, instagram — ~10k tweets). The Kaggle file has no
regular users; for those you need another CSV.

Usage:
    pip install kagglehub   # only if you don't pass --csv

    # Full dataset with per-author follower counts (~52k tweets)
    python examples/compare_dataset.py --sample 10000 --include-celebrities \\
        --runs 40 --out results/compare_10k.csv

    # Brand/news only (~10k tweets, no celeb handles)
    python examples/compare_dataset.py --sample 10000 --no-followers

    # Your own CSV (columns: tweet or content, likes, retweets; optional: author)
    python examples/compare_dataset.py --csv data/my_tweets.csv --sample 200
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from tweet_virality_simulator import Config, analyze

_W_LIKE = 1.0
_W_RT = 2.0

# Personal / entertainer accounts in mmmarchetti/tweets-dataset (not brands).
_KAGGLE_CELEBRITY_AUTHORS = frozenset(
    {
        "ArianaGrande",
        "BarackObama",
        "Cristiano",
        "KimKardashian",
        "TheEllenShow",
        "britneyspears",
        "ddlovato",
        "jimmyfallon",
        "jtimberlake",
        "justinbieber",
        "katyperry",
        "ladygaga",
        "rihanna",
        "selenagomez",
        "shakira",
        "taylorswift13",
    }
)

# Approximate follower counts when this Kaggle dump was collected (~2016–2017).
# The CSV has no follower column; these are used only for simulation seed size.
_KAGGLE_FOLLOWERS: dict[str, int] = {
    "justinbieber": 91_000_000,
    "taylorswift13": 83_000_000,
    "katyperry": 95_000_000,
    "BarackObama": 78_000_000,
    "ladygaga": 64_000_000,
    "TheEllenShow": 65_000_000,
    "rihanna": 59_000_000,
    "YouTube": 60_000_000,
    "Twitter": 55_000_000,
    "Cristiano": 52_000_000,
    "KimKardashian": 48_000_000,
    "instagram": 45_000_000,
    "selenagomez": 45_000_000,
    "ArianaGrande": 42_000_000,
    "jimmyfallon": 42_000_000,
    "shakira": 42_000_000,
    "ddlovato": 41_000_000,
    "jtimberlake": 41_000_000,
    "britneyspears": 36_000_000,
    "cnnbrk": 25_000_000,
}


def _effective_followers(raw: int | None, *, default: int = 60) -> int:
    """Map real follower counts into the sim's in-network seed cap (~150 at n=1000)."""
    if raw is None or raw <= 0:
        return default
    if raw < 5_000:
        return max(10, raw // 100)
    # log-scale: 10k→~30, 1M→~85, 100M→~148 (capped at 150)
    return int(min(150, max(10, (math.log10(raw) - 2.0) * 25.0)))


def _parse_followers(row: dict[str, str]) -> int | None:
    for key in ("followers", "follower_count", "follower_cnt", "num_followers"):
        raw = row.get(key)
        if raw not in (None, ""):
            try:
                return int(float(raw))
            except ValueError:
                pass
    return None


def _followers_for(author: str, row: dict[str, str], *, use_map: bool) -> int:
    raw = _parse_followers(row)
    if raw is None and use_map:
        raw = _KAGGLE_FOLLOWERS.get(author)
    return _effective_followers(raw)


def _engagement(likes: int, retweets: int) -> float:
    return _W_LIKE * likes + _W_RT * retweets


def _default_csv() -> Path:
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit("pip install kagglehub  (or pass --csv /path/to/file.csv)") from e
    root = Path(kagglehub.dataset_download("mmmarchetti/tweets-dataset"))
    return root / "tweets.csv"


def _load_rows(path: Path, *, exclude_celebrities: bool, use_followers: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            author = row.get("author") or row.get("user") or ""
            if exclude_celebrities and author in _KAGGLE_CELEBRITY_AUTHORS:
                skipped += 1
                continue
            text = (row.get("tweet") or row.get("content") or row.get("text") or "").strip()
            if not text or text.startswith("RT @"):
                continue
            try:
                likes = int(float(row.get("likes") or row.get("number_of_likes") or 0))
                rts = int(float(row.get("retweets") or row.get("number_of_shares") or 0))
            except ValueError:
                continue
            rows.append(
                {
                    "author": author,
                    "tweet": text,
                    "likes": likes,
                    "retweets": rts,
                    "followers_raw": _parse_followers(row)
                    or (_KAGGLE_FOLLOWERS.get(author) if use_followers else None),
                    "author_followers": _followers_for(author, row, use_map=use_followers),
                }
            )
    if exclude_celebrities and skipped:
        print(
            f"excluded {skipped} celebrity tweets "
            f"(kept {len(rows)} from brand/news accounts)",
            file=sys.stderr,
        )
    return rows


def _rel_amp(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_author: dict[str, list[float]] = {}
    for r in rows:
        by_author.setdefault(r["author"], []).append(_engagement(r["likes"], r["retweets"]))

    out: list[dict[str, Any]] = []
    for r in rows:
        vals = by_author.get(r["author"], [])
        obs = _engagement(r["likes"], r["retweets"])
        others = [v for v in vals if v != obs] or vals
        baseline = float(statistics.median(others)) if others else max(obs, 1.0)
        out.append({**r, "engagement": obs, "baseline": baseline, "rel_amp": obs / max(baseline, 1.0)})
    return out


def _spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 0.0

    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    denom = (sum(x * x for x in ra) * sum(x * x for x in rb)) ** 0.5
    return sum(x * y for x, y in zip(ra, rb)) / denom if denom else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Local score vs actual engagement (no API)")
    p.add_argument("--csv", help="Dataset CSV path (default: download Kaggle tweets-dataset)")
    p.add_argument("--sample", type=int, default=10_000, help="How many tweets to score (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=40, help="Monte Carlo runs per tweet")
    p.add_argument("--audience", type=int, default=1000, help="Synthetic audience size")
    p.add_argument("--out", help="Write results CSV here (still prints summary to stderr)")
    p.add_argument(
        "--include-celebrities",
        action="store_true",
        help="Keep personal celeb accounts (needed for 10k+ sample on Kaggle CSV)",
    )
    p.add_argument(
        "--no-followers",
        action="store_true",
        help="Ignore follower map; use default author_followers=60 for everyone",
    )
    args = p.parse_args()

    path = Path(args.csv) if args.csv else _default_csv()
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    exclude_celebs = not args.include_celebrities
    use_followers = not args.no_followers
    all_rows = _rel_amp(_load_rows(path, exclude_celebrities=exclude_celebs, use_followers=use_followers))
    if not all_rows:
        raise SystemExit(
            "no tweets left after filters — use --include-celebrities for the full Kaggle set "
            "or pass a different --csv"
        )
    if args.sample <= 0 or args.sample >= len(all_rows):
        rows = all_rows
    else:
        rng = random.Random(args.seed)
        rows = rng.sample(all_rows, args.sample)
    print(
        f"scoring {len(rows)} / {len(all_rows)} tweets from {path} "
        f"(followers={'mapped' if use_followers else 'default=60'})",
        file=sys.stderr,
    )

    base_cfg = Config(runs=args.runs, audience_size=args.audience)
    results: list[dict[str, Any]] = []

    for i, r in enumerate(rows, 1):
        has_media = "https://t.co/" in r["tweet"] or "pic.twitter.com" in r["tweet"]
        cfg = replace(base_cfg, author_followers=r["author_followers"])
        report = analyze(r["tweet"], cfg, has_media=has_media)
        row = {
            "author": r["author"],
            "followers_raw": r.get("followers_raw") or "",
            "author_followers": r["author_followers"],
            "score": report.virality_score,
            "verdict": report.verdict,
            "reach_median": report.stats.reach_median,
            "likes": r["likes"],
            "retweets": r["retweets"],
            "engagement": round(r["engagement"], 1),
            "rel_amp": round(r["rel_amp"], 3),
            "tweet": r["tweet"][:120],
        }
        results.append(row)
        print(
            f"[{i}/{len(rows)}] @{r['author']} followers={r['author_followers']:3d} "
            f"score={row['score']:3d}  likes={row['likes']}  rel_amp={row['rel_amp']:.2f}",
            file=sys.stderr,
        )

    scores = [float(r["score"]) for r in results]
    rels = [float(r["rel_amp"]) for r in results]
    likes = [float(r["likes"]) for r in results]
    engs = [float(r["engagement"]) for r in results]

    print("\n--- summary ---", file=sys.stderr)
    print(f"tweets scored:     {len(results)}", file=sys.stderr)
    print(f"score vs rel_amp:  spearman={_spearman(scores, rels):.3f}  (fair compare — same author baseline)", file=sys.stderr)
    print(f"score vs likes:    spearman={_spearman(scores, likes):.3f}  (biased by account size)", file=sys.stderr)
    print(f"score vs engagement (likes+2*RT): spearman={_spearman(scores, engs):.3f}", file=sys.stderr)

    fields = list(results[0].keys())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_stream = open(args.out, "w", newline="", encoding="utf-8") if args.out else sys.stdout
    try:
        w = csv.DictWriter(out_stream, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    finally:
        if args.out:
            out_stream.close()
            print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
