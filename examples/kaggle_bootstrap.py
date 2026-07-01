"""Prepare and score tweets from the Kaggle mmmarchetti/tweets-dataset.

Dataset: ~52k tweets from 20 celebrity accounts (2009–2017), columns:
  author, content, number_of_likes, number_of_shares, date_time, ...

Good for:
  * bulk engine testing (local simulation)
  * bootstrap training labels (author-relative amplification)

Not good for:
  * claiming SOTA on modern X (old platform era, no impressions/replies/quotes)

Usage:
    pip install kagglehub

    # Download + score 50 tweets locally (fast engine test)
    python examples/kaggle_bootstrap.py --sample 50 --score-local

    # Write a bootstrap CSV (for review or API ingest)
    python examples/kaggle_bootstrap.py --sample 200 --out data/kaggle_bootstrap.csv

    # Push to live TVS API (predict + manual outcome)
    python examples/kaggle_bootstrap.py --sample 100 --ingest-api http://157.245.233.161 \\
        --api-key YOUR_KEY
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tweet_virality_simulator import Config, analyze

# Match tvs-cloud/app/ingest/labels.py public-path weights (likes + RT only here).
_W_LIKE = 1.0
_W_RT = 2.0


def _engagement(likes: int, retweets: int) -> float:
    return _W_LIKE * likes + _W_RT * retweets


def _dataset_path(csv_path: str | None) -> Path:
    if csv_path:
        return Path(csv_path)
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit("pip install kagglehub") from e
    root = Path(kagglehub.dataset_download("mmmarchetti/tweets-dataset"))
    return root / "tweets.csv"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("content") or "").strip()
            if not text or text.startswith("RT @"):
                continue
            try:
                likes = int(float(row.get("number_of_likes") or 0))
                rts = int(float(row.get("number_of_shares") or 0))
            except ValueError:
                continue
            rows.append(
                {
                    "author": row.get("author") or "",
                    "tweet": text,
                    "likes": likes,
                    "retweets": rts,
                    "replies": 0,
                    "quotes": 0,
                    "date": row.get("date_time") or "",
                }
            )
    return rows


def _add_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Leave-one-out author median baseline → rel_amplification."""
    by_author: dict[str, list[float]] = {}
    for r in rows:
        by_author.setdefault(r["author"], []).append(_engagement(r["likes"], r["retweets"]))

    out: list[dict[str, Any]] = []
    for r in rows:
        vals = by_author.get(r["author"], [])
        obs = _engagement(r["likes"], r["retweets"])
        others = [v for v in vals if v != obs] or vals
        baseline = float(statistics.median(others)) if others else max(obs, 1.0)
        rel = obs / max(baseline, 1.0)
        out.append({**r, "baseline": baseline, "rel_amp": rel})
    return out


def _sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


def _score_local(rows: list[dict[str, Any]], runs: int) -> None:
    cfg = Config(runs=runs)
    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["author", "score", "verdict", "rel_amp", "likes", "retweets", "tweet"]
    )
    for i, r in enumerate(rows, 1):
        has_media = "https://t.co/" in r["tweet"] or "pic.twitter.com" in r["tweet"]
        report = analyze(r["tweet"], cfg, has_media=has_media)
        writer.writerow(
            [
                r["author"],
                report.virality_score,
                report.verdict,
                round(r["rel_amp"], 3),
                r["likes"],
                r["retweets"],
                r["tweet"][:100],
            ]
        )
        print(f"[{i}/{len(rows)}] {report.virality_score:3d}  rel={r['rel_amp']:.2f}", file=sys.stderr)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "author",
        "tweet",
        "likes",
        "retweets",
        "replies",
        "quotes",
        "baseline",
        "rel_amp",
        "date",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {len(rows)} rows → {path}", file=sys.stderr)


def _resolve_api_key(cli_key: str) -> str:
    import os

    if cli_key:
        return cli_key
    if os.environ.get("TVS_API_KEY"):
        return os.environ["TVS_API_KEY"]
    # .env uses TVS_API_KEYS (comma-separated); take the first key.
    keys = os.environ.get("TVS_API_KEYS", "")
    for k in keys.split(","):
        k = k.strip()
        if k:
            return k
    return ""


def _post(url: str, api_key: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def _ingest_api(rows: list[dict[str, Any]], base_url: str, api_key: str, delay: float) -> None:
    base = base_url.rstrip("/")
    for i, r in enumerate(rows, 1):
        has_media = "https://t.co/" in r["tweet"] or "pic.twitter.com" in r["tweet"]
        try:
            # Do NOT pass user_id here: API user_id is a FK to users.id (OAuth),
            # not a Kaggle author handle. Invalid IDs make PostgresTracer fail
            # silently → empty prediction_id.
            pred = _post(
                f"{base}/predict",
                api_key,
                {"tweet": r["tweet"], "has_media": has_media},
            )
            pred_id = (pred.get("prediction_id") or "").strip()
            if not pred_id:
                print(
                    f"[{i}] predict returned no prediction_id (DB log failed?). "
                    f"response={pred!r}",
                    file=sys.stderr,
                )
                continue
            out = _post(
                f"{base}/outcomes/manual",
                api_key,
                {
                    "prediction_id": pred_id,
                    "likes": r["likes"],
                    "retweets": r["retweets"],
                    "replies": 0,
                    "quotes": 0,
                    "source": "public",
                    "rel_amplification_override": round(r["rel_amp"], 4),
                    "author_baseline": round(r["baseline"], 2),
                },
            )
            print(
                f"[{i}/{len(rows)}] id={pred_id[:8]}… "
                f"score={pred['virality_score']} rel={out['rel_amplification']}",
                file=sys.stderr,
            )
        except urllib.error.HTTPError as e:
            print(f"[{i}] HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        time.sleep(delay)


def main() -> None:
    p = argparse.ArgumentParser(description="Kaggle tweets → TVS bootstrap")
    p.add_argument("--csv", help="Path to tweets.csv (default: download via kagglehub)")
    p.add_argument("--sample", type=int, default=50, help="How many tweets to use")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=60, help="Monte Carlo runs for --score-local")
    p.add_argument("--out", help="Write prepared CSV here")
    p.add_argument("--score-local", action="store_true", help="Run engine locally, print CSV")
    p.add_argument("--ingest-api", help="TVS base URL, e.g. http://157.245.233.161")
    p.add_argument("--api-key", default="", help="X-API-Key (or set TVS_API_KEY)")
    p.add_argument("--delay", type=float, default=0.25, help="Seconds between API calls")
    args = p.parse_args()

    api_key = _resolve_api_key(args.api_key)

    path = _dataset_path(args.csv)
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    rows = _add_labels(_load_rows(path))
    rows = _sample(rows, args.sample, args.seed)
    print(f"using {len(rows)} tweets from {path}", file=sys.stderr)

    if args.out:
        _write_csv(rows, Path(args.out))
    if args.score_local:
        _score_local(rows, args.runs)
    if args.ingest_api:
        if not api_key:
            raise SystemExit("need --api-key or TVS_API_KEY for --ingest-api")
        _ingest_api(rows, args.ingest_api, api_key, args.delay)

    if not (args.out or args.score_local or args.ingest_api):
        p.print_help()
        raise SystemExit("\nPick at least one of: --score-local, --out, --ingest-api")


if __name__ == "__main__":
    main()
