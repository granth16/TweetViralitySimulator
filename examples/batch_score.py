"""Score many tweets through the engine and print a CSV summary.

Usage:
    # Built-in face-validity benchmark (24 tweets)
    python examples/batch_score.py

    # One tweet per line
    python examples/batch_score.py tweets.txt

    # Faster smoke test (fewer Monte Carlo runs)
    python examples/batch_score.py tweets.txt --runs 40

    # Save full JSON per tweet
    python examples/batch_score.py tweets.txt --json-dir out/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from tweet_virality_simulator import Config, analyze
from tweet_virality_simulator.validation.dataset import LABELED


def _load_tweets(path: Path | None) -> list[tuple[str, int | None]]:
    if path is None:
        return [(t, tier) for t, tier in LABELED]

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"empty file: {path}")

    if path.suffix.lower() == ".csv":
        rows: list[tuple[str, int | None]] = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tweet = (row.get("tweet") or row.get("text") or "").strip()
                if not tweet:
                    continue
                tier_raw = row.get("tier") or row.get("expected_tier")
                tier = int(tier_raw) if tier_raw not in (None, "") else None
                rows.append((tweet, tier))
        return rows

    return [(line.strip(), None) for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score tweets with TVS")
    parser.add_argument(
        "file",
        nargs="?",
        help="Text file (one tweet per line) or CSV with 'tweet' column. Omit for built-in benchmark.",
    )
    parser.add_argument("--runs", type=int, default=80, help="Monte Carlo runs per tweet")
    parser.add_argument("--audience", type=int, default=1000)
    parser.add_argument("--followers", type=int, default=60)
    parser.add_argument("--profile", type=str, default=None, help="Calibration profile JSON")
    parser.add_argument("--media", action="store_true", help="Treat all tweets as having media")
    parser.add_argument("--json-dir", type=str, default=None, help="Write full report JSON per tweet here")
    args = parser.parse_args()

    path = Path(args.file) if args.file else None
    tweets = _load_tweets(path)
    cfg = Config(
        audience_size=args.audience,
        runs=args.runs,
        author_followers=args.followers,
        profile_path=args.profile,
    )

    json_dir = Path(args.json_dir) if args.json_dir else None
    if json_dir:
        json_dir.mkdir(parents=True, exist_ok=True)

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "idx",
            "score",
            "verdict",
            "reach_median",
            "reach_fraction",
            "p_viral",
            "reproduction_number",
            "in_network_share",
            "expected_tier",
            "tweet",
        ]
    )

    for i, (tweet, expected_tier) in enumerate(tweets, start=1):
        report = analyze(tweet, cfg, has_media=args.media)
        s = report.stats
        writer.writerow(
            [
                i,
                report.virality_score,
                report.verdict,
                s.reach_median,
                round(s.reach_fraction_median, 4),
                round(s.p_viral, 4),
                round(s.reproduction_number, 3),
                round(s.in_network_share, 3),
                expected_tier if expected_tier is not None else "",
                tweet[:120] + ("…" if len(tweet) > 120 else ""),
            ]
        )
        if json_dir:
            slug = f"{i:03d}.json"
            (json_dir / slug).write_text(
                json.dumps(report.model_dump(), indent=2),
                encoding="utf-8",
            )
        print(f"[{i}/{len(tweets)}] {report.virality_score:3d}  {report.verdict}", file=sys.stderr)


if __name__ == "__main__":
    main()
