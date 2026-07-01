"""Fetch a tweet by ID (or URL) and run it through TVS.

Requires an X API v2 app bearer token:
    export X_BEARER_TOKEN="your-token"
    # or: set -a && source tvs-cloud/.env && set +a

Usage:
    python examples/score_by_id.py 1234567890123456789
    python examples/score_by_id.py "https://x.com/user/status/1234567890123456789"
    python examples/score_by_id.py 123... --json
    python examples/score_by_id.py 123... --runs 40
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from tweet_virality_simulator import Config, analyze
from tweet_virality_simulator.report import render_report

_API = "https://api.twitter.com/2/tweets"


def parse_tweet_id(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"\d{10,25}", value):
        return value
    match = re.search(r"/status/(\d+)", value)
    if match:
        return match.group(1)
    raise SystemExit(f"could not parse tweet id from: {value!r}")


def fetch_tweet(tweet_id: str, bearer_token: str) -> dict:
    params = urllib.parse.urlencode(
        {
            "tweet.fields": "text,attachments,public_metrics,author_id,created_at",
        }
    )
    url = f"{_API}/{tweet_id}?{params}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {bearer_token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise SystemExit(f"X API error {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"network error: {e}") from e

    data = payload.get("data")
    if not data:
        raise SystemExit(f"tweet not found or not accessible: {tweet_id}")
    return data


def has_media(data: dict) -> bool:
    attachments = data.get("attachments") or {}
    return bool(attachments.get("media_keys"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an X tweet by ID via TVS")
    parser.add_argument("tweet", help="Tweet ID or https://x.com/.../status/ID URL")
    parser.add_argument("--runs", type=int, default=120)
    parser.add_argument("--audience", type=int, default=1000)
    parser.add_argument("--followers", type=int, default=60)
    parser.add_argument("--profile", default=None, help="Calibration profile JSON path")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print full report JSON")
    parser.add_argument(
        "--bearer",
        default=os.environ.get("X_BEARER_TOKEN", ""),
        help="X API bearer token (default: X_BEARER_TOKEN env)",
    )
    args = parser.parse_args()

    if not args.bearer:
        raise SystemExit(
            "missing bearer token — set X_BEARER_TOKEN or pass --bearer\n"
            "Get one from the X developer portal (app-only / bearer token)."
        )

    tweet_id = parse_tweet_id(args.tweet)
    data = fetch_tweet(tweet_id, args.bearer)
    text = (data.get("text") or "").strip()
    if not text:
        raise SystemExit("tweet has no text (may be media-only; not supported yet)")

    media = has_media(data)
    pm = data.get("public_metrics") or {}

    cfg = Config(
        audience_size=args.audience,
        runs=args.runs,
        author_followers=args.followers,
        profile_path=args.profile,
    )
    report = analyze(text, cfg, has_media=media)

    meta = {
        "tweet_id": tweet_id,
        "author_id": data.get("author_id"),
        "created_at": data.get("created_at"),
        "has_media": media,
        "public_metrics": pm,
        "text": text,
    }

    if args.as_json:
        out = report.model_dump()
        out["source"] = meta
        print(json.dumps(out, indent=2))
        return

    print(
        f"tweet id={tweet_id}  "
        f"likes={pm.get('like_count', 0)}  "
        f"rts={pm.get('retweet_count', 0)}  "
        f"replies={pm.get('reply_count', 0)}  "
        f"media={media}\n"
    )
    render_report(report)


if __name__ == "__main__":
    # rich markup in the print above needs a Console if we want colors;
    # keep plain for id line, render_report handles the card.
    main()
