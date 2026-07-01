"""Paper-grade evaluation: time-split holdout, baselines, ablations, metrics.

Loads the Kaggle mmmarchetti/tweets-dataset (~52k tweets, 20 accounts, 2009--2017),
splits by tweet timestamp (oldest 75% train / newest 25% test), fits a profile on a
train slice via random search, and reports the metric bundle from the paper on the
held-out test set.

Usage:
    pip install -e .
  pip install kagglehub

    # Full run (test set ~13k tweets; allow ~1--2 hours)
    python examples/paper_eval.py --out results/paper_eval.json

    # Quick smoke test (~5 min)
    python examples/paper_eval.py --quick --out results/paper_eval_quick.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tweet_virality_simulator import Config, analyze
from tweet_virality_simulator.platforms.x import tweet_dna
from tweet_virality_simulator.platforms.x.reactions import content_appeal
from tweet_virality_simulator.profile import Profile
from tweet_virality_simulator.providers import get_provider
from tweet_virality_simulator.validation.tune import _SEARCH, _sample

# Reuse follower map from compare_dataset.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_dataset import (  # noqa: E402
    _KAGGLE_FOLLOWERS,
    _effective_followers,
    _engagement,
    _followers_for,
    _parse_followers,
)

_W_LIKE = 1.0
_W_RT = 2.0


def _default_csv() -> Path:
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit("pip install kagglehub") from e
    root = Path(kagglehub.dataset_download("mmmarchetti/tweets-dataset"))
    return root / "tweets.csv"


def _parse_date(raw: str) -> datetime:
    return datetime.strptime(raw.strip(), "%d/%m/%Y %H:%M")


def _load_tweets(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("content") or row.get("tweet") or "").strip()
            if not text or text.startswith("RT @"):
                continue
            try:
                likes = int(float(row.get("number_of_likes") or row.get("likes") or 0))
                rts = int(float(row.get("number_of_shares") or row.get("retweets") or 0))
                dt = _parse_date(row["date_time"])
            except (ValueError, KeyError):
                continue
            author = row.get("author") or row.get("user") or ""
            rows.append(
                {
                    "author": author,
                    "tweet": text,
                    "likes": likes,
                    "retweets": rts,
                    "date": dt,
                    "followers_raw": _parse_followers(row) or _KAGGLE_FOLLOWERS.get(author),
                    "author_followers": _followers_for(author, row, use_map=True),
                    "has_media": "https://t.co/" in text or "pic.twitter.com" in text,
                }
            )
    rows.sort(key=lambda r: r["date"])
    return rows


def _author_baselines(train: list[dict[str, Any]]) -> dict[str, float]:
    by_author: dict[str, list[float]] = {}
    for r in train:
        by_author.setdefault(r["author"], []).append(_engagement(r["likes"], r["retweets"]))
    return {a: float(statistics.median(vs)) for a, vs in by_author.items() if vs}


def _attach_rel_amp(rows: list[dict[str, Any]], baselines: dict[str, float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        eng = _engagement(r["likes"], r["retweets"])
        baseline = max(baselines.get(r["author"], eng), 1.0)
        out.append({**r, "engagement": eng, "baseline": baseline, "rel_amp": eng / baseline})
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


def _minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return [0.5] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def _decile_labels(xs: list[float]) -> list[int]:
    if not xs:
        return []
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    labels = [0] * len(xs)
    n = len(xs)
    for rank, idx in enumerate(order):
        labels[idx] = min(9, int(10 * rank / n))
    return labels


def compute_metrics(preds: list[float], rel_amps: list[float]) -> dict[str, float]:
    n = len(preds)
    if n < 2:
        return {k: 0.0 for k in ("rank_corr", "tail_rank", "top_recall", "pair_acc", "calib_mae")}

    rank_corr = _spearman(preds, rel_amps)
    log_preds = [math.log(max(p, 1e-6)) for p in preds]
    log_amps = [math.log(max(a, 1e-6)) for a in rel_amps]
    tail_rank = _spearman(log_preds, log_amps)

    true_top = set(i for i, d in enumerate(_decile_labels(rel_amps)) if d >= 9)
    pred_top = set(i for i, d in enumerate(_decile_labels(preds)) if d >= 9)
    top_recall = len(true_top & pred_top) / len(true_top) if true_top else 0.0

    true_dec = _decile_labels(rel_amps)
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if true_dec[i] == true_dec[j]:
                continue
            total += 1
            if true_dec[i] < true_dec[j]:
                if preds[i] < preds[j]:
                    correct += 1
                elif preds[i] == preds[j]:
                    correct += 0.5
            elif preds[i] > preds[j]:
                correct += 1
            elif preds[i] == preds[j]:
                correct += 0.5
    pair_acc = correct / total if total else 0.0

    pn = _minmax(preds)
    an = _minmax(rel_amps)
    calib_mae = sum(abs(p - a) for p, a in zip(pn, an)) / n

    return {
        "rank_corr": round(rank_corr, 3),
        "tail_rank": round(tail_rank, 3),
        "top_recall": round(top_recall, 3),
        "pair_acc": round(pair_acc, 3),
        "calib_mae": round(calib_mae, 3),
    }


def _cache_key(row: dict[str, Any], profile: Profile, cfg: Config, mode: str) -> str:
    blob = json.dumps(
        {
            "tweet": row["tweet"],
            "mode": mode,
            "runs": cfg.runs,
            "audience": cfg.audience_size,
            "followers": row["author_followers"],
            "profile": profile.model_dump(),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _score_sim(row: dict[str, Any], cfg: Config, profile: Profile) -> float:
    c = replace(cfg, author_followers=row["author_followers"])
    return float(analyze(row["tweet"], c, has_media=row["has_media"], profile=profile).virality_score)


def _score_content(row: dict[str, Any], cfg: Config, profile: Profile) -> float:
    provider = get_provider(cfg.provider)
    dna = tweet_dna.extract(row["tweet"], provider, has_media=row["has_media"])
    return float(np.clip(content_appeal(dna, profile), 0.0, 1.0) * 100.0)


def score_rows(
    rows: list[dict[str, Any]],
    cfg: Config,
    profile: Profile,
    *,
    mode: str = "sim",
    cache: dict[str, float] | None = None,
    label: str = "",
) -> list[float]:
    cache = cache if cache is not None else {}
    scores: list[float] = []
    scorer: Callable[[dict[str, Any], Config, Profile], float]
    if mode == "content":
        scorer = _score_content
    else:
        scorer = _score_sim

    t0 = time.time()
    for i, row in enumerate(rows, 1):
        key = _cache_key(row, profile, cfg, mode)
        if key not in cache:
            cache[key] = scorer(row, cfg, profile)
        scores.append(cache[key])
        if label and i % 250 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            print(f"  [{label}] {i}/{len(rows)} ({rate:.1f} tweets/s)", file=sys.stderr)
    return scores


def _profile_spearman(
    rows: list[dict[str, Any]],
    profile: Profile,
    cfg: Config,
    cache: dict[str, float],
) -> float:
    preds = score_rows(rows, cfg, profile, mode="sim", cache=cache, label="calib")
    amps = [float(r["rel_amp"]) for r in rows]
    return _spearman(preds, amps)


def calibrate_profile(
    train_rows: list[dict[str, Any]],
    cfg: Config,
    *,
    n_samples: int,
    seed: int,
    cache: dict[str, float],
) -> Profile:
    base = Profile(name="calibrated")
    rng = np.random.default_rng(seed)
    best_profile = base
    best_corr = _profile_spearman(train_rows, base, cfg, cache)
    print(f"  calibration baseline spearman={best_corr:.3f}", file=sys.stderr)

    for i in range(n_samples):
        cand = _sample(base, rng)
        corr = _profile_spearman(train_rows, cand, cfg, cache)
        if corr > best_corr:
            best_corr = corr
            best_profile = cand.model_copy(update={"name": "calibrated"})
            print(f"  calibration [{i+1}/{n_samples}] new best spearman={best_corr:.3f}", file=sys.stderr)
    best_profile = best_profile.model_copy(update={"name": f"calibrated_{best_corr:.3f}"})
    return best_profile


def _error_rows(rows: list[dict[str, Any]], preds: list[float]) -> list[dict[str, Any]]:
    def pct_ranks(vals: list[float]) -> list[float]:
        n = len(vals)
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = 100.0 * avg / max(n - 1, 1)
            i = j + 1
        return out

    pred_pct = pct_ranks(preds)
    amp_pct = pct_ranks([float(r["rel_amp"]) for r in rows])
    err_rows: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        err = abs(pred_pct[i] - amp_pct[i])
        err_rows.append(
            {
                "rank_error": round(err, 2),
                "signed_error": round(pred_pct[i] - amp_pct[i], 2),
                "score": int(round(preds[i])),
                "rel_amp": round(float(r["rel_amp"]), 3),
                "author": r["author"],
                "likes": r["likes"],
                "retweets": r["retweets"],
                "tweet": r["tweet"][:140],
            }
        )
    err_rows.sort(key=lambda x: (-x["rank_error"], -abs(x["signed_error"])))
    return err_rows


def benchmark_speed(cfg: Config, n: int = 20) -> float:
    t0 = time.time()
    for i in range(n):
        analyze(f"Speed benchmark tweet number {i} #test", cfg, has_media=False)
    return (time.time() - t0) / n


def main() -> None:
    p = argparse.ArgumentParser(description="Paper evaluation harness")
    p.add_argument("--csv", help="Dataset path (default: Kaggle download)")
    p.add_argument("--train-frac", type=float, default=0.75)
    p.add_argument("--calib-sample", type=int, default=1200, help="Train tweets for profile search")
    p.add_argument("--calib-search", type=int, default=60, help="Random-search candidates")
    p.add_argument("--runs", type=int, default=40)
    p.add_argument("--audience", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="Small calib + test slices for smoke tests")
    p.add_argument("--out", default="results/paper_eval.json")
    p.add_argument("--errors-out", default="results/paper_errors.csv")
    args = p.parse_args()

    if args.quick:
        args.calib_sample = 200
        args.calib_search = 12
        args.runs = 20

    path = Path(args.csv) if args.csv else _default_csv()
    all_rows = _load_tweets(path)
    cut = int(len(all_rows) * args.train_frac)
    train_raw, test_raw = all_rows[:cut], all_rows[cut:]
    baselines = _author_baselines(train_raw)
    train = _attach_rel_amp(train_raw, baselines)
    test = _attach_rel_amp(test_raw, baselines)

    if args.quick:
        rng = random.Random(args.seed)
        test = rng.sample(test, min(800, len(test)))

    cfg = Config(runs=args.runs, audience_size=args.audience)
    cache: dict[str, float] = {}

    print(
        f"dataset: {len(all_rows)} tweets, {len(set(r['author'] for r in all_rows))} accounts\n"
        f"dates: {all_rows[0]['date'].date()} .. {all_rows[-1]['date'].date()}\n"
        f"train={len(train)} test={len(test)} runs={cfg.runs}",
        file=sys.stderr,
    )

    rng = random.Random(args.seed)
    calib_rows = rng.sample(train, min(args.calib_sample, len(train)))

    t0 = time.time()
    print("calibrating profile on train slice...", file=sys.stderr)
    calibrated = calibrate_profile(calib_rows, cfg, n_samples=args.calib_search, seed=args.seed, cache=cache)

    models: list[tuple[str, str, Profile, dict[str, Any]]] = [
        ("content_only", "content", Profile(name="content_only"), {}),
        ("uncalibrated", "sim", Profile(name="default"), {}),
        ("calibrated", "sim", calibrated, {}),
    ]

    # Ablations on the same default / calibrated profiles.
    no_algo = Profile(name="no_algo").model_copy(update={"promotion_threshold": 9.99})
    flat_audience = {}  # handled by zeroing followers per row below

    ablations: list[tuple[str, str, Profile, dict[str, Any]]] = [
        ("full_model", "sim", Profile(name="default"), {}),
        ("no_algorithmic_channel", "sim", no_algo, {}),
        ("no_author_audience", "sim", Profile(name="default"), {"followers": 60}),
    ]

    results: dict[str, Any] = {
        "dataset": {
            "source": "kaggle:mmmarchetti/tweets-dataset",
            "total_tweets": len(all_rows),
            "accounts": len(set(r["author"] for r in all_rows)),
            "date_min": all_rows[0]["date"].isoformat(),
            "date_max": all_rows[-1]["date"].isoformat(),
            "train_size": len(train),
            "test_size": len(test),
            "calib_size": len(calib_rows),
        },
        "config": {"runs": cfg.runs, "audience_size": cfg.audience_size, "train_frac": args.train_frac},
        "main_table": {},
        "ablation_table": {},
        "speed_seconds_per_tweet": round(benchmark_speed(cfg), 3),
        "calibrated_profile": calibrated.model_dump(),
    }

    rel_amps = [float(r["rel_amp"]) for r in test]

    print("scoring baselines on test...", file=sys.stderr)
    for name, mode, profile, opts in models:
        rows = test
        if opts.get("followers") is not None:
            rows = [{**r, "author_followers": opts["followers"]} for r in test]
        preds = score_rows(rows, cfg, profile, mode=mode, cache=cache, label=name)
        results["main_table"][name] = compute_metrics(preds, rel_amps)
        print(f"  {name}: {results['main_table'][name]}", file=sys.stderr)

    print("scoring ablations on test...", file=sys.stderr)
    for name, mode, profile, opts in ablations:
        rows = test
        if opts.get("followers") is not None:
            rows = [{**r, "author_followers": opts["followers"]} for r in test]
        preds = score_rows(rows, cfg, profile, mode=mode, cache=cache, label=name)
        results["ablation_table"][name] = {
            "rank_corr": compute_metrics(preds, rel_amps)["rank_corr"],
        }
        print(f"  {name}: rank_corr={results['ablation_table'][name]['rank_corr']}", file=sys.stderr)

    # Error analysis from the main uncalibrated simulator.
    uncal_preds = score_rows(test, cfg, Profile(name="default"), mode="sim", cache=cache)
    errors = _error_rows(test, uncal_preds)
    results["error_analysis"] = {
        "best": errors[-5:][::-1][:5],
        "worst": errors[:5],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out} in {time.time()-t0:.0f}s", file=sys.stderr)

    err_path = Path(args.errors_out)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rank_error", "signed_error", "score", "rel_amp", "author", "likes", "retweets", "tweet"]
    with err_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(errors)
    print(f"wrote {err_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
