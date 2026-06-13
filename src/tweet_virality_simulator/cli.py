"""`tvs` command-line interface."""

from __future__ import annotations

import hashlib
import json as _json
from typing import Optional

import typer
from rich.console import Console

from .config import Config
from .engine import analyze
from .report import render_comparison, render_report

app = typer.Typer(
    add_completion=False,
    help="Simulate whether a tweet spreads on X — before you post.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _root() -> None:
    """Simulate whether content spreads on a platform — before you post."""


@app.command()
def x(
    tweet: str = typer.Argument(..., help="The tweet text to simulate."),
    provider: str = typer.Option("heuristic", "--provider", "-p",
                                 help="Tweet scorer: heuristic | openai | ollama."),
    audience: int = typer.Option(1000, "--audience", "-a", help="Synthetic audience size."),
    runs: int = typer.Option(120, "--runs", "-r", help="Monte Carlo runs."),
    followers: int = typer.Option(120, "--followers", "-f",
                                  help="Author's in-network follower seed."),
    media: bool = typer.Option(False, "--media", help="Tweet includes image/video."),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed (defaults to tweet hash)."),
    as_json: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
) -> None:
    """Simulate an X (Twitter) post."""
    cfg = Config(
        provider=provider,
        audience_size=audience,
        runs=runs,
        author_followers=followers,
        seed=seed,
    )
    with console.status("[bold]simulating cascade...", spinner="dots"):
        report = analyze(tweet, cfg, has_media=media)

    if as_json:
        console.print_json(_json.dumps(report.model_dump()))
    else:
        render_report(report, console)


@app.command()
def compare(
    tweet_a: str = typer.Argument(..., help="Version A."),
    tweet_b: str = typer.Argument(..., help="Version B."),
    provider: str = typer.Option("heuristic", "--provider", "-p",
                                 help="Tweet scorer: heuristic | openai | ollama."),
    audience: int = typer.Option(1000, "--audience", "-a", help="Synthetic audience size."),
    runs: int = typer.Option(120, "--runs", "-r", help="Monte Carlo runs."),
    followers: int = typer.Option(60, "--followers", "-f",
                                  help="Author's in-network follower seed."),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed."),
) -> None:
    """A/B test two tweets on the same synthetic population."""
    # Both versions must run on the SAME population for a fair comparison, so we
    # force a shared seed (derived from both texts when not provided).
    if seed is None:
        seed = int(hashlib.md5((tweet_a + "||" + tweet_b).encode("utf-8")).hexdigest(), 16) % (2**32)
    cfg = Config(provider=provider, audience_size=audience, runs=runs,
                 author_followers=followers, seed=seed)
    with console.status("[bold]simulating both versions...", spinner="dots"):
        report_a = analyze(tweet_a, cfg)
        report_b = analyze(tweet_b, cfg)
    render_comparison(report_a, report_b, console)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
