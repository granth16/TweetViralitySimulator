"""`tvs` command-line interface."""

from __future__ import annotations

import json as _json
from typing import Optional

import typer
from rich.console import Console

from .config import Config
from .engine import analyze
from .report import render_report

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
