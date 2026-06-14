"""`tvs` command-line interface."""

from __future__ import annotations

import hashlib
import json as _json
from typing import Optional

import typer
from rich.console import Console

from .config import Config
from .engine import analyze
from .improve import improve as improve_tweet
from .report import render_comparison, render_improvement, render_report

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
                                 help="Tweet scorer: heuristic | openai | ollama | compat."),
    audience: int = typer.Option(1000, "--audience", "-a", help="Synthetic audience size."),
    runs: int = typer.Option(120, "--runs", "-r", help="Monte Carlo runs."),
    followers: int = typer.Option(120, "--followers", "-f",
                                  help="Author's in-network follower seed."),
    media: bool = typer.Option(False, "--media", help="Tweet includes image/video."),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed (defaults to tweet hash)."),
    profile: Optional[str] = typer.Option(None, "--profile",
                                          help="Path to a calibration profile JSON (defaults to built-in)."),
    as_json: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
    save: Optional[str] = typer.Option(None, "--save", help="Save the report card as an SVG image."),
) -> None:
    """Simulate an X (Twitter) post."""
    cfg = Config(
        provider=provider,
        audience_size=audience,
        runs=runs,
        author_followers=followers,
        seed=seed,
        profile_path=profile,
    )
    with console.status("[bold]simulating cascade...", spinner="dots"):
        report = analyze(tweet, cfg, has_media=media)

    if as_json:
        console.print_json(_json.dumps(report.model_dump()))
        return

    out = Console(record=True) if save else console
    render_report(report, out)
    if save:
        out.save_svg(save, title="tvs x")
        console.print(f"[dim]saved report card → {save}[/dim]")


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


@app.command()
def improve(
    tweet: str = typer.Argument(..., help="The tweet to rewrite for more spread."),
    provider: str = typer.Option("heuristic", "--provider", "-p",
                                 help="Rewriter/scorer: heuristic | openai | ollama | compat."),
    variants: int = typer.Option(6, "--variants", "-n", help="How many rewrites to try."),
    audience: int = typer.Option(1000, "--audience", "-a", help="Synthetic audience size."),
    runs: int = typer.Option(120, "--runs", "-r", help="Monte Carlo runs."),
    followers: int = typer.Option(60, "--followers", "-f",
                                  help="Author's in-network follower seed."),
    media: bool = typer.Option(False, "--media", help="Tweet includes image/video."),
    seed: Optional[int] = typer.Option(None, "--seed", help="RNG seed (defaults to tweet hash)."),
    profile: Optional[str] = typer.Option(None, "--profile",
                                          help="Path to a calibration profile JSON."),
    save: Optional[str] = typer.Option(None, "--save", help="Save the result card as an SVG image."),
) -> None:
    """Rewrite a tweet for spread — the simulator ranks the variants."""
    cfg = Config(
        provider=provider,
        audience_size=audience,
        runs=runs,
        author_followers=followers,
        seed=seed,
        profile_path=profile,
    )
    with console.status("[bold]generating + simulating rewrites...", spinner="dots"):
        result = improve_tweet(tweet, cfg, variants=variants, has_media=media)

    out = Console(record=True) if save else console
    render_improvement(result, out)
    if save:
        out.save_svg(save, title="tvs improve")
        console.print(f"[dim]saved result card → {save}[/dim]")


@app.command()
def validate(
    profile: Optional[str] = typer.Option(None, "--profile",
                                          help="Path to a calibration profile JSON (defaults to built-in)."),
    audience: int = typer.Option(600, "--audience", "-a", help="Synthetic audience size."),
    runs: int = typer.Option(60, "--runs", "-r", help="Monte Carlo runs per tweet."),
    do_tune: bool = typer.Option(False, "--tune", help="Search for better profile params."),
    samples: int = typer.Option(60, "--samples", help="Tuning samples (with --tune)."),
) -> None:
    """Score the model against the face-validity benchmark (honest sanity check)."""
    from .validation import evaluate, tune as tune_profile

    cfg = Config(audience_size=audience, runs=runs, profile_path=profile)
    if do_tune:
        with console.status("[bold]searching profile space...", spinner="dots"):
            res = tune_profile(n_samples=samples, config=cfg)
        console.print(res.summary())
        console.print("\n[dim]save the tuned profile and load it with --profile / TVS_PROFILE_PATH[/dim]")
        return

    with console.status("[bold]running benchmark...", spinner="dots"):
        result = evaluate(config=cfg)
    console.print(f"[bold]benchmark[/bold]  {result.summary()}")
    console.print("[dim]priors-based sanity check, not real-outcome accuracy — "
                  "real data plugs in via the storage seam.[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
