"""Render a Report as a terminal card — the shareable artifact."""

from __future__ import annotations

from typing import List

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..models import Driver, Improvement, Report

_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values: List[float]) -> str:
    if not values:
        return ""
    hi = max(values)
    if hi <= 0:
        return _SPARK[0] * len(values)
    out = []
    for v in values:
        level = int(round((v / hi) * (len(_SPARK) - 1)))
        out.append(_SPARK[level])
    return "".join(out)


def _score_color(score: int) -> str:
    if score >= 75:
        return "bold green"
    if score >= 50:
        return "bold yellow"
    if score >= 25:
        return "bold orange3"
    return "bold red"


def _bar(impact: float, width: int = 16) -> Text:
    filled = int(round(min(abs(impact), 1.0) * width))
    color = "green" if impact >= 0 else "red"
    glyph = "█" * filled + "░" * (width - filled)
    return Text(glyph, style=color)


def _drivers_table(title: str, drivers: List[Driver], positive: bool) -> Table:
    t = Table(show_header=False, box=None, padding=(0, 1, 0, 0), expand=True)
    t.add_column("bar", width=16, no_wrap=True)
    t.add_column("label")
    icon = "[green]+[/green]" if positive else "[red]-[/red]"
    if not drivers:
        t.add_row("", Text("(none detected)", style="dim"))
        return t
    for d in drivers:
        t.add_row(_bar(d.impact), Text.from_markup(f"{icon} [b]{d.label}[/b]  [dim]{d.detail}[/dim]"))
    return t


def render_report(report: Report, console: Console = None) -> None:
    console = console or Console()
    s = report.stats
    score = report.virality_score
    color = _score_color(score)

    header = Text()
    header.append("  ", style="")
    header.append(f"{score}", style=color)
    header.append("/100   ", style="dim")
    header.append(report.verdict, style=color)

    tweet_line = Text(f'\n  "{report.tweet.strip()}"\n', style="italic")

    spread = Table(show_header=True, header_style="bold dim", box=None, expand=True)
    spread.add_column("Predicted spread", justify="left")
    spread.add_column("", justify="right")
    spread.add_row("Reach (median)", f"{s.reach_median} / {s.audience_size} accounts")
    spread.add_row("Reach range (p10–p90)", f"{s.reach_p10} – {s.reach_p90}")
    spread.add_row("Best run", f"{s.reach_max}")
    spread.add_row("Viral odds", f"{s.p_viral*100:.0f}%  (≥{s.viral_threshold} reached)")
    spread.add_row("Reproduction number R", f"{s.reproduction_number:.2f}  "
                                            + ("[green](grows)[/green]" if s.reproduction_number >= 1 else "[red](dies)[/red]"))
    spread.add_row("Avg shares / depth", f"{s.avg_shares:.0f} shares · {s.avg_depth:.1f} hops")
    spread.add_row("In- vs out-of-network", f"{s.in_network_share*100:.0f}% follow graph · "
                                           f"{s.out_network_share*100:.0f}% For You")

    spark = Text(f"  spread per round  {_sparkline(s.reach_curve)}", style="cyan")

    drivers_tbl = _drivers_table("Why it works", report.drivers, positive=True)
    weak_tbl = _drivers_table("What's holding it back", report.weaknesses, positive=False)

    roast = Panel(Text(report.roast, style="italic"), title="roast", title_align="left",
                  border_style="dim", padding=(0, 1))

    body = Group(
        tweet_line,
        spread,
        spark,
        Text("\n  Why it works", style="bold green"),
        drivers_tbl,
        Text("\n  What's holding it back", style="bold red"),
        weak_tbl,
        Text(""),
        roast,
        Text(f"\n  scored by: {report.dna.scored_by}  ·  {s.runs} simulations  ·  "
             f"reach is a relative indicator, not a raw view count", style="dim"),
    )

    console.print(Panel(body, title=header, title_align="left",
                        border_style=color, padding=(1, 2)))


def render_comparison(a: Report, b: Report, console: Console = None) -> None:
    """Render a side-by-side A/B comparison of two tweets."""
    console = console or Console()

    def rank(r: Report):
        return (r.virality_score, r.stats.reach_median)

    a_wins = rank(a) >= rank(b)
    winner, loser = ("A", "B") if a_wins else ("B", "A")
    hi = max(a.stats.reach_median, b.stats.reach_median)
    lo = max(min(a.stats.reach_median, b.stats.reach_median), 1)
    ratio = hi / lo

    col_a = "bold green" if a_wins else "dim"
    col_b = "bold green" if not a_wins else "dim"

    t = Table(show_header=True, box=None, expand=True, header_style="bold")
    t.add_column("", justify="left")
    t.add_column(Text("Version A", style=col_a), justify="right")
    t.add_column(Text("Version B", style=col_b), justify="right")

    t.add_row("Tweet", Text(a.tweet[:34] + ("…" if len(a.tweet) > 34 else ""), style=col_a),
              Text(b.tweet[:34] + ("…" if len(b.tweet) > 34 else ""), style=col_b))
    t.add_row("Virality score", Text(f"{a.virality_score}/100", style=col_a),
              Text(f"{b.virality_score}/100", style=col_b))
    t.add_row("Verdict", Text(a.verdict, style=col_a), Text(b.verdict, style=col_b))
    t.add_row("Reach (median)", Text(str(a.stats.reach_median), style=col_a),
              Text(str(b.stats.reach_median), style=col_b))
    t.add_row("Viral odds", Text(f"{a.stats.p_viral*100:.0f}%", style=col_a),
              Text(f"{b.stats.p_viral*100:.0f}%", style=col_b))
    t.add_row("R", Text(f"{a.stats.reproduction_number:.2f}", style=col_a),
              Text(f"{b.stats.reproduction_number:.2f}", style=col_b))

    headline = Text()
    headline.append(f"  Version {winner} ", style="bold green")
    headline.append(f"is {ratio:.1f}× more likely to spread than Version {loser}",
                    style="bold")

    body = Group(t, Text(""), headline,
                 Text(f"\n  same population · {a.stats.runs} simulations each · "
                      f"scored by {a.dna.scored_by}", style="dim"))
    console.print(Panel(body, title=Text("A/B comparison", style="bold"),
                        title_align="left", border_style="cyan", padding=(1, 2)))


def render_improvement(imp: Improvement, console: Console = None, top: int = 3) -> None:
    """Render the original tweet, the best rewrite, and a few runners-up."""
    console = console or Console()
    original = imp.original
    best = imp.variants[0] if imp.variants else original
    lift = imp.lift()
    beat = bool(imp.variants) and best.virality_score > original.virality_score

    color = _score_color(best.virality_score) if beat else "yellow"

    if beat:
        title = Text()
        title.append("  ↑ +", style="bold green")
        title.append(f"{lift}", style="bold green")
        title.append(" points   ", style="bold green")
        title.append(
            f"{original.virality_score} → {best.virality_score}", style="bold"
        )
    else:
        title = Text("  Your draft already holds up", style="bold yellow")

    # Original.
    orig_block = Group(
        Text("  ORIGINAL", style="bold dim"),
        Text(f'  "{original.tweet.strip()}"', style="italic dim"),
        Text(
            f"  score {original.virality_score}/100 · {original.verdict} · "
            f"reach {original.stats.reach_median}/{original.stats.audience_size}",
            style="dim",
        ),
    )

    # Best rewrite (the copy-paste-ready payload).
    rewrite_panel = Panel(
        Text(best.tweet.strip(), style="bold"),
        title=Text(
            f"BEST REWRITE · {best.virality_score}/100 · {best.verdict}",
            style=color,
        ),
        title_align="left",
        border_style=color,
        padding=(0, 1),
    )
    rewrite_why = Text(
        f"  reach {best.stats.reach_median}/{best.stats.audience_size} · "
        f"viral odds {best.stats.p_viral*100:.0f}% · R {best.stats.reproduction_number:.2f}",
        style="dim",
    )

    # Runners-up.
    runners = imp.variants[1:top]
    others = Table(show_header=False, box=None, padding=(0, 1, 0, 0), expand=True)
    others.add_column("score", width=9, no_wrap=True)
    others.add_column("tweet")
    for r in runners:
        snippet = r.tweet.strip()
        snippet = snippet[:60] + ("…" if len(snippet) > 60 else "")
        others.add_row(Text(f"{r.virality_score}/100", style="cyan"), Text(snippet, style="dim"))

    parts = [orig_block, Text(""), rewrite_panel, rewrite_why]
    if runners:
        parts += [Text("\n  other options", style="bold dim"), others]
    parts.append(
        Text(
            f"\n  {len(imp.variants)} rewrites · same population · "
            f"generated by {imp.generated_by} · scored by {original.dna.scored_by}",
            style="dim",
        )
    )

    console.print(
        Panel(Group(*parts), title=title, title_align="left",
              border_style=color, padding=(1, 2))
    )
