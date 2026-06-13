"""Render a Report as a terminal card — the shareable artifact."""

from __future__ import annotations

from typing import List

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..models import Driver, Report

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
