"""Judge: render daily post-mortem markdown from pipeline outputs."""
from __future__ import annotations

from datetime import date
from pathlib import Path


def render(today: date, prices: list[dict], score: dict, matches: list[dict], headline: str) -> Path:
    """Write backtest_results/YYYY-MM-DD.md and return its path."""
    out = Path(f"backtest_results/{today.isoformat()}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# Sentinel Post-Mortem — {today.isoformat()}", "", "## Headline Analyzed", "", f"> {headline}", ""]
    lines += ["## Linguist Certainty Score", "", f"**{score.get('score', 'n/a')} / 100**  ", f"_{score.get('reasoning', '')}_", ""]
    lines += ["## Scout Watchlist (5d)", "", "| Ticker | Close | 5d % |", "|---|---:|---:|"]
    for p in prices:
        if "error" in p:
            lines.append(f"| {p['ticker']} | — | err |")
        else:
            lines.append(f"| {p['ticker']} | ${p['close']} | {p['pct_change']:+.2f}% |")
    lines += ["", "## Historian — Top Matches", ""]
    if matches:
        for m in matches:
            lines.append(f"- **{m['date']}** — {m['event']} (S&P 5d: {m['sp500_5d_pct']:+.1f}%, overlap={m['match_score']})")
    else:
        lines.append("_no matches above threshold_")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
