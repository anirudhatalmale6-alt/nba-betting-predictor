#!/usr/bin/env python3
"""
Generate a clean, readable picks page (TODAYS_PICKS.md) for easy viewing on GitHub.
"""

import sys
import os
import json
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import OUTPUT_DIR

PROJECT_ROOT = Path(__file__).resolve().parent


def generate_picks_page(target_date: date = None):
    if target_date is None:
        target_date = date.today()

    date_str = target_date.strftime("%Y-%m-%d")
    json_path = OUTPUT_DIR / f"picks_{date_str}.json"
    output_path = PROJECT_ROOT / "TODAYS_PICKS.md"

    lines = []
    lines.append(f"# NBA Betting Picks - {target_date.strftime('%A, %B %d, %Y')}")
    lines.append("")
    lines.append(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}*")
    lines.append("")

    if not json_path.exists():
        lines.append("## No Games Today")
        lines.append("")
        lines.append("No NBA games scheduled today.")
        _write(output_path, lines)
        print(f"No games today. Wrote {output_path}")
        return

    with open(json_path) as f:
        picks = json.load(f)

    if not picks:
        lines.append("## No Picks Today")
        lines.append("")
        lines.append("Games are scheduled but no picks met the model's criteria.")
        _write(output_path, lines)
        print(f"No picks today. Wrote {output_path}")
        return

    # Group by bet type
    by_type = {}
    for p in picks:
        bt = p.get("bet_type", "OTHER")
        by_type.setdefault(bt, []).append(p)

    for bet_type in ["MONEYLINE", "SPREAD", "TOTAL"]:
        type_picks = by_type.get(bet_type, [])
        if not type_picks:
            continue

        recommended = [p for p in type_picks if p.get("recommended")]
        others = [p for p in type_picks if not p.get("recommended")]

        type_label = {
            "MONEYLINE": "Moneyline Underdogs",
            "SPREAD": "Spread (ATS) Picks",
            "TOTAL": "Totals (Over/Under)",
        }[bet_type]

        lines.append(f"## {type_label}")
        lines.append("")

        if recommended:
            lines.append(f"### Recommended Plays ({len(recommended)})")
            lines.append("")
            lines.append("| Pick | Line | Win Prob | Edge | Confidence |")
            lines.append("|------|------|----------|------|------------|")
            for p in recommended:
                matchup = f"{p.get('away_team','')} @ {p.get('home_team','')}"
                prob = p.get("model_prob", 0)
                prob_str = f"{prob:.1%}" if isinstance(prob, float) else str(prob)
                lines.append(
                    f"| **{p.get('pick','')}** ({matchup}) | {p.get('line','')} | "
                    f"{prob_str} | {p.get('edge_pct','')} | {p.get('confidence','')} |"
                )
            lines.append("")

        if others:
            lines.append(f"Other qualifying ({len(others)} - below threshold):")
            lines.append("")
            for p in others:
                prob = p.get("model_prob", 0)
                prob_str = f"{prob:.1%}" if isinstance(prob, float) else str(prob)
                lines.append(
                    f"- {p.get('pick','')} ({p.get('away_team','')} @ {p.get('home_team','')}) "
                    f"| {p.get('line','')} | {prob_str} | {p.get('edge_pct','')}"
                )
            lines.append("")

        lines.append("---")
        lines.append("")

    # Footer
    lines.append("### How to Read This")
    lines.append("")
    lines.append("- **MONEYLINE**: Picking the underdog to win outright. Odds show the payout (e.g., +200 = $100 bet wins $200)")
    lines.append("- **SPREAD (ATS)**: Picking the underdog to cover the point spread (e.g., +5.5 means they can lose by up to 5)")
    lines.append("- **TOTAL**: Picking whether the combined score goes OVER or UNDER the line")
    lines.append("- **Edge**: How much the model's probability exceeds the market's (positive = value)")
    lines.append("- **Confidence**: LOW / MEDIUM / HIGH based on edge size")
    lines.append("")
    lines.append("*Disclaimer: Statistical model for informational purposes. Past performance does not guarantee future results. Gamble responsibly.*")

    _write(output_path, lines)
    n_rec = sum(1 for p in picks if p.get("recommended"))
    print(f"Wrote {n_rec} recommended picks to {output_path}")


def _write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    target = date.today()
    if len(sys.argv) > 1:
        try:
            target = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            pass
    generate_picks_page(target)
