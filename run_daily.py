#!/usr/bin/env python3
"""
Run the daily NBA prediction pipeline.
Usage: python run_daily.py [YYYY-MM-DD]
"""

import sys
import os
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.daily import run_daily_pipeline
from src.utils.logging import get_logger

log = get_logger("daily")


def main():
    target_date = date.today()
    if len(sys.argv) > 1:
        try:
            target_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid date format: {sys.argv[1]}. Use YYYY-MM-DD.")
            sys.exit(1)

    log.info(f"Running NBA daily pipeline for {target_date}...")
    results = run_daily_pipeline(target_date)

    total_picks = 0
    for bet_type, df in results.items():
        if df.empty:
            continue
        recommended = df[df["recommended"]] if "recommended" in df.columns else df
        total_picks += len(recommended)

        print(f"\n{'='*60}")
        print(f"  NBA {bet_type.upper()} PICKS — {target_date}")
        print(f"{'='*60}")

        if recommended.empty:
            print("  No recommended plays.")
        else:
            print(f"  {len(recommended)} RECOMMENDED:\n")
            for _, pick in recommended.iterrows():
                print(f"  {pick['pick']} ({pick['line']})")
                print(f"       {pick['away_team']} @ {pick['home_team']}")
                print(f"       Prob: {pick['model_prob']:.1%} | Edge: {pick['edge_pct']} | {pick['confidence']}")
                print()

    if total_picks == 0:
        print(f"\nNo recommended plays for {target_date}.")

    print(f"\n{'='*60}")
    print(f"  Output saved to data/output/picks_{target_date}.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
