#!/usr/bin/env python3
"""
Run the full NBA backtest.
Collects historical data, trains models, and evaluates performance.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backtest.runner import run_full_backtest
from src.utils.logging import get_logger

log = get_logger("backtest")


def main():
    log.info("Starting NBA full backtest...")
    results = run_full_backtest()

    if not results:
        print("Backtest failed — no results.")
        return

    print(f"\n{'='*60}")
    print(f"  NBA BACKTEST RESULTS")
    print(f"{'='*60}")

    for model_type, data in results.items():
        print(f"\n  {model_type.upper()} MODEL")
        print(f"  Overall accuracy: {data['overall_accuracy']:.1%}")
        print(f"  Games tested: {data['n_games']}")
        for season, sdata in data.get("per_season", {}).items():
            print(f"    {season}: {sdata['accuracy']:.1%} ({sdata['n_games']} games)")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
