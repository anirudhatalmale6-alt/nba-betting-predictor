"""
Backtest runner for all three NBA betting models.
Collects historical data, builds features, trains models, and evaluates performance.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

from config.settings import (
    PROCESSED_DIR, OUTPUT_DIR, HISTORICAL_SEASONS, BACKTEST_TEST_SEASONS,
)
from src.ingest.historical import (
    collect_all_seasons, build_rolling_features,
    merge_rolling_features_to_games,
)
from src.features.builder import (
    build_training_dataset, ML_FEATURES, SPREAD_FEATURES, TOTAL_FEATURES,
)
from src.model.train import walk_forward_validation
from src.model.registry import save_model
from src.utils.logging import get_logger

log = get_logger(__name__)


def run_full_backtest(
    seasons: list[int] = None,
    test_seasons: list[int] = None,
) -> dict:
    """
    Run full backtest pipeline:
    1. Collect historical game data
    2. Build rolling features
    3. Create training dataset with targets
    4. Walk-forward validation for each model
    5. Generate performance report
    """
    if seasons is None:
        seasons = HISTORICAL_SEASONS
    if test_seasons is None:
        test_seasons = BACKTEST_TEST_SEASONS

    # Step 1: Collect or load historical data
    log.info("Step 1: Loading historical game data...")
    hist_path = PROCESSED_DIR / "historical_games.parquet"
    if hist_path.exists():
        games = pd.read_parquet(hist_path)
        log.info(f"  Loaded {len(games)} games from cache")
    else:
        games = collect_all_seasons(seasons)

    if games.empty:
        log.error("No historical data available!")
        return {}

    # Step 2: Build rolling features
    log.info("Step 2: Building rolling features...")
    rolling = build_rolling_features(games, window=10)

    # Step 3: Merge and build training dataset
    log.info("Step 3: Building training dataset...")
    games_with_rolling = merge_rolling_features_to_games(games, rolling)
    dataset = build_training_dataset(games_with_rolling)

    # Add targets for each model type
    # Spread target: away team covers (as proxy for underdog covering)
    # In NBA, home team wins ~58%, so away = underdog proxy
    dataset["underdog_won"] = (dataset["home_won"] == 0).astype(int)
    dataset["underdog_is_home"] = 0  # proxy: underdog is away team
    dataset["spread_covered"] = dataset["underdog_won"]  # simplified
    dataset["went_over"] = 0  # will be set per-season below

    # For totals: compute season-average total as proxy line
    for season in dataset["season"].unique():
        mask = dataset["season"] == season
        avg_total = dataset.loc[mask, "total_pts"].mean()
        dataset.loc[mask, "total_line"] = avg_total
        dataset.loc[mask, "went_over"] = (dataset.loc[mask, "total_pts"] > avg_total).astype(int)

    # Simulate underdog odds from win pct differential
    dataset["ml_underdog_odds"] = _estimate_ml_odds(dataset)
    dataset["market_implied_prob"] = dataset["ml_underdog_odds"].apply(
        lambda x: 100.0 / (x + 100.0) if x > 0 else abs(x) / (abs(x) + 100.0)
    )

    # Save dataset
    dataset.to_parquet(PROCESSED_DIR / "training_dataset.parquet", index=False)
    log.info(f"  Training dataset: {len(dataset)} games")

    # Step 4: Walk-forward validation for each model
    results = {}

    for model_type in ["moneyline", "spread", "total"]:
        log.info(f"\nStep 4: Walk-forward validation for {model_type}...")
        final_model, preds = walk_forward_validation(dataset, test_seasons, model_type)

        if preds.empty:
            log.warning(f"  No predictions for {model_type}")
            continue

        # Calculate metrics
        acc = accuracy_score(preds["actual"], preds["model_pick"])
        n_games = len(preds)

        per_season = {}
        for season in test_seasons:
            sp = preds[preds["season"] == season]
            if len(sp) > 0:
                sacc = accuracy_score(sp["actual"], sp["model_pick"])
                per_season[season] = {
                    "n_games": len(sp),
                    "accuracy": round(sacc, 4),
                    "pick_rate": round(sp["model_pick"].mean(), 4),
                }

        results[model_type] = {
            "overall_accuracy": round(acc, 4),
            "n_games": n_games,
            "per_season": per_season,
        }

        log.info(f"  {model_type}: {acc:.1%} accuracy on {n_games} games")

        # Save final model
        metadata = {
            "model_type": model_type,
            "n_train": len(dataset),
            "test_accuracy": round(acc, 4),
            "test_seasons": test_seasons,
            "features": (ML_FEATURES if model_type == "moneyline"
                        else SPREAD_FEATURES if model_type == "spread"
                        else TOTAL_FEATURES),
        }
        name = f"xgb_{model_type}" if model_type != "total" else "xgb_total"
        if model_type == "moneyline":
            name = "xgb_moneyline"
        save_model(final_model, metadata, name=name)

    # Step 5: Generate report
    log.info("\nStep 5: Generating backtest report...")
    _save_report(results)

    return results


def _estimate_ml_odds(df: pd.DataFrame) -> pd.Series:
    """
    Estimate moneyline underdog odds from win pct differential.
    Maps win_pct diff to approximate ML odds for backtesting.
    """
    diff = df["home_season_win_pct"].fillna(0.5) - df["away_season_win_pct"].fillna(0.5)
    # Larger diff = bigger underdog
    # Map: diff of 0.1 ~ +150, diff of 0.2 ~ +200, diff of 0.3 ~ +300
    odds = 130 + (diff.abs() * 800).clip(0, 370)
    return odds.astype(int)


def _save_report(results: dict):
    """Save backtest report as JSON and text."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(OUTPUT_DIR / "backtest_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Text report
    lines = []
    lines.append("=" * 60)
    lines.append("  NBA BETTING MODEL — BACKTEST REPORT")
    lines.append("=" * 60)
    lines.append("")

    for model_type, data in results.items():
        lines.append(f"--- {model_type.upper()} MODEL ---")
        lines.append(f"Overall accuracy: {data['overall_accuracy']:.1%}")
        lines.append(f"Total games tested: {data['n_games']}")
        lines.append("")

        for season, sdata in data.get("per_season", {}).items():
            lines.append(f"  Season {season}: {sdata['accuracy']:.1%} "
                        f"({sdata['n_games']} games)")
        lines.append("")

    lines.append("=" * 60)

    report_text = "\n".join(lines)
    with open(OUTPUT_DIR / "backtest_report.txt", "w") as f:
        f.write(report_text)

    log.info(report_text)
