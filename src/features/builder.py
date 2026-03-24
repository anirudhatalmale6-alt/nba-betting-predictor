"""
Feature engineering for NBA betting models.
Builds feature vectors for three bet types: ML underdogs, spreads ATS, and totals.
"""

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

# ── Feature columns for each model ──────────────────────────────────────

# Shared features used by all models
SHARED_FEATURES = [
    # Rolling offensive stats
    "home_roll_pts_for", "away_roll_pts_for",
    "home_roll_pts_against", "away_roll_pts_against",
    "home_roll_fg_pct", "away_roll_fg_pct",
    "home_roll_fg3_pct", "away_roll_fg3_pct",
    "home_roll_ft_pct", "away_roll_ft_pct",
    "home_roll_reb", "away_roll_reb",
    "home_roll_ast", "away_roll_ast",
    "home_roll_tov", "away_roll_tov",
    "home_roll_oreb", "away_roll_oreb",
    "home_roll_stl", "away_roll_stl",
    "home_roll_blk", "away_roll_blk",
    # Season performance
    "home_season_win_pct", "away_season_win_pct",
    # Rest and momentum
    "home_rest_days", "away_rest_days",
    "home_streak", "away_streak",
]

# Differential features (computed from shared)
DIFF_FEATURES = [
    "diff_pts_for",       # home rolling pts - away rolling pts
    "diff_pts_against",
    "diff_fg_pct",
    "diff_fg3_pct",
    "diff_reb",
    "diff_ast",
    "diff_tov",
    "diff_win_pct",
    "diff_rest_days",
    "diff_streak",
]

# Model-specific features
ML_FEATURES = SHARED_FEATURES + DIFF_FEATURES + [
    "underdog_is_home",
    "ml_underdog_odds",
    "market_implied_prob",
]

SPREAD_FEATURES = SHARED_FEATURES + DIFF_FEATURES + [
    "spread_points",      # the spread line
    "underdog_is_home",
]

TOTAL_FEATURES = SHARED_FEATURES + DIFF_FEATURES + [
    "total_line",         # the over/under line
    "combined_roll_pts",  # home + away rolling pts
    "combined_roll_pace", # estimated pace factor
]

# All features for reference
ALL_FEATURE_COLUMNS = list(set(ML_FEATURES + SPREAD_FEATURES + TOTAL_FEATURES))


def build_features_for_game(game_row: pd.Series) -> dict:
    """
    Build derived features for a single game row that already has rolling stats.
    """
    features = {}

    # Copy existing rolling features
    for col in SHARED_FEATURES:
        features[col] = game_row.get(col, np.nan)

    # Differential features
    features["diff_pts_for"] = (
        game_row.get("home_roll_pts_for", 0) - game_row.get("away_roll_pts_for", 0)
    )
    features["diff_pts_against"] = (
        game_row.get("home_roll_pts_against", 0) - game_row.get("away_roll_pts_against", 0)
    )
    features["diff_fg_pct"] = (
        game_row.get("home_roll_fg_pct", 0) - game_row.get("away_roll_fg_pct", 0)
    )
    features["diff_fg3_pct"] = (
        game_row.get("home_roll_fg3_pct", 0) - game_row.get("away_roll_fg3_pct", 0)
    )
    features["diff_reb"] = (
        game_row.get("home_roll_reb", 0) - game_row.get("away_roll_reb", 0)
    )
    features["diff_ast"] = (
        game_row.get("home_roll_ast", 0) - game_row.get("away_roll_ast", 0)
    )
    features["diff_tov"] = (
        game_row.get("home_roll_tov", 0) - game_row.get("away_roll_tov", 0)
    )
    features["diff_win_pct"] = (
        game_row.get("home_season_win_pct", 0.5) - game_row.get("away_season_win_pct", 0.5)
    )
    features["diff_rest_days"] = (
        game_row.get("home_rest_days", 1) - game_row.get("away_rest_days", 1)
    )
    features["diff_streak"] = (
        game_row.get("home_streak", 0) - game_row.get("away_streak", 0)
    )

    # Totals features
    features["combined_roll_pts"] = (
        game_row.get("home_roll_pts_for", 100) + game_row.get("away_roll_pts_for", 100)
    )
    features["combined_roll_pace"] = (
        (game_row.get("home_roll_pts_for", 100) + game_row.get("home_roll_pts_against", 100) +
         game_row.get("away_roll_pts_for", 100) + game_row.get("away_roll_pts_against", 100)) / 2.0
    )

    return features


def build_training_dataset(games_with_rolling: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full training dataset with features and targets for all 3 models.
    Input: games DataFrame that already has rolling features merged.
    """
    records = []
    for _, row in games_with_rolling.iterrows():
        features = build_features_for_game(row)

        # Metadata
        features["game_id"] = row.get("game_id", "")
        features["game_date"] = row.get("game_date")
        features["home_team"] = row.get("home_team", "")
        features["away_team"] = row.get("away_team", "")
        features["season"] = row.get("season", 0)

        # Actual results (targets)
        features["home_pts"] = row.get("home_pts", 0)
        features["away_pts"] = row.get("away_pts", 0)
        features["total_pts"] = row.get("total_pts", 0)
        features["point_diff"] = row.get("point_diff", 0)
        features["home_won"] = row.get("home_won", 0)

        records.append(features)

    df = pd.DataFrame(records)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df
