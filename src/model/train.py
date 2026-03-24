"""
Model training for NBA betting: spreads, totals, and moneyline underdogs.
Uses XGBoost with walk-forward validation.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score

from config.settings import XGB_PARAMS
from src.features.builder import ML_FEATURES, SPREAD_FEATURES, TOTAL_FEATURES
from src.model.registry import save_model
from src.utils.logging import get_logger

log = get_logger(__name__)


def train_spread_model(df: pd.DataFrame, train_seasons: list[int]) -> tuple:
    """
    Train the ATS (spread) model.
    Target: did the underdog cover the spread? (1 = covered, 0 = did not)
    """
    log.info("=== Training SPREAD (ATS) Model ===")

    # Simulate spreads from point differentials for historical data
    # In backtest, we approximate: home spread = -(home_pts - away_pts predicted line)
    # For training, target is: did the underdog cover?
    # If home is underdog (positive spread): cover = point_diff > -spread_points
    # Since we don't have historical spreads, we use actual point diff as target
    # and train to predict if a game will be close (underdog covers)

    train_df = df[df["season"].isin(train_seasons)].copy()
    available_cols = [c for c in SPREAD_FEATURES if c in train_df.columns]

    X = train_df[available_cols].fillna(0)
    # Target: away team covers (since away is usually the underdog in spread)
    # For simplicity: home team did NOT win by more than expected
    y = train_df["spread_covered"].astype(int) if "spread_covered" in train_df else (train_df["home_won"] == 0).astype(int)

    log.info(f"Training spread model on {len(X)} games, {len(available_cols)} features")
    log.info(f"Cover rate: {y.mean():.3f}")

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, verbose=False)

    metadata = {
        "model_type": "spread_ats",
        "n_train": len(X),
        "features": available_cols,
        "cover_rate": float(y.mean()),
        "train_accuracy": float(accuracy_score(y, model.predict(X))),
    }

    save_model(model, metadata, name="xgb_spread")
    return model, metadata


def train_total_model(df: pd.DataFrame, train_seasons: list[int]) -> tuple:
    """
    Train the totals (over/under) model.
    Target: did the game go over? (1 = over, 0 = under)
    """
    log.info("=== Training TOTALS Model ===")

    train_df = df[df["season"].isin(train_seasons)].copy()

    # For historical games without actual lines, use season average total as proxy
    if "total_line" not in train_df.columns:
        season_avg = train_df.groupby("season")["total_pts"].transform("mean")
        train_df["total_line"] = season_avg

    available_cols = [c for c in TOTAL_FEATURES if c in train_df.columns]

    X = train_df[available_cols].fillna(0)
    # Target: game went over the line
    y = (train_df["total_pts"] > train_df["total_line"]).astype(int)

    log.info(f"Training totals model on {len(X)} games, {len(available_cols)} features")
    log.info(f"Over rate: {y.mean():.3f}")

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, verbose=False)

    metadata = {
        "model_type": "totals",
        "n_train": len(X),
        "features": available_cols,
        "over_rate": float(y.mean()),
        "train_accuracy": float(accuracy_score(y, model.predict(X))),
    }

    save_model(model, metadata, name="xgb_total")
    return model, metadata


def train_moneyline_model(df: pd.DataFrame, train_seasons: list[int]) -> tuple:
    """
    Train the moneyline underdog model.
    Target: did the underdog win outright? (1 = won, 0 = lost)
    """
    log.info("=== Training MONEYLINE UNDERDOG Model ===")

    train_df = df[df["season"].isin(train_seasons)].copy()

    # For historical data, the "underdog" is the away team (typically)
    # In reality this would be determined by odds, but for backtest
    # we use win_pct differential as proxy
    available_cols = [c for c in ML_FEATURES if c in train_df.columns]

    X = train_df[available_cols].fillna(0)
    y = train_df["underdog_won"].astype(int) if "underdog_won" in train_df else (train_df["home_won"] == 0).astype(int)

    log.info(f"Training ML underdog model on {len(X)} games, {len(available_cols)} features")
    log.info(f"Underdog win rate: {y.mean():.3f}")

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, verbose=False)

    metadata = {
        "model_type": "moneyline_underdog",
        "n_train": len(X),
        "features": available_cols,
        "underdog_win_rate": float(y.mean()),
        "train_accuracy": float(accuracy_score(y, model.predict(X))),
    }

    save_model(model, metadata, name="xgb_moneyline")
    return model, metadata


def walk_forward_validation(
    df: pd.DataFrame,
    test_seasons: list[int],
    model_type: str = "moneyline",
) -> tuple:
    """
    Walk-forward validation for any model type.
    For each test season, train on all prior seasons.
    """
    if model_type == "spread":
        feature_cols = SPREAD_FEATURES
        target_col = "spread_covered"
        model_name = "xgb_spread"
    elif model_type == "total":
        feature_cols = TOTAL_FEATURES
        target_col = "went_over"
        model_name = "xgb_total"
    else:
        feature_cols = ML_FEATURES
        target_col = "underdog_won"
        model_name = "xgb_moneyline"

    available_cols = [c for c in feature_cols if c in df.columns]
    all_predictions = []

    for test_season in test_seasons:
        train_mask = df["season"] < test_season
        test_mask = df["season"] == test_season

        X_train = df.loc[train_mask, available_cols].fillna(0)
        y_train = df.loc[train_mask, target_col].astype(int)
        X_test = df.loc[test_mask, available_cols].fillna(0)
        y_test = df.loc[test_mask, target_col].astype(int)

        if len(X_train) < 100 or len(X_test) < 10:
            log.warning(f"Skipping {test_season}: train={len(X_train)}, test={len(X_test)}")
            continue

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, verbose=False)
        probs = model.predict_proba(X_test)[:, 1]

        season_preds = df.loc[test_mask, [
            "game_id", "game_date", "home_team", "away_team", "season",
            "home_pts", "away_pts", "total_pts", "point_diff",
        ]].copy()
        season_preds["model_prob"] = probs
        season_preds["model_pick"] = (probs >= 0.5).astype(int)
        season_preds["actual"] = y_test.values
        season_preds["model_type"] = model_type

        acc = accuracy_score(y_test, (probs >= 0.5).astype(int))
        log.info(f"  {model_type} {test_season}: {len(X_test)} games, accuracy={acc:.3f}")

        all_predictions.append(season_preds)

    # Final model on all data
    X_all = df[available_cols].fillna(0)
    y_all = df[target_col].astype(int)
    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X_all, y_all, verbose=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    return final_model, predictions_df
