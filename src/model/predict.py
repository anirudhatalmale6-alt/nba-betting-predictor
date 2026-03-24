"""
Prediction module — generates daily picks from trained models.
Handles all three bet types: moneyline, spread, and totals.
"""

import pandas as pd
import numpy as np
from datetime import date

from config.settings import (
    EDGE_THRESHOLD_ML, EDGE_THRESHOLD_SPREAD, EDGE_THRESHOLD_TOTAL,
)
from src.features.builder import ML_FEATURES, SPREAD_FEATURES, TOTAL_FEATURES
from src.model.registry import load_latest_model
from src.utils.odds_math import american_to_implied, odds_to_decimal, calculate_kelly
from src.utils.logging import get_logger

log = get_logger(__name__)


def generate_ml_predictions(games: list[dict], model=None, metadata=None) -> pd.DataFrame:
    """Generate moneyline underdog predictions."""
    if model is None:
        model, metadata = load_latest_model("xgb_moneyline")

    if not games:
        return pd.DataFrame()

    feature_cols = metadata.get("features", [c for c in ML_FEATURES if c in games[0]])
    available_cols = [c for c in feature_cols if c in games[0]]

    df = pd.DataFrame(games)
    X = df[available_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]

    results = []
    for i, game in enumerate(games):
        model_prob = float(probs[i])
        underdog_odds = game.get("ml_underdog_odds", 200)
        market_prob = american_to_implied(underdog_odds)
        edge = model_prob - market_prob
        decimal_odds = odds_to_decimal(underdog_odds)
        kelly = calculate_kelly(model_prob, decimal_odds)

        results.append({
            "bet_type": "MONEYLINE",
            "game_date": game.get("game_date", str(date.today())),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
            "pick": game.get("ml_underdog_team", ""),
            "odds": underdog_odds,
            "line": f"+{underdog_odds}",
            "model_prob": round(model_prob, 4),
            "market_prob": round(market_prob, 4),
            "edge": round(edge, 4),
            "edge_pct": f"{edge * 100:.1f}%",
            "kelly": round(kelly, 4),
            "confidence": _confidence_label(edge, "ml"),
            "recommended": edge >= EDGE_THRESHOLD_ML,
        })

    return pd.DataFrame(results).sort_values("edge", ascending=False)


def generate_spread_predictions(games: list[dict], model=None, metadata=None) -> pd.DataFrame:
    """Generate spread (ATS) predictions."""
    if model is None:
        model, metadata = load_latest_model("xgb_spread")

    if not games:
        return pd.DataFrame()

    # Use only features the model was trained on
    trained_features = metadata.get("features", [])
    available_cols = [c for c in trained_features if c in games[0]]
    if not available_cols:
        available_cols = [c for c in SPREAD_FEATURES if c in games[0] and c != "spread_points"]

    df = pd.DataFrame(games)
    X = df[available_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]

    results = []
    for i, game in enumerate(games):
        model_prob = float(probs[i])
        spread_pts = game.get("spread_points", 0)
        # Standard spread is -110 on both sides, so fair line is 0.5
        edge = model_prob - 0.5

        results.append({
            "bet_type": "SPREAD",
            "game_date": game.get("game_date", str(date.today())),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
            "pick": game.get("spread_underdog_team", ""),
            "odds": -110,
            "line": f"+{spread_pts}",
            "model_prob": round(model_prob, 4),
            "market_prob": 0.5,
            "edge": round(edge, 4),
            "edge_pct": f"{edge * 100:.1f}%",
            "kelly": round(max(0, (model_prob * 1.909 - (1 - model_prob)) / 1.909), 4),
            "confidence": _confidence_label(edge, "spread"),
            "recommended": edge >= EDGE_THRESHOLD_SPREAD,
        })

    return pd.DataFrame(results).sort_values("edge", ascending=False)


def generate_total_predictions(games: list[dict], model=None, metadata=None) -> pd.DataFrame:
    """Generate totals (over/under) predictions."""
    if model is None:
        model, metadata = load_latest_model("xgb_total")

    if not games:
        return pd.DataFrame()

    feature_cols = metadata.get("features", [c for c in TOTAL_FEATURES if c in games[0]])
    available_cols = [c for c in feature_cols if c in games[0]]

    df = pd.DataFrame(games)
    X = df[available_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]  # probability of over

    results = []
    for i, game in enumerate(games):
        over_prob = float(probs[i])
        total_line = game.get("total_line", 220)
        # Pick: over if prob > 0.5, under if prob < 0.5
        is_over = over_prob > 0.5
        pick_prob = over_prob if is_over else (1 - over_prob)
        edge = pick_prob - 0.5

        results.append({
            "bet_type": "TOTAL",
            "game_date": game.get("game_date", str(date.today())),
            "home_team": game.get("home_team", ""),
            "away_team": game.get("away_team", ""),
            "pick": "OVER" if is_over else "UNDER",
            "odds": -110,
            "line": str(total_line),
            "model_prob": round(pick_prob, 4),
            "market_prob": 0.5,
            "edge": round(edge, 4),
            "edge_pct": f"{edge * 100:.1f}%",
            "kelly": round(max(0, (pick_prob * 1.909 - (1 - pick_prob)) / 1.909), 4),
            "confidence": _confidence_label(edge, "total"),
            "recommended": edge >= EDGE_THRESHOLD_TOTAL,
        })

    return pd.DataFrame(results).sort_values("edge", ascending=False)


def _confidence_label(edge: float, bet_type: str) -> str:
    if bet_type == "ml":
        if edge >= 0.10:
            return "HIGH"
        elif edge >= 0.05:
            return "MEDIUM"
        elif edge >= EDGE_THRESHOLD_ML:
            return "LOW"
        else:
            return "NO PLAY"
    else:
        if edge >= 0.06:
            return "HIGH"
        elif edge >= 0.04:
            return "MEDIUM"
        elif edge >= 0.02:
            return "LOW"
        else:
            return "NO PLAY"
