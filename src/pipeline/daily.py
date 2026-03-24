"""
Daily prediction pipeline for NBA.
Fetches today's games + odds, builds features, generates predictions for all 3 bet types.
"""

import json
import pandas as pd
from datetime import date, datetime

from config.settings import OUTPUT_DIR, ODDS_API_KEY, ABBREV_TO_TEAM_ID
from src.ingest.nba_stats import (
    get_todays_games, get_team_game_log, get_team_stats,
    get_standings, season_string, get_recent_games,
)
from src.ingest.odds_api import fetch_nba_odds
from src.features.builder import build_features_for_game, SHARED_FEATURES
from src.model.predict import (
    generate_ml_predictions, generate_spread_predictions, generate_total_predictions,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


def run_daily_pipeline(target_date: date = None) -> dict:
    """
    Run the full daily prediction pipeline.
    Returns dict with DataFrames for each bet type.
    """
    if target_date is None:
        target_date = date.today()

    season_year = target_date.year if target_date.month >= 10 else target_date.year - 1
    season_str = season_string(season_year)

    log.info(f"{'='*50}")
    log.info(f"NBA Daily Pipeline — {target_date} (season {season_str})")
    log.info(f"{'='*50}")

    # Step 1: Get today's games
    log.info("Step 1: Fetching today's games...")
    games = get_todays_games(target_date)
    if not games:
        log.info("No NBA games today.")
        return {"moneyline": pd.DataFrame(), "spread": pd.DataFrame(), "total": pd.DataFrame()}

    log.info(f"  Found {len(games)} games")

    # Step 2: Fetch live odds
    log.info("Step 2: Fetching live odds...")
    odds_data = []
    if ODDS_API_KEY:
        odds_data = fetch_nba_odds()
    else:
        log.warning("No ODDS_API_KEY set.")

    # Deduplicate games (scoreboard sometimes returns dupes)
    seen = set()
    unique_games = []
    for g in games:
        key = (g["home_team"], g["away_team"])
        if key not in seen:
            seen.add(key)
            unique_games.append(g)
    games = unique_games

    # Step 3: Match games with odds
    log.info("Step 3: Matching games with odds...")
    matched = _match_games_with_odds(games, odds_data)

    # Step 4: Build features for each game
    log.info("Step 4: Building features...")
    standings = get_standings(season_str)

    games_with_features = []
    for game in matched:
        try:
            features = _build_live_features(game, season_str, standings)
            if features:
                games_with_features.append(features)
        except Exception as e:
            log.warning(f"Failed features for {game.get('home_team')} vs {game.get('away_team')}: {e}")

    if not games_with_features:
        log.info("Could not build features for any games.")
        return {"moneyline": pd.DataFrame(), "spread": pd.DataFrame(), "total": pd.DataFrame()}

    # Step 5: Generate predictions for each bet type
    log.info("Step 5: Generating predictions...")

    results = {}

    # Moneyline underdogs
    ml_games = [g for g in games_with_features if g.get("is_qualifying_ml")]
    try:
        results["moneyline"] = generate_ml_predictions(ml_games)
    except Exception as e:
        log.warning(f"ML prediction error: {e}")
        results["moneyline"] = pd.DataFrame()

    # Spreads
    spread_games = [g for g in games_with_features if g.get("has_spread")]
    try:
        results["spread"] = generate_spread_predictions(spread_games)
    except Exception as e:
        log.warning(f"Spread prediction error: {e}")
        results["spread"] = pd.DataFrame()

    # Totals
    total_games = [g for g in games_with_features if g.get("has_total")]
    try:
        results["total"] = generate_total_predictions(total_games)
    except Exception as e:
        log.warning(f"Total prediction error: {e}")
        results["total"] = pd.DataFrame()

    # Step 6: Save output
    log.info("Step 6: Saving output...")
    _save_output(results, target_date)

    # Summary
    for bet_type, df in results.items():
        if not df.empty:
            rec = df[df["recommended"]].shape[0] if "recommended" in df.columns else 0
            log.info(f"  {bet_type}: {len(df)} picks, {rec} recommended")

    return results


def _match_games_with_odds(games: list[dict], odds_data: list[dict]) -> list[dict]:
    """Match scheduled games with live odds."""
    odds_lookup = {}
    for od in odds_data:
        key = (od.get("home_team"), od.get("away_team"))
        odds_lookup[key] = od

    for game in games:
        key = (game["home_team"], game["away_team"])
        if key in odds_lookup:
            game.update(odds_lookup[key])
        else:
            game["is_qualifying_ml"] = False
            game["has_spread"] = False
            game["has_total"] = False

    return games


def _build_live_features(game: dict, season_str: str, standings: dict) -> dict:
    """Build features for a live game using current stats."""
    home = game["home_team"]
    away = game["away_team"]
    home_id = ABBREV_TO_TEAM_ID.get(home)
    away_id = ABBREV_TO_TEAM_ID.get(away)

    if not home_id or not away_id:
        return None

    # Get recent game logs for rolling features
    home_recent = get_recent_games(home_id, season_str, 10)
    away_recent = get_recent_games(away_id, season_str, 10)

    features = {}

    # Build rolling stats from recent games
    features.update(_rolling_from_gamelog(home_recent, "home"))
    features.update(_rolling_from_gamelog(away_recent, "away"))

    # Standings
    home_stand = standings.get(home, {})
    away_stand = standings.get(away, {})
    features["home_season_win_pct"] = home_stand.get("win_pct", 0.5)
    features["away_season_win_pct"] = away_stand.get("win_pct", 0.5)

    # Streaks
    features["home_streak"] = _parse_streak(home_stand.get("streak", ""))
    features["away_streak"] = _parse_streak(away_stand.get("streak", ""))

    # Rest days
    if not home_recent.empty:
        last_home = pd.to_datetime(home_recent.iloc[0].get("GAME_DATE", ""))
        features["home_rest_days"] = max(1, (datetime.now() - last_home).days)
    else:
        features["home_rest_days"] = 3

    if not away_recent.empty:
        last_away = pd.to_datetime(away_recent.iloc[0].get("GAME_DATE", ""))
        features["away_rest_days"] = max(1, (datetime.now() - last_away).days)
    else:
        features["away_rest_days"] = 3

    # Build derived features
    game_series = pd.Series(features)
    derived = build_features_for_game(game_series)
    features.update(derived)

    # Copy odds data
    for key in ["is_qualifying_ml", "ml_underdog_team", "ml_underdog_odds",
                "ml_underdog_implied_prob", "has_spread", "spread_underdog_team",
                "spread_points", "has_total", "total_line",
                "home_ml", "away_ml", "home_spread", "away_spread",
                "over_odds", "under_odds", "spread_underdog", "ml_underdog"]:
        if key in game:
            features[key] = game[key]

    features["underdog_is_home"] = 1 if game.get("ml_underdog") == "home" else 0
    features["market_implied_prob"] = game.get("ml_underdog_implied_prob", 0.3)

    # Game metadata
    features["game_date"] = game.get("game_date", "")
    features["home_team"] = home
    features["away_team"] = away

    return features


def _rolling_from_gamelog(gl: pd.DataFrame, prefix: str) -> dict:
    """Compute rolling averages from a team game log."""
    if gl.empty:
        return {
            f"{prefix}_roll_pts_for": 105,
            f"{prefix}_roll_pts_against": 105,
            f"{prefix}_roll_fg_pct": 0.46,
            f"{prefix}_roll_fg3_pct": 0.36,
            f"{prefix}_roll_ft_pct": 0.78,
            f"{prefix}_roll_reb": 44,
            f"{prefix}_roll_ast": 25,
            f"{prefix}_roll_tov": 14,
            f"{prefix}_roll_oreb": 10,
            f"{prefix}_roll_stl": 7,
            f"{prefix}_roll_blk": 5,
        }

    # Estimate pts_against from win/loss margin
    # TeamGameLog doesn't have PLUS_MINUS, so we estimate from W/L and PTS
    pts_for = gl["PTS"].mean()
    # Use win pct to estimate average margin, then derive pts_against
    win_pct = (gl["WL"] == "W").mean() if "WL" in gl.columns else 0.5
    # Average NBA margin for a team with this win_pct (rough estimate)
    avg_margin = (win_pct - 0.5) * 20  # ~+10 for 100% win, ~-10 for 0%
    pts_against = pts_for - avg_margin

    return {
        f"{prefix}_roll_pts_for": pts_for,
        f"{prefix}_roll_pts_against": pts_against,
        f"{prefix}_roll_fg_pct": gl["FG_PCT"].mean(),
        f"{prefix}_roll_fg3_pct": gl["FG3_PCT"].mean(),
        f"{prefix}_roll_ft_pct": gl["FT_PCT"].mean(),
        f"{prefix}_roll_reb": gl["REB"].mean(),
        f"{prefix}_roll_ast": gl["AST"].mean(),
        f"{prefix}_roll_tov": gl["TOV"].mean(),
        f"{prefix}_roll_oreb": gl["OREB"].mean(),
        f"{prefix}_roll_stl": gl["STL"].mean(),
        f"{prefix}_roll_blk": gl["BLK"].mean(),
    }


def _parse_streak(streak_val) -> int:
    """Parse streak value to signed int. Handles string 'W 3' / 'L 2' or numeric."""
    if not streak_val:
        return 0
    if isinstance(streak_val, (int, float)):
        return int(streak_val)
    try:
        streak_str = str(streak_val).strip()
        parts = streak_str.split()
        if len(parts) == 2:
            n = int(parts[1])
            return n if parts[0] == "W" else -n
        return int(streak_str)
    except (ValueError, AttributeError):
        return 0


def _save_output(results: dict, target_date: date):
    """Save all predictions to files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = target_date.strftime("%Y-%m-%d")

    all_picks = []
    for bet_type, df in results.items():
        if not df.empty:
            all_picks.append(df)

    if all_picks:
        combined = pd.concat(all_picks, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / f"picks_{date_str}.csv", index=False)
        combined.to_json(OUTPUT_DIR / f"picks_{date_str}.json", orient="records", indent=2)
        log.info(f"  Saved to picks_{date_str}.csv / .json")
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / f"picks_{date_str}.csv", index=False)
        with open(OUTPUT_DIR / f"picks_{date_str}.json", "w") as f:
            json.dump([], f)
