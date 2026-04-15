"""
The Odds API data ingestion for NBA.
Fetches live NBA moneyline, spread, and totals odds.
"""

import json
import requests
from datetime import datetime
from pathlib import Path

from config.settings import (
    ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT,
    ODDS_REGIONS, ODDS_FORMAT, RAW_DIR,
    MIN_ML_UNDERDOG_ODDS, MAX_ML_UNDERDOG_ODDS,
    ODDS_TEAM_TO_ABBREV,
)
from src.utils.odds_math import american_to_implied, remove_vig, is_qualifying_ml_underdog
from src.utils.logging import get_logger

log = get_logger(__name__)


def fetch_nba_odds(markets: str = "h2h,spreads,totals") -> list[dict]:
    """
    Fetch current NBA odds from The Odds API.
    Returns list of game dicts with moneyline, spread, and total odds.
    """
    if not ODDS_API_KEY:
        log.error("ODDS_API_KEY not set.")
        return []

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": markets,
        "oddsFormat": ODDS_FORMAT,
    }

    resp = requests.get(url, params=params, timeout=30)

    # Handle quota exhaustion gracefully
    if resp.status_code in (401, 429):
        remaining = resp.headers.get("x-requests-remaining", "0")
        log.warning(f"Odds API quota exhausted (HTTP {resp.status_code}). Remaining: {remaining}. Trying cached snapshot...")
        cached = load_latest_snapshot()
        if cached:
            log.info(f"  Using cached odds ({len(cached)} games)")
            return cached
        log.warning("  No cached snapshot available. Proceeding without odds.")
        return []

    resp.raise_for_status()

    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    log.info(f"Odds API quota: {remaining} remaining, {used} used")

    data = resp.json()

    # Save raw snapshot
    snapshot_dir = RAW_DIR / "odds_api"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(snapshot_dir / f"nba_odds_{ts}.json", "w") as f:
        json.dump(data, f, indent=2)

    return _parse_odds_response(data)


def _parse_odds_response(data: list[dict]) -> list[dict]:
    """Parse The Odds API response into clean game records with all market types."""
    games = []
    for event in data:
        home_team_full = event.get("home_team", "")
        away_team_full = event.get("away_team", "")
        home_team = ODDS_TEAM_TO_ABBREV.get(home_team_full, home_team_full)
        away_team = ODDS_TEAM_TO_ABBREV.get(away_team_full, away_team_full)
        commence = event.get("commence_time", "")

        game = {
            "event_id": event.get("id", ""),
            "commence_time": commence,
            "home_team": home_team,
            "away_team": away_team,
            "home_team_full": home_team_full,
            "away_team_full": away_team_full,
        }

        # Collect odds from all bookmakers
        h2h_home = []
        h2h_away = []
        spread_home_points = []
        spread_home_odds = []
        spread_away_odds = []
        total_points = []
        over_odds = []
        under_odds = []

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                key = market.get("key")
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}

                if key == "h2h":
                    if home_team_full in outcomes and away_team_full in outcomes:
                        h2h_home.append(outcomes[home_team_full]["price"])
                        h2h_away.append(outcomes[away_team_full]["price"])

                elif key == "spreads":
                    if home_team_full in outcomes and away_team_full in outcomes:
                        spread_home_points.append(outcomes[home_team_full].get("point", 0))
                        spread_home_odds.append(outcomes[home_team_full]["price"])
                        spread_away_odds.append(outcomes[away_team_full]["price"])

                elif key == "totals":
                    if "Over" in outcomes and "Under" in outcomes:
                        total_points.append(outcomes["Over"].get("point", 0))
                        over_odds.append(outcomes["Over"]["price"])
                        under_odds.append(outcomes["Under"]["price"])

        # Moneyline consensus (median)
        if h2h_home and h2h_away:
            h2h_home.sort()
            h2h_away.sort()
            mid = len(h2h_home) // 2
            game["home_ml"] = h2h_home[mid]
            game["away_ml"] = h2h_away[mid]

            # Determine underdog
            if game["home_ml"] > game["away_ml"]:
                game["ml_underdog"] = "home"
                game["ml_underdog_team"] = home_team
                game["ml_underdog_odds"] = game["home_ml"]
                game["ml_favorite_odds"] = game["away_ml"]
            else:
                game["ml_underdog"] = "away"
                game["ml_underdog_team"] = away_team
                game["ml_underdog_odds"] = game["away_ml"]
                game["ml_favorite_odds"] = game["home_ml"]

            home_prob, away_prob = remove_vig(game["home_ml"], game["away_ml"])
            game["home_implied_prob"] = home_prob
            game["away_implied_prob"] = away_prob
            game["ml_underdog_implied_prob"] = home_prob if game["ml_underdog"] == "home" else away_prob
            game["is_qualifying_ml"] = is_qualifying_ml_underdog(
                game["ml_underdog_odds"], MIN_ML_UNDERDOG_ODDS, MAX_ML_UNDERDOG_ODDS
            )
        else:
            game["is_qualifying_ml"] = False

        # Spread consensus (median)
        if spread_home_points and spread_home_odds:
            spread_home_points.sort()
            spread_home_odds.sort()
            spread_away_odds.sort()
            mid = len(spread_home_points) // 2
            game["home_spread"] = spread_home_points[mid]
            game["away_spread"] = -spread_home_points[mid]
            game["home_spread_odds"] = spread_home_odds[mid]
            game["away_spread_odds"] = spread_away_odds[mid]

            # Underdog ATS is the team getting points (positive spread)
            if game["home_spread"] > 0:
                game["spread_underdog"] = "home"
                game["spread_underdog_team"] = home_team
                game["spread_points"] = game["home_spread"]
            else:
                game["spread_underdog"] = "away"
                game["spread_underdog_team"] = away_team
                game["spread_points"] = game["away_spread"]
            game["has_spread"] = True
        else:
            game["has_spread"] = False

        # Totals consensus (median)
        if total_points and over_odds:
            total_points.sort()
            over_odds.sort()
            under_odds.sort()
            mid = len(total_points) // 2
            game["total_line"] = total_points[mid]
            game["over_odds"] = over_odds[mid]
            game["under_odds"] = under_odds[mid]
            game["has_total"] = True
        else:
            game["has_total"] = False

        game["num_bookmakers"] = len(event.get("bookmakers", []))
        games.append(game)

    qualifying_ml = sum(1 for g in games if g.get("is_qualifying_ml"))
    with_spread = sum(1 for g in games if g.get("has_spread"))
    with_total = sum(1 for g in games if g.get("has_total"))
    log.info(f"Parsed {len(games)} games: {qualifying_ml} qualifying ML underdogs, "
             f"{with_spread} with spreads, {with_total} with totals")
    return games


def load_latest_snapshot():
    """Load the most recent odds snapshot from disk."""
    snapshot_dir = RAW_DIR / "odds_api"
    if not snapshot_dir.exists():
        return None
    files = sorted(snapshot_dir.glob("nba_odds_*.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        data = json.load(f)
    return _parse_odds_response(data)
