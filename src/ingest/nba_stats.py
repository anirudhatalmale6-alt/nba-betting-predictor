"""
NBA stats data ingestion using nba_api.
Fetches team stats, game logs, standings, and schedules.
"""

import time
import requests
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Optional

from nba_api.stats.endpoints import (
    leaguegamefinder,
    teamgamelog,
    leaguestandings,
    scoreboardv2,
    teamdashboardbygeneralsplits,
)
from nba_api.stats.static import teams as nba_teams

from config.settings import TEAM_ID_TO_ABBREV, ABBREV_TO_TEAM_ID
from src.utils.logging import get_logger

log = get_logger(__name__)

# Rate limit: nba_api recommends ~0.6s between requests
API_DELAY = 0.6

# Custom headers to avoid blocks from stats.nba.com on cloud IPs
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_RETRIES = 3


def _delay():
    time.sleep(API_DELAY)


def get_all_teams() -> dict:
    """Get all NBA teams as {abbreviation: team_info}."""
    all_teams = nba_teams.get_teams()
    return {t["abbreviation"]: t for t in all_teams}


def get_todays_games(target_date: date = None) -> list[dict]:
    """
    Get today's NBA games. Tries NBA CDN first (reliable), falls back to nba_api scoreboard.
    Returns list of game dicts with team info.
    """
    if target_date is None:
        target_date = date.today()

    date_str = target_date.strftime("%Y-%m-%d")

    # Try CDN schedule first (more reliable, not blocked by cloud IPs)
    games = _get_games_from_cdn(target_date, date_str)
    if games is not None:
        return games

    # Fallback to nba_api scoreboard
    log.info("CDN schedule failed, falling back to ScoreboardV2...")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            scoreboard = scoreboardv2.ScoreboardV2(
                game_date=date_str,
                league_id="00",
                headers=_HEADERS,
                timeout=60,
            )
            _delay()

            games_header = scoreboard.game_header.get_data_frame()

            if games_header.empty:
                log.info(f"No NBA games on {date_str}")
                return []

            games = []
            seen = set()
            for _, row in games_header.iterrows():
                home_id = row.get("HOME_TEAM_ID")
                away_id = row.get("VISITOR_TEAM_ID")
                pair = (home_id, away_id)
                if pair in seen:
                    continue
                seen.add(pair)
                home_abbrev = TEAM_ID_TO_ABBREV.get(home_id, "???")
                away_abbrev = TEAM_ID_TO_ABBREV.get(away_id, "???")

                games.append({
                    "game_id": row.get("GAME_ID", ""),
                    "game_date": date_str,
                    "home_team": home_abbrev,
                    "away_team": away_abbrev,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "game_status": row.get("GAME_STATUS_TEXT", ""),
                })

            log.info(f"Found {len(games)} NBA games on {date_str}")
            return games

        except Exception as e:
            log.warning(f"Scoreboard attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)
            else:
                log.error(f"All {MAX_RETRIES} attempts failed for scoreboard {date_str}")
                return []

    return []


def _get_games_from_cdn(target_date: date, date_str: str) -> list[dict] | None:
    """Fetch games from NBA CDN schedule (more reliable than stats.nba.com)."""
    try:
        url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        target_str = target_date.strftime("%m/%d/%Y")
        for game_date in data.get("leagueSchedule", {}).get("gameDates", []):
            gd = game_date.get("gameDate", "")
            if gd.startswith(target_str):
                raw_games = game_date.get("games", [])
                if not raw_games:
                    log.info(f"No NBA games on {date_str} (CDN)")
                    return []

                games = []
                for g in raw_games:
                    home = g.get("homeTeam", {})
                    away = g.get("awayTeam", {})
                    home_abbrev = home.get("teamTricode", "???")
                    away_abbrev = away.get("teamTricode", "???")
                    home_id = ABBREV_TO_TEAM_ID.get(home_abbrev)
                    away_id = ABBREV_TO_TEAM_ID.get(away_abbrev)

                    games.append({
                        "game_id": g.get("gameId", ""),
                        "game_date": date_str,
                        "home_team": home_abbrev,
                        "away_team": away_abbrev,
                        "home_team_id": home_id,
                        "away_team_id": away_id,
                        "game_status": g.get("gameStatusText", ""),
                    })

                log.info(f"Found {len(games)} NBA games on {date_str} (CDN)")
                return games

        log.info(f"No date match for {date_str} in CDN schedule")
        return []

    except Exception as e:
        log.warning(f"CDN schedule fetch failed: {e}")
        return None


def get_team_game_log(team_id: int, season: str) -> pd.DataFrame:
    """
    Get a team's game log for a season.
    season format: "2024-25"
    """
    try:
        gl = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star="Regular Season",
            headers=_HEADERS,
            timeout=15,
        )
        _delay()
        df = gl.get_data_frames()[0]
        return df
    except Exception as e:
        log.warning(f"Game log failed for team {team_id}: {e}")
        # Fallback to CDN
        abbrev = TEAM_ID_TO_ABBREV.get(team_id, "")
        if abbrev:
            return _get_game_log_from_cdn(abbrev)
        return pd.DataFrame()


def get_team_stats(team_id: int, season: str) -> dict:
    """
    Get a team's overall stats for a season.
    Returns dict of key stats (OffRtg, DefRtg, Pace, etc.)
    """
    try:
        dashboard = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
            team_id=team_id,
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Base",
            headers=_HEADERS,
            timeout=60,
        )
        _delay()

        overall = dashboard.overall_team_dashboard.get_data_frame()
        if overall.empty:
            return {}

        row = overall.iloc[0]
        return {
            "wins": int(row.get("W", 0)),
            "losses": int(row.get("L", 0)),
            "win_pct": float(row.get("W_PCT", 0.5)),
            "pts": float(row.get("PTS", 0)),
            "fgm": float(row.get("FGM", 0)),
            "fga": float(row.get("FGA", 0)),
            "fg_pct": float(row.get("FG_PCT", 0)),
            "fg3m": float(row.get("FG3M", 0)),
            "fg3a": float(row.get("FG3A", 0)),
            "fg3_pct": float(row.get("FG3_PCT", 0)),
            "ftm": float(row.get("FTM", 0)),
            "fta": float(row.get("FTA", 0)),
            "ft_pct": float(row.get("FT_PCT", 0)),
            "oreb": float(row.get("OREB", 0)),
            "dreb": float(row.get("DREB", 0)),
            "reb": float(row.get("REB", 0)),
            "ast": float(row.get("AST", 0)),
            "tov": float(row.get("TOV", 0)),
            "stl": float(row.get("STL", 0)),
            "blk": float(row.get("BLK", 0)),
            "plus_minus": float(row.get("PLUS_MINUS", 0)),
        }
    except Exception as e:
        log.warning(f"Error fetching team stats for {team_id}: {e}")
        return {}


def get_standings(season: str) -> dict:
    """
    Get current standings.
    Returns {team_abbrev: {wins, losses, win_pct, ...}}
    """
    try:
        standings = leaguestandings.LeagueStandings(
            league_id="00",
            season=season,
            season_type="Regular Season",
            headers=_HEADERS,
            timeout=15,
        )
        _delay()

        df = standings.get_data_frames()[0]
        result = {}
        for _, row in df.iterrows():
            team_id = row.get("TeamID")
            abbrev = TEAM_ID_TO_ABBREV.get(team_id, "???")
            result[abbrev] = {
                "wins": int(row.get("WINS", 0)),
                "losses": int(row.get("LOSSES", 0)),
                "win_pct": float(row.get("WinPCT", 0.5)),
                "home_record": row.get("HOME", "0-0"),
                "away_record": row.get("ROAD", "0-0"),
                "l10_record": row.get("L10", "0-0"),
                "streak": row.get("CurrentStreak", ""),
                "conference_rank": int(row.get("PlayoffRank", 15)),
            }
        return result
    except Exception as e:
        log.warning(f"Error fetching standings from nba_api: {e}")
        log.info("Falling back to CDN for standings...")
        return get_standings_from_cdn()


def get_season_games(season: str) -> pd.DataFrame:
    """
    Get all games for a season using LeagueGameFinder.
    season format: "2024-25"
    Returns DataFrame with one row per team per game.
    """
    try:
        finder = leaguegamefinder.LeagueGameFinder(
            league_id_nullable="00",
            season_nullable=season,
            season_type_nullable="Regular Season",
        )
        _delay()
        df = finder.get_data_frames()[0]
        log.info(f"Found {len(df)} team-game records for season {season}")
        return df
    except Exception as e:
        log.error(f"Error fetching season games for {season}: {e}")
        return pd.DataFrame()


def season_string(year: int) -> str:
    """Convert a year to NBA season string: 2024 -> '2024-25'."""
    next_yr = (year + 1) % 100
    return f"{year}-{next_yr:02d}"


def get_recent_games(team_id: int, season: str, n_games: int = 10) -> pd.DataFrame:
    """Get a team's most recent N games."""
    gl = get_team_game_log(team_id, season)
    if gl.empty:
        # Fallback: build from CDN schedule data
        abbrev = TEAM_ID_TO_ABBREV.get(team_id, "")
        if abbrev:
            gl = _get_game_log_from_cdn(abbrev)
    if gl.empty:
        return gl
    return gl.head(n_games)


# ── CDN-based fallbacks ──

_cdn_cache = {}


def _fetch_cdn_schedule():
    """Fetch and cache the full NBA CDN schedule."""
    if "schedule" in _cdn_cache:
        return _cdn_cache["schedule"]
    try:
        url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        _cdn_cache["schedule"] = data
        return data
    except Exception as e:
        log.warning(f"CDN schedule fetch failed: {e}")
        return None


def get_standings_from_cdn() -> dict:
    """
    Build standings from CDN schedule data.
    Returns {team_abbrev: {wins, losses, win_pct, streak, ...}}
    """
    data = _fetch_cdn_schedule()
    if not data:
        return {}

    # Track per-team stats from completed games
    teams = {}
    for gd in data.get("leagueSchedule", {}).get("gameDates", []):
        for g in gd.get("games", []):
            if g.get("gameStatus") != 3:  # not completed
                continue
            if g.get("gameLabel", "") == "Preseason":
                continue

            for side in ["homeTeam", "awayTeam"]:
                t = g.get(side, {})
                abbrev = t.get("teamTricode", "")
                if abbrev:
                    teams[abbrev] = {
                        "wins": t.get("wins", 0),
                        "losses": t.get("losses", 0),
                    }

    # Calculate win_pct and add defaults
    result = {}
    for abbrev, stats in teams.items():
        total = stats["wins"] + stats["losses"]
        result[abbrev] = {
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_pct": stats["wins"] / max(total, 1),
            "streak": "",
            "conference_rank": 15,
            "home_record": "0-0",
            "away_record": "0-0",
            "l10_record": "0-0",
        }

    log.info(f"Built standings for {len(result)} teams from CDN")
    return result


def _get_game_log_from_cdn(team_abbrev: str) -> pd.DataFrame:
    """
    Build a simplified game log from CDN schedule data.
    Returns DataFrame with columns matching what _rolling_from_gamelog expects.
    """
    data = _fetch_cdn_schedule()
    if not data:
        return pd.DataFrame()

    games = []
    for gd in data.get("leagueSchedule", {}).get("gameDates", []):
        for g in gd.get("games", []):
            if g.get("gameStatus") != 3:
                continue
            if g.get("gameLabel", "") == "Preseason":
                continue

            ht = g.get("homeTeam", {})
            at = g.get("awayTeam", {})
            is_home = ht.get("teamTricode") == team_abbrev
            is_away = at.get("teamTricode") == team_abbrev

            if not is_home and not is_away:
                continue

            if is_home:
                pts = ht.get("score", 0)
                opp_pts = at.get("score", 0)
            else:
                pts = at.get("score", 0)
                opp_pts = ht.get("score", 0)

            wl = "W" if pts > opp_pts else "L"
            game_date = g.get("gameDateEst", "")[:10]

            games.append({
                "GAME_DATE": game_date,
                "PTS": pts,
                "WL": wl,
                # CDN doesn't have detailed stats - use league averages
                "FG_PCT": 0.471,
                "FG3_PCT": 0.363,
                "FT_PCT": 0.781,
                "REB": 44.0,
                "AST": 25.5,
                "TOV": 13.5,
                "OREB": 10.5,
                "STL": 7.5,
                "BLK": 5.0,
            })

    if not games:
        return pd.DataFrame()

    df = pd.DataFrame(games)
    # Sort by date descending (most recent first) like TeamGameLog
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    # Adjust shooting stats based on scoring (higher scoring = better shooting)
    league_avg_pts = 112
    for idx in df.index:
        pts_ratio = df.at[idx, "PTS"] / league_avg_pts
        df.at[idx, "FG_PCT"] *= pts_ratio
        df.at[idx, "FG3_PCT"] *= min(pts_ratio, 1.1)  # cap adjustment

    log.info(f"Built {len(df)} game CDN log for {team_abbrev}")
    return df
