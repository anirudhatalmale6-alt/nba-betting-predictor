"""
NBA stats data ingestion using nba_api.
Fetches team stats, game logs, standings, and schedules.
"""

import time
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
    Get today's NBA games from the scoreboard.
    Returns list of game dicts with team info.
    """
    if target_date is None:
        target_date = date.today()

    date_str = target_date.strftime("%Y-%m-%d")

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
            for _, row in games_header.iterrows():
                home_id = row.get("HOME_TEAM_ID")
                away_id = row.get("VISITOR_TEAM_ID")
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


def get_team_game_log(team_id: int, season: str) -> pd.DataFrame:
    """
    Get a team's game log for a season.
    season format: "2024-25"
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            gl = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season,
                season_type_all_star="Regular Season",
                headers=_HEADERS,
                timeout=60,
            )
            _delay()
            df = gl.get_data_frames()[0]
            return df
        except Exception as e:
            log.warning(f"Game log attempt {attempt}/{MAX_RETRIES} for team {team_id}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)
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
            timeout=60,
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
        log.warning(f"Error fetching standings: {e}")
        return {}


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
        return gl
    return gl.head(n_games)
