"""
Collect and process historical NBA game data for backtesting.
Uses nba_api LeagueGameFinder to get complete season data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from config.settings import TEAM_ID_TO_ABBREV, PROCESSED_DIR, HISTORICAL_SEASONS
from src.ingest.nba_stats import get_season_games, season_string, get_team_game_log, _delay
from src.utils.logging import get_logger

log = get_logger(__name__)


def collect_all_seasons(seasons: list[int] = None) -> pd.DataFrame:
    """
    Collect game data for multiple seasons.
    Returns DataFrame with one row per game (home perspective merged with away).
    """
    if seasons is None:
        seasons = HISTORICAL_SEASONS

    all_games = []
    for year in seasons:
        season_str = season_string(year)
        log.info(f"Collecting season {season_str}...")
        df = get_season_games(season_str)
        if df.empty:
            continue
        games = _process_season_games(df, year)
        all_games.append(games)
        log.info(f"  -> {len(games)} games for {season_str}")

    if not all_games:
        return pd.DataFrame()

    combined = pd.concat(all_games, ignore_index=True)
    combined = combined.sort_values("game_date").reset_index(drop=True)
    log.info(f"Total: {len(combined)} games across {len(seasons)} seasons")

    # Save
    out_path = PROCESSED_DIR / "historical_games.parquet"
    combined.to_parquet(out_path, index=False)
    log.info(f"Saved to {out_path}")

    return combined


def _process_season_games(df: pd.DataFrame, season_year: int) -> pd.DataFrame:
    """
    Process raw LeagueGameFinder output into one row per game.
    Each row has home/away team stats and the result.
    """
    # Add team abbreviation
    df = df.copy()
    df["team_abbrev"] = df["TEAM_ID"].map(TEAM_ID_TO_ABBREV)

    # Parse matchup to determine home/away
    # Matchup format: "ATL vs. BOS" (home) or "ATL @ BOS" (away)
    df["is_home"] = df["MATCHUP"].str.contains(" vs. ", na=False)

    # Parse game date
    df["game_date"] = pd.to_datetime(df["GAME_DATE"])

    # Split into home and away
    home_df = df[df["is_home"]].copy()
    away_df = df[~df["is_home"]].copy()

    # Rename columns for merge
    home_cols = {
        "TEAM_ID": "home_team_id",
        "team_abbrev": "home_team",
        "PTS": "home_pts",
        "FGM": "home_fgm", "FGA": "home_fga", "FG_PCT": "home_fg_pct",
        "FG3M": "home_fg3m", "FG3A": "home_fg3a", "FG3_PCT": "home_fg3_pct",
        "FTM": "home_ftm", "FTA": "home_fta", "FT_PCT": "home_ft_pct",
        "OREB": "home_oreb", "DREB": "home_dreb", "REB": "home_reb",
        "AST": "home_ast", "TOV": "home_tov",
        "STL": "home_stl", "BLK": "home_blk",
        "PLUS_MINUS": "home_plus_minus",
        "WL": "home_wl",
    }

    away_cols = {
        "TEAM_ID": "away_team_id",
        "team_abbrev": "away_team",
        "PTS": "away_pts",
        "FGM": "away_fgm", "FGA": "away_fga", "FG_PCT": "away_fg_pct",
        "FG3M": "away_fg3m", "FG3A": "away_fg3a", "FG3_PCT": "away_fg3_pct",
        "FTM": "away_ftm", "FTA": "away_fta", "FT_PCT": "away_ft_pct",
        "OREB": "away_oreb", "DREB": "away_dreb", "REB": "away_reb",
        "AST": "away_ast", "TOV": "away_tov",
        "STL": "away_stl", "BLK": "away_blk",
        "PLUS_MINUS": "away_plus_minus",
        "WL": "away_wl",
    }

    home_renamed = home_df.rename(columns=home_cols)[
        ["GAME_ID", "game_date"] + list(home_cols.values())
    ]
    away_renamed = away_df.rename(columns=away_cols)[
        ["GAME_ID"] + list(away_cols.values())
    ]

    # Merge home and away
    games = home_renamed.merge(away_renamed, on="GAME_ID", how="inner")

    # Derived columns
    games["total_pts"] = games["home_pts"] + games["away_pts"]
    games["point_diff"] = games["home_pts"] - games["away_pts"]
    games["home_won"] = (games["home_wl"] == "W").astype(int)
    games["season"] = season_year

    # Game ID
    games = games.rename(columns={"GAME_ID": "game_id"})

    return games


def build_rolling_features(games_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Build rolling team-level features from game history.
    For each game, compute rolling stats from PRIOR games only (no leakage).
    """
    df = games_df.sort_values("game_date").copy()

    # Build team-level game history
    team_games = []
    for _, row in df.iterrows():
        # Home team entry
        team_games.append({
            "game_id": row["game_id"],
            "game_date": row["game_date"],
            "team": row["home_team"],
            "opponent": row["away_team"],
            "is_home": 1,
            "pts_for": row["home_pts"],
            "pts_against": row["away_pts"],
            "won": row["home_won"],
            "fg_pct": row["home_fg_pct"],
            "fg3_pct": row["home_fg3_pct"],
            "ft_pct": row["home_ft_pct"],
            "reb": row["home_reb"],
            "ast": row["home_ast"],
            "tov": row["home_tov"],
            "oreb": row["home_oreb"],
            "stl": row["home_stl"],
            "blk": row["home_blk"],
        })
        # Away team entry
        team_games.append({
            "game_id": row["game_id"],
            "game_date": row["game_date"],
            "team": row["away_team"],
            "opponent": row["home_team"],
            "is_home": 0,
            "pts_for": row["away_pts"],
            "pts_against": row["home_pts"],
            "won": 1 - row["home_won"],
            "fg_pct": row["away_fg_pct"],
            "fg3_pct": row["away_fg3_pct"],
            "ft_pct": row["away_ft_pct"],
            "reb": row["away_reb"],
            "ast": row["away_ast"],
            "tov": row["away_tov"],
            "oreb": row["away_oreb"],
            "stl": row["away_stl"],
            "blk": row["away_blk"],
        })

    tg = pd.DataFrame(team_games).sort_values(["team", "game_date"])

    # Rolling stats per team (shifted to avoid leakage)
    rolling_cols = ["pts_for", "pts_against", "won", "fg_pct", "fg3_pct",
                    "ft_pct", "reb", "ast", "tov", "oreb", "stl", "blk"]

    for col in rolling_cols:
        tg[f"roll_{col}"] = (
            tg.groupby("team")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
        )

    # Streak
    tg["streak"] = tg.groupby("team")["won"].transform(
        lambda x: x.shift(1).groupby((x.shift(1) != x.shift(1).shift(1)).cumsum()).cumcount() + 1
    )
    tg["streak"] = tg["streak"] * tg.groupby("team")["won"].shift(1).map({1: 1, 0: -1})
    tg["streak"] = tg["streak"].fillna(0)

    # Rest days
    tg["prev_game_date"] = tg.groupby("team")["game_date"].shift(1)
    tg["rest_days"] = (tg["game_date"] - tg["prev_game_date"]).dt.days
    tg["rest_days"] = tg["rest_days"].fillna(3).clip(0, 7)

    # Season win pct (expanding, shifted)
    tg["season_win_pct"] = tg.groupby("team")["won"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    return tg


def merge_rolling_features_to_games(games_df: pd.DataFrame, rolling_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rolling team features back onto the games DataFrame.
    Each game gets home_roll_* and away_roll_* features.
    """
    # Home team features
    home_feats = rolling_df.copy()
    home_feats = home_feats.add_prefix("home_")
    home_feats = home_feats.rename(columns={
        "home_game_id": "game_id",
        "home_team": "home_team_check",
    })

    # Away team features
    away_feats = rolling_df.copy()
    away_feats = away_feats.add_prefix("away_")
    away_feats = away_feats.rename(columns={
        "away_game_id": "game_id",
        "away_team": "away_team_check",
    })

    # Filter to home and away entries
    home_feats = home_feats[home_feats["home_is_home"] == 1]
    away_feats = away_feats[away_feats["away_is_home"] == 0]

    # Merge
    result = games_df.merge(
        home_feats[["game_id"] + [c for c in home_feats.columns if c.startswith("home_roll_") or c in ["home_streak", "home_rest_days", "home_season_win_pct"]]],
        on="game_id", how="left"
    )
    result = result.merge(
        away_feats[["game_id"] + [c for c in away_feats.columns if c.startswith("away_roll_") or c in ["away_streak", "away_rest_days", "away_season_win_pct"]]],
        on="game_id", how="left"
    )

    return result
