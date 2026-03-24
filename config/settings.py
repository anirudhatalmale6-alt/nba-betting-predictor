"""
Central configuration for the NBA Betting Predictor.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"

for d in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ─────────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ── Odds API ─────────────────────────────────────────────────────────────
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "basketball_nba"
ODDS_REGIONS = "us"
ODDS_FORMAT = "american"

# ── Bet Type Settings ────────────────────────────────────────────────────
# Moneyline underdogs
MIN_ML_UNDERDOG_ODDS = int(os.getenv("MIN_ML_UNDERDOG_ODDS", "130"))
MAX_ML_UNDERDOG_ODDS = int(os.getenv("MAX_ML_UNDERDOG_ODDS", "500"))

# Spreads: model picks underdog ATS
# Totals: model picks over or under

# ── Model Settings ───────────────────────────────────────────────────────
EDGE_THRESHOLD_ML = float(os.getenv("EDGE_THRESHOLD_ML", "0.03"))
EDGE_THRESHOLD_SPREAD = float(os.getenv("EDGE_THRESHOLD_SPREAD", "0.02"))
EDGE_THRESHOLD_TOTAL = float(os.getenv("EDGE_THRESHOLD_TOTAL", "0.02"))

# Training seasons for historical backtest
HISTORICAL_SEASONS = list(range(2019, 2026))  # 2019-2025
BACKTEST_TEST_SEASONS = [2022, 2023, 2024, 2025]

# XGBoost default hyperparameters
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "learning_rate": 0.03,
    "n_estimators": 500,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

# ── Feature Engineering ──────────────────────────────────────────────────
ROLLING_WINDOW_GAMES = 10       # Games for rolling stats
REST_DAYS_THRESHOLD = 2         # Back-to-back = 1, well-rested >= 3

# ── Logging ──────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Output ───────────────────────────────────────────────────────────────
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "both")  # csv, json, or both

# ── NBA Team Mapping ─────────────────────────────────────────────────────
# nba_api team IDs to abbreviations
TEAM_ID_TO_ABBREV = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN", 1610612766: "CHA",
    1610612741: "CHI", 1610612739: "CLE", 1610612742: "DAL", 1610612743: "DEN",
    1610612765: "DET", 1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
    1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612740: "NOP", 1610612752: "NYK",
    1610612760: "OKC", 1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612761: "TOR",
    1610612762: "UTA", 1610612764: "WAS",
}

ABBREV_TO_TEAM_ID = {v: k for k, v in TEAM_ID_TO_ABBREV.items()}

# Odds API team name to abbreviation
ODDS_TEAM_TO_ABBREV = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

# Reverse lookup
ABBREV_TO_ODDS_TEAM = {v: k for k, v in ODDS_TEAM_TO_ABBREV.items()}

# Home court advantage (points, approximate)
HOME_COURT_ADVANTAGE = 3.0

# Pace factors (possessions per 48 min, approximate league avg ~100)
# These get updated from actual data during feature engineering
DEFAULT_PACE = 100.0
