# NBA Betting Predictor

Automated NBA betting model that generates daily picks for three bet types:

1. **Moneyline Underdogs** (+130 to +500) — picking underdogs to win outright
2. **Spread (ATS)** — picking underdogs to cover the point spread
3. **Totals (Over/Under)** — picking whether the combined score goes over or under

## Daily Picks

Check today's picks: [TODAYS_PICKS.md](TODAYS_PICKS.md)

Picks update automatically every morning at 10:00 AM ET during the NBA season.

## How It Works

- **Data**: NBA stats from nba_api + live odds from The Odds API
- **Model**: XGBoost classifiers trained on 5+ seasons of historical data
- **Features**: Rolling team stats, rest days, streaks, pace, shooting efficiency, and more
- **Automation**: GitHub Actions runs the pipeline daily on a schedule

## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API key: `cp .env.example .env` and edit
4. Run: `python run_daily.py`

## Backtest

Run the historical backtest: `python run_backtest.py`
