"""
Odds conversion utilities for NBA betting.
Handles American odds, spreads, totals conversions.
"""


def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability (0-1)."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def implied_to_american(prob: float) -> int:
    """Convert implied probability (0-1) to American odds."""
    if prob <= 0 or prob >= 1:
        raise ValueError(f"Probability must be between 0 and 1, got {prob}")
    if prob >= 0.5:
        return int(round(-prob / (1 - prob) * 100))
    else:
        return int(round((1 - prob) / prob * 100))


def remove_vig(odds_a: int, odds_b: int) -> tuple[float, float]:
    """Remove bookmaker vig from a two-way market. Returns true probabilities."""
    imp_a = american_to_implied(odds_a)
    imp_b = american_to_implied(odds_b)
    total = imp_a + imp_b
    return imp_a / total, imp_b / total


def odds_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    else:
        return (100.0 / abs(american_odds)) + 1.0


def calculate_edge(model_prob: float, market_odds: int) -> float:
    """Model's edge over the market."""
    market_prob = american_to_implied(market_odds)
    return model_prob - market_prob


def calculate_kelly(model_prob: float, decimal_odds: float) -> float:
    """Kelly criterion for optimal bet sizing."""
    q = 1 - model_prob
    b = decimal_odds - 1
    kelly = (model_prob * b - q) / b
    return max(0.0, kelly)


def spread_to_implied_cover(spread: float) -> float:
    """
    Rough conversion: spread to implied cover probability.
    Standard assumption: each point of spread ~ 3% probability shift from 50%.
    """
    return 0.5 + (spread * 0.03)


def is_qualifying_ml_underdog(odds: int, min_odds: int = 130, max_odds: int = 500) -> bool:
    """Check if American odds fall within the qualifying ML underdog range."""
    return min_odds <= odds <= max_odds
