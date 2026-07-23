"""Shared validation for decimal-odds markets.

Provider payloads occasionally use zeroes or one-sided prices as missing-value
sentinels.  Those values must never be promoted to real markets: downstream
fair-probability calculations assume decimal odds are finite and greater than
one.
"""

from __future__ import annotations

import math
from numbers import Real


def finite_number(value: object) -> bool:
    """Return whether *value* is a finite, non-boolean real number."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def valid_decimal_odds(value: object) -> bool:
    """A usable decimal price is finite and strictly greater than one."""
    return finite_number(value) and float(value) > 1.0


def valid_price_pair(first: object, second: object) -> bool:
    """Return whether both sides of a two-way market have usable prices."""
    return valid_decimal_odds(first) and valid_decimal_odds(second)


def validated_market_values(values: dict) -> dict:
    """Return only complete, internally valid market families.

    A handicap of exactly zero is a legitimate pick'em when both prices are
    present.  A zero handicap paired with zero/missing prices is not.
    """
    result: dict[str, float] = {}

    home_h2h = values.get("h2h_odds_home")
    away_h2h = values.get("h2h_odds_away")
    if valid_price_pair(home_h2h, away_h2h):
        result["h2h_odds_home"] = float(home_h2h)
        result["h2h_odds_away"] = float(away_h2h)

    for field in (
        "h2h_odds_home_min",
        "h2h_odds_home_max",
        "h2h_odds_away_min",
        "h2h_odds_away_max",
    ):
        if valid_decimal_odds(values.get(field)):
            result[field] = float(values[field])

    line = values.get("line_amount_home")
    line_home = values.get("line_odds_home")
    line_away = values.get("line_odds_away")
    if finite_number(line) and valid_price_pair(line_home, line_away):
        result["line_amount_home"] = float(line)
        result["line_odds_home"] = float(line_home)
        result["line_odds_away"] = float(line_away)

    total = values.get("total_line")
    total_over = values.get("total_over_odds")
    total_under = values.get("total_under_odds")
    if finite_number(total) and float(total) > 0 and valid_price_pair(
        total_over, total_under
    ):
        result["total_line"] = float(total)
        result["total_over_odds"] = float(total_over)
        result["total_under_odds"] = float(total_under)

    return result
