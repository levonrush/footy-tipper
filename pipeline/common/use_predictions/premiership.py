"""Premiership race: P(premiership) and P(grand final) for every surviving team.

The regular-season email answers "who wins on Saturday". A finals email can answer
the only question that matters in September, and it can answer it with machinery
that already exists: the calibrated model probability for fixtures that have been
drawn, and Tier-A team ratings for the rounds that have not.

The bracket is the NRL top-eight system. Its pairing rules are not asserted from
memory: `tests/test_premiership.py` reproduces the actual 2023, 2024 and 2025 week
two and week three matchups from each season's week one results.

Everything here fails soft. A bracket that does not match the template, a missing
ladder, or any other surprise returns `available: False` and the email drops one
section rather than printing a fabricated premiership table.
"""

import sqlite3

import numpy as np
import pandas as pd

from pipeline.common import rounds
from pipeline.common.model_prediciton.prediction_functions import GAME_SEED_BASE
from pipeline.common.model_training.tier_a_baseline import compute_tier_a_ratings
from pipeline.common.use_predictions.probabilities import two_way_home_probability

DEFAULT_SIMULATIONS = 20000

# The four week-one fixtures, identified by the ladder seeds they pair. Game
# number is not usable: 2023 and 2025 both ran the qualifying finals in a
# different order to 2024.
QUALIFYING_1 = (1, 4)
QUALIFYING_2 = (2, 3)
ELIMINATION_1 = (5, 8)
ELIMINATION_2 = (6, 7)
WEEK_ONE_PAIRS = (QUALIFYING_1, QUALIFYING_2, ELIMINATION_1, ELIMINATION_2)


class BracketError(RuntimeError):
    """The observed finals series does not match the top-eight template."""


def _unavailable(reason):
    return {
        "available": False,
        "reason": reason,
        "stage": None,
        "teams": [],
        "simulations": 0,
    }


# ---------------------------------------------------------------------------
# Bracket structure
# ---------------------------------------------------------------------------


def semi_final_pairs(qualifying_losers, elimination_winners):
    """Week two: the loser of each qualifying final hosts an elimination winner.

    The loser of QF1 (1v4) meets the winner of EF1 (5v8); the loser of QF2 (2v3)
    meets the winner of EF2 (6v7). Confirmed against 2023, 2024 and 2025.
    """
    return (
        (qualifying_losers[0], elimination_winners[0]),
        (qualifying_losers[1], elimination_winners[1]),
    )


def preliminary_final_pairs(qualifying_winners, semi_winners):
    """Week three: the qualifying winners host, and the semi winners cross over.

    The winner of QF1 meets the winner of SF2, and the winner of QF2 meets the
    winner of SF1. Confirmed against 2023, 2024 and 2025.
    """
    return (
        (qualifying_winners[0], semi_winners[1]),
        (qualifying_winners[1], semi_winners[0]),
    )


# ---------------------------------------------------------------------------
# Reading the season
# ---------------------------------------------------------------------------


_FINALS_QUERY = """
SELECT CAST(game_id AS INTEGER) AS game_id,
       CAST(round_id AS INTEGER) AS round_id,
       round_name,
       game_state_name,
       team_home,
       team_away,
       CAST(position_home_ladder AS INTEGER) AS seed_home,
       CAST(position_away_ladder AS INTEGER) AS seed_away,
       CAST(team_final_score_home AS REAL) AS score_home,
       CAST(team_final_score_away AS REAL) AS score_away
FROM footy_tipping_data
WHERE CAST(competition_year AS INTEGER) = ?
ORDER BY CAST(round_id AS REAL), CAST(start_time AS REAL), CAST(game_number AS REAL)
"""


def _load_season(db_path, competition_year):
    con = sqlite3.connect(str(db_path))
    try:
        season = pd.read_sql_query(_FINALS_QUERY, con, params=(int(competition_year),))
    finally:
        con.close()
    if season.empty:
        raise BracketError("no fixtures found for the season")
    last_regular = rounds.last_regular_round(
        zip(season["round_id"], season["round_name"])
    )
    season["stage"] = [
        rounds.round_stage(name, round_id, last_regular)
        for name, round_id in zip(season["round_name"], season["round_id"])
    ]
    finals = season[season["stage"] != rounds.REGULAR].reset_index(drop=True)
    return finals, last_regular


def _seed_map(week_one):
    """Team name to ladder seed, taken from the week-one fixtures."""
    seeds = {}
    for _, row in week_one.iterrows():
        for team, seed in (
            (row["team_home"], row["seed_home"]),
            (row["team_away"], row["seed_away"]),
        ):
            if pd.isna(seed) or not str(team or "").strip():
                raise BracketError("a week-one finalist has no ladder position")
            seed = int(seed)
            if seeds.setdefault(str(team), seed) != seed:
                raise BracketError(f"{team} appears with two ladder positions")
    if sorted(seeds.values()) != list(range(1, 9)):
        raise BracketError(
            "week-one finalists are not ladder positions 1 to 8: "
            + ", ".join(str(value) for value in sorted(seeds.values()))
        )
    observed = {
        tuple(sorted((int(row["seed_home"]), int(row["seed_away"]))))
        for _, row in week_one.iterrows()
    }
    if observed != {tuple(sorted(pair)) for pair in WEEK_ONE_PAIRS}:
        raise BracketError("week-one pairings do not match the top-eight template")
    return seeds


def _resolved_results(season, seeds):
    """Completed finals games as `frozenset(seed pair) -> winning seed`."""
    resolved = {}
    played = season[
        (season["game_state_name"] == "Final")
        & season["score_home"].notna()
        & season["score_away"].notna()
        & (season["score_home"] != season["score_away"])
    ]
    for _, row in played.iterrows():
        home, away = str(row["team_home"]), str(row["team_away"])
        if home not in seeds or away not in seeds:
            continue
        winner = home if row["score_home"] > row["score_away"] else away
        resolved[frozenset((seeds[home], seeds[away]))] = seeds[winner]
    return resolved


def _surviving_seeds(stage, predictions, seeds, resolved):
    """Seeds still in the competition: playing this week, or on a week-two bye.

    Deriving this from simulation counts would call a long shot eliminated, so it
    is read off the draw instead.
    """
    alive = {
        seeds[str(team)]
        for column in ("team_home", "team_away")
        for team in predictions[column]
        if str(team) in seeds
    }
    if stage == rounds.FINALS_WEEK_2:
        # Qualifying-final winners have the week off and are very much alive.
        for pair in (QUALIFYING_1, QUALIFYING_2):
            winner = resolved.get(frozenset(pair))
            if winner is not None:
                alive.add(winner)
    return alive


def _model_probabilities(predictions, seeds):
    """`frozenset(seed pair) -> P(lower seed wins)` for the scheduled round."""
    priced = {}
    if predictions is None or getattr(predictions, "empty", True):
        return priced
    for _, row in predictions.iterrows():
        home, away = str(row["team_home"]), str(row["team_away"])
        if home not in seeds or away not in seeds:
            continue
        home_probability = two_way_home_probability(
            row.get("home_team_win_prob"), row.get("home_team_lose_prob")
        )
        if home_probability is None or not np.isfinite(home_probability):
            continue
        seed_home, seed_away = seeds[home], seeds[away]
        favourite = min(seed_home, seed_away)
        probability = (
            float(home_probability)
            if seed_home == favourite
            else 1.0 - float(home_probability)
        )
        priced[frozenset((seed_home, seed_away))] = probability
    return priced


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _pair_probability(pair, host_seed, seeds_by_number, ratings, priced):
    """P(the first seed of `pair` wins), from the model where possible.

    `pair` is ordered by bracket role, not by seed, so both lookups convert into
    that orientation before returning.
    """
    key = frozenset(pair)
    if key in priced:
        # `priced` is keyed on the better seed winning.
        probability = priced[key]
        return probability if pair[0] == min(pair) else 1.0 - probability
    neutral = host_seed is None
    host = pair[0] if neutral else host_seed
    guest = pair[1] if host == pair[0] else pair[0]
    host_probability = ratings.home_win_probability(
        seeds_by_number[host], seeds_by_number[guest], neutral=neutral
    )
    return host_probability if pair[0] == host else 1.0 - host_probability


def _simulate(seeds_by_number, ratings, priced, resolved, simulations, seed):
    """Monte Carlo over the remaining bracket, returning per-seed tallies."""
    rng = np.random.default_rng(seed)
    premierships = {number: 0 for number in seeds_by_number}
    grand_finals = {number: 0 for number in seeds_by_number}

    def play(pair, host_seed, draws):
        """Vectorised outcome for one bracket slot across every simulation."""
        key = frozenset(pair)
        if key in resolved:
            winner = resolved[key]
            loser = pair[0] if winner == pair[1] else pair[1]
            return (
                np.full(draws.shape, winner, dtype=int),
                np.full(draws.shape, loser, dtype=int),
            )
        probability = _pair_probability(pair, host_seed, seeds_by_number, ratings, priced)
        first_wins = draws < probability
        winners = np.where(first_wins, pair[0], pair[1])
        losers = np.where(first_wins, pair[1], pair[0])
        return winners, losers

    uniforms = rng.random((9, simulations))

    qf1_w, qf1_l = play(QUALIFYING_1, QUALIFYING_1[0], uniforms[0])
    qf2_w, qf2_l = play(QUALIFYING_2, QUALIFYING_2[0], uniforms[1])
    ef1_w, _ = play(ELIMINATION_1, ELIMINATION_1[0], uniforms[2])
    ef2_w, _ = play(ELIMINATION_2, ELIMINATION_2[0], uniforms[3])

    # From here the matchups differ between simulations, so the remaining rounds
    # are resolved per distinct pairing rather than as one vectorised draw.
    sf1_w = _play_dynamic(qf1_l, ef1_w, uniforms[4], seeds_by_number, ratings, priced, resolved)
    sf2_w = _play_dynamic(qf2_l, ef2_w, uniforms[5], seeds_by_number, ratings, priced, resolved)
    pf1_w = _play_dynamic(qf1_w, sf2_w, uniforms[6], seeds_by_number, ratings, priced, resolved)
    pf2_w = _play_dynamic(qf2_w, sf1_w, uniforms[7], seeds_by_number, ratings, priced, resolved)
    champion = _play_dynamic(
        pf1_w, pf2_w, uniforms[8], seeds_by_number, ratings, priced, resolved, neutral=True
    )

    for number in seeds_by_number:
        premierships[number] = int(np.sum(champion == number))
        grand_finals[number] = int(np.sum((pf1_w == number) | (pf2_w == number)))
    return premierships, grand_finals


def _play_dynamic(hosts, guests, draws, seeds_by_number, ratings, priced, resolved, neutral=False):
    """Resolve a bracket slot whose matchup varies across simulations."""
    winners = np.empty(hosts.shape, dtype=int)
    pairings = np.stack([hosts, guests])
    for host, guest in {tuple(column) for column in pairings.T}:
        mask = (hosts == host) & (guests == guest)
        key = frozenset((host, guest))
        if key in resolved:
            winners[mask] = resolved[key]
            continue
        probability = _pair_probability(
            (host, guest), None if neutral else host, seeds_by_number, ratings, priced
        )
        winners[mask] = np.where(draws[mask] < probability, host, guest)
    return winners


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def premiership_race(db_path, predictions, simulations=DEFAULT_SIMULATIONS):
    """Premiership and grand-final probabilities for the current finals series.

    Never raises. Returns `{"available": False, "reason": ...}` when the bracket
    cannot be resolved.
    """
    try:
        return _premiership_race(db_path, predictions, simulations)
    except BracketError as exc:
        return _unavailable(str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Premiership race failed soft ({exc}).")
        return _unavailable(f"premiership simulation failed ({exc})")


def _premiership_race(db_path, predictions, simulations):
    if predictions is None or getattr(predictions, "empty", True):
        raise BracketError("no predictions to anchor the finals series")

    competition_year = int(predictions.iloc[0]["competition_year"])
    round_name = predictions.iloc[0].get("round_name")
    round_id = predictions.iloc[0].get("round_id")

    finals, last_regular = _load_season(db_path, competition_year)
    if finals.empty:
        raise BracketError("the season has no finals fixtures yet")

    week_one = finals[finals["stage"] == rounds.FINALS_WEEK_1]
    if len(week_one) != 4:
        raise BracketError(
            f"expected four week-one finals, found {len(week_one)}"
        )
    seeds = _seed_map(week_one)
    seeds_by_number = {seed: team for team, seed in seeds.items()}

    resolved = _resolved_results(finals, seeds)
    priced = _model_probabilities(predictions, seeds)

    con = sqlite3.connect(str(db_path))
    try:
        history = pd.read_sql_query("SELECT * FROM footy_tipping_data", con)
    finally:
        con.close()
    ratings = compute_tier_a_ratings(history)
    unrated = [team for team in seeds if not ratings.knows(team)]
    if unrated:
        raise BracketError("no team ratings for " + ", ".join(sorted(unrated)))

    stage = rounds.round_stage(round_name, round_id, last_regular)
    simulations = max(1000, int(simulations))
    seed = int(GAME_SEED_BASE + 7919 * int(round_id if pd.notna(round_id) else 0))
    premierships, grand_finals = _simulate(
        seeds_by_number, ratings, priced, resolved, simulations, seed
    )

    alive = _surviving_seeds(stage, predictions, seeds, resolved)
    teams = [
        {
            "team": seeds_by_number[number],
            "seed": number,
            "p_premiership": premierships[number] / simulations,
            "p_grand_final": grand_finals[number] / simulations,
            "alive": number in alive,
        }
        for number in sorted(seeds_by_number)
    ]
    teams.sort(key=lambda entry: (-entry["p_premiership"], entry["seed"]))

    scheduled = len(priced)
    return {
        "available": True,
        "reason": None,
        "stage": stage,
        "competition_year": competition_year,
        "teams": teams,
        "simulations": simulations,
        "scheduled_games_priced_by_model": scheduled,
        "ratings_calibrated": ratings.calibrated,
        "method": (
            "This week's games use the full model. Matchups that have not been "
            "drawn yet are priced from team ratings, which are blunter."
        ),
    }
