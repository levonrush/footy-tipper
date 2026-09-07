"""Round stage classification shared by ingestion, prediction and delivery.

`round_name` is free text from the nrl.com draw (`roundTitle`). Across 2008-2026 it
takes exactly five shapes: `Round N` and the four finals names `Finals Week 1`,
`Finals Week 2`, `Finals Week 3` and `Grand Final`. Lineup article titles use the
older broadcast vocabulary instead (`Qualifying Final`, `Elimination Final`,
`Semi Final`, `Preliminary Final`), so both vocabularies are recognised here.

This module is deliberately stdlib-only: the ladder builder and the delivery layer
both classify rounds, and neither should have to import the other.

The `round_id` fallback is defensive and opt-in. Finals round numbers are
season-dependent (2020 used 21-24, 2022 used 26-29, 2023-2025 use 28-31), so a
caller must supply the season's last regular round before numbers can be trusted.
Without it an unrecognised name stays `REGULAR`, which is the current behaviour.
"""

import re


REGULAR = "regular"
FINALS_WEEK_1 = "finals_week_1"
FINALS_WEEK_2 = "finals_week_2"
PRELIMINARY = "preliminary_final"
GRAND_FINAL = "grand_final"

# Ordered by how far through the series they sit.
FINALS_STAGES = (FINALS_WEEK_1, FINALS_WEEK_2, PRELIMINARY, GRAND_FINAL)

_STAGE_BY_WEEK = {1: FINALS_WEEK_1, 2: FINALS_WEEK_2, 3: PRELIMINARY, 4: GRAND_FINAL}
_WEEK_BY_STAGE = {stage: week for week, stage in _STAGE_BY_WEEK.items()}

# The name the feed uses, so joins and log lines stay recognisable.
_FEED_LABELS = {
    FINALS_WEEK_1: "Finals Week 1",
    FINALS_WEEK_2: "Finals Week 2",
    PRELIMINARY: "Finals Week 3",
    GRAND_FINAL: "Grand Final",
}

# What people actually call them.
_DISPLAY_NAMES = {
    FINALS_WEEK_1: "Finals Week 1",
    FINALS_WEEK_2: "Semi Finals",
    PRELIMINARY: "Preliminary Finals",
    GRAND_FINAL: "Grand Final",
}

_SLUGS = {
    FINALS_WEEK_1: "finals-week-1",
    FINALS_WEEK_2: "semi-finals",
    PRELIMINARY: "preliminary-finals",
    GRAND_FINAL: "grand-final",
}

_FINALS_WEEK_PATTERN = re.compile(r"finals?\s*week\s*(\d+)")
_ROUND_NUMBER_PATTERN = re.compile(r"^round\s*(\d+)$")


def _normalize(round_name) -> str:
    return re.sub(r"\s+", " ", str(round_name or "").strip().lower())


def _stage_from_name(name: str) -> str | None:
    """Classify a normalised round name, or None when it is not finals-shaped."""
    if not name:
        return None

    # Order matters: every finals name below also contains "final".
    if "grand final" in name:
        return GRAND_FINAL
    if "preliminary" in name or "prelim" in name:
        return PRELIMINARY

    week_match = _FINALS_WEEK_PATTERN.search(name)
    if week_match:
        week = int(week_match.group(1))
        # A week beyond the series is the decider; week 0 is not a thing.
        return _STAGE_BY_WEEK.get(week, GRAND_FINAL if week > 4 else FINALS_WEEK_1)

    if "semi final" in name or "semi-final" in name:
        return FINALS_WEEK_2
    if "qualifying final" in name or "elimination final" in name:
        return FINALS_WEEK_1
    if _ROUND_NUMBER_PATTERN.match(name):
        # A bare "Round N" is regular unless a round number says otherwise. The
        # draw synthesises this name when `roundTitle` is missing, so the caller's
        # `last_regular_round` is the only thing that can catch a mislabelled
        # finals round.
        return None
    if "final" in name:
        # An unrecognised finals-ish name resolves to the earliest finals stage
        # rather than to a regular round: under-hyping one email is recoverable,
        # accumulating a ladder through the finals is not.
        return FINALS_WEEK_1
    return None


def _stage_from_round_id(round_id, last_regular_round) -> str | None:
    if round_id is None or last_regular_round is None:
        return None
    try:
        offset = int(round_id) - int(last_regular_round)
    except (TypeError, ValueError):
        return None
    if offset <= 0:
        return None
    return _STAGE_BY_WEEK.get(offset, GRAND_FINAL)


def round_stage(round_name, round_id=None, last_regular_round=None) -> str:
    """Return the stage constant for a round.

    `round_name` wins whenever it is recognisable. `round_id` is consulted only
    when `last_regular_round` is supplied and the name did not resolve.
    """
    stage = _stage_from_name(_normalize(round_name))
    if stage is not None:
        return stage
    stage = _stage_from_round_id(round_id, last_regular_round)
    if stage is not None:
        return stage
    return REGULAR


def is_finals(round_name, round_id=None, last_regular_round=None) -> bool:
    return round_stage(round_name, round_id, last_regular_round) != REGULAR


def stage_week(stage) -> int | None:
    """1 through 4 for the finals stages, None for a regular round."""
    return _WEEK_BY_STAGE.get(stage)


def stage_feed_label(stage) -> str | None:
    """The name the nrl.com draw uses, or None for a regular round."""
    return _FEED_LABELS.get(stage)


def stage_display_name(stage) -> str | None:
    """The name a supporter would use, or None for a regular round."""
    return _DISPLAY_NAMES.get(stage)


def round_slug(round_name, round_id=None, last_regular_round=None) -> str:
    """Filesystem-safe identifier for a round, used for site archive pages."""
    stage = round_stage(round_name, round_id, last_regular_round)
    if stage in _SLUGS:
        return _SLUGS[stage]
    number = None
    match = _ROUND_NUMBER_PATTERN.match(_normalize(round_name))
    if match:
        number = int(match.group(1))
    elif round_id is not None:
        try:
            number = int(round_id)
        except (TypeError, ValueError):
            number = None
    return f"round-{number}" if number is not None else "round"


def last_regular_round(rounds) -> int | None:
    """Highest round number whose name is not finals-shaped.

    `rounds` is an iterable of `(round_id, round_name)`. Returns None when the
    season has no recognisable regular rounds.
    """
    highest = None
    for round_id, round_name in rounds:
        if _stage_from_name(_normalize(round_name)) is not None:
            continue
        try:
            number = int(round_id)
        except (TypeError, ValueError):
            continue
        if highest is None or number > highest:
            highest = number
    return highest
