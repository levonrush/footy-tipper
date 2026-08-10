"""Reading the deployed probability stack, exactly as inference assembles it.

The published conditional home-win probability is built in three steps:

    experts (Tier A / Tier B / Tier C / market)
      -> SimplexLogitPool:   pooled_logit = sum_i w_i * logit(p_i)
      -> TemperatureCalibrator: published_logit = pooled_logit / T
      -> consensus guard (may replace the row wholesale with Tier B)

The pool and the temperature are both linear in logit space, so a feature's
effect on the published probability is its effect on its expert's logit scaled
by ``w_i / T``. That single number is the bridge between TreeSHAP output and
the number in the email, and it is what ``chain_multiplier`` returns.

Nothing here re-runs a model or a simulation.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import asdict, dataclass, field

from pipeline.common.model_training import calibration as calib

# The expert that carries the LightGBM binary classifier, i.e. the one whose
# logit TreeSHAP on binary_model explains.
TIER_C = "tier_c"

_ARTIFACTS = (
    ("stacker", "stacker.pkl"),
    ("calibrator", "win_prob_calibrator.pkl"),
    ("stacker_no_market", "stacker_no_market.pkl"),
    ("calibrator_no_market", "win_prob_calibrator_no_market.pkl"),
    ("binary_model", "binary_model.pkl"),
)


@dataclass(frozen=True)
class ProbabilityStack:
    """The deployed probability artifacts plus the manifest that describes them."""

    manifest: dict = field(default_factory=dict)
    stacker: object = None
    calibrator: object = None
    stacker_no_market: object = None
    calibrator_no_market: object = None
    binary_model: object = None

    @property
    def weight_map(self) -> dict:
        return expert_weights(self.stacker)

    @property
    def temperature(self) -> float:
        return temperature_of(self.calibrator)

    @property
    def chain_multiplier(self) -> float:
        return chain_multiplier(self.stacker, self.calibrator)

    def describe(self) -> dict:
        weights = self.weight_map
        return {
            "experts": weights,
            "temperature": self.temperature,
            "chain_multiplier": self.chain_multiplier,
            "dominant_expert": max(weights, key=weights.get) if weights else None,
            "fallback_reason": (
                (self.manifest.get("probability_stack") or {})
                .get("market", {})
                .get("selection", {})
                .get("fallback_reason")
            ),
        }


def load_probability_stack(models_dir) -> ProbabilityStack:
    """Load the stack artifacts, tolerating any that are absent."""
    models_dir = pathlib.Path(models_dir)
    manifest = {}
    manifest_path = models_dir / "model_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    loaded = {}
    for attr, filename in _ARTIFACTS:
        path = models_dir / filename
        loaded[attr] = calib.load_artifact(path) if path.exists() else None
    return ProbabilityStack(manifest=manifest, **loaded)


def expert_weights(stacker) -> dict:
    """Simplex weights per expert, empty when the stacker is absent or legacy."""
    weight_map = getattr(stacker, "weight_map", None)
    if isinstance(weight_map, dict):
        return {str(name): float(weight) for name, weight in weight_map.items()}
    return {}


def temperature_of(calibrator) -> float:
    """Temperature of a TemperatureCalibrator; 1.0 for anything else."""
    temperature = getattr(calibrator, "temperature_", None)
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        return 1.0
    return temperature if temperature > 0 else 1.0


def is_logit_linear(stacker, calibrator) -> bool:
    """Whether the published logit is an exactly linear function of expert logits.

    True for SimplexLogitPool + TemperatureCalibrator, which is what is deployed.
    False for the legacy LogisticStacker/BetaCalibrator pair, where a scalar
    multiplier would misstate the chain and callers should say so rather than
    quietly reporting an approximation.
    """
    return bool(expert_weights(stacker)) and hasattr(calibrator, "temperature_")


ATTRIBUTION_BINARY = "binary_model"
ATTRIBUTION_SCORE = "score_models"
ATTRIBUTION_EXPERTS = "experts_only"


@dataclass(frozen=True)
class ProbabilityTrace:
    """How one game's published win probability was assembled, exactly."""

    game_id: int
    tier_a: float
    tier_b: float
    tier_c: float = float("nan")
    market: float = float("nan")
    valid_market: bool = False
    weights: dict = field(default_factory=dict)
    expert_logit_terms: dict = field(default_factory=dict)
    pooled_logit: float = 0.0
    temperature: float = 1.0
    calibrated_logit: float = 0.0
    route: str = "tier_b"
    guard_fired: bool = False
    published_cond: float = 0.5
    draw_prob: float = 0.0
    published_home_win_prob: float = 0.5
    attribution_source: str = ATTRIBUTION_BINARY
    feature_multiplier: float = 1.0
    exact_chain: bool = True

    @property
    def tipped_home(self) -> bool:
        return self.published_cond >= 0.5

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["tipped_home"] = self.tipped_home
        return payload


@dataclass(frozen=True)
class ScoreTrace:
    """How one game's scoreline was assembled, from raw model means outward."""

    game_id: int
    mu_model_home: float = float("nan")
    mu_model_away: float = float("nan")
    mu_baseline_home: float = float("nan")
    mu_baseline_away: float = float("nan")
    blend_weight_home: float = 1.0
    blend_weight_away: float = 1.0
    mu_blended_home: float = float("nan")
    mu_blended_away: float = float("nan")
    mu_final_home: float = float("nan")
    mu_final_away: float = float("nan")
    line_applied: bool = False
    total_applied: bool = False
    reconciled: bool = False
    displayed_home: int = 0
    displayed_away: int = 0
    displayed_margin: int = 0
    tier_a_attack_home: float = float("nan")
    tier_a_defence_home: float = float("nan")
    tier_a_attack_away: float = float("nan")
    tier_a_defence_away: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


def prob_per_margin_point(mu_home, mu_away, step=1.0) -> float:
    """d(conditional home win probability) / d(margin), at this fixture.

    Used to price score-model feature contributions in probability terms on the
    rows where the published probability comes from the score models rather
    than the classifier. A deterministic central difference along the
    total-preserving ray: no simulation, no RNG, so the determinism contract is
    untouched.
    """
    from pipeline.common.model_prediciton import prediction_functions as pf

    half = step / 2.0
    up = pf.conditional_home_win_prob(max(mu_home + half, 1e-6), max(mu_away - half, 1e-6))
    down = pf.conditional_home_win_prob(max(mu_home - half, 1e-6), max(mu_away + half, 1e-6))
    return (up - down) / step


def chain_multiplier(stacker, calibrator, expert: str = TIER_C) -> float:
    """d(published logit) / d(expert logit), i.e. ``w_expert / T``.

    Returns 0.0 when the expert carries no weight, which is the honest answer:
    that model's features do not move the published probability at all.
    """
    weights = expert_weights(stacker)
    if not weights:
        # No simplex pool deployed. Treat the expert as passed through so
        # attribution still has a usable scale, and let is_logit_linear flag it.
        return 1.0 / temperature_of(calibrator)
    return weights.get(expert, 0.0) / temperature_of(calibrator)


def _logit(p, eps=1e-12):
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def build_probability_traces(
    *,
    game_ids,
    stack: ProbabilityStack,
    tier_a,
    tier_b,
    tier_c=None,
    market=None,
    valid_market=None,
    routes: dict = None,
    published_cond,
    draw_prob=None,
    mu_home=None,
    mu_away=None,
):
    """One ProbabilityTrace per game, reconstructed from values inference kept.

    The attribution source is the part that has to be right. On the deployed
    stack the published logit is ``w_tier_c / T`` times the classifier's logit,
    so the classifier's TreeSHAP explains the tip. But a row that the consensus
    guard reversed, or that fell back to Tier B, was NOT decided by the
    classifier, and attributing it there would be quietly wrong. Those rows are
    routed to the score models instead, priced through d(p)/d(margin).
    """
    routes = routes or {}
    row_route = routes.get("row_route") or []
    row_guarded = routes.get("row_guarded") or []

    weights = expert_weights(stack.stacker)
    temperature = temperature_of(stack.calibrator)
    exact = is_logit_linear(stack.stacker, stack.calibrator)
    tier_c_multiplier = chain_multiplier(stack.stacker, stack.calibrator)

    traces = []
    for i, game_id in enumerate(game_ids):
        route = str(row_route[i]) if i < len(row_route) else "tier_b"
        guarded = bool(row_guarded[i]) if i < len(row_guarded) else False

        experts = {"tier_a": float(tier_a[i]), "tier_b": float(tier_b[i])}
        if tier_c is not None:
            experts["tier_c"] = float(tier_c[i])
        has_market = bool(valid_market[i]) if valid_market is not None else False
        if market is not None:
            experts["market"] = float(market[i])

        # For a no-market row the pool that ran excluded the market expert, so
        # its term must not appear in the decomposition of that row's logit.
        active = {
            name: value
            for name, value in experts.items()
            if name in weights and (name != "market" or has_market)
        }
        terms = {name: weights[name] * _logit(value) for name, value in active.items()}
        pooled = sum(terms.values())

        cond = float(published_cond[i])
        if guarded or route == "tier_b":
            source = ATTRIBUTION_SCORE
            multiplier = (
                prob_per_margin_point(float(mu_home[i]), float(mu_away[i]))
                if mu_home is not None and mu_away is not None
                else 0.0
            )
        elif tier_c_multiplier == 0.0:
            # The classifier carries no weight, so no feature of it moves this
            # tip. Saying so is more useful than showing drivers that do nothing.
            source = ATTRIBUTION_EXPERTS
            multiplier = 0.0
        else:
            source = ATTRIBUTION_BINARY
            multiplier = tier_c_multiplier

        draw = float(draw_prob[i]) if draw_prob is not None else 0.0
        traces.append(
            ProbabilityTrace(
                game_id=int(game_id),
                tier_a=experts["tier_a"],
                tier_b=experts["tier_b"],
                tier_c=experts.get("tier_c", float("nan")),
                market=experts.get("market", float("nan")),
                valid_market=has_market,
                weights=dict(weights),
                expert_logit_terms=terms,
                pooled_logit=pooled,
                temperature=temperature,
                calibrated_logit=pooled / temperature if temperature else pooled,
                route=route,
                guard_fired=guarded,
                published_cond=cond,
                draw_prob=draw,
                published_home_win_prob=cond * max(0.0, 1.0 - draw),
                attribution_source=source,
                feature_multiplier=float(multiplier),
                exact_chain=exact,
            )
        )
    return traces
