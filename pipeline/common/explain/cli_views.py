"""Console rendering for `footy-tipper advanced explain`.

Kept out of operator_cli.py so that module stays a thin dispatcher, matching
how the other advanced subcommands delegate. Every table goes through
console.panel with (label, value) rows, the same shape cli._render_model_results
uses, so output looks like the rest of the CLI in both rich and plain modes.
"""

from __future__ import annotations

import pathlib

from pipeline.common import console

# Reported after every in-sample table. Contributions computed on rows the
# model trained on overstate how useful a family looks, and a table that does
# not say so will be read as honest.
IN_SAMPLE_CAVEAT = (
    "In-sample attribution: it shows what the model uses, not whether that "
    "helps on unseen games. Run `advanced model evaluate --explain` for the "
    "honest version."
)


def _fmt(value, decimals=3, width=9):
    if value is None:
        return " " * width
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{str(value):>{width}}"
    if number != number:  # NaN
        return f"{'-':>{width}}"
    return f"{number:>{width}.{decimals}f}"


def _table(df, columns, *, label_column, limit=None):
    """Build console.panel rows: a header string plus one row per record.

    A plain string row renders in the value column, so the header lines up with
    the numbers underneath it.
    """
    header = "".join(f"{title:>{width}}" for _, title, width, _ in columns)
    rows = [header]
    frame = df.head(limit) if limit else df
    for _, record in frame.iterrows():
        cells = "".join(
            _fmt(record.get(key), decimals, width) for key, _, width, decimals in columns
        )
        rows.append((str(record.get(label_column, "")), cells))
    return rows


def render_stack(stack_description: dict) -> None:
    """The deployed probability stack: which expert actually decides the tip."""
    weights = stack_description.get("experts") or {}
    rows = [
        (f"weight {name}", f"{weight:.3f}") for name, weight in sorted(weights.items())
    ]
    rows.append(("temperature", f"{stack_description.get('temperature', 1.0):.4f}"))
    rows.append(
        ("chain multiplier", f"{stack_description.get('chain_multiplier', 1.0):.4f}")
    )
    rows.append(("dominant expert", str(stack_description.get("dominant_expert"))))
    if stack_description.get("fallback_reason"):
        rows.append(("pool fallback", str(stack_description["fallback_reason"])))
    console.panel("Deployed probability stack", rows, style="green")

    zeroed = [name for name, weight in weights.items() if weight <= 0.0]
    if zeroed and len(zeroed) < len(weights):
        console.warn(
            f"{', '.join(sorted(zeroed))} carry zero weight: they do not move the "
            "published tip at all."
        )


_DIRECTION_GLOSS = {
    "helps": "-> winner",
    "hurts": "-> loser",
    "unclear": "no direction",
    "silent": "silent",
}


def render_families(signal, *, top=None, honest=False) -> None:
    """Two panels: how loud each family is, then whether being loud helps."""
    columns = (
        ("n_features", "feats", 7, 0),
        ("mean_abs_prob_points", "prob pts", 10, 2),
        ("lift_log_loss", "lift LL", 9, 4),
        ("mean_abs_margin_points", "marg pts", 10, 2),
    )
    console.panel(
        "Feature families, ranked by skill not volume",
        _table(signal, columns, label_column="label", limit=top),
        style="cyan",
    )
    console.note(
        "prob pts is volume: mean absolute swing in percentage points. lift LL "
        "is skill: log-loss improvement from that family's contribution alone. "
        "Loud with no lift means variance you are paying for and not using."
    )

    rows = []
    frame = signal.head(top) if top else signal
    for _, record in frame.iterrows():
        if record["direction"] == "silent":
            detail = "never speaks"
        else:
            detail = (
                f"{record['agreement_rate']:.3f} "
                f"[{record['agreement_lo']:.3f}-{record['agreement_hi']:.3f}]  "
                f"{_DIRECTION_GLOSS[record['direction']]}"
            )
        rows.append((record["label"], detail))
    console.panel("Does the family point at the winner?", rows, style="cyan")
    console.note(
        "Agreement rate with a 95% interval. An interval covering 0.500 means "
        "the family has shown no directional value however loudly it speaks."
    )
    if not honest:
        console.warn(IN_SAMPLE_CAVEAT)


def render_dead(dead: dict, *, top=20) -> None:
    never = dead.get("never_split") or []
    soft = dead.get("soft_dead") or []
    rare = dead.get("rare_but_strong") or []
    total = int(dead.get("n_features", 0) or 0)

    rows = [
        ("never split", f"{len(never)} / {total} predictors, provably unused"),
        ("effectively dead", f"{len(soft)} predictors move nothing on 99% of games"),
        ("rare but strong", f"{len(rare)} predictors do niche work: keep these"),
    ]
    console.panel("Dead weight", rows, style="yellow")

    by_family = dead.get("by_family")
    if by_family is not None and len(by_family):
        columns = (
            ("never_split", "never", 8, 0),
            ("soft_dead", "quiet", 8, 0),
            ("rare_but_strong", "niche", 8, 0),
            ("active", "active", 8, 0),
        )
        console.panel(
            "Dead weight by family",
            _table(by_family.sort_values("never_split", ascending=False), columns,
                   label_column="family"),
            style="yellow",
        )

    if never:
        console.panel(
            f"Never split (first {min(top, len(never))})",
            [str(name) for name in never[:top]],
            style="dim",
        )
        console.note(
            "These are safe to delete from training_config.predictors: no "
            "booster ever split on them."
        )


def render_coverage(coverage: dict) -> None:
    buckets = coverage.get("residual_buckets")
    if buckets is not None and len(buckets):
        columns = (
            ("games", "games", 7, 0),
            ("mean_residual", "resid", 9, 2),
            ("mean_abs_margin_attribution", "marg attr", 11, 2),
            ("mean_abs_prob_attribution", "prob attr", 11, 2),
        )
        rows = _table(
            buckets.assign(bucket_label=lambda d: ["Q" + str(int(b) + 1) for b in d["bucket"]]),
            columns,
            label_column="bucket_label",
        )
        console.panel("Margin error by quintile", rows, style="magenta")
        console.note(
            "A high-error quintile with low attribution is a data gap: the model "
            "has no information about those games. High error with loud "
            "attribution is a modelling problem instead."
        )

    missing = coverage.get("missingness")
    if missing is not None and len(missing):
        columns = (
            ("games_present", "present", 9, 0),
            ("games_missing", "missing", 9, 0),
            ("mean_abs_prob_points_present", "pts here", 10, 3),
            ("mean_abs_prob_points_missing", "pts gone", 10, 3),
            ("uses_when_present", "delta", 9, 3),
        )
        console.panel(
            "Does the model use each family when it is there?",
            _table(missing.sort_values("uses_when_present"), columns, label_column="label"),
            style="magenta",
        )
        console.note(
            "A delta near zero means the model gets no more out of that family "
            "when the data is present than when it is missing."
        )

    sides = coverage.get("side_balance")
    if sides is not None and len(sides):
        columns = (
            ("n_features", "feats", 7, 0),
            ("mean_abs_prob_points", "prob pts", 10, 2),
            ("mean_abs_margin_points", "marg pts", 10, 2),
        )
        console.panel(
            "Home/away balance",
            _table(sides, columns, label_column="side"),
            style="magenta",
        )


def render_game(explanation, *, top=8, target="both", by="family", trace=False) -> None:
    """One fixture's card: the tip, its drivers, and optionally the arithmetic."""
    from pipeline.common.explain import trace as xt

    probability = explanation.probability
    header = f"{explanation.team_home} vs {explanation.team_away}"
    rows = [
        ("tip", f"{explanation.tipped_team} at {probability.published_cond:.1%}"),
        ("why", explanation.why_line),
        (
            "scoreline",
            f"{explanation.score.displayed_home}-{explanation.score.displayed_away} "
            f"({explanation.score.displayed_margin:+d})",
        ),
        ("decided by", probability.attribution_source),
        ("route", probability.route + (" + guard" if probability.guard_fired else "")),
    ]
    console.panel(header, rows, style="green")

    if probability.attribution_source == xt.ATTRIBUTION_EXPERTS:
        console.warn(
            "The classifier carries zero weight in the deployed pool, so no "
            "feature of it moves this tip."
        )

    drivers_for = {
        ("probability", "family"): explanation.prob_families,
        ("probability", "feature"): explanation.prob_drivers,
        ("margin", "family"): explanation.margin_families,
        ("margin", "feature"): explanation.margin_drivers,
    }
    targets = ("probability", "margin") if target == "both" else (target,)
    for name in targets:
        drivers = drivers_for[(name, by)][:top]
        unit = "pts" if name == "probability" else "pt margin"
        rows = [
            (
                driver.label if by == "family" else driver.key,
                f"{driver.points:+7.2f} {unit}   {driver.share:+.0%}"
                + (f"   {driver.detail}" if by == "feature" and driver.detail else ""),
            )
            for driver in drivers
        ]
        console.panel(
            f"{name.capitalize()} drivers ({by}), toward {explanation.tipped_team}",
            rows or ["nothing above the noise floor"],
            style="cyan",
        )

    if trace:
        render_trace(explanation)


def render_trace(explanation) -> None:
    """The exact arithmetic, so the published number can be checked by hand."""
    probability = explanation.probability
    score = explanation.score

    rows = []
    for name, weight in sorted(probability.weights.items()):
        value = getattr(probability, name, float("nan"))
        term = probability.expert_logit_terms.get(name)
        term_text = f"  ->{term:+.4f}" if term is not None else "   (not in this route)"
        rows.append((f"{name}", f"p={value:.4f}  w={weight:.3f}{term_text}"))
    rows.append(("pooled logit", f"{probability.pooled_logit:+.4f}"))
    rows.append(
        ("/ temperature", f"{probability.temperature:.4f} -> {probability.calibrated_logit:+.4f}")
    )
    rows.append(("consensus guard", "fired" if probability.guard_fired else "not fired"))
    rows.append(("published", f"{probability.published_cond:.4f} conditional"))
    rows.append(("feature multiplier", f"{probability.feature_multiplier:.4f}"))
    console.panel("Probability chain", rows, style="magenta")

    console.panel(
        "Score chain",
        [
            ("model mu", f"{score.mu_model_home:.2f} / {score.mu_model_away:.2f}"),
            ("baseline mu", f"{score.mu_baseline_home:.2f} / {score.mu_baseline_away:.2f}"),
            (
                "blend weights",
                f"{score.blend_weight_home:.2f} / {score.blend_weight_away:.2f}",
            ),
            ("blended mu", f"{score.mu_blended_home:.2f} / {score.mu_blended_away:.2f}"),
            ("market line", "applied" if score.line_applied else "not applied"),
            ("market total", "applied" if score.total_applied else "not applied"),
            ("simulated mu", f"{score.mu_final_home:.2f} / {score.mu_final_away:.2f}"),
            ("reconciliation", "moved the means" if score.reconciled else "not needed"),
            (
                "tier A attack/defence",
                f"home {score.tier_a_attack_home:.3f}/{score.tier_a_defence_home:.3f}  "
                f"away {score.tier_a_attack_away:.3f}/{score.tier_a_defence_away:.3f}",
            ),
        ],
        style="magenta",
    )


def render_round(args, *, root: pathlib.Path, db_path: pathlib.Path) -> int:
    from pipeline.common.explain import store as xstore

    console.section("Explain: this round")
    explanations = xstore.load_game_explanations(db_path)
    if not explanations:
        console.fail(
            "No stored explanations for the published round.",
            hint="Run `footy-tipper advanced model infer` to regenerate tips; "
            "explanations are written alongside them.",
        )
        return 1

    game_id = getattr(args, "game_id", None)
    if game_id:
        explanations = [e for e in explanations if e.game_id == int(game_id)]
        if not explanations:
            console.fail(f"Game {game_id} is not in the published round.")
            return 1

    if getattr(args, "json", False):
        from pipeline.operator_cli import _emit_json

        _emit_json(
            "advanced explain round",
            True,
            games=[e.as_dict() for e in explanations],
        )
        return 0

    for explanation in explanations:
        render_game(
            explanation,
            top=getattr(args, "top", 8),
            target=getattr(args, "target", "both"),
            by=getattr(args, "by", "family"),
            trace=getattr(args, "trace", False),
        )
    return 0


def render_disagreement(result: dict) -> None:
    if not result.get("available", False):
        console.note(f"Market disagreement skipped: {result.get('reason')}.")
        return

    console.panel(
        "Model vs market, where they disagree most",
        [
            ("market games", result["market_games"]),
            ("top-decile disagreements", result["disagreement_games"]),
            ("model accuracy there", f"{result['overall_model_accuracy']:.1%}"),
            ("market accuracy there", f"{result['overall_market_accuracy']:.1%}"),
        ],
        style="cyan",
    )

    columns = (
        ("games_leading", "games", 7, 0),
        ("deviation_share", "share", 8, 3),
        ("model_accuracy", "model", 8, 3),
        ("market_accuracy", "market", 8, 3),
        ("edge_when_family_leads", "edge", 8, 3),
    )
    console.panel(
        "When this family drives the departure, who is right?",
        _table(result["families"], columns, label_column="label"),
        style="cyan",
    )
    console.note(
        "'edge' is model accuracy minus market accuracy on the games that "
        "family led. Negative with enough games is a family the model would "
        "do better to ignore. Blank rows had too few games to score."
    )
    if not result.get("honest", False):
        console.warn(IN_SAMPLE_CAVEAT)


def render_confident_wrong(result: dict, *, top=10) -> None:
    console.panel(
        "Confident tips",
        [
            ("threshold", f"{result['threshold']:.0%}"),
            ("confident games", result["confident_games"]),
            ("of which wrong", result["confident_wrong"]),
            ("accuracy when confident", f"{result['confident_accuracy']:.1%}"),
        ],
        style="yellow",
    )

    families = result.get("families")
    if families is not None and len(families):
        columns = (
            ("mean_when_wrong", "wrong", 9, 3),
            ("mean_when_right", "right", 9, 3),
            ("difference", "diff", 8, 3),
            ("standardized", "std eff", 9, 3),
        )
        console.panel(
            "Which families push hardest when the tip is confident and wrong",
            _table(families, columns, label_column="label"),
            style="yellow",
        )
        console.note(
            "Standardized so loud families do not top the table by volume "
            "alone. A large positive effect means that family was pushing the "
            "tip hardest precisely when the tip was wrong."
        )

    worst = result.get("worst_games") or []
    if worst:
        rows = []
        for record in worst[:top]:
            # Feature names, not family names: two drivers from one family read
            # as a duplicate otherwise, and the point of this table is to name
            # the specific thing that misled the model.
            drivers = ", ".join(
                f"{str(d['feature'])[:26]} {d['points']:+.1f}"
                for d in record["top_drivers"]
            )
            rows.append(
                (
                    str(record.get("game_id") or f"row {record['row']}"),
                    f"{record['confidence']:.0%} confident, lost by "
                    f"{abs(record['actual_margin']):.0f}   {drivers}",
                )
            )
        console.panel("Worst calls, with what drove them", rows, style="red")
        console.note(
            "These are the concrete matches worth reading: a pattern here is "
            "what tells you which dataset is missing."
        )
    if not result.get("honest", False):
        console.warn(IN_SAMPLE_CAVEAT)


def render_cohort(args, *, root: pathlib.Path, db_path: pathlib.Path) -> int:
    from pipeline.common.explain import cohort as xco
    from pipeline.common.explain import report as xreport

    years = None
    start = getattr(args, "start_year", None)
    end = getattr(args, "end_year", None)
    if start or end:
        years = range(int(start or 1998), int(end or 2100) + 1)

    console.section("Explain: cohort attribution")
    console.progress("attributing the deployed models over history (slow)")
    inputs = xco.build_deployed_cohort(root, db_path, years=years)
    results = xco.run_analyses(inputs, getattr(args, "analysis", "all"))

    if getattr(args, "json", False):
        from pipeline.operator_cli import _emit_json

        _emit_json(
            "advanced explain cohort",
            True,
            **xreport.build_explain_report(results, source=inputs.source),
        )
        return 0

    console.panel(
        "Cohort",
        [
            ("games", inputs.n_games),
            ("predictors", len(inputs.feature_names)),
            ("source", inputs.source),
            ("seasons", f"{inputs.meta['years'][0]}-{inputs.meta['years'][-1]}"),
        ],
        style="cyan",
    )
    render_stack(inputs.meta.get("stack", {}))

    top = getattr(args, "top", 20)
    if "families" in results:
        render_families(results["families"], honest=inputs.honest)
    if "dead" in results:
        render_dead(results["dead"], top=top)
    if "coverage" in results:
        render_coverage(results["coverage"])
    if "disagreement" in results:
        render_disagreement(results["disagreement"])
    if "confident-wrong" in results:
        render_confident_wrong(results["confident-wrong"])

    if getattr(args, "write_report", False):
        path = xreport.write_explain_report(
            xreport.build_explain_report(
                results,
                source=inputs.source,
                config={"analysis": getattr(args, "analysis", "all"), **inputs.meta},
            ),
            root,
        )
        if path:
            console.ok(f"Report written: {path.relative_to(root)}")
    return 0


def render_report(args, *, root: pathlib.Path) -> int:
    """Render a previously written report, honest or otherwise."""
    import pandas as pd

    from pipeline.common.explain import report as xreport

    path = pathlib.Path(getattr(args, "path", None) or xreport.latest_report_path(root))
    if not path.exists():
        console.fail(
            f"No explain report at {path}.",
            hint="Run `footy-tipper advanced explain cohort --write-report`, or "
            "`advanced model evaluate --explain` for the honest version.",
        )
        return 1

    document = xreport.load_explain_report(path)
    if getattr(args, "json", False):
        print(pathlib.Path(path).read_text(encoding="utf-8"))
        return 0

    console.section("Explain: stored report", str(path))
    console.panel(
        "Report",
        [
            ("generated", document.get("generated_at")),
            ("source", document.get("source")),
            ("honest", document.get("honest")),
            ("games", document.get("n_games")),
        ],
        style="cyan",
    )

    which = getattr(args, "analysis", "all")
    honest = bool(document.get("honest"))
    if document.get("families") and which in ("all", "families"):
        render_families(pd.DataFrame(document["families"]), honest=honest)
    if document.get("dead") and which in ("all", "dead"):
        dead = dict(document["dead"])
        if isinstance(dead.get("by_family"), list):
            dead["by_family"] = pd.DataFrame(dead["by_family"])
        render_dead(dead, top=getattr(args, "top", 20))
    if document.get("coverage") and which in ("all", "coverage"):
        coverage = {
            key: pd.DataFrame(value) if isinstance(value, list) else value
            for key, value in document["coverage"].items()
        }
        render_coverage(coverage)
    if document.get("disagreement") and which in ("all", "disagreement"):
        disagreement = dict(document["disagreement"])
        if isinstance(disagreement.get("families"), list):
            disagreement["families"] = pd.DataFrame(disagreement["families"])
        render_disagreement(disagreement)
    # The stored key uses an underscore; the analysis name uses a hyphen.
    if document.get("confident_wrong") and which in ("all", "confident-wrong"):
        confident = dict(document["confident_wrong"])
        if isinstance(confident.get("families"), list):
            confident["families"] = pd.DataFrame(confident["families"])
        render_confident_wrong(confident, top=getattr(args, "top", 10))
    return 0
