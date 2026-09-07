"""LLM-generated (and fallback) email copy plus full email payload assembly."""

import html
import json
import os
import re

import pandas as pd

# For direct Anthropic API calls
try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

from pipeline.common import rounds
from pipeline.common.use_predictions.banner import (
    _generate_dynamic_banner,
    _resolve_banner_path,
)
from pipeline.common.use_predictions.distribution import _sort_predictions_for_display
from pipeline.common.use_predictions.email_render import (
    _default_subject,
    _format_percent,
    _format_market_price,
    _format_predicted_margin,
    _format_predicted_score_numbers,
    _format_price,
    _format_probability,
    _joker_prompt_block,
    _market_coverage_notice,
    _prediction_winner,
    _render_html_email,
    _render_plain_email,
    two_way_home_probability,
)
from pipeline.common.use_predictions.llm import DEFAULT_CLAUDE_MODEL
from pipeline.common.use_predictions.news import _fetch_nrl_news_context
from pipeline.common.use_predictions.scoreboard import scoreboard_summary_line


def _finals_prompt_block(finals):
    """The finals-only material the copywriter gets, or None for a regular round."""
    if not isinstance(finals, dict) or not finals.get("is_finals"):
        return None

    lines = []
    ledger = finals.get("ledger") or {}
    regular = ledger.get("regular_season")
    if regular:
        market = (
            f", market favourite {regular['market_accuracy']:.0%}"
            if regular.get("market_accuracy") is not None
            else ""
        )
        lines.append(
            f"- Regular season finished: {regular['correct']}/{regular['games']} "
            f"({regular['accuracy']:.0%}){market}. The tipping comp is over; this is a "
            "special edition, not a comp round."
        )
    finals_record = ledger.get("finals")
    if finals_record:
        lines.append(
            f"- Finals so far: {finals_record['correct']}/{finals_record['games']}."
        )
    best = ledger.get("best_call")
    if best:
        lines.append(
            f"- Best call of the year: {best['tipped_team']} over {best['opponent']} in "
            f"{best['round_name']}, {best['score']}, when the market gave them "
            f"{best['market_probability']:.0%}."
        )
    worst = ledger.get("worst_call")
    if worst:
        lines.append(
            f"- Worst call of the year: {worst['tipped_team']} at "
            f"{worst['tip_probability']:.0%} against {worst['opponent']} in "
            f"{worst['round_name']}, lost {worst['score']}."
        )

    for game_id, stakes in (finals.get("stakes") or {}).items():
        history = (finals.get("head_to_head") or {}).get(game_id)
        lines.append(f"- Stakes: {stakes}" + (f" {history}" if history else ""))

    race = finals.get("premiership")
    if isinstance(race, dict) and race.get("available"):
        live = [team for team in race["teams"] if team["alive"]]
        summary = ", ".join(
            f"{team['team']} {team['p_premiership']:.0%}" for team in live
        )
        lines.append(f"- Premiership probabilities: {summary}.")
    return "\n".join(lines) if lines else None


def _build_fallback_copy(predictions, folder_url, joker_recommendation=None, finals=None):
    if predictions.empty:
        return {
            "subject": "Footy Tipper Update",
            "opening": "No pre-game NRL fixtures were found for the current run, so there are no tips to send this week.",
            "closing": "Bring back the biff.",
        }

    round_name = predictions['round_name'].iloc[0]
    competition_year = predictions['competition_year'].iloc[0]
    is_finals = bool(isinstance(finals, dict) and finals.get("is_finals"))
    special_event_context = _special_event_context(
        round_name, competition_year, stage=(finals or {}).get("stage")
    )
    opening = (
        f"Welcome to {round_name} {competition_year}. {special_event_context['opening_line']}\n"
        "The model has done the hard yakka and lined up this week's tips.\n"
    )
    if is_finals:
        # No comp to torch: it finished with the regular season, and there is no
        # joker to watch either.
        opening += (
            "This is a special edition. The tipping comp finished with the regular "
            "season, so these tips are purely for bragging rights."
        )
    else:
        opening += (
            "If these picks torch your tipping comp, remember this was all done "
            "with science and zero accountability."
        )
    if not is_finals and isinstance(joker_recommendation, dict):
        opening += (
            "\nJoker watch: "
            f"{joker_recommendation.get('headline', 'Joker call unavailable')}."
        )
    if folder_url:
        opening += f"\nFull details are in the tips folder: {folder_url}"
    closing = (
        "If you're in tipping comps at the Seven Seas Hotel in Carrington or the work comp at Hunter Water, "
        "you did not get this from us.\nBring back the biff."
    )
    return {
        "subject": _default_subject(predictions, finals=finals),
        "opening": opening,
        "closing": closing,
    }


# Keyed on the shared round-stage constants rather than on guessed round names.
# The previous version matched "qualifying final", "elimination final" and
# "semi final", none of which the nrl.com draw has ever emitted: it says
# "Finals Week 1/2/3". Every finals week except the decider therefore fell
# through to the regular-round copy below.
_STAGE_EVENTS = {
    rounds.FINALS_WEEK_1: {
        "event_name": "Finals Week 1",
        "opening_line": (
            "Finals footy is here. Four games, two of them sudden death, and no "
            "next round to fix a bad week in."
        ),
        "prompt_angle": (
            "Week one of the finals. Two qualifying finals with a double chance and "
            "two elimination finals where the loser's season ends on the night. "
            "Write it as the shift from a long season into knockout footy."
        ),
    },
    rounds.FINALS_WEEK_2: {
        "event_name": "Semi Finals",
        "opening_line": (
            "Semi-final weekend. Two games, four teams, and two of them are done "
            "by Sunday night."
        ),
        "prompt_angle": (
            "Semi-final weekend: pure sudden death, with the two qualifying-final "
            "winners resting up and watching. Desperation is the theme."
        ),
    },
    rounds.PRELIMINARY: {
        "event_name": "Preliminary Finals",
        "opening_line": (
            "Preliminary finals. Win and you are playing for the premiership; lose "
            "and you spend summer thinking about it."
        ),
        "prompt_angle": (
            "Preliminary finals: the cruellest weekend of the year, one game short "
            "of the grand final. Reputations made and seasons buried."
        ),
    },
    rounds.GRAND_FINAL: {
        "event_name": "Grand Final",
        "opening_line": (
            "It's the Grand Final. One game, one trophy, and a whole season "
            "reduced to eighty minutes."
        ),
        "prompt_angle": (
            "The premiership decider. One game, the biggest of the year. Go big on "
            "the occasion and the two clubs; no generic weekly intro."
        ),
    },
}


def _special_event_context(round_name, competition_year, stage=None):
    stage = stage or rounds.round_stage(round_name)
    if stage in _STAGE_EVENTS:
        return dict(_STAGE_EVENTS[stage])
    if re.search(r"\bround\s*1\b", str(round_name).strip().lower()):
        return {
            "event_name": "Round 1",
            "opening_line": f"It's Round 1, the season opener for {competition_year}, so optimism is irrationally high.",
            "prompt_angle": "Treat as season opener energy: fresh starts, overreactions, and new-year storylines.",
        }
    return {
        "event_name": "Regular Round",
        "opening_line": "Another week, another chance to make objectively questionable tipping decisions.",
        "prompt_angle": "Treat as a regular season round with concise but lively banter.",
    }


def _build_prompt_input(predictions, tipper_picks, joker_recommendation=None):
    fixture_lines = []
    for _, row in predictions.iterrows():
        winner = _prediction_winner(row)
        home_prob = two_way_home_probability(row["home_team_win_prob"], row["home_team_lose_prob"])
        away_prob = None if pd.isna(home_prob) else 1.0 - home_prob
        fixture_lines.append(
            f"- {row['team_home']} vs {row['team_away']}: tip {winner} "
            f"(home win {_format_probability(home_prob)}, "
            f"away win {_format_probability(away_prob)}, "
            f"score tip {_format_predicted_score_numbers(row)}, "
            f"margin {_format_predicted_margin(row)}, "
            f"market {row['team_home']} "
            f"{_format_market_price(row['team_head_to_head_odds_home'], row.get('market_odds_fresh', True))}, "
            f"{row['team_away']} "
            f"{_format_market_price(row['team_head_to_head_odds_away'], row.get('market_odds_fresh', True))})"
        )

    pick_lines = []
    if tipper_picks.empty:
        pick_lines.append("- None flagged by the model.")
    else:
        for _, row in tipper_picks.iterrows():
            pick_lines.append(
                f"- {row['team']} vs {row['opponent']}: market {_format_price(row['price'])}, "
                f"fair {_format_price(row['price_min'])}, edge {_format_percent(row['edge'])}, "
                f"stake share {_format_percent(row['stake_fraction'])}"
            )

    return "\n".join(fixture_lines), "\n".join(pick_lines), _joker_prompt_block(joker_recommendation)


def _sanitize_json_newlines(text):
    """Replace literal newlines inside JSON string values with escaped \\n."""
    result = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\":
            result.append(ch)
            escape_next = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            pass  # strip CR
        else:
            result.append(ch)
    return "".join(result)


def _parse_json_object(text):
    if not text:
        return None
    # Strip markdown code fences if present
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        stripped = stripped[start:end + 1]

    sanitized = _sanitize_json_newlines(stripped)
    candidates = [stripped, sanitized]
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return None


def _generate_claude_copy(predictions, tipper_picks, api_key, folder_url, temperature, joker_recommendation=None, news_context=None, scoreboard_line=None, finals=None):
    if predictions.empty:
        return None
    if not api_key:
        print("ANTHROPIC_API_KEY is not configured. Using fallback email content.")
        return None
    if Anthropic is None:
        print("Anthropic SDK is unavailable. Using fallback email content.")
        return None

    fixtures_text, picks_text, joker_text = _build_prompt_input(
        predictions,
        tipper_picks,
        joker_recommendation=joker_recommendation,
    )
    round_name = predictions['round_name'].iloc[0]
    competition_year = predictions['competition_year'].iloc[0]
    is_finals = bool(isinstance(finals, dict) and finals.get("is_finals"))
    special_event_context = _special_event_context(
        round_name, competition_year, stage=(finals or {}).get("stage")
    )
    finals_block = _finals_prompt_block(finals)
    folder_line = folder_url if folder_url else "No public folder URL is configured this run."
    market_notice = _market_coverage_notice(predictions)
    prompt = f"""
Write Reg Reagan's weekly NRL tipping email. Reg is loud, passionate, and deeply invested — he doesn't hedge, he doesn't whisper, and he definitely doesn't forgive bad footy. Write like he's been awake since 5am thinking about this round.

Round: {round_name} {competition_year}
Special event context: {special_event_context['event_name']}
Special event writing angle: {special_event_context['prompt_angle']}
Tips folder: {folder_line}

Fixtures and model picks:
{fixtures_text}

Value picks:
{picks_text}

Market data status:
{market_notice if market_notice else "Every fixture has valid paired H2H prices."}

Joker recommendation:
{"Not applicable. This is a finals special edition: the tipping comp finished with the regular season, so there is no joker and no comp strategy." if is_finals else joker_text}

Finals context (only present during the finals):
{finals_block if finals_block else "Not a finals week."}

Season scoreboard (the model's real tipping record so far — brag or cop it on the chin as appropriate):
{scoreboard_line if scoreboard_line else "No completed rounds yet this season."}

Current NRL news this week (use if something is funny or worth a dig — otherwise ignore):
{news_context if news_context else "Nothing notable found this week."}

Return JSON only with this exact schema:
{{
  "subject": "short email subject line, max 75 chars",
  "news_hit": "1 punchy paragraph where Reg calls out the biggest scandal or story from the news this week — opinionated, direct, sets the tone before the tips. If news is provided above, you MUST write this. Only use null if no news was provided.",
  "opening": "{"3-4 paragraphs — this is a finals special edition, so go bigger than a normal week: the occasion, what each game is worth, and who Reg thinks is actually winning the premiership" if is_finals else "2-3 paragraphs — Reg's take on the round with some personality and genuine opinions on the key games"}",
  "closing": "1-2 short paragraphs. Must end with: Bring back the biff."
}}

Rules:
- If news is provided in "Current NRL news", you MUST write news_hit — do not bury it in the opening and do not set it to null.
- Reg is a one-eyed Newcastle Knights and NSW fan — mention them positively.
- Reg's fictional backstory is that he is secretly Andrew "Joey" Johns' brother. Joey was the 8th Immortal and is widely considered one of the best to ever play rugby league. Reg loves him at heart, but gives him a hard time for fun with a bit of genuine needle, often calling him "barge arse".
- Reg hates QLD and Manly — take digs at both.
- In international footy, Reg backs Australia but has genuine love for minor nations and their underdog stories.
- Reg absolutely despises England and Great Britain — any reference should be dismissive.
- Include this disclaimer naturally: if people are in tipping comps at Seven Seas Hotel in Carrington or the Hunter Water work comp, they should not use these tips.
{"- The regular season tipping comp is OVER. Do not mention the joker, the comp ladder, points gaps, or a comp strategy. This is a bonus special edition played for pride. Say so once, early, and then talk about the football." if is_finals else '- Include one explicit sentence that starts with "Joker call:" and states PLAY or HOLD for this round.'}
{"- Name the premiership favourite and the percentage from the finals context, and be clear those numbers came from the model, not a hunch." if is_finals else "- Mention the season scoreboard once so readers know how the model has actually been going."}
- Never describe a market edge, price, or betting value for a fixture whose odds are unavailable.
- Keep it punchy and readable — a touch of colour, not a wall of slang.
- Output raw JSON only. No markdown fences, no preamble, no text before {{ or after }}.
- Do not include markdown, HTML, or extra keys.
"""

    client = Anthropic(api_key=api_key)
    configured_model = os.getenv("CLAUDE_MODEL")
    model_candidates = (
        [configured_model]
        if configured_model
        else [DEFAULT_CLAUDE_MODEL]
    )
    last_exception = None

    for model_name in model_candidates:
        if not model_name:
            continue
        try:
            response = client.messages.create(
                model=model_name,
                system="You are Reg Reagan — an opinionated Australian NRL tragic who writes weekly tipping emails. You're a one-eyed Newcastle Knights and NSW fan. In your fictional backstory, you're secretly Andrew \"Joey\" Johns' brother: you love him at heart, but you also love giving him a hard time for fun with a bit of genuine needle, often calling him \"barge arse\". Joey was the 8th Immortal and is widely considered one of the best to ever play rugby league. You hate QLD and Manly with a passion. You back Australia in internationals but have genuine love for minor nations' underdog stories — and you absolutely despise England and Great Britain. You're enthusiastic and direct, use occasional Australian slang, and have genuine strong opinions on footy. You're entertaining but not over the top — think passionate pub regular, not raving lunatic.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=temperature,
            )
            if getattr(response, "stop_reason", None) == "max_tokens":
                print(
                    f"Claude email generation hit the max_tokens cap for model '{model_name}'; "
                    "JSON is likely truncated."
                )
            raw_text = response.content[0].text or ""
            payload = _parse_json_object(raw_text)
            if not payload:
                print(f"Claude email generation returned non-JSON payload for model '{model_name}'.")
                continue
            subject = str(payload.get("subject", "")).strip()
            opening = str(payload.get("opening", "")).strip()
            closing = str(payload.get("closing", "")).strip()
            if not subject or not opening or not closing:
                print(f"Claude email generation returned incomplete JSON keys for model '{model_name}'.")
                continue
            news_hit_raw = payload.get("news_hit")
            news_hit = str(news_hit_raw).strip() if news_hit_raw and str(news_hit_raw).lower() != "null" else None
            print(f"Claude email generation model: {model_name}")
            return {
                "subject": subject,
                "news_hit": news_hit,
                "opening": opening,
                "closing": closing,
            }
        except Exception as exc:
            last_exception = exc
            print(f"Claude email generation failed for model '{model_name}' ({exc}).")
            if configured_model:
                break

    if last_exception is not None:
        print(f"Claude email generation failed ({last_exception}). Using fallback email content.")
    return None


def generate_reg_regan_email_payload(
    predictions,
    tipper_picks,
    api_key,
    folder_url,
    temperature,
    use_openai=True,
    joker_recommendation=None,
    openai_api_key=None,
    scoreboard=None,
    use_llm=None,
    comp_strategy=None,
    finals=None,
    context_cards=None,
    db_path=None,
):
    # `use_openai` is a deprecated alias for `use_llm` (the copy actually comes
    # from Claude; only the banner image uses OpenAI).
    use_llm = use_openai if use_llm is None else use_llm

    predictions = _sort_predictions_for_display(predictions)
    if context_cards is None and db_path is not None and not predictions.empty:
        try:
            from pipeline.common.club_context.product import load_context_cards

            context_cards = load_context_cards(
                db_path,
                predictions["game_id"].tolist() if "game_id" in predictions else None,
            )
        except Exception:
            context_cards = []
    try:
        from pipeline.common.club_context.product import normalize_context_cards

        context_cards = normalize_context_cards(context_cards)
    except Exception:
        context_cards = []
    sensitive_context = any(bool(card.get("sensitive")) for card in context_cards)
    is_finals = bool(isinstance(finals, dict) and finals.get("is_finals"))
    fallback_copy = _build_fallback_copy(
        predictions,
        folder_url,
        joker_recommendation=joker_recommendation,
        finals=finals,
    )
    news_context = None
    if use_llm and api_key and Anthropic is not None:
        news_context = _fetch_nrl_news_context(Anthropic(api_key=api_key))

    if not use_llm:
        print("Claude generation disabled. Using fallback email content.")
    llm_copy = (
        _generate_claude_copy(
            predictions,
            tipper_picks,
            api_key,
            folder_url,
            temperature,
            joker_recommendation=joker_recommendation,
            news_context=news_context,
            scoreboard_line=scoreboard_summary_line(scoreboard),
            finals=finals,
        )
        if use_llm
        else None
    )
    copy = llm_copy or fallback_copy

    # Strategy corner: surfaced in both plain and HTML closings so every
    # deviation from pure model tips is visible with its P(win comp) math.
    # Suppressed in finals: there is no comp left to optimise against.
    if not is_finals and isinstance(comp_strategy, dict) and comp_strategy.get("available"):
        note = f"{comp_strategy.get('headline', '').strip()}. {comp_strategy.get('detail', '').strip()}"
        copy["closing"] = f"{copy['closing']}\n\n{note}" if copy.get("closing") else note

    if predictions.empty:
        plain = copy["opening"]
        html_email = (
            "<html><body style=\"font-family:Arial,sans-serif; background:#eef2f7; padding:20px;\">"
            "<div style=\"max-width:680px; margin:0 auto; background:#fff; border-radius:12px; padding:24px;\">"
            f"<p style=\"margin:0; color:#111827; font-size:16px; line-height:1.5;\">{html.escape(copy['opening'])}</p>"
            "</div></body></html>"
        )
        return {
            "subject": copy["subject"],
            "plain_text": plain,
            "html_text": html_email,
            "inline_images": [],
        }

    news_hit = copy.get("news_hit")
    # A sensitive fact may appear in Context Watch but can never seed banner
    # imagery.  The context cards are not in the banner prompt, and this guard
    # also prevents accidental thematic overlap with legacy editorial news.
    generated_banner = None
    if not sensitive_context:
        generated_banner = _generate_dynamic_banner(
            copy,
            api_key,
            openai_api_key,
            news_context=news_context,
            news_hit=news_hit,
            finals=finals,
        )
    banner_path = generated_banner or _resolve_banner_path()
    plain_email = _render_plain_email(
        predictions,
        tipper_picks,
        folder_url,
        copy["subject"],
        copy["opening"],
        copy["closing"],
        joker_recommendation=joker_recommendation,
        news_hit=news_hit,
        scoreboard=scoreboard,
        finals=finals,
        context_cards=context_cards,
    )
    html_email = _render_html_email(
        predictions,
        tipper_picks,
        folder_url,
        copy["opening"],
        copy["closing"],
        banner_available=bool(banner_path),
        joker_recommendation=joker_recommendation,
        news_hit=news_hit,
        scoreboard=scoreboard,
        finals=finals,
        context_cards=context_cards,
    )

    inline_images = []
    if banner_path:
        inline_images.append({"cid": "footy_tipper_email_banner", "path": banner_path})

    return {
        "subject": copy["subject"],
        "plain_text": plain_email,
        "html_text": html_email,
        "inline_images": inline_images,
    }


# Backward-compatible wrapper: returns plain text body only.
def generate_reg_regan_email(
    predictions,
    tipper_picks,
    api_key,
    folder_url,
    temperature,
    joker_recommendation=None,
):
    payload = generate_reg_regan_email_payload(
        predictions,
        tipper_picks,
        api_key,
        folder_url,
        temperature,
        use_openai=bool(api_key),
        joker_recommendation=joker_recommendation,
    )
    return payload["plain_text"]
