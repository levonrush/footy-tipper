"""Backward-compatible facade for the use_predictions package.

The implementation lives in focused modules:

- ``joker``         — joker policy, recommendation, and usage state
- ``staking``       — value-pick selection and Kelly staking
- ``distribution``  — prediction retrieval, Drive upload, SMTP sending
- ``email_render``  — plain-text and HTML email rendering
- ``email_copy``    — LLM/fallback copy and email payload assembly
- ``news``          — NRL news context fetching
- ``banner``        — banner resolution and dynamic banner generation

This module re-exports the public (and historically used private) names so
existing imports such as ``from ... import sending_functions as sf`` keep
working unchanged.
"""

from pipeline.common.use_predictions.banner import (  # noqa: F401
    _build_banner_edit_instruction,
    _generate_dynamic_banner,
    _resolve_banner_path,
)
from pipeline.common.use_predictions.distribution import (  # noqa: F401
    _attach_inline_images,
    _build_mime_message,
    _ensure_email_sends_table,
    _ensure_predictions_table_columns,
    _sort_predictions_for_display,
    backup_db_to_drive,
    email_send_already_recorded,
    get_predictions,
    record_email_send,
    send_emails,
    send_test_email,
    upload_df_to_drive,
)
from pipeline.common.use_predictions.email_copy import (  # noqa: F401
    _build_fallback_copy,
    _build_prompt_input,
    _generate_claude_copy,
    _parse_json_object,
    _sanitize_json_newlines,
    _special_event_context,
    generate_reg_regan_email,
    generate_reg_regan_email_payload,
)
from pipeline.common.use_predictions.email_render import (  # noqa: F401
    _coerce_int,
    _default_subject,
    _first_game_callout,
    _format_number,
    _format_percent,
    _format_predicted_margin,
    _format_predicted_score_numbers,
    _format_predicted_scoreline,
    _format_price,
    _format_probability,
    _joker_prompt_block,
    _joker_summary_lines,
    _prediction_winner,
    _render_html_email,
    _render_plain_email,
    _to_html_paragraphs,
)
from pipeline.common.use_predictions.joker import (  # noqa: F401
    _apply_joker_usage_state,
    _coerce_competition_year,
    _coerce_env_float,
    _coerce_env_int,
    _ensure_joker_usage_table,
    _infer_joker_competition_year,
    _joker_objective_meta,
    _load_joker_policy,
    _load_json_file,
    _resolve_joker_strategy,
    _resolve_joker_strategy_context,
    _resolve_joker_strategy_value,
    _round_label,
    _unavailable_joker_recommendation,
    compute_joker_round_metrics,
    get_joker_round_candidates,
    get_joker_round_recommendation,
    get_joker_usage_for_year,
    persist_joker_usage_if_applicable,
    recommend_joker_round,
)
from pipeline.common.use_predictions.news import (  # noqa: F401
    _fetch_nrl_news_context,
    _fetch_rss_headlines,
)
from pipeline.common.use_predictions.scoreboard import (  # noqa: F401
    get_season_scoreboard,
    scoreboard_summary_line,
)
from pipeline.common.use_predictions.staking import get_tipper_picks  # noqa: F401
