"""Shared LLM configuration for email copy, news, and banner generation."""

import os

DEFAULT_CLAUDE_MODEL = "claude-fable-5-1"
DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-2.5-sunburst"


def resolve_claude_model():
    """Model for Claude calls: CLAUDE_MODEL env override, else the default."""
    configured = os.getenv("CLAUDE_MODEL", "").strip()
    return configured or DEFAULT_CLAUDE_MODEL


def resolve_openai_image_model():
    """Image-edit model: OPENAI_MODEL env override, else the current default."""
    configured = os.getenv("OPENAI_MODEL", "").strip()
    return configured or DEFAULT_OPENAI_IMAGE_MODEL


def claude_generation_options(*, max_tokens, temperature=None):
    """Keep short editorial calls compatible with current and rollback models."""
    model = resolve_claude_model()
    options = {"model": model, "max_tokens": max_tokens}
    if model.startswith((
        "claude-fable-5", "claude-opus-5", "claude-sonnet-5",
        "claude-opus-4-7", "claude-opus-4-8",
    )):
        # These models think by default and reject custom sampling settings.
        # The output cap includes thinking; reserve room beyond the text budget.
        # Low effort suits these short writing tasks, which do not use tools.
        options["max_tokens"] += 4096
        options["output_config"] = {"effort": "low"}
    elif temperature is not None:
        options["temperature"] = temperature
    return options


def claude_response_text(response):
    """Read only complete text, never thinking, tool data or truncated output."""
    if getattr(response, "stop_reason", None) in {
        "max_tokens", "refusal", "model_context_window_exceeded",
    }:
        return ""
    return "".join(
        block.text for block in (getattr(response, "content", None) or [])
        if getattr(block, "type", None) == "text"
    ).strip()
