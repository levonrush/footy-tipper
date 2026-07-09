"""Shared LLM configuration for email copy, news, and banner generation."""

import os

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"


def resolve_claude_model():
    """Model for Claude calls: CLAUDE_MODEL env override, else the default."""
    configured = os.getenv("CLAUDE_MODEL", "").strip()
    return configured or DEFAULT_CLAUDE_MODEL
