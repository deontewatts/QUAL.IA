"""
QUALIA Configuration — Environment & Settings
==============================================
All runtime configuration lives here, pulled from environment variables.
Never hardcode secrets — use a .env file locally and Docker secrets or
a secrets manager (AWS Secrets Manager, Doppler, etc.) in production.

Copy this to .env.example and commit it. The actual .env stays gitignored.
"""

import os
from dataclasses import dataclass


@dataclass
class QUALIASettings:
    # ── LLM / AI ────────────────────────────────────────────────────────────
    OPENROUTER_API_KEY: str    = os.environ.get("OPENROUTER_API_KEY", "")
    QUALIA_MODEL: str          = os.environ.get("QUALIA_MODEL", "anthropic/claude-sonnet-4-5")
    # Alternative models: "openai/gpt-4o", "mistralai/mistral-large", "meta-llama/llama-3.1-70b"

    # ── Instagram / Meta ────────────────────────────────────────────────────
    IG_PAGE_ACCESS_TOKEN: str  = os.environ.get("IG_PAGE_ACCESS_TOKEN", "")
    IG_VERIFY_TOKEN: str       = os.environ.get("IG_VERIFY_TOKEN", "qualia_webhook_secret")
    IG_PAGE_ID: str            = os.environ.get("IG_PAGE_ID", "")

    # ── Database / Memory ───────────────────────────────────────────────────
    QUALIA_DB_DIR: str         = os.environ.get("QUALIA_DB_DIR", "/data/qualia_memory")
    REDIS_URL: str             = os.environ.get("REDIS_URL", "")  # optional, for state sharing

    # ── Server ──────────────────────────────────────────────────────────────
    PORT: int                  = int(os.environ.get("PORT", 8000))
    ENV: str                   = os.environ.get("ENV", "production")
    LOG_LEVEL: str             = os.environ.get("LOG_LEVEL", "INFO")
    ALLOWED_ORIGINS: str       = os.environ.get("ALLOWED_ORIGINS", "*")

    # ── Feature Flags ───────────────────────────────────────────────────────
    MOCK_DATA: bool            = os.environ.get("MOCK_DATA", "true").lower() == "true"
    ENABLE_MEMORY: bool        = os.environ.get("ENABLE_MEMORY", "true").lower() == "true"
    ENABLE_LANGCHAIN: bool     = os.environ.get("ENABLE_LANGCHAIN", "true").lower() == "true"

    # ── Monitoring ──────────────────────────────────────────────────────────
    POSTHOG_API_KEY: str       = os.environ.get("POSTHOG_API_KEY", "")
    SENTRY_DSN: str            = os.environ.get("SENTRY_DSN", "")


settings = QUALIASettings()


# .env.example content — commit this, not the real .env
ENV_EXAMPLE = """
# ── Copy to .env and fill in your values ──

# LLM (OpenRouter gives you access to Claude, GPT-4, Mistral, etc.)
OPENROUTER_API_KEY=sk-or-...
QUALIA_MODEL=anthropic/claude-sonnet-4-5

# Instagram / Meta for Developers
IG_PAGE_ACCESS_TOKEN=
IG_VERIFY_TOKEN=qualia_webhook_secret_change_me
IG_PAGE_ID=

# Memory storage (Docker volume path)
QUALIA_DB_DIR=/data/qualia_memory

# Server
PORT=8000
ENV=production
LOG_LEVEL=INFO

# Feature flags
MOCK_DATA=false        # Set true for dev/demo
ENABLE_MEMORY=true
ENABLE_LANGCHAIN=true

# Monitoring (optional)
POSTHOG_API_KEY=
SENTRY_DSN=
"""
