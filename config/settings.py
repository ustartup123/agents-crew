"""
config/settings.py — Central configuration loaded from environment variables.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# ── Gemini ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GeminiConfig:
    api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-3-flash-preview"))
    temperature: float = 0.7


# ── Slack ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SlackConfig:
    bot_token: str = field(default_factory=lambda: _env("SLACK_BOT_TOKEN"))
    app_token: str = field(default_factory=lambda: _env("SLACK_APP_TOKEN"))
    signing_secret: str = field(default_factory=lambda: _env("SLACK_SIGNING_SECRET"))

    # Channels
    channel_standup: str = field(default_factory=lambda: _env("SLACK_CHANNEL_STANDUP"))
    channel_general: str = field(default_factory=lambda: _env("SLACK_CHANNEL_GENERAL"))
    channel_engineering: str = field(default_factory=lambda: _env("SLACK_CHANNEL_ENGINEERING"))
    channel_business: str = field(default_factory=lambda: _env("SLACK_CHANNEL_BUSINESS"))
    channel_executive: str = field(default_factory=lambda: _env("SLACK_CHANNEL_EXECUTIVE"))


# ── Notion ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NotionConfig:
    api_key: str = field(default_factory=lambda: _env("NOTION_API_KEY"))
    root_page_id: str = field(default_factory=lambda: _env("NOTION_ROOT_PAGE_ID"))


# ── E2B ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class E2BConfig:
    api_key: str = field(default_factory=lambda: _env("E2B_API_KEY"))


# ── GitHub ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GitHubConfig:
    token: str = field(default_factory=lambda: _env("GITHUB_TOKEN"))
    username: str = field(default_factory=lambda: _env("GITHUB_USERNAME"))
    default_private: bool = True  # all repos private by default


# ── App ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AppConfig:
    startup_name: str = field(default_factory=lambda: _env("STARTUP_NAME", "MyStartup"))
    startup_idea: str = field(default_factory=lambda: _env("STARTUP_IDEA", ""))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))


# ── Singleton instances ───────────────────────────────────────────────────────

gemini_cfg = GeminiConfig()
slack_cfg = SlackConfig()
notion_cfg = NotionConfig()
e2b_cfg = E2BConfig()
github_cfg = GitHubConfig()
app_cfg = AppConfig()
