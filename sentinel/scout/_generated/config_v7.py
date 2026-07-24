"""
Sentinel configuration loader.

Reads YAML configuration and environment variables into a typed Settings
dataclass. Used by all Scout, Linguist, Historian, and Judge modules to
fetch API keys, database paths, scraper timeouts, and model selections.
Part of the scout pillar's data ingestion initialization layer.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Settings:
    """Typed configuration container for Sentinel system."""

    # API Keys (from environment)
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    discord_webhook_url: Optional[str] = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL"))
    reddit_client_id: Optional[str] = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_ID"))
    reddit_client_secret: Optional[str] = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET"))
    reddit_user_agent: Optional[str] = field(default_factory=lambda: os.getenv("REDDIT_USER_AGENT"))

    # Database and storage paths
    db_path: Path = field(default_factory=lambda: Path(os.getenv("SENTINEL_DB_PATH", "sentinel.db")))
    chroma_path: Path = field(default_factory=lambda: Path(os.getenv("SENTINEL_CHROMA_PATH", ".chroma")))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("SENTINEL_CACHE_DIR", ".cache")))

    # Scraper timeouts and limits
    scraper_timeout_sec: int = 30
    scraper_max_retries: int = 3
    scraper_backoff_sec: float = 1.0

    # Model selections
    reasoning_model: str = "claude-sonnet-4-6"
    scraping_model: str = "gemini-3.1-flash-lite-preview"

    # RAG and vector search
    embedding_batch_size: int = 100
    chroma_similarity_threshold: float = 0.5

    # Predictor and Judge parameters
    min_confidence_for_alert: float = 0.65
    prediction_horizon_days: int = 5

    # Feature flags
    enable_reddit_scraping: bool = True
    enable_sec_scraping: bool = True
    enable_news_scraping: bool = True
    enable_discord_notifications: bool = False

    def validate(self) -> None:
        """Validate required API keys and paths; raise ValueError if missing."""
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Load Settings from YAML config file and environment variables, environment taking precedence."""
    settings = Settings()

    if config_path is None:
        config_path = Path(os.getenv("SENTINEL_CONFIG", "sentinel.yaml"))

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        # Overlay YAML values onto defaults
        for key, value in config_dict.items():
            if hasattr(settings, key):
                field_type = Settings.__dataclass_fields__[key].type
                if field_type == Path or (hasattr(field_type, "__origin__") and field_type.__origin__ is Path):
                    value = Path(value) if value else settings.__getattribute__(key)
                setattr(settings, key, value)

    # Environment variables override YAML
    env_overrides = {
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "gemini_api_key": "GEMINI_API_KEY",
        "discord_webhook_url": "DISCORD_WEBHOOK_URL",
        "reddit_client_id": "REDDIT_CLIENT_ID",
        "reddit_client_secret": "REDDIT_CLIENT_SECRET",
        "reddit_user_agent": "REDDIT_USER_AGENT",
        "scraper_timeout_sec": "SENTINEL_SCRAPER_TIMEOUT_SEC",
        "scraper_max_retries": "SENTINEL_SCRAPER_MAX_RETRIES",
        "min_confidence_for_alert": "SENTINEL_MIN_CONFIDENCE_FOR_ALERT",
        "enable_discord_notifications": "SENTINEL_ENABLE_DISCORD_NOTIFICATIONS",
    }

    for field_name, env_var in env_overrides.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            field_type = Settings.__dataclass_fields__[field_name].type
            if field_type == int:
                setattr(settings, field_name, int(env_value))
            elif field_type == float:
                setattr(settings, field_name, float(env_value))
            elif field_type == bool or (hasattr(field_type, "__origin__") and "bool" in str(field_type)):
                setattr(settings, field_name, env_value.lower() in ("true", "1", "yes"))
            else:
                setattr(settings, field_name, env_value)

    return settings
