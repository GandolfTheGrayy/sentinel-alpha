"""
Sentinel configuration loader — reads YAML config files and environment variables.

This module provides a typed Settings dataclass and loader functions for Sentinel's
configuration. It centralizes all runtime settings (API keys, model names, data paths,
thresholds) in a single source of truth, with environment variable overrides and
YAML-based defaults.

Part of sentinel.scout: Initializes shared config for all downstream pillars.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class Settings:
    """Typed configuration container for Sentinel Financial Intelligence Engine."""

    # API Keys & Auth
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    reddit_client_id: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_ID", ""))
    reddit_client_secret: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET", ""))
    discord_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))

    # Model Selection
    claude_model: str = "claude-sonnet-4-6"
    gemini_model: str = "gemini-3.1-flash-lite-preview"

    # Data Paths
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("SENTINEL_DATA_DIR", "./data")))
    chromadb_path: Path = field(default_factory=lambda: Path(os.getenv("SENTINEL_CHROMADB_PATH", "./data/chromadb")))
    sqlite_db_path: Path = field(default_factory=lambda: Path(os.getenv("SENTINEL_DB_PATH", "./data/sentinel.db")))

    # Prediction & Scoring Thresholds
    certainty_threshold: float = 0.65
    momentum_weight: float = 0.3
    sentiment_weight: float = 0.4
    regulatory_weight: float = 0.3
    min_confidence_for_alert: float = 0.75

    # Scraping & Rate Limiting
    request_timeout_sec: int = 10
    sec_filings_check_interval_hours: int = 6
    news_fetch_interval_minutes: int = 30
    max_retries: int = 3
    retry_backoff_sec: float = 2.0

    # Ticker Watchlist
    tickers: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"])

    # Feature Flags
    enable_discord_notifications: bool = True
    enable_reddit_sentiment: bool = True
    enable_sec_filing_analysis: bool = True
    debug_mode: bool = False


def load_settings(config_file: Optional[str] = None) -> Settings:
    """Load Settings from YAML config file and environment variable overrides."""
    settings = Settings()

    if config_file and Path(config_file).exists():
        with open(config_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        _apply_yaml_to_settings(settings, yaml_data)

    _apply_env_overrides(settings)
    return settings


def _apply_yaml_to_settings(settings: Settings, yaml_data: Dict[str, Any]) -> None:
    """Merge YAML configuration dict into Settings dataclass instance."""
    for key, value in yaml_data.items():
        if hasattr(settings, key):
            field_type = Settings.__annotations__.get(key)
            if field_type == Path or (hasattr(field_type, "__origin__") and field_type.__origin__ is type(Path)):
                setattr(settings, key, Path(value) if isinstance(value, str) else value)
            elif field_type == list or (hasattr(field_type, "__origin__") and field_type.__origin__ is list):
                setattr(settings, key, value if isinstance(value, list) else [value])
            else:
                setattr(settings, key, value)


def _apply_env_overrides(settings: Settings) -> None:
    """Apply environment variable overrides to Settings instance (highest precedence)."""
    env_mapping = {
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "GEMINI_API_KEY": "gemini_api_key",
        "REDDIT_CLIENT_ID": "reddit_client_id",
        "REDDIT_CLIENT_SECRET": "reddit_client_secret",
        "DISCORD_WEBHOOK_URL": "discord_webhook_url",
        "SENTINEL_DATA_DIR": "data_dir",
        "SENTINEL_CHROMADB_PATH": "chromadb_path",
        "SENTINEL_DB_PATH": "sqlite_db_path",
        "SENTINEL_CLAUDE_MODEL": "claude_model",
        "SENTINEL_GEMINI_MODEL": "gemini_model",
        "SENTINEL_CERTAINTY_THRESHOLD": "certainty_threshold",
        "SENTINEL_MIN_CONFIDENCE_ALERT": "min_confidence_for_alert",
        "SENTINEL_DEBUG_MODE": "debug_mode",
    }

    for env_key, attr_name in env_mapping.items():
        env_value = os.getenv(env_key)
        if env_value is not None:
            field_type = Settings.__annotations__.get(attr_name)
            if field_type == Path or (hasattr(field_type, "__origin__") and field_type.__origin__ is type(Path)):
                setattr(settings, attr_name, Path(env_value))
            elif field_type == bool:
                setattr(settings, attr_name, env_value.lower() in ("true", "1", "yes"))
            elif field_type == float:
                setattr(settings, attr_name, float(env_value))
            elif field_type == int:
                setattr(settings, attr_name, int(env_value))
            elif field_type == list or (hasattr(field_type, "__origin__") and field_type.__origin__ is list):
                setattr(settings, attr_name, env_value.split(","))
            else:
                setattr(settings, attr_name, env_value)


def settings_to_dict(settings: Settings) -> Dict[str, Any]:
    """Convert Settings instance to dict, serializing Path objects."""
    data = asdict(settings)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def get_default_config_path() -> Path:
    """Return the default config file path (project root or current dir)."""
    candidates = [
        Path("sentinel.yaml"),
        Path("sentinel.yml"),
        Path(".sentinel/config.yaml"),
        Path("config/sentinel.yaml"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("sentinel.yaml")
