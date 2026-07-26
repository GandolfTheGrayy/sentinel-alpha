"""
Sentinel configuration loader: reads YAML config and environment variables.

This module provides a typed Settings dataclass and loader for the Sentinel
Sentiment Engine configuration. It bridges YAML configuration files with
environment variable overrides, ensuring all downstream modules have access
to consistent, validated settings for API keys, database paths, scraping
parameters, and model selections.

Part of sentinel/scout/ — ingests configuration as a foundational setup step
before data scrapers begin.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


@dataclass
class Settings:
    """Typed configuration container for Sentinel Sentiment Engine."""

    # API Keys (read from env or config)
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    discord_webhook_url: Optional[str] = None
    praw_client_id: Optional[str] = None
    praw_client_secret: Optional[str] = None
    praw_user_agent: Optional[str] = None

    # Model selections
    claude_model: str = "claude-sonnet-4-6"
    gemini_model: str = "gemini-3.1-flash-lite-preview"

    # Database and storage
    chromadb_path: str = "./data/chromadb"
    sqlite_db_path: str = "./data/sentinel.db"
    cache_dir: str = "./data/cache"

    # Scraping parameters
    sec_scraper_enabled: bool = True
    news_scraper_enabled: bool = True
    reddit_scraper_enabled: bool = True
    github_scraper_enabled: bool = False
    request_timeout_seconds: int = 30
    rate_limit_delay_seconds: float = 1.0

    # Prediction and analysis
    prediction_horizon_days: int = 5
    confidence_threshold: float = 0.65
    min_historical_samples: int = 10

    # Feature flags
    postmortem_enabled: bool = True
    paper_trading_enabled: bool = False
    verbose_logging: bool = False

    # Tickers to monitor (comma-separated or list)
    watch_tickers: str = "AAPL,GOOGL,MSFT,NVDA,TSLA"

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)


def load_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load Settings from YAML config file and environment variables.

    Environment variables override YAML values. Looks for config.yaml in
    current directory or specified path. API keys are always read from env.

    Args:
        config_path: Optional path to YAML config file. Defaults to ./config.yaml

    Returns:
        Fully initialized Settings dataclass with all values populated.
    """
    # Determine config file path
    if config_path is None:
        config_path = "./config.yaml"
    config_path_obj = Path(config_path)

    # Load YAML if file exists
    yaml_config: Dict[str, Any] = {}
    if config_path_obj.exists():
        with open(config_path_obj, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}

    # Initialize Settings with YAML values
    settings_dict = yaml_config.get("sentinel", {})
    settings = Settings(**{k: v for k, v in settings_dict.items() if hasattr(Settings, k)})

    # Override with environment variables (these always take precedence)
    settings.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    settings.gemini_api_key = os.getenv("GEMINI_API_KEY", settings.gemini_api_key)
    settings.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", settings.discord_webhook_url)
    settings.praw_client_id = os.getenv("PRAW_CLIENT_ID", settings.praw_client_id)
    settings.praw_client_secret = os.getenv("PRAW_CLIENT_SECRET", settings.praw_client_secret)
    settings.praw_user_agent = os.getenv("PRAW_USER_AGENT", settings.praw_user_agent)

    # Create necessary directories
    Path(settings.chromadb_path).mkdir(parents=True, exist_ok=True)
    Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)

    return settings


def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Convenience function to load and cache Settings singleton.

    Args:
        config_path: Optional path to YAML config file.

    Returns:
        Cached Settings instance (creates on first call, returns same instance thereafter).
    """
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = load_settings(config_path)  # type: ignore
    return get_settings._instance  # type: ignore
