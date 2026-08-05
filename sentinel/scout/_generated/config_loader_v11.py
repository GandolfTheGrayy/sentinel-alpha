"""
Configuration loader for Sentinel Sentiment Engine.

Reads YAML config files and environment variables, providing a typed Settings
dataclass for centralized access to Scout module parameters (API keys, scraper
timeouts, data sources, etc.). Used by all Scout modules during initialization
to configure data ingestion behavior.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


@dataclass
class ScoutSettings:
    """Typed settings container for Scout data ingestion configuration."""

    # API Keys & Credentials
    anthropic_api_key: str
    gemini_api_key: str
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: Optional[str] = None

    # Scraping & Timeouts
    scraper_timeout_seconds: int = 30
    scraper_retries: int = 3
    scraper_backoff_factor: float = 1.5

    # Data Sources
    price_source: str = "yfinance"
    fallback_price_source: str = "stooq"
    news_sources: list[str] = field(
        default_factory=lambda: ["cnbc", "reuters", "bloomberg"]
    )
    sec_filing_types: list[str] = field(default_factory=lambda: ["8-K", "10-Q", "10-K"])

    # Rate Limiting
    reddit_requests_per_second: float = 1.0
    sec_requests_per_second: float = 0.5

    # Storage & Caching
    cache_dir: str = "./cache"
    db_path: str = "./sentinel.db"

    # Feature Flags
    enable_reddit_scraping: bool = True
    enable_github_signals: bool = True
    enable_sec_scraping: bool = True
    enable_news_scraping: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None


def load_config(config_path: Optional[str] = None) -> ScoutSettings:
    """
    Load configuration from YAML file and environment variables.

    Precedence: environment variables override YAML values, which override
    dataclass defaults. If config_path is None, searches for 'sentinel.yaml'
    or 'config.yaml' in current directory and parent directories.

    Args:
        config_path: Optional path to YAML config file.

    Returns:
        ScoutSettings instance with merged configuration.

    Raises:
        FileNotFoundError: If config_path is specified but file not found.
        yaml.YAMLError: If YAML parsing fails.
        ValueError: If required API keys are missing.
    """
    yaml_config: Dict[str, Any] = {}

    # Attempt to find and load YAML config file
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    else:
        # Search for default config files
        for candidate in ["sentinel.yaml", "config.yaml"]:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                config_file = candidate_path
                break
        else:
            config_file = None

    if config_file:
        with open(config_file, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

    # Extract typed fields from YAML or env, with env taking precedence
    anthropic_api_key = os.getenv(
        "ANTHROPIC_API_KEY", yaml_config.get("anthropic_api_key", "")
    )
    gemini_api_key = os.getenv(
        "GEMINI_API_KEY", yaml_config.get("gemini_api_key", "")
    )

    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in env or config")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not set in env or config")

    # Build settings dict from YAML, then overlay env vars
    settings_dict: Dict[str, Any] = yaml_config.get("scout", {}).copy()
    settings_dict["anthropic_api_key"] = anthropic_api_key
    settings_dict["gemini_api_key"] = gemini_api_key

    # Apply environment variable overrides (with explicit prefix checks)
    if os.getenv("SCOUT_SCRAPER_TIMEOUT"):
        settings_dict["scraper_timeout_seconds"] = int(
            os.getenv("SCOUT_SCRAPER_TIMEOUT")
        )
    if os.getenv("SCOUT_REDDIT_CLIENT_ID"):
        settings_dict["reddit_client_id"] = os.getenv("SCOUT_REDDIT_CLIENT_ID")
    if os.getenv("SCOUT_REDDIT_CLIENT_SECRET"):
        settings_dict["reddit_client_secret"] = os.getenv("SCOUT_REDDIT_CLIENT_SECRET")
    if os.getenv("SCOUT_REDDIT_USER_AGENT"):
        settings_dict["reddit_user_agent"] = os.getenv("SCOUT_REDDIT_USER_AGENT")
    if os.getenv("SCOUT_CACHE_DIR"):
        settings_dict["cache_dir"] = os.getenv("SCOUT_CACHE_DIR")
    if os.getenv("SCOUT_DB_PATH"):
        settings_dict["db_path"] = os.getenv("SCOUT_DB_PATH")
    if os.getenv("SCOUT_LOG_LEVEL"):
        settings_dict["log_level"] = os.getenv("SCOUT_LOG_LEVEL")
    if os.getenv("SCOUT_LOG_FILE"):
        settings_dict["log_file"] = os.getenv("SCOUT_LOG_FILE")

    # Instantiate and return typed settings
    return ScoutSettings(**settings_dict)


def validate_settings(settings: ScoutSettings) -> bool:
    """
    Validate that critical settings are present and sensible.

    Args:
        settings: ScoutSettings instance to validate.

    Returns:
        True if all critical checks pass.

    Raises:
        ValueError: If validation fails.
    """
    if not settings.anthropic_api_key or not settings.anthropic_api_key.strip():
        raise ValueError("anthropic_api_key must not be empty")
    if not settings.gemini_api_key or not settings.gemini_api_key.strip():
        raise ValueError("gemini_api_key must not be empty")
    if settings.scraper_timeout_seconds <= 0:
        raise ValueError("scraper_timeout_seconds must be positive")
    if settings.scraper_retries < 0:
        raise ValueError("scraper_retries must be non-negative")
    if settings.log_level.upper() not in [
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ]:
        raise ValueError(f"Invalid log_level: {settings.log_level}")
    return True
