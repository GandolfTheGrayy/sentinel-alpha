"""
Sentinel configuration loader: reads YAML config files and environment variables
into a typed Settings dataclass for use across all Scout data ingestion modules.

This module provides the single source of truth for Sentinel's runtime configuration,
including API keys, data source URLs, scraping parameters, and output paths.
Integrates with environment variables for secure credential management.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ScoutSettings:
    """Typed configuration container for Scout data ingestion module."""

    # API Keys (from environment or config)
    anthropic_api_key: str
    gemini_api_key: str
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: Optional[str] = None
    newsapi_key: Optional[str] = None
    discord_webhook_url: Optional[str] = None

    # Data source URLs
    sec_edgar_base_url: str = "https://www.sec.gov/cgi-bin/browse-edgar"
    news_sources: list[str] = field(
        default_factory=lambda: [
            "financial-times",
            "bloomberg",
            "cnbc",
            "reuters",
            "wall-street-journal",
        ]
    )

    # Scraping parameters
    request_timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 1.5
    user_agent: str = "Sentinel/1.0 (Financial Intelligence System)"
    rate_limit_requests_per_second: float = 1.0

    # Data storage
    data_dir: Path = field(default_factory=lambda: Path("./sentinel_data"))
    cache_dir: Path = field(default_factory=lambda: Path("./sentinel_data/cache"))
    chromadb_path: Path = field(
        default_factory=lambda: Path("./sentinel_data/chromadb")
    )

    # SEC filing parameters
    sec_filing_types: list[str] = field(default_factory=lambda: ["8-K", "10-Q", "10-K"])
    sec_lookback_days: int = 30

    # Reddit sentiment scraping
    reddit_subreddits: list[str] = field(
        default_factory=lambda: ["stocks", "investing", "wallstreetbets"]
    )
    reddit_post_limit: int = 100

    # Logging and debug
    debug: bool = False
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate and ensure all required directories exist."""
        self.data_dir = Path(self.data_dir)
        self.cache_dir = Path(self.cache_dir)
        self.chromadb_path = Path(self.chromadb_path)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chromadb_path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> ScoutSettings:
    """Load Sentinel configuration from YAML file and environment variables.

    Merges YAML config with environment variable overrides. Environment variables
    take precedence. Raises FileNotFoundError if config_path is specified but
    does not exist. Raises ValueError if required API keys are missing.

    Args:
        config_path: Path to YAML config file. If None, uses ./sentinel_config.yaml
                     or env var SENTINEL_CONFIG_PATH.

    Returns:
        Populated ScoutSettings dataclass instance.

    Raises:
        FileNotFoundError: If specified config file does not exist.
        ValueError: If required API keys (anthropic_api_key, gemini_api_key) are missing.
    """
    # Determine config file path
    if config_path is None:
        config_path = os.environ.get("SENTINEL_CONFIG_PATH", "./sentinel_config.yaml")

    config_path_obj = Path(config_path)

    config_dict: dict = {}

    # Load from YAML if file exists
    if config_path_obj.exists():
        with open(config_path_obj, "r") as f:
            yaml_content = yaml.safe_load(f)
            if yaml_content:
                config_dict.update(yaml_content)
    elif str(config_path) != "./sentinel_config.yaml":
        # Only raise error if a non-default path was explicitly specified
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Override with environment variables
    if os.environ.get("ANTHROPIC_API_KEY"):
        config_dict["anthropic_api_key"] = os.environ["ANTHROPIC_API_KEY"]

    if os.environ.get("GEMINI_API_KEY"):
        config_dict["gemini_api_key"] = os.environ["GEMINI_API_KEY"]

    if os.environ.get("REDDIT_CLIENT_ID"):
        config_dict["reddit_client_id"] = os.environ["REDDIT_CLIENT_ID"]

    if os.environ.get("REDDIT_CLIENT_SECRET"):
        config_dict["reddit_client_secret"] = os.environ["REDDIT_CLIENT_SECRET"]

    if os.environ.get("REDDIT_USER_AGENT"):
        config_dict["reddit_user_agent"] = os.environ["REDDIT_USER_AGENT"]

    if os.environ.get("NEWSAPI_KEY"):
        config_dict["newsapi_key"] = os.environ["NEWSAPI_KEY"]

    if os.environ.get("DISCORD_WEBHOOK_URL"):
        config_dict["discord_webhook_url"] = os.environ["DISCORD_WEBHOOK_URL"]

    if os.environ.get("SENTINEL_DEBUG"):
        config_dict["debug"] = os.environ["SENTINEL_DEBUG"].lower() == "true"

    if os.environ.get("SENTINEL_LOG_LEVEL"):
        config_dict["log_level"] = os.environ["SENTINEL_LOG_LEVEL"]

    if os.environ.get("SENTINEL_DATA_DIR"):
        config_dict["data_dir"] = os.environ["SENTINEL_DATA_DIR"]

    # Validate required keys
    if "anthropic_api_key" not in config_dict or not config_dict["anthropic_api_key"]:
        raise ValueError(
            "anthropic_api_key is required. Set via ANTHROPIC_API_KEY or config file."
        )

    if "gemini_api_key" not in config_dict or not config_dict["gemini_api_key"]:
        raise ValueError(
            "gemini_api_key is required. Set via GEMINI_API_KEY or config file."
        )

    return ScoutSettings(**config_dict)


def get_config(config_path: Optional[str] = None) -> ScoutSettings:
    """Cached config getter (convenience wrapper for load_config).

    Args:
        config_path: Path to YAML config file.

    Returns:
        ScoutSettings instance.
    """
    return load_config(config_path)
