"""
Sentinel configuration loader — reads YAML config and environment variables.

This module provides a typed Settings dataclass and loader function for the
Scout pillar. It centralizes all configuration (API keys, scraper endpoints,
retry logic, vector DB paths, etc.) into a single source of truth, with
environment variable overrides for production deployments.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ScoutConfig:
    """Configuration for Scout data ingestion (prices, news, filings, sentiment)."""

    yfinance_enabled: bool = True
    stooq_fallback_enabled: bool = True
    news_api_key: Optional[str] = None
    news_sources: list[str] = field(
        default_factory=lambda: [
            "bbc-news",
            "bloomberg",
            "cnbc",
            "financial-times",
        ]
    )
    reddit_enabled: bool = False
    reddit_subreddits: list[str] = field(
        default_factory=lambda: ["stocks", "investing", "wallstreetbets"]
    )
    sec_edgar_enabled: bool = True
    sec_filing_types: list[str] = field(default_factory=lambda: ["8-K", "10-Q", "10-K"])
    github_token: Optional[str] = None
    request_timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_seconds: int = 2


@dataclass
class LinguistConfig:
    """Configuration for Linguist reasoning (LLM analysis, certainty scoring)."""

    anthropic_api_key: Optional[str] = None
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 2000
    temperature: float = 0.7
    certainty_threshold: float = 0.6
    drift_detection_enabled: bool = True
    regulatory_whisper_enabled: bool = True


@dataclass
class HistorianConfig:
    """Configuration for Historian RAG (vector DB, embeddings, historical lookups)."""

    gemini_api_key: Optional[str] = None
    chroma_db_path: str = ".sentinel_chroma"
    embedding_model: str = "gemini-embedding-001"
    max_search_results: int = 5
    similarity_threshold: float = 0.7
    historical_lookback_days: int = 365


@dataclass
class JudgeConfig:
    """Configuration for Judge post-mortem and prediction calibration."""

    anthropic_api_key: Optional[str] = None
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1500
    temperature: float = 0.5
    discord_webhook_url: Optional[str] = None
    sqlite_db_path: str = ".sentinel_predictions.db"
    baseline_strategies: list[str] = field(
        default_factory=lambda: ["momentum", "mean_reversion", "volatility"]
    )


@dataclass
class Settings:
    """Root configuration object for the entire Sentinel system."""

    scout: ScoutConfig = field(default_factory=ScoutConfig)
    linguist: LinguistConfig = field(default_factory=LinguistConfig)
    historian: HistorianConfig = field(default_factory=HistorianConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    tickers: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    log_level: str = "INFO"
    debug: bool = False


def load_config(config_path: Optional[str] = None) -> Settings:
    """Load configuration from YAML file and environment variables, with env overrides.

    Args:
        config_path: Path to YAML config file. If None, looks for ./sentinel.yaml.

    Returns:
        Fully resolved Settings dataclass with env var overrides applied.
    """
    # Determine config file path
    if config_path is None:
        config_path = os.getenv("SENTINEL_CONFIG_PATH", "./sentinel.yaml")

    settings = Settings()

    # Load YAML if it exists
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}

        # Merge YAML into dataclass fields
        if "scout" in yaml_data:
            for key, val in yaml_data["scout"].items():
                if hasattr(settings.scout, key):
                    setattr(settings.scout, key, val)

        if "linguist" in yaml_data:
            for key, val in yaml_data["linguist"].items():
                if hasattr(settings.linguist, key):
                    setattr(settings.linguist, key, val)

        if "historian" in yaml_data:
            for key, val in yaml_data["historian"].items():
                if hasattr(settings.historian, key):
                    setattr(settings.historian, key, val)

        if "judge" in yaml_data:
            for key, val in yaml_data["judge"].items():
                if hasattr(settings.judge, key):
                    setattr(settings.judge, key, val)

        if "tickers" in yaml_data:
            settings.tickers = yaml_data["tickers"]

        if "log_level" in yaml_data:
            settings.log_level = yaml_data["log_level"]

        if "debug" in yaml_data:
            settings.debug = yaml_data["debug"]

    # Environment variable overrides (highest priority)
    settings.scout.yfinance_enabled = _parse_bool(
        os.getenv("SCOUT_YFINANCE_ENABLED", str(settings.scout.yfinance_enabled))
    )
    settings.scout.stooq_fallback_enabled = _parse_bool(
        os.getenv("SCOUT_STOOQ_FALLBACK_ENABLED", str(settings.scout.stooq_fallback_enabled))
    )
    settings.scout.news_api_key = os.getenv("SCOUT_NEWS_API_KEY", settings.scout.news_api_key)
    settings.scout.reddit_enabled = _parse_bool(
        os.getenv("SCOUT_REDDIT_ENABLED", str(settings.scout.reddit_enabled))
    )
    settings.scout.sec_edgar_enabled = _parse_bool(
        os.getenv("SCOUT_SEC_EDGAR_ENABLED", str(settings.scout.sec_edgar_enabled))
    )
    settings.scout.github_token = os.getenv("SCOUT_GITHUB_TOKEN", settings.scout.github_token)
    settings.scout.request_timeout_seconds = int(
        os.getenv("SCOUT_REQUEST_TIMEOUT_SECONDS", settings.scout.request_timeout_seconds)
    )
    settings.scout.max_retries = int(os.getenv("SCOUT_MAX_RETRIES", settings.scout.max_retries))

    settings.linguist.anthropic_api_key = os.getenv(
        "ANTHROPIC_API_KEY", settings.linguist.anthropic_api_key
    )
    settings.linguist.temperature = float(
        os.getenv("LINGUIST_TEMPERATURE", settings.linguist.temperature)
    )
    settings.linguist.certainty_threshold = float(
        os.getenv("LINGUIST_CERTAINTY_THRESHOLD", settings.linguist.certainty_threshold)
    )
    settings.linguist.drift_detection_enabled = _parse_bool(
        os.getenv("LINGUIST_DRIFT_DETECTION_ENABLED", str(settings.linguist.drift_detection_enabled))
    )

    settings.historian.gemini_api_key = os.getenv(
        "GEMINI_API_KEY", settings.historian.gemini_api_key
    )
    settings.historian.chroma_db_path = os.getenv(
        "HISTORIAN_CHROMA_DB_PATH", settings.historian.chroma_db_path
    )
    settings.historian.max_search_results = int(
