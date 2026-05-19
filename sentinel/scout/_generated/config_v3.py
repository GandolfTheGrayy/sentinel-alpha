"""
Config loader for Sentinel Sentiment Engine.

Reads YAML configuration files and environment variables, merging them into
a typed Settings dataclass. Used by Scout modules to access API keys, scraper
timeouts, data source URLs, and other runtime parameters without hardcoding.

Integrates with all Scout data ingestion pipelines (live prices, SEC filings,
Reddit sentiment, GitHub signals) to provide consistent, validated configuration.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class Settings:
    """Typed configuration container for Sentinel runtime parameters."""

    # API Keys
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    discord_webhook_url: str = ""
    praw_client_id: str = ""
    praw_client_secret: str = ""
    praw_user_agent: str = ""

    # Scout: Live prices
    live_prices_timeout: int = 10
    live_prices_retry_count: int = 3
    stooq_fallback_enabled: bool = True

    # Scout: News
    news_sources: list = field(default_factory=lambda: ["newsapi", "finnhub"])
    news_max_results: int = 50
    news_timeout: int = 15

    # Scout: SEC filings
    sec_filings_types: list = field(default_factory=lambda: ["8-K", "10-Q", "10-K"])
    sec_filings_lookback_days: int = 90
    sec_filings_batch_size: int = 5

    # Scout: Reddit sentiment
    reddit_subreddits: list = field(default_factory=lambda: ["stocks", "investing", "wallstreetbets"])
    reddit_post_limit: int = 100
    reddit_min_score: int = 10

    # Scout: GitHub signals
    github_token: str = ""
    github_lookback_days: int = 30

    # Linguist: Reasoning
    linguist_model: str = "claude-sonnet-4-6"
    linguist_certainty_threshold: float = 0.65
    linguist_drift_window_days: int = 30

    # Historian: RAG
    chromadb_path: str = "./data/chromadb"
    chromadb_collection_name: str = "sentinel_corpus"
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.5

    # Judge: Prediction
    judge_model: str = "claude-sonnet-4-6"
    judge_confidence_floor: float = 0.55
    judge_post_mortem_enabled: bool = True

    # Pipeline: General
    pipeline_ticker_list: list = field(default_factory=list)
    pipeline_batch_size: int = 10
    pipeline_log_level: str = "INFO"
    pipeline_dry_run: bool = False

    # Paths
    config_dir: str = "./config"
    data_dir: str = "./data"
    logs_dir: str = "./logs"

    def to_dict(self) -> Dict[str, Any]:
        """Convert Settings to dictionary for serialization."""
        return asdict(self)

    def to_env_dict(self) -> Dict[str, str]:
        """Convert Settings to flat string dict suitable for os.environ updates."""
        result = {}
        for key, value in self.to_dict().items():
            if isinstance(value, (list, dict)):
                result[key.upper()] = yaml.dump(value, default_flow_style=True).strip()
            else:
                result[key.upper()] = str(value)
        return result


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Load Settings from YAML config file, merge with environment variables.

    Environment variables take precedence. Expects config file at
    `config_path` (default: ./config/sentinel.yaml). Missing keys default
    to Settings() field defaults.

    Args:
        config_path: Path to YAML config file. Defaults to ./config/sentinel.yaml.

    Returns:
        Populated Settings dataclass with merged values.

    Raises:
        FileNotFoundError: If config_path is specified but does not exist.
    """
    if config_path is None:
        config_path = "./config/sentinel.yaml"

    # Load YAML if file exists
    yaml_config: Dict[str, Any] = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}

    # Extract top-level keys from YAML
    settings_dict = {}
    for key in Settings.__dataclass_fields__:
        if key in yaml_config:
            settings_dict[key] = yaml_config[key]

    # Override with environment variables (UPPER_CASE with underscores)
    for key in Settings.__dataclass_fields__:
        env_key = key.upper()
        env_value = os.environ.get(env_key)
        if env_value is not None:
            # Attempt YAML parsing for list/dict env vars
            field_type = Settings.__dataclass_fields__[key].type
            if field_type in (list, dict) or "list" in str(field_type).lower():
                try:
                    settings_dict[key] = yaml.safe_load(env_value)
                except yaml.YAMLError:
                    settings_dict[key] = env_value
            elif field_type == bool or "bool" in str(field_type).lower():
                settings_dict[key] = env_value.lower() in ("true", "1", "yes")
            elif field_type in (int, float):
                try:
                    settings_dict[key] = field_type(env_value)
                except (ValueError, TypeError):
                    settings_dict[key] = env_value
            else:
                settings_dict[key] = env_value

    return Settings(**settings_dict)


def validate_config(settings: Settings) -> bool:
    """
    Validate critical Settings fields for completeness and correctness.

    Checks that API keys are non-empty, numeric thresholds are in valid ranges,
    paths are writable, and list fields are properly typed.

    Args:
        settings: Settings object to validate.

    Returns:
        True if validation passes. Raises ValueError on failure.

    Raises:
        ValueError: If required fields are missing or out of range.
    """
    errors = []

    # Check API keys
    if not settings.anthropic_api_key:
        errors.append("ANTHROPIC_API_KEY is required.")
    if not settings.gemini_api_key:
        errors.append("GEMINI_API_KEY is required.")

    # Check thresholds
    if not 0.0 <= settings.linguist_certainty_threshold <= 1.0:
        errors.append("linguist_certainty_threshold must be in [0.0, 1.0].")
    if not 0.0 <= settings.judge_confidence_floor <= 1.0:
        errors.append("judge_confidence_floor must be in [0.0, 1.0].")
    if not 0.0 <= settings.rag_similarity_threshold <= 1.0:
        errors.append("rag_similarity_threshold must be in [0.0, 1.0].")

    # Check positive integers
    if settings.live_prices_timeout <= 0:
        errors.append("live_prices_timeout must be positive.")
    if settings.news_max_results <= 0:
        errors.append("news_max_results must be positive.")
    if settings.sec_filings_lookback_days <= 0:
        errors.append("sec_filings_lookback_days must be positive.")

    # Check list types
    if not isinstance(settings.sec_filings_types, list):
        errors.append("sec_filings_types must be a list.")
    if not isinstance(settings.reddit_subreddits, list):
        errors.append("reddit_subredd
