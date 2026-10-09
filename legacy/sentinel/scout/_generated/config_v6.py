"""
Sentinel configuration loader — reads YAML config and environment variables.

This module provides a typed Settings dataclass and a config loader function
that merges YAML configuration with environment variable overrides. It serves
as the single source of truth for Sentinel's runtime parameters across all
pillars (scout, linguist, historian, judge).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ScoutConfig:
    """Configuration for the scout pillar (data ingestion)."""

    live_price_symbols: list[str] = field(default_factory=lambda: ["AAPL", "TSLA", "GOOGL"])
    news_fetch_limit: int = 50
    news_sources: list[str] = field(default_factory=lambda: ["hn", "reddit"])
    sec_filings_lookback_days: int = 30
    sec_filing_types: list[str] = field(default_factory=lambda: ["8-K", "10-Q", "10-K"])
    reddit_subreddits: list[str] = field(default_factory=lambda: ["stocks", "investing"])
    reddit_post_limit: int = 100


@dataclass
class LinguistConfig:
    """Configuration for the linguist pillar (LLM reasoning)."""

    claude_model: str = "claude-sonnet-4-6"
    gemini_model: str = "gemini-3.1-flash-lite-preview"
    certainty_threshold: float = 0.65
    linguistic_drift_window_days: int = 7
    max_tokens_reasoning: int = 2000


@dataclass
class HistorianConfig:
    """Configuration for the historian pillar (RAG pipeline)."""

    chromadb_path: str = "./chromadb_data"
    embedding_model: str = "models/embedding-001"
    rag_similarity_threshold: float = 0.5
    rag_top_k_results: int = 5
    historical_event_lookback_days: int = 365


@dataclass
class JudgeConfig:
    """Configuration for the judge pillar (prediction & post-mortem)."""

    prediction_confidence_threshold: float = 0.6
    baseline_strategies: list[str] = field(
        default_factory=lambda: ["momentum", "mean_reversion", "sentiment_only"]
    )
    discord_webhook_url: Optional[str] = None
    postmortem_include_details: bool = True
    anomaly_detection_zscore: float = 2.5


@dataclass
class Settings:
    """Root settings object for Sentinel — aggregates all pillar configs."""

    scout: ScoutConfig = field(default_factory=ScoutConfig)
    linguist: LinguistConfig = field(default_factory=LinguistConfig)
    historian: HistorianConfig = field(default_factory=HistorianConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    # Global settings
    log_level: str = "INFO"
    debug_mode: bool = False
    openapi_key: Optional[str] = None


def load_config(config_path: Optional[str] = None) -> Settings:
    """Load configuration from YAML file and environment variable overrides.

    Args:
        config_path: Path to YAML config file. If None, looks for ./config.yaml
                     or uses built-in defaults.

    Returns:
        Settings dataclass populated with merged config + env vars.
    """
    settings = Settings()

    # Try to load YAML if it exists
    if config_path is None:
        config_path = "./config.yaml"

    config_path_obj = Path(config_path)
    if config_path_obj.exists():
        with open(config_path_obj, "r") as f:
            yaml_data = yaml.safe_load(f) or {}
        _merge_dict_into_settings(settings, yaml_data)

    # Override with environment variables
    _apply_env_overrides(settings)

    return settings


def _merge_dict_into_settings(settings: Settings, data: dict) -> None:
    """Recursively merge YAML dictionary into Settings dataclass.

    Args:
        settings: Settings object to mutate.
        data: Dictionary from parsed YAML.
    """
    if "scout" in data:
        _merge_dataclass(settings.scout, data["scout"])
    if "linguist" in data:
        _merge_dataclass(settings.linguist, data["linguist"])
    if "historian" in data:
        _merge_dataclass(settings.historian, data["historian"])
    if "judge" in data:
        _merge_dataclass(settings.judge, data["judge"])

    if "log_level" in data:
        settings.log_level = data["log_level"]
    if "debug_mode" in data:
        settings.debug_mode = data["debug_mode"]
    if "openapi_key" in data:
        settings.openapi_key = data["openapi_key"]


def _merge_dataclass(target: object, data: dict) -> None:
    """Merge dictionary into a dataclass instance.

    Args:
        target: Dataclass instance to mutate.
        data: Dictionary of field values.
    """
    for key, value in data.items():
        if hasattr(target, key):
            setattr(target, key, value)


def _apply_env_overrides(settings: Settings) -> None:
    """Apply environment variable overrides to settings.

    Supports env vars like SENTINEL_SCOUT_LIVE_PRICE_SYMBOLS, etc.

    Args:
        settings: Settings object to mutate.
    """
    # Global overrides
    if "SENTINEL_LOG_LEVEL" in os.environ:
        settings.log_level = os.environ["SENTINEL_LOG_LEVEL"]
    if "SENTINEL_DEBUG_MODE" in os.environ:
        settings.debug_mode = os.environ["SENTINEL_DEBUG_MODE"].lower() in ("true", "1")

    # API keys from environment (standard locations)
    if "ANTHROPIC_API_KEY" in os.environ:
        # Stored for reference, but Claude SDK reads directly from env
        pass
    if "GEMINI_API_KEY" in os.environ:
        # Stored for reference, but Gemini SDK reads directly from env
        pass

    # Scout overrides
    if "SENTINEL_SCOUT_NEWS_FETCH_LIMIT" in os.environ:
        settings.scout.news_fetch_limit = int(os.environ["SENTINEL_SCOUT_NEWS_FETCH_LIMIT"])
    if "SENTINEL_SCOUT_LIVE_PRICE_SYMBOLS" in os.environ:
        settings.scout.live_price_symbols = os.environ[
            "SENTINEL_SCOUT_LIVE_PRICE_SYMBOLS"
        ].split(",")

    # Linguist overrides
    if "SENTINEL_LINGUIST_CERTAINTY_THRESHOLD" in os.environ:
        settings.linguist.certainty_threshold = float(
            os.environ["SENTINEL_LINGUIST_CERTAINTY_THRESHOLD"]
        )

    # Historian overrides
    if "SENTINEL_HISTORIAN_CHROMADB_PATH" in os.environ:
        settings.historian.chromadb_path = os.environ["SENTINEL_HISTORIAN_CHROMADB_PATH"]

    # Judge overrides
    if "SENTINEL_JUDGE_DISCORD_WEBHOOK_URL" in os.environ:
        settings.judge.discord_webhook_url = os.environ["SENTINEL_JUDGE_DISCORD_WEBHOOK_URL"]
    if "SENTINEL_JUDGE_PREDICTION_CONFIDENCE_THRESHOLD" in os.environ:
        settings.judge.prediction_confidence_threshold = float(
            os.environ["SENTINEL_JUDGE_PREDICTION_CONFIDENCE_THRESHOLD"]
        )


def get_settings() -> Settings:
    """Convenience singleton getter for global settings instance.

    Returns:
        Cached Settings object (or loads fresh if not yet initialized).
    """
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = load_config()
    return get_settings._instance
