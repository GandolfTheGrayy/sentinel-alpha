"""
Config loader for Sentinel Sentiment Engine.

Reads YAML configuration files and environment variables into a typed Settings
dataclass. Used by Scout ingestion modules to configure data sources, API keys,
and scraping parameters. Supports environment variable overrides for all fields.
"""

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Dict, List, Any

import yaml


@dataclass
class Settings:
    """Typed configuration container for Sentinel Sentinel Engine."""

    # API Keys (from environment)
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    discord_webhook_url: str = ""

    # Scout: Live Prices
    price_symbols: List[str] = field(default_factory=lambda: ["AAPL", "TSLA", "GOOGL"])
    price_fetch_interval_seconds: int = 300
    price_source: str = "yfinance"

    # Scout: News
    news_sources: List[str] = field(default_factory=lambda: ["reddit", "hn"])
    news_fetch_interval_seconds: int = 1800
    reddit_subreddits: List[str] = field(default_factory=lambda: ["stocks", "investing", "wallstreetbets"])

    # Scout: SEC Filings
    sec_form_types: List[str] = field(default_factory=lambda: ["8-K", "10-Q", "10-K"])
    sec_fetch_interval_seconds: int = 3600
    sec_lookback_days: int = 30

    # Linguist: Sentiment Analysis
    sentiment_model: str = "claude-sonnet-4-6"
    certainty_threshold: float = 0.65
    linguistic_drift_window_days: int = 14

    # Historian: RAG
    chromadb_path: str = "./data/chromadb"
    embedding_model: str = "gemini-3.1-flash-lite-preview"
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.5

    # Judge: Predictions
    judge_model: str = "claude-sonnet-4-6"
    prediction_horizon_days: int = 5
    confidence_floor: float = 0.55

    # Judge: Post-mortem
    postmortem_lookback_days: int = 7
    anomaly_detection_enabled: bool = True

    # General
    debug_mode: bool = False
    log_level: str = "INFO"


def load_settings(config_path: Optional[str] = None) -> Settings:
    """Load Settings from YAML file and environment variables."""
    settings_dict: Dict[str, Any] = {}

    # Load from YAML if provided
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                yaml_data = yaml.safe_load(f) or {}
                settings_dict.update(yaml_data)

    # Override with environment variables
    env_overrides = _extract_env_overrides()
    settings_dict.update(env_overrides)

    return Settings(**settings_dict)


def _extract_env_overrides() -> Dict[str, Any]:
    """Extract Sentinel-prefixed environment variables into typed settings dict."""
    overrides: Dict[str, Any] = {}
    prefix = "SENTINEL_"

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Strip prefix and lowercase
        field_name = key[len(prefix):].lower()

        # Parse value type based on field definition
        field_type = _get_field_type(field_name)
        if field_type:
            overrides[field_name] = _coerce_value(value, field_type)

    return overrides


def _get_field_type(field_name: str) -> Optional[type]:
    """Return the type annotation for a Settings field."""
    for f in fields(Settings):
        if f.name == field_name:
            # Handle generic types like List[str]
            origin = getattr(f.type, "__origin__", None)
            if origin is list:
                return list
            return f.type
    return None


def _coerce_value(value: str, target_type: type) -> Any:
    """Coerce environment variable string to target type."""
    if target_type is str:
        return value
    elif target_type is int:
        return int(value)
    elif target_type is float:
        return float(value)
    elif target_type is bool:
        return value.lower() in ("true", "1", "yes")
    elif target_type is list:
        # CSV split for lists
        return [v.strip() for v in value.split(",")]
    return value


def merge_settings(base: Settings, overrides: Settings) -> Settings:
    """Merge an overrides Settings object into a base Settings object."""
    merged_dict = {}
    for f in fields(Settings):
        base_val = getattr(base, f.name)
        override_val = getattr(overrides, f.name)
        # Use override if it differs from default
        merged_dict[f.name] = override_val if override_val != f.default else base_val
    return Settings(**merged_dict)
