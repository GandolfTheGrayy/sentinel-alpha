"""
Sentinel config loader: reads YAML configuration and environment variables.

Provides a typed Settings dataclass that centralizes all configuration for
the Scout pillar (data ingestion) and other Sentinel components. Merges
environment variable overrides with YAML defaults for flexible deployment.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class ScoutSettings:
    """Scout pillar configuration: data sources, API keys, ingestion parameters."""
    
    live_prices_enabled: bool = True
    live_prices_interval_sec: int = 300
    live_prices_fallback_stooq: bool = True
    
    news_enabled: bool = True
    news_sources: list = field(default_factory=lambda: ["newsapi", "reddit"])
    news_limit_per_source: int = 50
    
    sec_filings_enabled: bool = True
    sec_filings_lookback_days: int = 30
    sec_filings_form_types: list = field(default_factory=lambda: ["8-K", "10-Q", "10-K"])
    
    reddit_enabled: bool = True
    reddit_subreddits: list = field(default_factory=lambda: ["stocks", "investing", "wallstreetbets"])
    reddit_post_limit: int = 100
    reddit_sentiment_keywords: list = field(default_factory=lambda: ["bullish", "bearish", "moon", "dump"])


@dataclass
class LinguistSettings:
    """Linguist pillar configuration: LLM reasoning, certainty scoring."""
    
    llm_model: str = "claude-sonnet-4-6"
    certainty_threshold: float = 0.65
    linguistic_drift_window_days: int = 90
    max_tokens_per_request: int = 2000
    temperature: float = 0.3


@dataclass
class HistorianSettings:
    """Historian pillar configuration: RAG, vector DB, historical lookups."""
    
    rag_enabled: bool = True
    chroma_db_path: str = "./data/chroma_db"
    embedding_model: str = "gemini-3.1-flash-lite-preview"
    vector_batch_size: int = 100
    similarity_threshold: float = 0.7
    lookback_days: int = 365


@dataclass
class JudgeSettings:
    """Judge pillar configuration: predictions, post-mortems, notifications."""
    
    prediction_enabled: bool = True
    prediction_horizon_days: int = 7
    baseline_strategy: str = "weighted"
    
    postmortem_enabled: bool = True
    postmortem_schedule_utc: str = "22:00"
    
    notify_discord: bool = False
    discord_webhook_url: Optional[str] = None
    notify_email: bool = False
    email_recipients: list = field(default_factory=list)


@dataclass
class Settings:
    """Root configuration: merges all pillars + global API keys."""
    
    # Global API keys (override via environment)
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    newsapi_key: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: Optional[str] = None
    
    # Global settings
    log_level: str = "INFO"
    data_dir: str = "./data"
    cache_enabled: bool = True
    cache_ttl_sec: int = 3600
    
    # Pillar-specific settings
    scout: ScoutSettings = field(default_factory=ScoutSettings)
    linguist: LinguistSettings = field(default_factory=LinguistSettings)
    historian: HistorianSettings = field(default_factory=HistorianSettings)
    judge: JudgeSettings = field(default_factory=JudgeSettings)
    
    # Ticker universe
    tickers: list = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])


def load_config(config_path: Optional[str] = None) -> Settings:
    """Load configuration from YAML file and environment variables, with env overrides."""
    
    settings = Settings()
    
    # Load YAML if provided
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r") as f:
                yaml_data = yaml.safe_load(f) or {}
                _apply_yaml_to_settings(settings, yaml_data)
    
    # Override with environment variables
    _apply_env_to_settings(settings)
    
    return settings


def _apply_yaml_to_settings(settings: Settings, yaml_data: Dict[str, Any]) -> None:
    """Recursively apply YAML data to Settings dataclass and nested pillar configs."""
    
    for key, value in yaml_data.items():
        if key == "scout" and isinstance(value, dict):
            for sk, sv in value.items():
                if hasattr(settings.scout, sk):
                    setattr(settings.scout, sk, sv)
        
        elif key == "linguist" and isinstance(value, dict):
            for sk, sv in value.items():
                if hasattr(settings.linguist, sk):
                    setattr(settings.linguist, sk, sv)
        
        elif key == "historian" and isinstance(value, dict):
            for sk, sv in value.items():
                if hasattr(settings.historian, sk):
                    setattr(settings.historian, sk, sv)
        
        elif key == "judge" and isinstance(value, dict):
            for sk, sv in value.items():
                if hasattr(settings.judge, sk):
                    setattr(settings.judge, sk, sv)
        
        elif hasattr(settings, key):
            setattr(settings, key, value)


def _apply_env_to_settings(settings: Settings) -> None:
    """Apply environment variable overrides to Settings, respecting type conversions."""
    
    env_map = {
        "ANTHROPIC_API_KEY": ("anthropic_api_key", str),
        "GEMINI_API_KEY": ("gemini_api_key", str),
        "NEWSAPI_KEY": ("newsapi_key", str),
        "REDDIT_CLIENT_ID": ("reddit_client_id", str),
        "REDDIT_CLIENT_SECRET": ("reddit_client_secret", str),
        "REDDIT_USER_AGENT": ("reddit_user_agent", str),
        "LOG_LEVEL": ("log_level", str),
        "DATA_DIR": ("data_dir", str),
        "CACHE_ENABLED": ("cache_enabled", lambda x: x.lower() in ("true", "1", "yes")),
        "CACHE_TTL_SEC": ("cache_ttl_sec", int),
        "TICKERS": ("tickers", lambda x: x.split(",")),
    }
    
    for env_var, (attr, converter) in env_map.items():
        value = os.getenv(env_var)
        if value is not None:
            setattr(settings, attr, converter(value))
    
    # Scout pillar env overrides
    if os.getenv("LIVE_PRICES_ENABLED"):
        settings.scout.live_prices_enabled = os.getenv("LIVE_PRICES_ENABLED").lower() in ("true", "1")
    if os.getenv("LIVE_PRICES_INTERVAL_SEC"):
        settings.scout.live_prices_interval_sec = int(os.getenv("LIVE_PRICES_INTERVAL_SEC"))
    if os.getenv("NEWS_ENABLED"):
        settings.scout.news_enabled = os.getenv("NEWS_ENABLED").lower() in ("true", "1")
    if os.getenv("SEC_FILINGS_ENABLED"):
        settings.scout.sec_filings_enabled = os.getenv("SEC_FILINGS_ENABLED").lower
