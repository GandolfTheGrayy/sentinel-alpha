"""
Config loader for Sentinel Sentiment Engine.

Reads YAML configuration files and environment variables, merging them into
a typed Settings dataclass. Used by scout modules (live_prices, news, sec_filings)
and upstream linguist/historian/judge pipelines to access credentials, endpoints,
and runtime parameters without hardcoding.

Precedence: env vars > YAML file > built-in defaults.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path

import yaml


@dataclass
class Settings:
    """
    Typed configuration container for Sentinel.
    
    All fields have defaults; override via YAML or environment variables.
    Env var names follow pattern: SENTINEL_<SECTION>_<KEY> (uppercase).
    """
    
    # API Keys & Auth
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    discord_webhook_url: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "sentinel-bot/1.0"
    
    # Data sources
    yfinance_timeout: int = 30
    sec_edgar_base_url: str = "https://www.sec.gov/cgi-bin/browse-edgar"
    news_api_key: str = ""
    news_api_base_url: str = "https://newsapi.org/v2"
    
    # Database & Storage
    chromadb_path: str = "./data/chromadb"
    sqlite_db_path: str = "./data/sentinel.db"
    
    # Model Configuration
    claude_model: str = "claude-sonnet-4-6"
    gemini_model: str = "gemini-3.1-flash-lite-preview"
    embedding_model: str = "models/embedding-001"
    
    # Pipeline behavior
    tickers_to_monitor: list = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT"])
    max_sec_filings_per_ticker: int = 10
    sentiment_threshold_positive: float = 0.65
    sentiment_threshold_negative: float = 0.35
    rag_top_k: int = 5
    
    # Logging & Debug
    log_level: str = "INFO"
    debug_mode: bool = False
    
    # Post-mortem & Resolver
    postmortem_retention_days: int = 30
    resolver_confidence_threshold: float = 0.72


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Load and merge YAML config, environment variables, and defaults into Settings.
    
    Args:
        config_path: Optional path to YAML config file. If None, searches for
                     ./config.yaml or ./config/sentinel.yaml.
    
    Returns:
        Fully populated Settings dataclass with precedence: env > YAML > defaults.
    
    Raises:
        FileNotFoundError: If config_path is explicitly provided but not found.
        yaml.YAMLError: If YAML parsing fails.
    """
    # Start with defaults
    settings = Settings()
    
    # Determine config file location
    if config_path is None:
        candidates = [Path("config.yaml"), Path("config/sentinel.yaml")]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break
    
    # Load YAML if found
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        
        # Flatten nested YAML (e.g., api.anthropic_api_key -> anthropic_api_key)
        flat_yaml = _flatten_yaml(yaml_data)
        
        # Merge YAML into settings
        for key, value in flat_yaml.items():
            if hasattr(settings, key) and value is not None:
                setattr(settings, key, value)
    
    # Override with environment variables (SENTINEL_* pattern)
    for key, value in os.environ.items():
        if key.startswith("SENTINEL_"):
            setting_key = key[9:].lower()  # Strip "SENTINEL_" prefix
            if hasattr(settings, setting_key) and value:
                # Attempt type coercion
                current_value = getattr(settings, setting_key)
                if isinstance(current_value, bool):
                    setattr(settings, setting_key, value.lower() in ("true", "1", "yes"))
                elif isinstance(current_value, int):
                    try:
                        setattr(settings, setting_key, int(value))
                    except ValueError:
                        setattr(settings, setting_key, value)
                elif isinstance(current_value, list):
                    # Comma-separated list
                    setattr(settings, setting_key, [v.strip() for v in value.split(",")])
                else:
                    setattr(settings, setting_key, value)
    
    return settings


def _flatten_yaml(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested YAML dict into single-level keys (e.g., api.key -> api_key).
    
    Args:
        data: Nested dictionary from YAML.
        prefix: Internal recursion prefix.
    
    Returns:
        Flattened dictionary with underscore-joined keys.
    """
    flat = {}
    for key, value in data.items():
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_yaml(value, full_key))
        else:
            flat[full_key] = value
    return flat


def settings_to_dict(settings: Settings) -> Dict[str, Any]:
    """
    Convert Settings dataclass to plain dict (e.g., for logging or export).
    
    Args:
        settings: Settings instance.
    
    Returns:
        Dictionary representation, with API keys redacted for safety.
    """
    data = asdict(settings)
    # Redact sensitive fields
    for key in data:
        if "api_key" in key or "secret" in key or "webhook" in key:
            if data[key]:
                data[key] = data[key][:8] + "***" if len(data[key]) > 8 else "***"
    return data
