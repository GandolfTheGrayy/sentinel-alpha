"""
Sentinel config loader — reads YAML configuration and environment variables.

Provides a typed Settings dataclass and loader functions for use across Scout
(data ingestion), Linguist (LLM reasoning), Historian (RAG pipeline), and Judge
(post-mortem analysis) pillars. Merges environment overrides with YAML defaults.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml


@dataclass
class ScoutConfig:
    """Configuration for Scout data ingestion modules."""
    live_prices_enabled: bool = True
    live_prices_symbols: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    news_enabled: bool = True
    news_sources: list[str] = field(default_factory=lambda: ["hacker_news", "reddit"])
    sec_filings_enabled: bool = True
    sec_filings_delay_hours: int = 24
    reddit_subreddits: list[str] = field(default_factory=lambda: ["stocks", "investing"])
    github_trending_enabled: bool = False


@dataclass
class LinguistConfig:
    """Configuration for Linguist LLM reasoning modules."""
    certainty_model: str = "claude-sonnet-4-6"
    certainty_max_tokens: int = 1000
    drift_detector_enabled: bool = False
    regulatory_whispers_enabled: bool = False
    confidence_threshold: float = 0.6


@dataclass
class HistorianConfig:
    """Configuration for Historian RAG pipeline."""
    vector_db_path: str = "./chromadb"
    embedding_model: str = "gemini-3.1-flash-lite-preview"
    historical_lookback_days: int = 365
    chunk_size: int = 512
    top_k_results: int = 5
    similarity_threshold: float = 0.5


@dataclass
class JudgeConfig:
    """Configuration for Judge post-mortem and prediction modules."""
    prediction_model: str = "claude-sonnet-4-6"
    prediction_max_tokens: int = 2000
    baseline_strategies: list[str] = field(default_factory=lambda: ["momentum", "mean_reversion", "sentiment"])
    discord_webhook_enabled: bool = False
    discord_webhook_url: Optional[str] = None
    postmortem_enabled: bool = True
    postmortem_output_dir: str = "./postmortems"


@dataclass
class Settings:
    """Root Sentinel configuration container."""
    environment: str = "development"
    log_level: str = "INFO"
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    scout: ScoutConfig = field(default_factory=ScoutConfig)
    linguist: LinguistConfig = field(default_factory=LinguistConfig)
    historian: HistorianConfig = field(default_factory=HistorianConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Settings to dictionary representation."""
        return asdict(self)


def load_config(yaml_path: Optional[str] = None) -> Settings:
    """Load configuration from YAML file and environment variables, merging with defaults."""
    settings = Settings()
    
    # Load from YAML if provided or if default exists
    if yaml_path is None:
        yaml_path = os.getenv("SENTINEL_CONFIG_PATH", "./config.yaml")
    
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f) or {}
        
        # Merge YAML into settings
        if "environment" in config_data:
            settings.environment = config_data["environment"]
        if "log_level" in config_data:
            settings.log_level = config_data["log_level"]
        
        if "scout" in config_data:
            settings.scout = _merge_dataclass(ScoutConfig, config_data["scout"])
        if "linguist" in config_data:
            settings.linguist = _merge_dataclass(LinguistConfig, config_data["linguist"])
        if "historian" in config_data:
            settings.historian = _merge_dataclass(HistorianConfig, config_data["historian"])
        if "judge" in config_data:
            settings.judge = _merge_dataclass(JudgeConfig, config_data["judge"])
    
    # Override with environment variables
    settings.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", settings.anthropic_api_key)
    settings.gemini_api_key = os.getenv("GEMINI_API_KEY", settings.gemini_api_key)
    
    if env := os.getenv("SENTINEL_ENVIRONMENT"):
        settings.environment = env
    if log_level := os.getenv("SENTINEL_LOG_LEVEL"):
        settings.log_level = log_level
    
    return settings


def _merge_dataclass(dc_class: type, data: Dict[str, Any]) -> Any:
    """Merge dictionary into a dataclass instance, preserving defaults for missing keys."""
    if not isinstance(data, dict):
        return dc_class()
    
    # Filter to only keys that exist in the dataclass
    field_names = {f.name for f in dc_class.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in field_names}
    
    return dc_class(**filtered)


def get_settings() -> Settings:
    """Retrieve or initialize the global Settings singleton."""
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = load_config()
    return get_settings._instance


def reset_settings() -> None:
    """Reset the global Settings singleton (useful for testing)."""
    if hasattr(get_settings, "_instance"):
        delattr(get_settings, "_instance")
