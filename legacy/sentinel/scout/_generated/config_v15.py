"""
Sentinel configuration loader — reads YAML config and environment variables.

This module provides a typed Settings dataclass and a load_config() function
that merges YAML configuration with environment variable overrides. It serves
as the single source of truth for all Sentinel subsystem parameters across
scout, linguist, historian, and judge pillars.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class ScoutSettings:
    """Scout pillar configuration (data ingestion)."""
    live_prices_enabled: bool = True
    live_prices_symbols: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    news_sources: list[str] = field(default_factory=lambda: ["reuters", "bloomberg", "cnbc"])
    sec_filings_enabled: bool = True
    sec_filing_types: list[str] = field(default_factory=lambda: ["8-K", "10-Q", "10-K"])
    reddit_enabled: bool = False
    reddit_subreddits: list[str] = field(default_factory=lambda: ["wallstreetbets", "stocks"])
    github_enabled: bool = False
    github_search_terms: list[str] = field(default_factory=list)
    scrape_timeout_seconds: int = 30
    max_retries: int = 3


@dataclass
class LinguistSettings:
    """Linguist pillar configuration (LLM reasoning)."""
    model_id: str = "claude-sonnet-4-6"
    certainty_threshold: float = 0.65
    hesitation_keywords: list[str] = field(
        default_factory=lambda: ["may", "might", "could", "unclear", "uncertain"]
    )
    regulatory_whisper_enabled: bool = True
    linguistic_drift_enabled: bool = False
    max_tokens: int = 2000
    temperature: float = 0.3


@dataclass
class HistorianSettings:
    """Historian pillar configuration (RAG pipeline)."""
    chroma_db_path: str = "./data/chroma_db"
    embedding_model: str = "gemini-3.1-flash-lite-preview"
    vector_similarity_threshold: float = 0.5
    max_context_documents: int = 5
    historical_lookback_days: int = 365
    corpus_expansion_enabled: bool = False


@dataclass
class JudgeSettings:
    """Judge pillar configuration (predictions & post-mortems)."""
    predictor_model: str = "claude-sonnet-4-6"
    baseline_strategy: str = "momentum"
    confidence_weighting_enabled: bool = True
    anomaly_detection_enabled: bool = True
    discord_webhook_url: Optional[str] = None
    discord_enabled: bool = False
    postmortem_enabled: bool = True
    postmortem_lookback_hours: int = 24


@dataclass
class Settings:
    """Root Sentinel configuration container."""
    scout: ScoutSettings = field(default_factory=ScoutSettings)
    linguist: LinguistSettings = field(default_factory=LinguistSettings)
    historian: HistorianSettings = field(default_factory=HistorianSettings)
    judge: JudgeSettings = field(default_factory=JudgeSettings)
    api_keys: Dict[str, str] = field(default_factory=dict)
    log_level: str = "INFO"
    dry_run: bool = False


def load_config(config_path: Optional[str] = None) -> Settings:
    """Load configuration from YAML file and environment variables, with env overrides."""
    settings = Settings()

    # Load YAML if provided
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                raw_config = yaml.safe_load(f) or {}
                _merge_config(settings, raw_config)

    # Apply environment variable overrides
    _apply_env_overrides(settings)

    return settings


def _merge_config(settings: Settings, raw_config: Dict[str, Any]) -> None:
    """Recursively merge raw YAML config dict into Settings dataclass."""
    if "scout" in raw_config:
        _update_dataclass(settings.scout, raw_config["scout"])
    if "linguist" in raw_config:
        _update_dataclass(settings.linguist, raw_config["linguist"])
    if "historian" in raw_config:
        _update_dataclass(settings.historian, raw_config["historian"])
    if "judge" in raw_config:
        _update_dataclass(settings.judge, raw_config["judge"])
    if "api_keys" in raw_config and isinstance(raw_config["api_keys"], dict):
        settings.api_keys.update(raw_config["api_keys"])
    if "log_level" in raw_config:
        settings.log_level = raw_config["log_level"]
    if "dry_run" in raw_config:
        settings.dry_run = raw_config["dry_run"]


def _update_dataclass(obj: Any, updates: Dict[str, Any]) -> None:
    """Update dataclass fields from a dict, respecting types."""
    if not isinstance(updates, dict):
        return
    for key, value in updates.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def _apply_env_overrides(settings: Settings) -> None:
    """Apply environment variable overrides (SENTINEL_* prefix convention)."""
    # API keys
    if api_key := os.getenv("ANTHROPIC_API_KEY"):
        settings.api_keys["anthropic"] = api_key
    if api_key := os.getenv("GEMINI_API_KEY"):
        settings.api_keys["gemini"] = api_key

    # Scout overrides
    if symbols := os.getenv("SENTINEL_SCOUT_SYMBOLS"):
        settings.scout.live_prices_symbols = symbols.split(",")
    if val := os.getenv("SENTINEL_SCOUT_TIMEOUT"):
        settings.scout.scrape_timeout_seconds = int(val)

    # Linguist overrides
    if threshold := os.getenv("SENTINEL_LINGUIST_CERTAINTY_THRESHOLD"):
        settings.linguist.certainty_threshold = float(threshold)
    if tokens := os.getenv("SENTINEL_LINGUIST_MAX_TOKENS"):
        settings.linguist.max_tokens = int(tokens)

    # Historian overrides
    if db_path := os.getenv("SENTINEL_HISTORIAN_CHROMA_PATH"):
        settings.historian.chroma_db_path = db_path
    if threshold := os.getenv("SENTINEL_HISTORIAN_SIMILARITY_THRESHOLD"):
        settings.historian.vector_similarity_threshold = float(threshold)

    # Judge overrides
    if webhook := os.getenv("SENTINEL_JUDGE_DISCORD_WEBHOOK"):
        settings.judge.discord_webhook_url = webhook
        settings.judge.discord_enabled = True
    if strategy := os.getenv("SENTINEL_JUDGE_BASELINE"):
        settings.judge.baseline_strategy = strategy

    # Global overrides
    if log_level := os.getenv("SENTINEL_LOG_LEVEL"):
        settings.log_level = log_level
    if dry_run := os.getenv("SENTINEL_DRY_RUN"):
        settings.dry_run = dry_run.lower() in ("true", "1", "yes")


def settings_to_dict(settings: Settings) -> Dict[str, Any]:
    """Serialize Settings dataclass to dict (excludes sensitive API keys)."""
    result = {
        "scout": asdict(settings.scout),
        "linguist": asdict(settings.linguist),
        "historian": asdict(settings.historian),
        "judge": asdict(settings.judge),
        "api_keys": {k: "***" for k in settings.api_keys.keys()},
        "log_level": settings.log_level,
        "dry_run": settings.dry
