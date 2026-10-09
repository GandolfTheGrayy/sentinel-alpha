"""
Config loader for Sentinel Sentiment Engine.

Reads YAML configuration files and environment variables into a typed Settings
dataclass. Used by Scout modules (live_prices, news, sec_filings) and throughout
the pipeline to access consistent, validated configuration.

Environment variables override YAML file values. All paths are resolved relative
to the Sentinel project root.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ScoutConfig:
    """Scout module configuration (data ingestion)."""

    yfinance_timeout: int = 10
    stooq_fallback_enabled: bool = True
    sec_edgar_base_url: str = "https://www.sec.gov/cgi-bin"
    sec_request_delay: float = 0.5
    news_sources: list[str] = field(
        default_factory=lambda: ["reuters", "bloomberg", "cnbc"]
    )
    reddit_subreddits: list[str] = field(default_factory=lambda: ["stocks", "investing"])
    hn_enabled: bool = True
    github_health_enabled: bool = False


@dataclass
class LinguistConfig:
    """Linguist module configuration (LLM reasoning)."""

    model: str = "claude-sonnet-4-6"
    certainty_threshold: float = 0.65
    hesitation_keywords: list[str] = field(
        default_factory=lambda: ["may", "might", "could", "uncertain", "risk"]
    )
    regulatory_whispers_enabled: bool = True
    tone_drift_lookback_days: int = 30


@dataclass
class HistorianConfig:
    """Historian module configuration (RAG pipeline)."""

    vector_db_path: str = "data/chroma_db"
    embedding_model: str = "gemini-3.1-flash-lite-preview"
    similarity_threshold: float = 0.6
    max_context_chunks: int = 5
    historical_event_table: str = "data/events.db"
    confidence_weighting_enabled: bool = True


@dataclass
class JudgeConfig:
    """Judge module configuration (prediction & post-mortem)."""

    model: str = "claude-sonnet-4-6"
    prediction_horizon_days: int = 5
    confidence_floor: float = 0.5
    baseline_strategies: list[str] = field(
        default_factory=lambda: ["momentum", "mean_reversion", "sentiment"]
    )
    discord_webhook_url: Optional[str] = None
    discord_enabled: bool = False
    post_mortem_lookback_days: int = 7


@dataclass
class Settings:
    """Root Sentinel configuration."""

    project_root: Path = field(default_factory=lambda: Path.cwd())
    debug: bool = False
    log_level: str = "INFO"
    scout: ScoutConfig = field(default_factory=ScoutConfig)
    linguist: LinguistConfig = field(default_factory=LinguistConfig)
    historian: HistorianConfig = field(default_factory=HistorianConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path against project root."""
        return self.project_root / relative_path


def load_config(config_path: Optional[str] = None) -> Settings:
    """
    Load Settings from YAML file and environment variables.

    Environment variables override YAML values. Vars prefixed with SENTINEL_ are
    mapped to nested config keys (e.g., SENTINEL_SCOUT_YFINANCE_TIMEOUT).
    If config_path is None, defaults to sentinel/config.yaml relative to cwd.
    """
    if config_path is None:
        config_path = "sentinel/config.yaml"

    config_file = Path(config_path)
    settings_dict: dict = {}

    # Load YAML if it exists
    if config_file.exists():
        with open(config_file, "r") as f:
            file_data = yaml.safe_load(f) or {}
            settings_dict.update(file_data)

    # Override with environment variables
    for key, value in os.environ.items():
        if key.startswith("SENTINEL_"):
            # Remove prefix and convert to lowercase
            config_key = key[len("SENTINEL_") :].lower()
            # Parse nested keys (e.g., SCOUT_YFINANCE_TIMEOUT -> scout.yfinance_timeout)
            parts = config_key.split("_")
            current = settings_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

    # Construct Settings with nested dataclasses
    return Settings(
        project_root=Path(
            settings_dict.get("project_root", Path.cwd())
        ),
        debug=_parse_bool(settings_dict.get("debug", False)),
        log_level=settings_dict.get("log_level", "INFO"),
        scout=_build_scout_config(settings_dict.get("scout", {})),
        linguist=_build_linguist_config(settings_dict.get("linguist", {})),
        historian=_build_historian_config(settings_dict.get("historian", {})),
        judge=_build_judge_config(settings_dict.get("judge", {})),
    )


def _parse_bool(value) -> bool:
    """Convert string or bool to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


def _build_scout_config(data: dict) -> ScoutConfig:
    """Build ScoutConfig from dict, coercing types."""
    return ScoutConfig(
        yfinance_timeout=int(data.get("yfinance_timeout", 10)),
        stooq_fallback_enabled=_parse_bool(data.get("stooq_fallback_enabled", True)),
        sec_edgar_base_url=str(data.get("sec_edgar_base_url", "https://www.sec.gov/cgi-bin")),
        sec_request_delay=float(data.get("sec_request_delay", 0.5)),
        news_sources=data.get("news_sources", ["reuters", "bloomberg", "cnbc"]),
        reddit_subreddits=data.get("reddit_subreddits", ["stocks", "investing"]),
        hn_enabled=_parse_bool(data.get("hn_enabled", True)),
        github_health_enabled=_parse_bool(data.get("github_health_enabled", False)),
    )


def _build_linguist_config(data: dict) -> LinguistConfig:
    """Build LinguistConfig from dict, coercing types."""
    return LinguistConfig(
        model=str(data.get("model", "claude-sonnet-4-6")),
        certainty_threshold=float(data.get("certainty_threshold", 0.65)),
        hesitation_keywords=data.get(
            "hesitation_keywords", ["may", "might", "could", "uncertain", "risk"]
        ),
        regulatory_whispers_enabled=_parse_bool(
            data.get("regulatory_whispers_enabled", True)
        ),
        tone_drift_lookback_days=int(data.get("tone_drift_lookback_days", 30)),
    )


def _build_historian_config(data: dict) -> HistorianConfig:
    """Build HistorianConfig from dict, coercing types."""
    return HistorianConfig(
        vector_db_path=str(data.get("vector_db_path", "data/chroma_db")),
        embedding_model=str(
            data.get("embedding_model", "gemini-3.1-flash-lite-preview")
        ),
        similarity_threshold=float(data.get("similarity_threshold", 0.6)),
        max_context_chunks=int(data.get("
