"""
Config loader for Sentinel Sentiment Engine.

Reads YAML configuration file and environment variables, merging them into a
typed Settings dataclass. Used by Scout modules (price fetchers, scrapers, etc.)
to access API keys, rate limits, data sources, and other runtime parameters.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml


@dataclass
class APISettings:
    """API credentials and endpoints."""
    anthropic_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    reddit_client_id: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_ID", ""))
    reddit_client_secret: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET", ""))
    reddit_user_agent: str = field(default_factory=lambda: os.getenv("REDDIT_USER_AGENT", ""))
    discord_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))


@dataclass
class ScraperSettings:
    """Scout scraper configuration."""
    yfinance_timeout: int = 10
    sec_edgar_delay: float = 0.5
    news_sources: list = field(default_factory=lambda: ["reuters", "bloomberg", "cnbc"])
    reddit_subreddits: list = field(default_factory=lambda: ["stocks", "investing", "wallstreetbets"])
    max_reddit_posts: int = 100
    sentiment_lookback_days: int = 7


@dataclass
class RAGSettings:
    """Historian RAG pipeline configuration."""
    chroma_db_path: str = "./data/chroma_db"
    embedding_model: str = "gemini-3.1-flash-lite-preview"
    top_k_results: int = 5
    similarity_threshold: float = 0.6
    enable_historical_indexing: bool = True
    historical_sec_lookback_years: int = 3


@dataclass
class PredictionSettings:
    """Judge prediction configuration."""
    claude_model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.7
    enable_baseline_strategies: bool = True
    confidence_threshold: float = 0.65
    prediction_horizon_days: int = 5


@dataclass
class NotificationSettings:
    """Judge notification configuration."""
    enable_discord: bool = False
    enable_email: bool = False
    email_recipients: list = field(default_factory=list)
    alert_on_high_confidence: bool = True
    alert_threshold: float = 0.75


@dataclass
class Settings:
    """Root Sentinel configuration dataclass."""
    api: APISettings = field(default_factory=APISettings)
    scraper: ScraperSettings = field(default_factory=ScraperSettings)
    rag: RAGSettings = field(default_factory=RAGSettings)
    prediction: PredictionSettings = field(default_factory=PredictionSettings)
    notification: NotificationSettings = field(default_factory=NotificationSettings)
    debug: bool = False
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Convert Settings to nested dictionary for inspection."""
        return asdict(self)


def load_config(config_path: str = "config.yaml") -> Settings:
    """Load and merge YAML config file with environment variables into Settings."""
    settings = Settings()
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}
    else:
        yaml_data = {}
    
    # Merge API settings
    if "api" in yaml_data:
        api_cfg = yaml_data["api"]
        settings.api.anthropic_key = api_cfg.get("anthropic_key", settings.api.anthropic_key)
        settings.api.gemini_key = api_cfg.get("gemini_key", settings.api.gemini_key)
        settings.api.reddit_client_id = api_cfg.get("reddit_client_id", settings.api.reddit_client_id)
        settings.api.reddit_client_secret = api_cfg.get("reddit_client_secret", settings.api.reddit_client_secret)
        settings.api.reddit_user_agent = api_cfg.get("reddit_user_agent", settings.api.reddit_user_agent)
        settings.api.discord_webhook_url = api_cfg.get("discord_webhook_url", settings.api.discord_webhook_url)
    
    # Merge Scraper settings
    if "scraper" in yaml_data:
        scraper_cfg = yaml_data["scraper"]
        settings.scraper.yfinance_timeout = scraper_cfg.get("yfinance_timeout", settings.scraper.yfinance_timeout)
        settings.scraper.sec_edgar_delay = scraper_cfg.get("sec_edgar_delay", settings.scraper.sec_edgar_delay)
        settings.scraper.news_sources = scraper_cfg.get("news_sources", settings.scraper.news_sources)
        settings.scraper.reddit_subreddits = scraper_cfg.get("reddit_subreddits", settings.scraper.reddit_subreddits)
        settings.scraper.max_reddit_posts = scraper_cfg.get("max_reddit_posts", settings.scraper.max_reddit_posts)
        settings.scraper.sentiment_lookback_days = scraper_cfg.get("sentiment_lookback_days", settings.scraper.sentiment_lookback_days)
    
    # Merge RAG settings
    if "rag" in yaml_data:
        rag_cfg = yaml_data["rag"]
        settings.rag.chroma_db_path = rag_cfg.get("chroma_db_path", settings.rag.chroma_db_path)
        settings.rag.embedding_model = rag_cfg.get("embedding_model", settings.rag.embedding_model)
        settings.rag.top_k_results = rag_cfg.get("top_k_results", settings.rag.top_k_results)
        settings.rag.similarity_threshold = rag_cfg.get("similarity_threshold", settings.rag.similarity_threshold)
        settings.rag.enable_historical_indexing = rag_cfg.get("enable_historical_indexing", settings.rag.enable_historical_indexing)
        settings.rag.historical_sec_lookback_years = rag_cfg.get("historical_sec_lookback_years", settings.rag.historical_sec_lookback_years)
    
    # Merge Prediction settings
    if "prediction" in yaml_data:
        pred_cfg = yaml_data["prediction"]
        settings.prediction.claude_model = pred_cfg.get("claude_model", settings.prediction.claude_model)
        settings.prediction.max_tokens = pred_cfg.get("max_tokens", settings.prediction.max_tokens)
        settings.prediction.temperature = pred_cfg.get("temperature", settings.prediction.temperature)
        settings.prediction.enable_baseline_strategies = pred_cfg.get("enable_baseline_strategies", settings.prediction.enable_baseline_strategies)
        settings.prediction.confidence_threshold = pred_cfg.get("confidence_threshold", settings.prediction.confidence_threshold)
        settings.prediction.prediction_horizon_days = pred_cfg.get("prediction_horizon_days", settings.prediction.prediction_horizon_days)
    
    # Merge Notification settings
    if "notification" in yaml_data:
        notif_cfg = yaml_data["notification"]
        settings.notification.enable_discord = notif_cfg.get("enable_discord", settings.notification.enable_discord)
        settings.notification.enable_email = notif_cfg.get("enable_email", settings.notification.enable_email)
        settings.notification.email_recipients
