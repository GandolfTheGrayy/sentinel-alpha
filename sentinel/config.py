"""Typed configuration loaded from config.yaml (+ .env for secrets).

Everything has a sensible default so `python -m sentinel run --sim` works with no
config file at all. Secrets are read from the environment only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent

PLAN_WEEKLY_ALLOWANCE_USD = {
    # Conservative estimates of a plan's weekly allowance expressed in API-equivalent USD.
    # They only matter for the budget cap; calibrate from `/usage` after the first week.
    "pro": 50.0,
    "max5": 250.0,
    "max20": 1000.0,
    "custom": 250.0,
}


class BrokerCfg(BaseModel):
    provider: Literal["alpaca", "sim"] = "alpaca"
    mode: Literal["paper", "live"] = "paper"
    starting_equity: float = 100_000.0


class DataCfg(BaseModel):
    backfill_days_minute: int = 30
    backfill_days_daily: int = 400
    poll_interval_s: int = 60
    feed: str = "iex"


class UniverseCfg(BaseModel):
    equities: list[str] = Field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "AMD", "AVGO", "NFLX", "JPM", "XOM", "XLF", "XLE", "XLK", "TLT", "GLD",
    ])
    crypto: list[str] = Field(default_factory=lambda: ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "AVAX/USD"])
    benchmark_equity: str = "SPY"
    benchmark_crypto: str = "BTC/USD"

    @property
    def all_symbols(self) -> list[str]:
        return list(self.equities) + list(self.crypto)


class RiskCfg(BaseModel):
    risk_per_trade_pct: float = 1.0
    max_position_pct_of_sleeve: float = 60.0
    max_position_pct_of_equity: float = 10.0
    max_open_per_variant: int = 4
    max_gross_exposure_pct: float = 100.0
    daily_loss_pause_pct: float = 3.0
    daily_loss_halt_pct: float = 6.0
    variant_drawdown_pause_pct: float = 15.0
    allow_shorts_equities: bool = True
    extended_hours: bool = False


class CostsCfg(BaseModel):
    equity_slippage_bps: float = 2.0
    crypto_slippage_bps: float = 8.0
    equity_fee_bps: float = 0.0
    crypto_fee_bps: float = 25.0


class PopulationCfg(BaseModel):
    max_variants: int = 24
    incubation_min_trades: int = 30
    incubation_max_days: int = 21
    exploration_floor: float = 0.02
    allocation_cap: float = 0.20
    control_weight: float = 0.01


class SchedulesCfg(BaseModel):
    metrics_every_minutes: int = 15
    tournament: str = "16:35"
    attribution: str = "16:40"
    optimizer: str = "Sat 09:00"
    analyst: str = "16:45"
    strategist: str = "Sun 10:00"


class LearnCfg(BaseModel):
    min_trades_for_attribution: int = 30
    attribution_window_days: int = 45
    backtest_days: int = 45
    walk_forward_splits: int = 3
    optimizer_trials: int = 24


class ClaudeSessionCfg(BaseModel):
    model: str = "sonnet"
    max_budget_usd: float = 1.5
    max_turns: int = 15
    min_new_trades: int = 0


class ClaudeCfg(BaseModel):
    enabled: bool = True
    cli: str = "claude"
    plan: Literal["pro", "max5", "max20", "custom"] = "max5"
    weekly_share: float = 0.25
    weekly_allowance_usd_est: float | None = None
    allow_code_proposals: bool = True
    analyst: ClaudeSessionCfg = Field(default_factory=lambda: ClaudeSessionCfg(model="sonnet", max_budget_usd=1.5, max_turns=15, min_new_trades=10))
    strategist: ClaudeSessionCfg = Field(default_factory=lambda: ClaudeSessionCfg(model="opus", max_budget_usd=6.0, max_turns=30, min_new_trades=0))

    @property
    def weekly_allowance_usd(self) -> float:
        if self.weekly_allowance_usd_est:
            return float(self.weekly_allowance_usd_est)
        return PLAN_WEEKLY_ALLOWANCE_USD[self.plan]

    @property
    def weekly_cap_usd(self) -> float:
        return round(self.weekly_allowance_usd * self.weekly_share, 2)


class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8787


class Settings(BaseModel):
    broker: BrokerCfg = Field(default_factory=BrokerCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    universe: UniverseCfg = Field(default_factory=UniverseCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    costs: CostsCfg = Field(default_factory=CostsCfg)
    population: PopulationCfg = Field(default_factory=PopulationCfg)
    schedules: SchedulesCfg = Field(default_factory=SchedulesCfg)
    learn: LearnCfg = Field(default_factory=LearnCfg)
    claude: ClaudeCfg = Field(default_factory=ClaudeCfg)
    server: ServerCfg = Field(default_factory=ServerCfg)

    # runtime (not from yaml)
    data_dir: Path = Field(default_factory=lambda: ROOT / "data")
    sim: bool = False

    @property
    def db_path(self) -> Path:
        """Sim and paper/live never share a database."""
        return self.data_dir / ("sentinel-sim.db" if self.sim else "sentinel.db")

    @property
    def lab_dir(self) -> Path:
        return ROOT / ("lab-sim" if self.sim else "lab")

    @property
    def mode(self) -> str:
        if self.sim or self.broker.provider == "sim":
            return "sim"
        return self.broker.mode

    def sanitized(self) -> dict[str, Any]:
        d = self.model_dump(mode="json", exclude={"data_dir"})
        d["mode"] = self.mode
        d["db_path"] = str(self.db_path.name)
        return d


def alpaca_keys() -> tuple[str | None, str | None]:
    return os.environ.get("ALPACA_API_KEY") or None, os.environ.get("ALPACA_SECRET_KEY") or None


def load_settings(path: str | Path | None = None, *, sim: bool = False) -> Settings:
    """Load config.yaml (or the given path) merged over defaults; .env is loaded for secrets."""
    load_dotenv(ROOT / ".env", override=False)
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    settings = Settings.model_validate(raw)
    settings.sim = sim or settings.broker.provider == "sim" or os.environ.get("SENTINEL_SIM") == "1"
    if settings.broker.mode == "live" and os.environ.get("SENTINEL_CONFIRM_LIVE") != "yes" and not settings.sim:
        raise RuntimeError("broker.mode is 'live' but SENTINEL_CONFIRM_LIVE=yes is not set; refusing to start")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.lab_dir.mkdir(parents=True, exist_ok=True)
    return settings
