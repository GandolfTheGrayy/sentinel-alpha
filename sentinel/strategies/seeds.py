"""The seed population: a few hand-picked variants per family, plus the controls."""
from __future__ import annotations

from typing import Any

SEEDS: list[dict[str, Any]] = [
    {"family": "trend_ema", "params": {"tf": "15m", "fast": 12, "slow": 40, "adx_min": 22.0}, "markets": ["equities", "crypto"]},
    {"family": "trend_ema", "params": {"tf": "1h", "fast": 9, "slow": 30, "adx_min": 20.0, "trail_atr": 3.0, "max_hold": 150}, "markets": ["equities", "crypto"]},
    {"family": "meanrev_bb", "params": {"tf": "15m", "bb_n": 20, "bb_k": 2.0, "rsi_lo": 10.0, "rsi_hi": 90.0, "trend_filter": 1}, "markets": ["equities"]},
    {"family": "meanrev_bb", "params": {"tf": "5m", "bb_n": 20, "bb_k": 2.5, "rsi_lo": 8.0, "rsi_hi": 92.0, "trend_filter": 0, "max_hold": 24}, "markets": ["equities"]},
    {"family": "breakout_orb", "params": {"or_minutes": 30, "vol_mult": 1.2, "r_mult": 2.0}, "markets": ["equities"]},
    {"family": "breakout_orb", "params": {"or_minutes": 15, "vol_mult": 1.5, "r_mult": 1.5, "latest_entry_min": 90}, "markets": ["equities"]},
    {"family": "breakout_donchian", "params": {"tf": "1h", "n": 20, "trail_atr": 3.0}, "markets": ["equities", "crypto"]},
    {"family": "breakout_donchian", "params": {"tf": "4h", "n": 30, "trail_atr": 4.0, "max_hold": 300}, "markets": ["crypto"]},
    {"family": "vwap_revert", "params": {"tf": "5m", "z": 2.0, "vol_mult": 1.3}, "markets": ["equities"]},
    {"family": "squeeze", "params": {"tf": "15m", "min_squeeze": 6, "r_mult": 2.0}, "markets": ["equities", "crypto"]},
    {"family": "squeeze", "params": {"tf": "1h", "min_squeeze": 4, "r_mult": 3.0, "max_hold": 120}, "markets": ["equities", "crypto"]},
    {"family": "xs_momentum", "params": {"lookback": 60, "skip": 5, "n_long": 4}, "markets": ["equities"]},
    {"family": "overnight", "params": {"n_symbols": 4, "trend_sma": 50}, "markets": ["equities"]},
    {"family": "crypto_mtf", "params": {"trend_ema": 50, "fast_ema": 20, "rsi_pullback": 35.0}, "markets": ["crypto"]},
    {"family": "random_entry", "params": {}, "markets": ["equities", "crypto"], "is_control": True},
    {"family": "buy_hold", "params": {}, "markets": ["equities", "crypto"], "is_control": True},
]
