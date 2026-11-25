"""Vectorised technical indicators on pandas Series / OHLCV DataFrames.

DataFrames are expected to have columns o, h, l, c, v and a UTC DatetimeIndex.
All functions are pure and NaN-safe at the warm-up edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder smoothing
    au = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(ad != 0.0, 100.0)
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_c = df["c"].shift(1)
    tr = pd.concat([df["h"] - df["l"], (df["h"] - prev_c).abs(), (df["l"] - prev_c).abs()], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up_move = df["h"].diff()
    down_move = -df["l"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_n = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_n.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_n.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def bollinger_pos(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """Position of close within the bands: 0 = lower band, 1 = upper band."""
    lo, mid, hi = bollinger(close, n, k)
    width = (hi - lo).replace(0.0, np.nan)
    return (close - lo) / width


def keltner(df: pd.DataFrame, n: int = 20, mult: float = 1.5) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(df["c"], n)
    a = atr(df, n)
    return mid - mult * a, mid, mid + mult * a


def donchian(df: pd.DataFrame, n: int = 20) -> tuple[pd.Series, pd.Series]:
    """Upper/lower channel over the previous n bars (excluding the current bar)."""
    return df["h"].shift(1).rolling(n, min_periods=n).max(), df["l"].shift(1).rolling(n, min_periods=n).min()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP reset at each America/New_York session date."""
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    typical = (df["h"] + df["l"] + df["c"]) / 3.0
    vol = df["v"].fillna(0.0)
    ny_date = df.index.tz_convert("America/New_York").date
    pv = (typical * vol).groupby(ny_date).cumsum()
    vv = vol.groupby(ny_date).cumsum().replace(0.0, np.nan)
    out = pv / vv
    out.index = df.index
    return out


def realized_vol(close: pd.Series, n: int = 20, periods_per_year: float = 252.0) -> pd.Series:
    r = np.log(close).diff()
    return r.rolling(n, min_periods=n).std(ddof=0) * np.sqrt(periods_per_year)


def returns(close: pd.Series, n: int) -> pd.Series:
    return close / close.shift(n) - 1.0


def volume_ratio(vol: pd.Series, n: int = 20) -> pd.Series:
    avg = vol.shift(1).rolling(n, min_periods=max(2, n // 2)).mean()
    return vol / avg.replace(0.0, np.nan)


def slope(s: pd.Series, n: int) -> pd.Series:
    """Normalised slope: (s - s.shift(n)) / (n * s)."""
    return (s - s.shift(n)) / (n * s.replace(0.0, np.nan))
