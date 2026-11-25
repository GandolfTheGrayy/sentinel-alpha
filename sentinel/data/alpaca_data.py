"""Alpaca market-data provider (historical bars + latest prices) for stocks and crypto."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from sentinel.data.market import COLS, empty_frame, is_crypto
from sentinel.util.clock import NY, UTC


class AlpacaProvider:
    def __init__(self, api_key: str, secret_key: str, feed: str = "iex", extended_hours: bool = False):
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

        self.stocks = StockHistoricalDataClient(api_key, secret_key)
        self.crypto = CryptoHistoricalDataClient(api_key, secret_key)
        self.feed = DataFeed(feed)
        self.extended_hours = extended_hours

    @staticmethod
    def _timeframe(tf: str):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        return {"1m": TimeFrame(1, TimeFrameUnit.Minute), "1d": TimeFrame(1, TimeFrameUnit.Day)}[tf]

    def fetch_bars(self, symbols: list[str], tf: str, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest

        out: dict[str, pd.DataFrame] = {s: empty_frame() for s in symbols}
        eq = [s for s in symbols if not is_crypto(s)]
        cr = [s for s in symbols if is_crypto(s)]
        # Alpaca rejects end times too close to now on the free feed; back off a little.
        end = min(end, datetime.now(UTC) - timedelta(seconds=20)) if tf == "1m" else end
        if eq:
            req = StockBarsRequest(symbol_or_symbols=eq, timeframe=self._timeframe(tf), start=start, end=end, feed=self.feed, adjustment="split" if tf == "1d" else None)
            bars = self.stocks.get_stock_bars(req)
            out.update(self._frames(bars, eq, regular_session_only=(tf == "1m" and not self.extended_hours)))
        if cr:
            req = CryptoBarsRequest(symbol_or_symbols=cr, timeframe=self._timeframe(tf), start=start, end=end)
            bars = self.crypto.get_crypto_bars(req)
            out.update(self._frames(bars, cr))
        return out

    @staticmethod
    def _frames(barset, symbols: list[str], regular_session_only: bool = False) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        data = barset.data if hasattr(barset, "data") else {}
        for s in symbols:
            items = data.get(s) or []
            if not items:
                out[s] = empty_frame()
                continue
            rows = [{"ts": b.timestamp, "o": b.open, "h": b.high, "l": b.low, "c": b.close, "v": b.volume or 0.0, "vwap": b.vwap} for b in items]
            df = pd.DataFrame(rows)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.set_index("ts").sort_index()
            df = df[~df.index.duplicated(keep="last")]
            if regular_session_only and not is_crypto(s):
                # IEX minute bars include pre/post-market prints; intraday indicators are built on the regular session only
                ny = df.index.tz_convert(NY)
                mins = ny.hour * 60 + ny.minute
                df = df[(mins >= 9 * 60 + 30) & (mins < 16 * 60) & (ny.weekday < 5)]
            out[s] = df[COLS].astype(float)
        return out

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        from alpaca.data.requests import CryptoLatestTradeRequest, StockLatestTradeRequest

        out: dict[str, float] = {}
        eq = [s for s in symbols if not is_crypto(s)]
        cr = [s for s in symbols if is_crypto(s)]
        if eq:
            trades = self.stocks.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=eq, feed=self.feed))
            for s, t in trades.items():
                out[s] = float(t.price)
        if cr:
            trades = self.crypto.get_crypto_latest_trade(CryptoLatestTradeRequest(symbol_or_symbols=cr))
            for s, t in trades.items():
                out[s] = float(t.price)
        return out
