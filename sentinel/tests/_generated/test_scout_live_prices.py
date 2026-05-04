"""
Unit tests for the Scout live price fetcher module.

This test module validates that sentinel/scout/live_prices.py correctly:
1. Mocks yfinance API responses for historical and real-time price data.
2. Writes fetched prices to the SQLite price cache with proper schema.
3. Handles fallback to stooq when yfinance unavailable.
4. Gracefully degrades on network errors.
5. Validates ticker symbols and date ranges before querying.

Part of the Sentinel post-mortem validation pipeline: ensures Scout
data ingestion integrity before Linguist and Judge consume prices.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

import pytest
import pandas as pd
import yfinance as yf


# Mock data fixtures
@pytest.fixture
def mock_prices_df() -> pd.DataFrame:
    """Return mock price DataFrame matching yfinance.Ticker.history() schema."""
    dates = pd.date_range(start="2025-01-01", periods=5, freq="D")
    return pd.DataFrame({
        "Open": [100.0, 101.0, 102.0, 101.5, 103.0],
        "High": [101.0, 102.0, 103.0, 102.5, 104.0],
        "Low": [99.5, 100.5, 101.5, 101.0, 102.5],
        "Close": [100.5, 101.5, 102.5, 102.0, 103.5],
        "Volume": [1000000, 1100000, 950000, 1050000, 1200000],
    }, index=dates)


@pytest.fixture
def temp_db() -> str:
    """Return path to temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def initialized_db(temp_db: str) -> str:
    """Initialize a temporary SQLite database with price schema."""
    conn = sqlite3.connect(temp_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()
    conn.close()
    return temp_db


class TestScoutPriceFetcher:
    """Test suite for Scout live price fetcher."""

    def test_fetch_and_write_prices_yfinance(
        self, initialized_db: str, mock_prices_df: pd.DataFrame
    ) -> None:
        """Verify yfinance prices are fetched and written to SQLite correctly."""
        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_prices_df
            mock_ticker_class.return_value = mock_ticker

            from sentinel.scout import live_prices

            result = live_prices.fetch_and_cache_prices(
                ticker="AAPL",
                db_path=initialized_db,
                days_back=5,
            )

            # Verify return structure
            assert result is not None
            assert len(result) == 5, "Expected 5 price records"

            # Verify SQLite writes
            conn = sqlite3.connect(initialized_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prices WHERE ticker = ?", ("AAPL",))
            count = cursor.fetchone()[0]
            assert count == 5, "Expected 5 rows in SQLite for AAPL"

            # Verify data integrity
            cursor.execute(
                "SELECT close FROM prices WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                ("AAPL",),
            )
            latest_close = cursor.fetchone()[0]
            assert latest_close == 103.5, "Latest close price mismatch"
            conn.close()

    def test_fetch_prices_invalid_ticker(self) -> None:
        """Verify fetch_and_cache_prices rejects invalid tickers."""
        from sentinel.scout import live_prices

        with pytest.raises(ValueError, match="Invalid ticker"):
            live_prices.fetch_and_cache_prices(
                ticker="INVALID@#$",
                db_path=":memory:",
            )

    def test_fetch_prices_date_range_validation(self) -> None:
        """Verify fetch_and_cache_prices rejects invalid date ranges."""
        from sentinel.scout import live_prices

        with pytest.raises(ValueError, match="days_back must be positive"):
            live_prices.fetch_and_cache_prices(
                ticker="AAPL",
                db_path=":memory:",
                days_back=-5,
            )

    def test_yfinance_network_error_fallback(
        self, initialized_db: str
    ) -> None:
        """Verify graceful fallback to stooq on yfinance network error."""
        mock_stooq_df = pd.DataFrame({
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.5],
            "Close": [100.5],
            "Volume": [1000000],
        }, index=pd.date_range(start="2025-01-01", periods=1, freq="D"))

        with patch("yfinance.Ticker") as mock_yf, \
             patch("yfinance.download") as mock_stooq:
            # yfinance raises connection error
            mock_yf.side_effect = ConnectionError("Network unreachable")
            mock_stooq.return_value = mock_stooq_df

            from sentinel.scout import live_prices

            result = live_prices.fetch_and_cache_prices(
                ticker="AAPL",
                db_path=initialized_db,
                fallback_source="stooq",
            )

            assert result is not None
            assert len(result) == 1, "Expected 1 record from stooq fallback"

    def test_write_duplicate_prices_upsert(
        self, initialized_db: str, mock_prices_df: pd.DataFrame
    ) -> None:
        """Verify duplicate date entries are upserted (not duplicated)."""
        conn = sqlite3.connect(initialized_db)
        conn.execute("""
            INSERT INTO prices (ticker, date, close, open, high, low, volume)
            VALUES ('AAPL', ?, 100.0, 99.5, 101.0, 99.0, 500000)
        """, (mock_prices_df.index[0].strftime("%Y-%m-%d"),))
        conn.commit()
        conn.close()

        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = mock_prices_df
            mock_ticker_class.return_value = mock_ticker

            from sentinel.scout import live_prices

            live_prices.fetch_and_cache_prices(
                ticker="AAPL",
                db_path=initialized_db,
                days_back=5,
            )

            conn = sqlite3.connect(initialized_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prices WHERE ticker = 'AAPL'")
            count = cursor.fetchone()[0]
            conn.close()

            assert count == 5, "Expected 5 rows (upsert, not duplicate insert)"

    def test_fetch_prices_empty_response(
        self, initialized_db: str
    ) -> None:
        """Verify graceful handling of empty price responses."""
        empty_df = pd.DataFrame()

        with patch("yfinance.Ticker") as mock_ticker_class:
            mock_ticker = MagicM
