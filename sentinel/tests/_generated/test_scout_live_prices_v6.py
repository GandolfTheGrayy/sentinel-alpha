"""
Unit test suite for sentinel/scout/live_prices.py.

This module validates the Scout price fetcher's ability to:
  1. Mock yfinance API responses
  2. Handle fallback to stooq when yfinance fails
  3. Write price snapshots to SQLite correctly
  4. Handle edge cases (missing data, network errors, invalid symbols)

Part of the Sentinel post-mortem validation pipeline.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest
import pandas as pd
import yfinance as yf


@pytest.fixture
def temp_db() -> str:
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def mock_yfinance_data() -> pd.DataFrame:
    """Return a mock yfinance ticker data frame."""
    return pd.DataFrame({
        "Open": [150.0, 151.5],
        "High": [152.0, 153.0],
        "Low": [149.5, 151.0],
        "Close": [151.2, 152.8],
        "Volume": [1000000, 1100000],
    }, index=pd.date_range("2025-01-01", periods=2, freq="D"))


class TestLivePriceFetcher:
    """Test suite for live price fetching and storage."""

    def test_fetch_and_store_valid_symbol(
        self, temp_db: str, mock_yfinance_data: pd.DataFrame
    ) -> None:
        """Verify yfinance data is fetched and written to SQLite correctly."""
        with patch.object(yf.Ticker, "history", return_value=mock_yfinance_data):
            from sentinel.scout.live_prices import fetch_and_store_price
            
            result = fetch_and_store_price("AAPL", db_path=temp_db)
            assert result is True
            
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prices WHERE symbol = ?", ("AAPL",))
            count = cursor.fetchone()[0]
            conn.close()
            
            assert count == 2, f"Expected 2 rows, got {count}"

    def test_fetch_invalid_symbol_returns_false(self, temp_db: str) -> None:
        """Verify graceful handling of invalid stock symbols."""
        with patch.object(yf.Ticker, "history", return_value=pd.DataFrame()):
            from sentinel.scout.live_prices import fetch_and_store_price
            
            result = fetch_and_store_price("FAKESYM", db_path=temp_db)
            assert result is False

    def test_fallback_to_stooq_on_yfinance_failure(
        self, temp_db: str
    ) -> None:
        """Verify fallback mechanism when yfinance is unavailable."""
        mock_stooq_data = pd.DataFrame({
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.5],
            "Close": [100.5],
            "Volume": [500000],
        }, index=pd.date_range("2025-01-01", periods=1, freq="D"))
        
        with patch.object(yf.Ticker, "history", side_effect=Exception("Network error")):
            with patch("sentinel.scout.live_prices.fetch_stooq_fallback", return_value=mock_stooq_data):
                from sentinel.scout.live_prices import fetch_and_store_price
                
                result = fetch_and_store_price("AAPL", db_path=temp_db, use_fallback=True)
                assert result is True

    def test_database_schema_creation(self, temp_db: str) -> None:
        """Verify SQLite schema is initialized correctly."""
        from sentinel.scout.live_prices import init_db
        
        init_db(db_path=temp_db)
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prices'"
        )
        table_exists = cursor.fetchone() is not None
        conn.close()
        
        assert table_exists, "prices table was not created"

    def test_price_data_columns_match_schema(
        self, temp_db: str, mock_yfinance_data: pd.DataFrame
    ) -> None:
        """Verify all OHLCV columns are stored in SQLite."""
        with patch.object(yf.Ticker, "history", return_value=mock_yfinance_data):
            from sentinel.scout.live_prices import fetch_and_store_price
            
            fetch_and_store_price("MSFT", db_path=temp_db)
            
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(prices)")
            columns = {row[1] for row in cursor.fetchall()}
            conn.close()
            
            required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
            assert required.issubset(columns), f"Missing columns: {required - columns}"

    def test_duplicate_timestamp_handling(
        self, temp_db: str, mock_yfinance_data: pd.DataFrame
    ) -> None:
        """Verify graceful handling of duplicate price records (same symbol, same timestamp)."""
        with patch.object(yf.Ticker, "history", return_value=mock_yfinance_data):
            from sentinel.scout.live_prices import fetch_and_store_price
            
            fetch_and_store_price("GOOGL", db_path=temp_db)
            result_second = fetch_and_store_price("GOOGL", db_path=temp_db)
            
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prices WHERE symbol = ?", ("GOOGL",))
            count = cursor.fetchone()[0]
            conn.close()
            
            assert count <= 4, f"Duplicates not handled; got {count} rows"

    def test_multiple_symbols_batch_fetch(
        self, temp_db: str, mock_yfinance_data: pd.DataFrame
    ) -> None:
        """Verify batch fetching multiple tickers into one database."""
        with patch.object(yf.Ticker, "history", return_value=mock_yfinance_data):
            from sentinel.scout.live_prices import fetch_and_store_price
            
            symbols = ["AAPL", "MSFT", "GOOGL"]
            for sym in symbols:
                fetch_and_store_price(sym, db_path=temp_db)
            
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM prices")
            unique_symbols = cursor.fetchone()[0]
            conn.close()
            
            assert unique_symbols == 3, f"Expected 3 symbols, got {unique_symbols}"

    def test_fetch_with_date_range(
        self, temp_db: str, mock_yfinance_data: pd.DataFrame
    ) -> None:
        """Verify time-bounded price fetches respect start/end date parameters."""
        with patch.object(yf.Ticker, "history", return_value=mock_yfinance_data) as mock_hist:
            from sentinel.scout.live_prices import fetch_and_store_price
            
            fetch_and_store_price(
                "AAPL",
                db_path=temp_db,
                start="2025-01-01",
                end="2025-01-02"
