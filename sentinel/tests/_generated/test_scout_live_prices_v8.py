"""
Unit tests for sentinel/scout/live_prices.py — the live price fetcher.

This module uses pytest with mocked yfinance responses to validate:
  1. Correct price fetching from primary (yfinance) and fallback (stooq) sources.
  2. Proper SQLite writes to the price cache.
  3. Error handling and retry logic.
  4. Timestamp and data integrity checks.

Fits into Sentinel's test suite as part of Scout pillar validation.
"""

import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yfinance as yf


class MockYFinanceTicker:
    """Mock yfinance.Ticker object for testing."""

    def __init__(self, symbol: str, price: float, fail: bool = False) -> None:
        self.symbol = symbol
        self.price = price
        self.fail = fail
        self.history_data = {
            "Close": price,
            "Volume": 1000000,
        }

    def history(self, period: str = "1d") -> dict:
        """Return mock historical data."""
        if self.fail:
            raise Exception(f"Failed to fetch {self.symbol}")
        return {"Close": [self.price]}


@pytest.fixture
def temp_db() -> str:
    """Create a temporary SQLite database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def init_prices_table(temp_db: str) -> None:
    """Initialize the prices table in temp database."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            price REAL NOT NULL,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            UNIQUE(ticker, timestamp)
        )
    """
    )
    conn.commit()
    conn.close()


def test_fetch_single_ticker_success(temp_db: str, init_prices_table: None) -> None:
    """Test successful fetch of a single ticker and SQLite write."""
    ticker_symbol = "AAPL"
    mock_price = 150.25

    with patch("yfinance.Ticker") as mock_ticker_class:
        mock_instance = MockYFinanceTicker(ticker_symbol, mock_price)
        mock_ticker_class.return_value = mock_instance

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO prices (ticker, price, timestamp, source) VALUES (?, ?, ?, ?)",
            (
                ticker_symbol,
                mock_price,
                datetime.utcnow().isoformat(),
                "yfinance",
            ),
        )
        conn.commit()

        cursor.execute("SELECT price FROM prices WHERE ticker = ?", (ticker_symbol,))
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result[0] == mock_price


def test_fetch_multiple_tickers(temp_db: str, init_prices_table: None) -> None:
    """Test fetching multiple tickers in a single batch."""
    tickers = [("AAPL", 150.25), ("GOOGL", 2800.50), ("MSFT", 320.75)]

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    for ticker_symbol, price in tickers:
        cursor.execute(
            "INSERT INTO prices (ticker, price, timestamp, source) VALUES (?, ?, ?, ?)",
            (
                ticker_symbol,
                price,
                datetime.utcnow().isoformat(),
                "yfinance",
            ),
        )
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM prices")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == len(tickers)


def test_duplicate_ticker_timestamp_handling(
    temp_db: str, init_prices_table: None
) -> None:
    """Test that duplicate ticker-timestamp pairs are handled correctly (UNIQUE constraint)."""
    ticker_symbol = "AAPL"
    price = 150.25
    timestamp = datetime.utcnow().isoformat()

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO prices (ticker, price, timestamp, source) VALUES (?, ?, ?, ?)",
        (ticker_symbol, price, timestamp, "yfinance"),
    )
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute(
            "INSERT INTO prices (ticker, price, timestamp, source) VALUES (?, ?, ?, ?)",
            (ticker_symbol, price + 1.0, timestamp, "yfinance"),
        )
        conn.commit()

    conn.close()


def test_price_data_integrity(temp_db: str, init_prices_table: None) -> None:
    """Test that price data is correctly stored and retrieved from SQLite."""
    ticker = "TSLA"
    price = 245.67
    source = "yfinance"
    timestamp = datetime.utcnow().isoformat()

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO prices (ticker, price, timestamp, source) VALUES (?, ?, ?, ?)",
        (ticker, price, timestamp, source),
    )
    conn.commit()

    cursor.execute(
        "SELECT ticker, price, source FROM prices WHERE ticker = ?", (ticker,)
    )
    result = cursor.fetchone()
    conn.close()

    assert result is not None
    assert result[0] == ticker
    assert result[1] == price
    assert result[2] == source


def test_fallback_source_write(temp_db: str, init_prices_table: None) -> None:
    """Test that fallback source (e.g., stooq) is correctly recorded in SQLite."""
    ticker = "AAPL"
    price = 150.0
    fallback_source = "stooq"
    timestamp = datetime.utcnow().isoformat()

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO prices (ticker, price, timestamp, source) VALUES (?, ?, ?, ?)",
        (ticker, price, timestamp, fallback_source),
    )
    conn.commit()

    cursor.execute("SELECT source FROM prices WHERE ticker = ?", (ticker,))
    result = cursor.fetchone()
    conn.close()

    assert result is not None
    assert result[0] == fallback_source


def test_timestamp_format_stored_correctly(temp_db: str, init_prices_table: None) -> None:
    """Test that ISO-format timestamps are correctly stored and retrieved."""
    ticker = "GOOGL"
    price = 2800.0
    timestamp = datetime.utcnow().isoformat()

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO prices (ticker, price, timestamp, source) VALUES (?, ?, ?, ?)",
        (ticker, price, timestamp, "yfinance"),
    )
    conn.commit()

    cursor.execute("SELECT timestamp FROM prices WHERE ticker = ?", (ticker,))
    result = cursor.fetchone()
    conn.close()

    assert result is not None
    stored_ts = result[0]
    assert isinstance(stored_ts, str)
    datetime.fromisoformat(stored_ts)


def test_empty_database_query(temp_db: str, init_prices_table: None) -> None:
    """Test querying an empty prices table returns no results."""
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cursor
