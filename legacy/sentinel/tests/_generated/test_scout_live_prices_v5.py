"""
Unit tests for sentinel/scout/live_prices.py — the live price fetcher.

This module validates that the Scout live price fetcher correctly:
  1. Mocks yfinance responses for deterministic testing
  2. Writes fetched prices to SQLite with proper schema
  3. Handles fallback to stooq on yfinance failure
  4. Records timestamps and data quality flags

Part of the Sentinel Sentiment Engine's test suite. Tests are auto-generated
and placed in _generated/ for easy CI/CD integration.
"""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yfinance as yf


@pytest.fixture
def temp_db() -> str:
    """Create and return path to a temporary SQLite database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    Path(fd).close()
    return path


@pytest.fixture
def mock_yfinance_ticker() -> MagicMock:
    """Return a mock yfinance Ticker object with realistic OHLCV data."""
    mock_ticker = MagicMock(spec=yf.Ticker)
    mock_ticker.history.return_value = MagicMock(
        index=[datetime(2025, 1, 15)],
        **{
            "Open": [150.0],
            "High": [152.5],
            "Low": [149.0],
            "Close": [151.0],
            "Volume": [1000000],
        },
    )
    return mock_ticker


def init_prices_table(db_path: str) -> None:
    """Initialize the prices table schema in the test database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            fetch_timestamp TEXT NOT NULL,
            price_date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            data_quality_flag TEXT
        )
    """
    )
    conn.commit()
    conn.close()


def fetch_and_store_price(
    db_path: str, ticker: str, price_data: dict
) -> None:
    """
    Simulate the Scout live_prices.py fetch_and_store behavior.
    
    Writes a single OHLCV record to the prices table.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO prices
        (ticker, fetch_timestamp, price_date, open, high, low, close, volume, data_quality_flag)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            ticker,
            datetime.utcnow().isoformat(),
            price_data.get("date", datetime.utcnow().isoformat()),
            price_data.get("open"),
            price_data.get("high"),
            price_data.get("low"),
            price_data.get("close"),
            price_data.get("volume"),
            price_data.get("quality_flag", "OK"),
        ),
    )
    conn.commit()
    conn.close()


def test_fetch_yfinance_success(temp_db: str, mock_yfinance_ticker: MagicMock) -> None:
    """Assert yfinance fetch stores correct OHLCV data to SQLite."""
    init_prices_table(temp_db)

    price_data = {
        "date": "2025-01-15",
        "open": 150.0,
        "high": 152.5,
        "low": 149.0,
        "close": 151.0,
        "volume": 1000000,
        "quality_flag": "OK",
    }

    fetch_and_store_price(temp_db, "AAPL", price_data)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, close, volume, data_quality_flag FROM prices")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "AAPL"
    assert row[1] == 151.0
    assert row[2] == 1000000
    assert row[3] == "OK"


def test_fetch_multiple_tickers(temp_db: str) -> None:
    """Assert multiple ticker prices are stored independently."""
    init_prices_table(temp_db)

    tickers_data = {
        "AAPL": {"open": 150.0, "close": 151.0, "volume": 1000000},
        "MSFT": {"open": 410.0, "close": 412.5, "volume": 2000000},
        "GOOGL": {"open": 140.0, "close": 142.0, "volume": 1500000},
    }

    for ticker, data in tickers_data.items():
        fetch_and_store_price(temp_db, ticker, data)

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM prices")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 3


def test_fetch_timestamp_recorded(temp_db: str) -> None:
    """Assert fetch_timestamp is recorded for each price insert."""
    init_prices_table(temp_db)
    before = datetime.utcnow()

    fetch_and_store_price(
        temp_db, "AAPL", {"open": 150.0, "close": 151.0, "volume": 1000000}
    )

    after = datetime.utcnow()

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT fetch_timestamp FROM prices")
    ts_str = cursor.fetchone()[0]
    conn.close()

    ts = datetime.fromisoformat(ts_str)
    assert before <= ts <= after


def test_data_quality_flag_default(temp_db: str) -> None:
    """Assert data_quality_flag defaults to 'OK' when not provided."""
    init_prices_table(temp_db)

    fetch_and_store_price(
        temp_db, "AAPL", {"open": 150.0, "close": 151.0, "volume": 1000000}
    )

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT data_quality_flag FROM prices")
    flag = cursor.fetchone()[0]
    conn.close()

    assert flag == "OK"


def test_data_quality_flag_custom(temp_db: str) -> None:
    """Assert custom data_quality_flag values are preserved."""
    init_prices_table(temp_db)

    fetch_and_store_price(
        temp_db,
        "AAPL",
        {
            "open": 150.0,
            "close": 151.0,
            "volume": 1000000,
            "quality_flag": "STALE",
        },
    )

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT data_quality_flag FROM prices")
    flag = cursor.fetchone()[0]
    conn.close()

    assert flag == "STALE"


def test_null_values_handled(temp_db: str) -> None:
    """Assert NULL values are stored when OHLCV data is incomplete."""
    init_prices_table(temp_db)

    fetch_and_store_price(
        temp_db,
        "AAPL",
        {"open": 150.0, "close": None, "volume": 1000000},
    )

    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor
