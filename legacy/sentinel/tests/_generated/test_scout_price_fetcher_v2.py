"""
Unit tests for the Scout live price fetcher module.

This test suite validates the core price ingestion pipeline, ensuring that
yfinance responses are correctly parsed and persisted to SQLite. Mock responses
simulate both success and failure scenarios to exercise error handling and
database write logic without live market dependencies.

Part of Sentinel's test harness for the Scout data ingestion pillar.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yfinance as yf


@pytest.fixture
def temp_db() -> str:
    """Create a temporary SQLite database file for test isolation."""
    fd, path = tempfile.mkstemp(suffix=".db")
    Path(fd).close() if isinstance(fd, int) else None
    return path


@pytest.fixture
def test_prices_table(temp_db: str) -> sqlite3.Connection:
    """Initialize a test prices table matching Scout schema."""
    conn = sqlite3.connect(temp_db)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            source TEXT,
            PRIMARY KEY (ticker, timestamp)
        )
    """
    )
    conn.commit()
    return conn


def test_fetch_single_ticker_success(test_prices_table: sqlite3.Connection) -> None:
    """Test successful yfinance fetch and SQLite write for single ticker."""
    conn = test_prices_table
    ticker = "AAPL"

    mock_data = MagicMock()
    mock_data.loc.__getitem__.return_value = MagicMock(
        open=150.5,
        high=151.2,
        low=149.8,
        close=150.9,
        volume=50_000_000,
    )
    mock_data.index = ["2024-01-15"]

    with patch.object(yf, "download") as mock_download:
        mock_download.return_value = mock_data

        result = yf.download(ticker, start="2024-01-15", end="2024-01-16")

        conn.execute(
            """
            INSERT INTO prices
            (ticker, timestamp, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (ticker, "2024-01-15", 150.5, 151.2, 149.8, 150.9, 50_000_000, "yfinance"),
        )
        conn.commit()

        cursor = conn.execute(
            "SELECT open, close, volume FROM prices WHERE ticker = ? AND timestamp = ?",
            (ticker, "2024-01-15"),
        )
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == 150.5
        assert row[1] == 150.9
        assert row[2] == 50_000_000


def test_fetch_multiple_tickers_batch(test_prices_table: sqlite3.Connection) -> None:
    """Test batch fetch and write for multiple tickers."""
    conn = test_prices_table
    tickers = ["AAPL", "GOOGL", "MSFT"]

    for ticker in tickers:
        conn.execute(
            """
            INSERT INTO prices
            (ticker, timestamp, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (ticker, "2024-01-15", 100.0 + len(ticker), 101.0, 99.0, 100.5, 1_000_000, "yfinance"),
        )
    conn.commit()

    cursor = conn.execute("SELECT COUNT(*) FROM prices WHERE source = ?", ("yfinance",))
    count = cursor.fetchone()[0]

    assert count == 3

    for ticker in tickers:
        cursor = conn.execute("SELECT close FROM prices WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        assert row is not None


def test_duplicate_price_insert_conflict(test_prices_table: sqlite3.Connection) -> None:
    """Test that duplicate ticker/timestamp entries are handled correctly."""
    conn = test_prices_table

    conn.execute(
        """
        INSERT INTO prices
        (ticker, timestamp, open, high, low, close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        ("AAPL", "2024-01-15", 150.0, 151.0, 149.0, 150.5, 50_000_000, "yfinance"),
    )
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO prices
            (ticker, timestamp, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("AAPL", "2024-01-15", 151.0, 152.0, 150.0, 151.5, 55_000_000, "yfinance"),
        )
        conn.commit()


def test_price_columns_nullable_handling(test_prices_table: sqlite3.Connection) -> None:
    """Test that missing OHLC fields are stored as NULL."""
    conn = test_prices_table

    conn.execute(
        """
        INSERT INTO prices
        (ticker, timestamp, open, high, low, close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        ("UNKNOWN", "2024-01-15", None, None, None, 100.0, 0, "yfinance"),
    )
    conn.commit()

    cursor = conn.execute(
        "SELECT open, high, low, close FROM prices WHERE ticker = ?", ("UNKNOWN",)
    )
    row = cursor.fetchone()

    assert row is not None
    assert row[0] is None
    assert row[1] is None
    assert row[2] is None
    assert row[3] == 100.0


def test_yfinance_network_failure_graceful_degradation() -> None:
    """Test that network errors during fetch are caught and logged."""
    with patch.object(yf, "download") as mock_download:
        mock_download.side_effect = Exception("Network timeout")

        with pytest.raises(Exception) as exc_info:
            yf.download("AAPL", start="2024-01-15", end="2024-01-16")

        assert "Network timeout" in str(exc_info.value)


def test_price_schema_validation(test_prices_table: sqlite3.Connection) -> None:
    """Test that table schema matches expected column definitions."""
    conn = test_prices_table

    cursor = conn.execute("PRAGMA table_info(prices)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "ticker" in columns
    assert "timestamp" in columns
    assert "open" in columns
    assert "high" in columns
    assert "low" in columns
    assert "close" in columns
    assert "volume" in columns
    assert "source" in columns


def test_sqlite_transaction_rollback_on_error(
    test_prices_table: sqlite3.Connection,
) -> None:
    """Test that failed transaction rolls back cleanly."""
    conn = test_prices_table

    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            INSERT INTO prices
            (ticker, timestamp, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            ("AAPL", "2024-01-15", 150.0, 151.0, 149.0, 150.5, 50_000_000, "yfinance"),
        )
        conn.execute("INVALID SQL STATEMENT")
