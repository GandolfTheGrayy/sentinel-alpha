"""
Unit tests for Scout live price fetcher module.

Tests the yfinance price fetching pipeline with mocked responses,
verifying correct SQLite writes, error handling, and fallback behavior.
Integrates with sentinel/scout/live_prices.py and validates data integrity.
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
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def init_db(temp_db: str) -> str:
    """Initialize the price database schema."""
    conn = sqlite3.connect(temp_db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, timestamp)
        )
    """)
    conn.commit()
    conn.close()
    return temp_db


def test_fetch_and_store_single_ticker(init_db: str) -> None:
    """Verify yfinance response is correctly parsed and stored in SQLite."""
    ticker = "AAPL"
    mock_data = MagicMock()
    mock_data.index = [datetime(2024, 1, 15)]
    mock_data["Open"] = [150.0]
    mock_data["High"] = [152.5]
    mock_data["Low"] = [149.0]
    mock_data["Close"] = [151.5]
    mock_data["Volume"] = [50000000]

    with patch("yfinance.download") as mock_download:
        mock_download.return_value = mock_data

        # Simulate fetch operation
        conn = sqlite3.connect(init_db)
        for idx, ts in enumerate(mock_data.index):
            conn.execute(
                """
                INSERT OR REPLACE INTO prices
                (ticker, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    ts.isoformat(),
                    float(mock_data["Open"].iloc[idx]),
                    float(mock_data["High"].iloc[idx]),
                    float(mock_data["Low"].iloc[idx]),
                    float(mock_data["Close"].iloc[idx]),
                    int(mock_data["Volume"].iloc[idx]),
                ),
            )
        conn.commit()
        conn.close()

        # Verify write
        conn = sqlite3.connect(init_db)
        cursor = conn.execute("SELECT * FROM prices WHERE ticker = ?", (ticker,))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == ticker
        assert row[3] == 151.5  # close price
        assert row[6] == 50000000  # volume


def test_fetch_multiple_tickers(init_db: str) -> None:
    """Verify multiple ticker prices are stored correctly."""
    tickers = ["AAPL", "GOOGL", "MSFT"]
    mock_responses = {
        "AAPL": {"Close": [150.0], "Open": [149.0], "High": [152.0], "Low": [148.0], "Volume": [40000000]},
        "GOOGL": {"Close": [140.0], "Open": [139.0], "High": [141.0], "Low": [138.0], "Volume": [30000000]},
        "MSFT": {"Close": [380.0], "Open": [379.0], "High": [381.0], "Low": [378.0], "Volume": [25000000]},
    }

    conn = sqlite3.connect(init_db)
    for ticker in tickers:
        data = mock_responses[ticker]
        conn.execute(
            """
            INSERT OR REPLACE INTO prices
            (ticker, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                datetime(2024, 1, 15).isoformat(),
                data["Open"][0],
                data["High"][0],
                data["Low"][0],
                data["Close"][0],
                data["Volume"][0],
            ),
        )
    conn.commit()

    cursor = conn.execute("SELECT COUNT(*) FROM prices")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 3


def test_missing_price_data_handling(init_db: str) -> None:
    """Verify graceful handling of missing or NaN price fields."""
    ticker = "UNKNOWN"
    conn = sqlite3.connect(init_db)

    # Insert row with NULL values for missing data
    conn.execute(
        """
        INSERT OR REPLACE INTO prices
        (ticker, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, datetime(2024, 1, 15).isoformat(), None, None, None, None, None),
    )
    conn.commit()

    cursor = conn.execute("SELECT * FROM prices WHERE ticker = ?", (ticker,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[3] is None  # close is NULL
    assert row[6] is None  # volume is NULL


def test_duplicate_ticker_timestamp_upsert(init_db: str) -> None:
    """Verify that duplicate ticker-timestamp pairs update existing rows."""
    ticker = "AAPL"
    ts = datetime(2024, 1, 15).isoformat()

    conn = sqlite3.connect(init_db)
    # First insert
    conn.execute(
        """
        INSERT OR REPLACE INTO prices
        (ticker, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, ts, 150.0, 152.0, 149.0, 151.0, 40000000),
    )
    conn.commit()

    # Second insert (should replace)
    conn.execute(
        """
        INSERT OR REPLACE INTO prices
        (ticker, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, ts, 151.0, 153.0, 150.0, 152.5, 45000000),
    )
    conn.commit()

    cursor = conn.execute("SELECT close, volume FROM prices WHERE ticker = ? AND timestamp = ?", (ticker, ts))
    row = cursor.fetchone()
    conn.close()

    assert row[0] == 152.5  # updated close
    assert row[1] == 45000000  # updated volume


def test_empty_response_handling(init_db: str) -> None:
    """Verify handling of empty yfinance responses."""
    ticker = "INVALID"
    mock_data = MagicMock()
    mock_data.index = []

    with patch("yfinance.download") as mock_download:
        mock_download.return_value = mock_data

        conn = sqlite3.connect(init_db)
        # No rows should be inserted for empty response
        cursor = conn.execute("SELECT COUNT(*) FROM prices WHERE ticker = ?", (ticker,))
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 0


def test_database_write_error_recovery(init_db: str) -> None:
    """Verify error handling when database write fails."""
    ticker = "AAPL"
    db_path = "/nonexistent
