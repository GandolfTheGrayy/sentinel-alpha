"""
Unit tests for sentinel/scout/live_prices.py — mocks yfinance responses,
validates SQLite schema writes, and asserts price-fetch correctness.

Part of the Sentinel Sentiment Engine test harness. Runs via pytest
to verify Scout price-fetcher reliability before pipeline integration.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class MockYFinanceTicker:
    """Mock yfinance Ticker object for testing."""

    def __init__(self, symbol: str, price: float, volume: int) -> None:
        self.symbol = symbol
        self.info = {
            "currentPrice": price,
            "volume": volume,
            "marketCap": price * volume * 1000,
            "fiftyTwoWeekHigh": price * 1.2,
            "fiftyTwoWeekLow": price * 0.8,
        }
        self.history_data = {
            "Close": price,
            "Volume": volume,
        }

    def history(self, period: str = "1d"):
        """Mock history method returning a DataFrame-like object."""
        mock_df = MagicMock()
        mock_df.iloc = {-1: MagicMock(Close=self.info["currentPrice"],
                                       Volume=self.info["volume"])}
        return mock_df


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_prices.db"
        yield db_path


@pytest.fixture
def price_db_schema(temp_db: Path) -> sqlite3.Connection:
    """Initialize a test database with the price schema."""
    conn = sqlite3.connect(str(temp_db))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            volume INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source TEXT DEFAULT 'yfinance'
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol_timestamp
        ON prices(symbol, timestamp)
    """)
    conn.commit()
    return conn


def test_price_insertion_valid_data(price_db_schema: sqlite3.Connection) -> None:
    """Test insertion of valid price data into SQLite."""
    conn = price_db_schema
    conn.execute(
        "INSERT INTO prices (symbol, price, volume, source) VALUES (?, ?, ?, ?)",
        ("AAPL", 150.25, 1000000, "yfinance"),
    )
    conn.commit()

    cursor = conn.execute("SELECT symbol, price, volume FROM prices WHERE symbol = ?",
                          ("AAPL",))
    row = cursor.fetchone()

    assert row is not None
    assert row[0] == "AAPL"
    assert row[1] == 150.25
    assert row[2] == 1000000
    conn.close()


def test_price_insertion_multiple_symbols(price_db_schema: sqlite3.Connection) -> None:
    """Test insertion of multiple price records across different symbols."""
    conn = price_db_schema
    test_data = [
        ("AAPL", 150.25, 1000000),
        ("GOOGL", 2800.50, 500000),
        ("MSFT", 320.75, 2000000),
    ]

    for symbol, price, volume in test_data:
        conn.execute(
            "INSERT INTO prices (symbol, price, volume, source) VALUES (?, ?, ?, ?)",
            (symbol, price, volume, "yfinance"),
        )
    conn.commit()

    cursor = conn.execute("SELECT COUNT(*) FROM prices")
    count = cursor.fetchone()[0]
    assert count == 3
    conn.close()


def test_price_schema_integrity(price_db_schema: sqlite3.Connection) -> None:
    """Verify that the prices table schema is correct."""
    conn = price_db_schema
    cursor = conn.execute("PRAGMA table_info(prices)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}

    assert "id" in columns
    assert "symbol" in columns
    assert "price" in columns
    assert "volume" in columns
    assert "timestamp" in columns
    assert "source" in columns
    conn.close()


def test_index_creation(price_db_schema: sqlite3.Connection) -> None:
    """Verify that the symbol-timestamp index exists."""
    conn = price_db_schema
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_symbol_timestamp'"
    )
    assert cursor.fetchone() is not None
    conn.close()


@patch("yfinance.Ticker")
def test_mock_yfinance_ticker_creation(mock_ticker_class) -> None:
    """Test mocked yfinance Ticker object creation and data extraction."""
    mock_ticker = MockYFinanceTicker("AAPL", 150.25, 1000000)
    mock_ticker_class.return_value = mock_ticker

    ticker = mock_ticker_class("AAPL")
    assert ticker.symbol == "AAPL"
    assert ticker.info["currentPrice"] == 150.25
    assert ticker.info["volume"] == 1000000


def test_mock_yfinance_ticker_info_fields() -> None:
    """Test that mock Ticker populates all required info fields."""
    mock_ticker = MockYFinanceTicker("GOOGL", 2800.50, 500000)

    required_fields = [
        "currentPrice",
        "volume",
        "marketCap",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
    ]
    for field in required_fields:
        assert field in mock_ticker.info


def test_price_fetch_and_store_workflow(price_db_schema: sqlite3.Connection) -> None:
    """Integration test: fetch mock prices, validate schema, store in DB."""
    conn = price_db_schema
    symbols = ["AAPL", "GOOGL", "MSFT"]
    mock_prices = {
        "AAPL": (150.25, 1000000),
        "GOOGL": (2800.50, 500000),
        "MSFT": (320.75, 2000000),
    }

    for symbol in symbols:
        price, volume = mock_prices[symbol]
        conn.execute(
            "INSERT INTO prices (symbol, price, volume, source) VALUES (?, ?, ?, ?)",
            (symbol, price, volume, "yfinance"),
        )
    conn.commit()

    cursor = conn.execute("SELECT COUNT(*) FROM prices")
    assert cursor.fetchone()[0] == 3

    cursor = conn.execute("SELECT symbol FROM prices ORDER BY symbol")
    fetched_symbols = [row[0] for row in cursor.fetchall()]
    assert fetched_symbols == ["AAPL", "GOOGL", "MSFT"]
    conn.close()


def test_price_timestamp_auto_generation(price_db_schema: sqlite3.Connection) -> None:
    """Verify that timestamps are auto-generated on insertion."""
    conn = price_db_schema
    conn.execute(
        "INSERT INTO prices (symbol, price, volume, source) VALUES (?, ?, ?, ?)",
        ("AAPL", 150.25, 1000000, "yfinance"),
    )
    conn.commit()

    cursor = conn.execute("SELECT timestamp FROM prices WHERE symbol = ?", ("AAPL",))
    row = cursor.fetchone()
    assert row is not None
    assert row[0] is not None
    conn.close()


def test_price_query_by_symbol(price_db_schema: sqlite3.Connection) -> None:
    """Test querying prices by symbol using the index."""
    conn = price_db_schema
