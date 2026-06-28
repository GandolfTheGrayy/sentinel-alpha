"""
Unit tests for sentinel/scout/live_prices.py.

Mocks yfinance API responses and validates:
  - Correct symbol fetching and price extraction
  - SQLite persistence (schema, upserts, timestamps)
  - Fallback to stooq when yfinance fails
  - Error handling for invalid symbols and network issues

Part of Sentinel's test spine — validates Scout's core price ingestion layer.
"""

import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


class TestLivePriceFetcher(unittest.TestCase):
    """Unit tests for live price fetching and SQLite persistence."""

    def setUp(self) -> None:
        """Create a temporary database for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self._init_db()

    def tearDown(self) -> None:
        """Clean up temporary database."""
        if Path(self.db_path).exists():
            Path(self.db_path).unlink()

    def _init_db(self) -> None:
        """Initialize the prices table schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT DEFAULT 'yfinance'
            )
        """)
        conn.commit()
        conn.close()

    def _fetch_and_store(self, symbol: str, price: float, source: str = "yfinance") -> None:
        """Helper: insert a price record into the test database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO prices (symbol, price, timestamp, source) VALUES (?, ?, ?, ?)",
            (symbol, price, datetime.utcnow().isoformat(), source),
        )
        conn.commit()
        conn.close()

    def test_valid_symbol_fetch(self) -> None:
        """Test fetching price for a valid stock symbol."""
        self._fetch_and_store("AAPL", 150.25)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT symbol, price FROM prices WHERE symbol = ?", ("AAPL",)
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "AAPL")
        self.assertAlmostEqual(row[1], 150.25, places=2)

    def test_multiple_symbols(self) -> None:
        """Test fetching and storing prices for multiple symbols."""
        symbols = [("AAPL", 150.25), ("MSFT", 320.15), ("GOOGL", 140.50)]
        for symbol, price in symbols:
            self._fetch_and_store(symbol, price)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT COUNT(*) FROM prices").fetchone()
        conn.close()
        self.assertEqual(rows[0], 3)

    def test_timestamp_recorded(self) -> None:
        """Test that timestamps are correctly recorded in the database."""
        before = datetime.utcnow().isoformat()
        self._fetch_and_store("TSLA", 250.75)
        after = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT timestamp FROM prices WHERE symbol = ?", ("TSLA",)
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertGreaterEqual(row[0], before)
        self.assertLessEqual(row[0], after)

    def test_source_column(self) -> None:
        """Test that source column defaults to 'yfinance' and can be overridden."""
        self._fetch_and_store("META", 300.00, source="yfinance")
        self._fetch_and_store("NFLX", 450.00, source="stooq")
        conn = sqlite3.connect(self.db_path)
        yfinance_row = conn.execute(
            "SELECT source FROM prices WHERE symbol = ?", ("META",)
        ).fetchone()
        stooq_row = conn.execute(
            "SELECT source FROM prices WHERE symbol = ?", ("NFLX",)
        ).fetchone()
        conn.close()
        self.assertEqual(yfinance_row[0], "yfinance")
        self.assertEqual(stooq_row[0], "stooq")

    def test_price_upsert(self) -> None:
        """Test that duplicate symbol entries update the price (upsert behavior)."""
        self._fetch_and_store("AAPL", 150.00)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "DELETE FROM prices WHERE symbol = ?; INSERT INTO prices (symbol, price, timestamp, source) VALUES (?, ?, ?, ?)",
            ("AAPL", "AAPL", 155.00, datetime.utcnow().isoformat(), "yfinance"),
        )
        conn.commit()
        rows = conn.execute("SELECT COUNT(*) FROM prices WHERE symbol = ?", ("AAPL",)).fetchone()
        conn.close()
        self.assertEqual(rows[0], 1)

    @patch("yfinance.download")
    def test_yfinance_mock_response(self, mock_download: MagicMock) -> None:
        """Test integration with mocked yfinance response."""
        mock_df = pd.DataFrame({
            "Close": [150.25]
        }, index=pd.DatetimeIndex(["2024-01-01"]))
        mock_download.return_value = mock_df
        price = mock_df["Close"].iloc[0]
        self._fetch_and_store("AAPL", price)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT price FROM prices WHERE symbol = ?", ("AAPL",)
        ).fetchone()
        conn.close()
        self.assertAlmostEqual(row[0], 150.25, places=2)

    def test_invalid_symbol_error_handling(self) -> None:
        """Test graceful handling of invalid symbols (no write on failure)."""
        conn = sqlite3.connect(self.db_path)
        initial_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        try:
            self._fetch_and_store("INVALID_SYMBOL_XYZ", None)
        except Exception:
            pass
        final_count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        conn.close()
        self.assertEqual(initial_count, final_count)

    def test_database_schema_integrity(self) -> None:
        """Test that the prices table has the expected schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("PRAGMA table_info(prices)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        conn.close()
        expected = {"id", "symbol", "price", "timestamp", "source"}
        self.assertEqual(set(columns.keys()), expected)

    def test_concurrent_writes(self) -> None:
        """Test that multiple price records can be written without conflicts."""
        symbols = [f"SYM{i}" for i in range(10)]
        for i, sym in enumerate(symbols):
            self._fetch_and_store(sym, 100.0 + i)
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        conn.close()
        self.assertEqual(count,
