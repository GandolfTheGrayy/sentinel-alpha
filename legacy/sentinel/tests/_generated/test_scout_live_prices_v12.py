"""
Unit tests for sentinel/scout/live_prices.py — mocks yfinance responses and
asserts correct SQLite writes. Part of Sentinel's automated test suite.
"""

import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sentinel.scout.live_prices import fetch_live_prices, store_price_snapshot


class TestFetchLivePrices(unittest.TestCase):
    """Test live price fetching with mocked yfinance."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_tickers = ["AAPL", "GOOGL", "MSFT"]
        self.mock_price_data = {
            "AAPL": 150.25,
            "GOOGL": 2800.50,
            "MSFT": 380.75,
        }

    @patch("sentinel.scout.live_prices.yf.download")
    def test_fetch_live_prices_success(self, mock_download: MagicMock) -> None:
        """Test successful price fetch from mocked yfinance."""
        # Create a mock DataFrame matching yfinance output
        mock_df = pd.DataFrame({
            "Close": [150.25, 2800.50, 380.75],
        }, index=pd.Index(self.test_tickers, name="Ticker"))
        mock_download.return_value = mock_df

        result = fetch_live_prices(self.test_tickers)

        self.assertEqual(len(result), 3)
        self.assertEqual(result["AAPL"], 150.25)
        self.assertEqual(result["GOOGL"], 2800.50)
        self.assertEqual(result["MSFT"], 380.75)
        mock_download.assert_called_once()

    @patch("sentinel.scout.live_prices.yf.download")
    def test_fetch_live_prices_empty_ticker_list(self, mock_download: MagicMock) -> None:
        """Test fetch with empty ticker list."""
        result = fetch_live_prices([])
        self.assertEqual(result, {})
        mock_download.assert_not_called()

    @patch("sentinel.scout.live_prices.yf.download")
    def test_fetch_live_prices_network_error(self, mock_download: MagicMock) -> None:
        """Test graceful handling of network errors."""
        mock_download.side_effect = Exception("Network timeout")

        with self.assertRaises(Exception):
            fetch_live_prices(self.test_tickers)

    @patch("sentinel.scout.live_prices.yf.download")
    def test_fetch_live_prices_partial_data(self, mock_download: MagicMock) -> None:
        """Test handling of partial data (some tickers missing)."""
        # Simulate yfinance returning only 2 of 3 tickers
        mock_df = pd.DataFrame({
            "Close": [150.25, 2800.50],
        }, index=pd.Index(["AAPL", "GOOGL"], name="Ticker"))
        mock_download.return_value = mock_df

        result = fetch_live_prices(self.test_tickers)

        self.assertEqual(len(result), 2)
        self.assertIn("AAPL", result)
        self.assertIn("GOOGL", result)
        self.assertNotIn("MSFT", result)


class TestStorePriceSnapshot(unittest.TestCase):
    """Test SQLite price storage."""

    def setUp(self) -> None:
        """Set up temporary database for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self._init_db()

    def tearDown(self) -> None:
        """Clean up temporary database."""
        Path(self.db_path).unlink(missing_ok=True)

    def _init_db(self) -> None:
        """Initialize test database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE(ticker, timestamp)
            )
        """)
        conn.commit()
        conn.close()

    def test_store_price_snapshot_single(self) -> None:
        """Test storing a single price snapshot."""
        store_price_snapshot(
            db_path=self.db_path,
            ticker="AAPL",
            price=150.25,
            timestamp=datetime(2025, 1, 15, 10, 30, 0),
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT ticker, price FROM price_snapshots WHERE ticker = ?",
            ("AAPL",),
        )
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "AAPL")
        self.assertEqual(row[1], 150.25)

    def test_store_price_snapshot_multiple(self) -> None:
        """Test storing multiple price snapshots."""
        prices = {
            "AAPL": 150.25,
            "GOOGL": 2800.50,
            "MSFT": 380.75,
        }
        ts = datetime(2025, 1, 15, 10, 30, 0)

        for ticker, price in prices.items():
            store_price_snapshot(self.db_path, ticker, price, ts)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM price_snapshots")
        count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(count, 3)

    def test_store_price_snapshot_duplicate(self) -> None:
        """Test that duplicate ticker/timestamp entries are handled."""
        ts = datetime(2025, 1, 15, 10, 30, 0)

        store_price_snapshot(self.db_path, "AAPL", 150.25, ts)
        # Second insert with same ticker and timestamp should not duplicate
        store_price_snapshot(self.db_path, "AAPL", 150.50, ts)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM price_snapshots WHERE ticker = ?",
            ("AAPL",),
        )
        count = cursor.fetchone()[0]
        conn.close()

        # With UNIQUE constraint, only 1 row should exist (or update if implemented)
        self.assertLessEqual(count, 1)

    def test_store_price_snapshot_data_integrity(self) -> None:
        """Test that stored data matches input exactly."""
        ticker, price = "TSLA", 245.89
        ts = datetime(2025, 1, 15, 14, 45, 30)

        store_price_snapshot(self.db_path, ticker, price, ts)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT ticker, price, timestamp FROM price_snapshots WHERE ticker = ?",
            (ticker,),
        )
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], ticker)
        self.assertEqual(row[1], price)
        self.assertIn("2025-01-15", row[2])


class TestPriceSnapshotIntegration(unittest.TestCase):
    """Integration tests: fetch → store → retrieve cycle."""

    def setUp(self) -> None
