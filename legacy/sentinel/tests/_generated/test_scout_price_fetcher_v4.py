"""
Unit test module for Scout live price fetcher.

This module tests the yfinance-based price fetching logic in sentinel/scout/live_prices.py.
It mocks yfinance responses, verifies correct parsing, and asserts SQLite writes to the
price history database. Tests cover happy paths, fallback logic, and error handling.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys

# Add sentinel root to path for imports
sentinel_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sentinel_root))

from scout.live_prices import fetch_live_price, init_price_db, store_price


class TestPriceFetcher(unittest.TestCase):
    """Test suite for Scout price fetcher module."""

    def setUp(self) -> None:
        """Set up test fixtures: temp SQLite DB and mock responses."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_price_db(self.db_path)

    def tearDown(self) -> None:
        """Clean up temporary database file."""
        try:
            Path(self.db_path).unlink()
        except FileNotFoundError:
            pass

    @patch("scout.live_prices.yf.download")
    def test_fetch_live_price_success(self, mock_download: Mock) -> None:
        """Assert fetch_live_price parses yfinance response and returns price dict."""
        mock_data = MagicMock()
        mock_data.iloc[-1]["Close"] = 150.25
        mock_data.index[-1] = datetime(2024, 1, 15)
        mock_download.return_value = mock_data

        result = fetch_live_price("AAPL")

        self.assertIsNotNone(result)
        self.assertEqual(result["ticker"], "AAPL")
        self.assertAlmostEqual(result["price"], 150.25, places=2)
        self.assertEqual(result["timestamp"].date(), datetime(2024, 1, 15).date())
        mock_download.assert_called_once()

    @patch("scout.live_prices.yf.download")
    def test_fetch_live_price_empty_response(self, mock_download: Mock) -> None:
        """Assert fetch_live_price returns None on empty yfinance response."""
        mock_download.return_value = MagicMock()
        mock_download.return_value.__len__.return_value = 0

        result = fetch_live_price("INVALID")

        self.assertIsNone(result)

    @patch("scout.live_prices.yf.download")
    def test_fetch_live_price_exception(self, mock_download: Mock) -> None:
        """Assert fetch_live_price gracefully handles yfinance exceptions."""
        mock_download.side_effect = Exception("Network error")

        result = fetch_live_price("AAPL")

        self.assertIsNone(result)

    def test_store_price_writes_to_db(self) -> None:
        """Assert store_price inserts record into SQLite price history table."""
        price_dict = {
            "ticker": "MSFT",
            "price": 380.50,
            "timestamp": datetime(2024, 1, 15, 10, 30, 0),
        }

        store_price(price_dict, self.db_path)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, price, timestamp FROM price_history WHERE ticker = ?",
            ("MSFT",),
        )
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "MSFT")
        self.assertAlmostEqual(row[1], 380.50, places=2)

    def test_store_price_multiple_records(self) -> None:
        """Assert store_price handles multiple writes without collision."""
        prices = [
            {
                "ticker": "GOOGL",
                "price": 140.00,
                "timestamp": datetime(2024, 1, 15, 10, 0, 0),
            },
            {
                "ticker": "GOOGL",
                "price": 141.50,
                "timestamp": datetime(2024, 1, 15, 11, 0, 0),
            },
        ]

        for price_dict in prices:
            store_price(price_dict, self.db_path)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM price_history WHERE ticker = ?", ("GOOGL",))
        count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(count, 2)

    def test_init_price_db_creates_schema(self) -> None:
        """Assert init_price_db creates price_history table with correct schema."""
        test_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        test_db.close()

        init_price_db(test_db.name)

        conn = sqlite3.connect(test_db.name)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='price_history'"
        )
        table_exists = cursor.fetchone() is not None
        conn.close()

        self.assertTrue(table_exists)
        Path(test_db.name).unlink()

    @patch("scout.live_prices.yf.download")
    def test_fetch_live_price_multiple_tickers(self, mock_download: Mock) -> None:
        """Assert fetch_live_price works across different ticker symbols."""
        tickers = ["AAPL", "MSFT", "GOOGL"]

        for ticker in tickers:
            mock_data = MagicMock()
            mock_data.iloc[-1]["Close"] = 100.0 + len(ticker)
            mock_data.index[-1] = datetime.now()
            mock_download.return_value = mock_data

            result = fetch_live_price(ticker)

            self.assertIsNotNone(result)
            self.assertEqual(result["ticker"], ticker)

    @patch("scout.live_prices.yf.download")
    def test_fetch_live_price_preserves_precision(self, mock_download: Mock) -> None:
        """Assert fetch_live_price preserves floating-point precision in prices."""
        mock_data = MagicMock()
        mock_data.iloc[-1]["Close"] = 123.456789
        mock_data.index[-1] = datetime.now()
        mock_download.return_value = mock_data

        result = fetch_live_price("TSLA")

        self.assertAlmostEqual(result["price"], 123.456789, places=5)

    def test_store_price_timestamp_preserved(self) -> None:
        """Assert store_price preserves exact timestamp from price dict."""
        original_time = datetime(2024, 1, 15, 14, 23, 45)
        price_dict = {
            "ticker": "NVDA",
            "price": 500.00,
            "timestamp": original_time,
        }

        store_price(price_dict, self.db_path)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp FROM price_history WHERE ticker = ?", ("NVDA",))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        # Timestamp stored as ISO string, parse back for comparison
        stored_time = datetime.fromisoformat(row[0])
        self.assertEqual(stored_time, original_time)


class TestP
