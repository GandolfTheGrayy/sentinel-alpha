"""
Unit tests for sentinel/scout/live_prices.py — the live price fetcher.

This module validates that:
  - yfinance responses are correctly parsed and stored in SQLite
  - Fallback to stooq works when yfinance fails
  - Concurrent fetches for multiple tickers succeed
  - Database writes are atomic and indexed correctly
  - Malformed or missing data is handled gracefully

Part of the Sentinel test spine — exercises Scout's core data ingestion layer.
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
from sentinel.scout.live_prices import (
    fetch_live_prices,
    store_prices_to_db,
    get_latest_price,
    initialize_price_db,
)


class TestLivePriceFetcher(unittest.TestCase):
    """Test suite for live price fetching and storage."""

    def setUp(self) -> None:
        """Create a temporary SQLite database for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_path = self.temp_db.name
        self.temp_db.close()
        initialize_price_db(self.db_path)

    def tearDown(self) -> None:
        """Clean up temporary database."""
        if Path(self.db_path).exists():
            Path(self.db_path).unlink()

    def test_initialize_price_db_creates_table(self) -> None:
        """Verify price_history table is created with correct schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='price_history'"
        )
        result = cursor.fetchone()
        conn.close()
        self.assertIsNotNone(result, "price_history table should exist")

    def test_store_prices_to_db_single_ticker(self) -> None:
        """Test storing a single ticker's price data."""
        mock_data = {
            "AAPL": {
                "price": 150.25,
                "timestamp": datetime(2024, 1, 15, 14, 30, 0),
                "volume": 45000000,
            }
        }
        store_prices_to_db(mock_data, self.db_path)

        # Verify the record was inserted
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, price, volume FROM price_history WHERE ticker = ?", ("AAPL",))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "AAPL")
        self.assertAlmostEqual(row[1], 150.25, places=2)
        self.assertEqual(row[2], 45000000)

    def test_store_prices_to_db_multiple_tickers(self) -> None:
        """Test storing multiple tickers in one operation."""
        mock_data = {
            "AAPL": {
                "price": 150.25,
                "timestamp": datetime(2024, 1, 15, 14, 30, 0),
                "volume": 45000000,
            },
            "MSFT": {
                "price": 375.50,
                "timestamp": datetime(2024, 1, 15, 14, 30, 0),
                "volume": 22000000,
            },
            "GOOGL": {
                "price": 140.75,
                "timestamp": datetime(2024, 1, 15, 14, 30, 0),
                "volume": 18500000,
            },
        }
        store_prices_to_db(mock_data, self.db_path)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM price_history")
        count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(count, 3, "Should have 3 price records")

    def test_get_latest_price_returns_most_recent(self) -> None:
        """Test that get_latest_price returns the most recent entry."""
        mock_data_1 = {
            "AAPL": {
                "price": 150.00,
                "timestamp": datetime(2024, 1, 15, 14, 0, 0),
                "volume": 40000000,
            }
        }
        mock_data_2 = {
            "AAPL": {
                "price": 151.50,
                "timestamp": datetime(2024, 1, 15, 15, 0, 0),
                "volume": 45000000,
            }
        }
        store_prices_to_db(mock_data_1, self.db_path)
        store_prices_to_db(mock_data_2, self.db_path)

        latest = get_latest_price("AAPL", self.db_path)
        self.assertAlmostEqual(latest, 151.50, places=2)

    def test_get_latest_price_nonexistent_ticker(self) -> None:
        """Test get_latest_price returns None for missing ticker."""
        latest = get_latest_price("XYZ", self.db_path)
        self.assertIsNone(latest, "Should return None for nonexistent ticker")

    @patch("sentinel.scout.live_prices.yf.download")
    def test_fetch_live_prices_yfinance_success(self, mock_download: MagicMock) -> None:
        """Test successful yfinance fetch for a single ticker."""
        # Mock yfinance response
        mock_df = pd.DataFrame(
            {
                "Open": [149.50],
                "High": [151.00],
                "Low": [149.00],
                "Close": [150.25],
                "Volume": [45000000],
                "Adj Close": [150.25],
            },
            index=pd.DatetimeIndex([datetime(2024, 1, 15, 16, 0, 0)]),
        )
        mock_download.return_value = mock_df

        result = fetch_live_prices(["AAPL"])
        self.assertIn("AAPL", result)
        self.assertEqual(result["AAPL"]["price"], 150.25)

    @patch("sentinel.scout.live_prices.yf.download")
    def test_fetch_live_prices_handles_missing_data(self, mock_download: MagicMock) -> None:
        """Test graceful handling of NaN or missing data from yfinance."""
        mock_df = pd.DataFrame(
            {
                "Close": [float("nan")],
                "Volume": [0],
            },
            index=pd.DatetimeIndex([datetime(2024, 1, 15, 16, 0, 0)]),
        )
        mock_download.return_value = mock_df

        result = fetch_live_prices(["BADTICKER"])
        # Should skip or log; result should be empty or None
        self.assertNotIn("BADTICKER", result)

    @patch("sentinel.scout.live_prices.yf.download")
    def test_fetch_live_prices_empty_list(self, mock_download: MagicMock) -> None:
        """Test fetch with empty ticker list."""
        result = fetch_live_prices([])
        self.assertEqual(result, {})
        mock_download.assert_not_called()

    def test_store_prices_empty_dict(self) -> None:
        """Test storing empty price dict (no-op)."""
        store_prices_to_db({}, self.db_path)

        conn = sqlite3.connect(self.db_
