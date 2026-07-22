"""
Unit tests for sentinel.scout.live_prices — the real-time price fetcher.

This module validates that the Scout price fetcher correctly:
  1. Calls yfinance to retrieve OHLCV data
  2. Writes records to the SQLite prices table with correct schema
  3. Handles missing/malformed tickers gracefully
  4. Falls back to stooq on yfinance failure
  5. Records fetch timestamps and data freshness

Uses unittest.mock to stub yfinance and sqlite3, ensuring test isolation
and repeatability without live market or network calls.
"""

import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import pandas as pd
import sys

# Assume sentinel package is importable from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sentinel.scout.live_prices import fetch_live_prices, init_prices_db


class TestScoutPriceFetcher(unittest.TestCase):
    """Unit test suite for Scout live_prices module."""

    def setUp(self) -> None:
        """Create a temporary SQLite database for each test."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_prices_db(self.db_path)

    def tearDown(self) -> None:
        """Clean up temporary database file."""
        try:
            Path(self.db_path).unlink()
        except FileNotFoundError:
            pass

    @patch("yfinance.download")
    def test_fetch_single_ticker_success(
        self, mock_yf_download: MagicMock
    ) -> None:
        """Verify successful fetch and SQLite write for a single ticker."""
        # Arrange: Mock yfinance response for AAPL
        mock_data = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [152.0],
                "Low": [149.5],
                "Close": [151.5],
                "Volume": [1000000],
            },
            index=pd.DatetimeIndex(["2024-01-15"]),
        )
        mock_yf_download.return_value = mock_data

        # Act
        result = fetch_live_prices(["AAPL"], db_path=self.db_path)

        # Assert
        self.assertEqual(len(result), 1)
        self.assertIn("AAPL", result)
        self.assertEqual(result["AAPL"]["close"], 151.5)
        self.assertEqual(result["AAPL"]["volume"], 1000000)

        # Verify SQLite write
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, close, volume FROM prices WHERE ticker = ?", ("AAPL",))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "AAPL")
        self.assertEqual(row[1], 151.5)
        self.assertEqual(row[2], 1000000)

    @patch("yfinance.download")
    def test_fetch_multiple_tickers(self, mock_yf_download: MagicMock) -> None:
        """Verify batch fetch writes all tickers to database."""
        # Arrange
        mock_data = pd.DataFrame(
            {
                "Open": [150.0, 200.0],
                "High": [152.0, 202.0],
                "Low": [149.5, 199.5],
                "Close": [151.5, 201.5],
                "Volume": [1000000, 2000000],
            },
            index=pd.DatetimeIndex(["2024-01-15", "2024-01-15"]),
        )
        mock_data.index.name = "Date"
        mock_yf_download.return_value = mock_data

        # Act
        result = fetch_live_prices(["AAPL", "MSFT"], db_path=self.db_path)

        # Assert
        self.assertEqual(len(result), 2)
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)

        # Verify all rows in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM prices")
        count = cursor.fetchone()[0]
        conn.close()

        self.assertGreaterEqual(count, 2)

    @patch("yfinance.download")
    def test_fetch_malformed_ticker_graceful_fail(
        self, mock_yf_download: MagicMock
    ) -> None:
        """Verify graceful handling of invalid/non-existent tickers."""
        # Arrange
        mock_yf_download.return_value = pd.DataFrame()

        # Act
        result = fetch_live_prices(["INVALID_TICKER_XYZ"], db_path=self.db_path)

        # Assert: Should return empty dict or skip without crashing
        self.assertEqual(result, {})

    @patch("yfinance.download")
    def test_fetch_writes_timestamp(self, mock_yf_download: MagicMock) -> None:
        """Verify that fetch timestamp is recorded in database."""
        # Arrange
        before = datetime.utcnow()
        mock_data = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [152.0],
                "Low": [149.5],
                "Close": [151.5],
                "Volume": [1000000],
            },
            index=pd.DatetimeIndex(["2024-01-15"]),
        )
        mock_yf_download.return_value = mock_data

        # Act
        fetch_live_prices(["AAPL"], db_path=self.db_path)
        after = datetime.utcnow()

        # Assert
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT fetched_at FROM prices WHERE ticker = ?", ("AAPL",))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        fetched_time = datetime.fromisoformat(row[0])
        self.assertGreaterEqual(fetched_time, before)
        self.assertLessEqual(fetched_time, after)

    @patch("sentinel.scout.live_prices.fetch_live_prices_stooq")
    @patch("yfinance.download")
    def test_fallback_to_stooq_on_yfinance_failure(
        self, mock_yf_download: MagicMock, mock_stooq: MagicMock
    ) -> None:
        """Verify fallback to stooq when yfinance fails."""
        # Arrange
        mock_yf_download.side_effect = Exception("yfinance network error")
        mock_stooq.return_value = {
            "AAPL": {
                "open": 150.0,
                "high": 152.0,
                "low": 149.5,
                "close": 151.5,
                "volume": 1000000,
            }
        }

        # Act
        result = fetch_live_prices(["AAPL"], db_path=self.db_path)

        # Assert: stooq should be called as fallback
        mock_stooq.assert_called_once()
        self.assertEqual(result["AAPL"]["close"], 151.5)

    def test_init_prices_db_creates_schema(self) -> None:
        """Verify that init_prices_db creates correct table schema."""
        # Use a fresh temp db
        fresh_db = tempfile.Named
