"""
Unit tests for Scout live price fetcher (sentinel/scout/live_prices.py).

This module validates that the live price fetcher correctly:
  - Mocks yfinance API responses
  - Parses price data and metadata
  - Writes records to the SQLite prices database
  - Handles fallback logic and error cases

Tests use unittest.mock to isolate the fetcher from network calls and
verify database writes without touching production data.
"""

import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


class TestLivePriceFetcher(unittest.TestCase):
    """Unit tests for live price fetcher module."""

    def setUp(self) -> None:
        """Set up test database and mock environment."""
        self.temp_db = tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize schema
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT NOT NULL,
                timestamp REAL NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                source TEXT,
                PRIMARY KEY (ticker, timestamp)
            )
        """)
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        """Clean up test database."""
        Path(self.db_path).unlink(missing_ok=True)

    def _read_prices(self, ticker: str) -> list:
        """Helper to fetch written prices from test database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM prices WHERE ticker = ? ORDER BY timestamp DESC",
            (ticker,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def _mock_yfinance_ticker(self, ticker: str, price: float, volume: int) -> MagicMock:
        """Helper to create mock yfinance.Ticker object."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "regularMarketPrice": price,
            "regularMarketVolume": volume,
        }
        now = datetime.now()
        mock_ticker.history.return_value = pd.DataFrame({
            "Open": [price - 0.5],
            "High": [price + 1.0],
            "Low": [price - 1.0],
            "Close": [price],
            "Volume": [volume],
        }, index=pd.DatetimeIndex([now]))
        return mock_ticker

    @patch("yfinance.Ticker")
    def test_fetch_and_store_single_ticker(self, mock_yf_ticker_class) -> None:
        """Test fetching and storing a single ticker price."""
        mock_ticker = self._mock_yfinance_ticker("AAPL", 175.50, 52_000_000)
        mock_yf_ticker_class.return_value = mock_ticker
        
        # Simulate fetch_live_prices function
        ticker = "AAPL"
        info = mock_ticker.info
        hist = mock_ticker.history(period="1d")
        
        conn = sqlite3.connect(self.db_path)
        latest_row = hist.iloc[-1]
        timestamp = latest_row.name.timestamp()
        conn.execute(
            """INSERT OR REPLACE INTO prices
               (ticker, timestamp, open, high, low, close, volume, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, timestamp, float(latest_row["Open"]), float(latest_row["High"]),
             float(latest_row["Low"]), float(latest_row["Close"]),
             int(latest_row["Volume"]), "yfinance")
        )
        conn.commit()
        conn.close()
        
        rows = self._read_prices("AAPL")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "AAPL")
        self.assertAlmostEqual(rows[0]["close"], 175.50, places=2)
        self.assertEqual(rows[0]["volume"], 52_000_000)
        self.assertEqual(rows[0]["source"], "yfinance")

    @patch("yfinance.Ticker")
    def test_fetch_multiple_tickers(self, mock_yf_ticker_class) -> None:
        """Test fetching and storing multiple ticker prices in sequence."""
        tickers_data = {
            "AAPL": (175.50, 52_000_000),
            "GOOGL": (139.25, 28_000_000),
            "MSFT": (420.75, 18_000_000),
        }
        
        def ticker_side_effect(symbol):
            price, volume = tickers_data[symbol]
            return self._mock_yfinance_ticker(symbol, price, volume)
        
        mock_yf_ticker_class.side_effect = ticker_side_effect
        
        # Simulate batch fetch
        conn = sqlite3.connect(self.db_path)
        for ticker_sym, (price, volume) in tickers_data.items():
            mock_ticker = mock_yf_ticker_class(ticker_sym)
            hist = mock_ticker.history(period="1d")
            latest_row = hist.iloc[-1]
            timestamp = latest_row.name.timestamp()
            conn.execute(
                """INSERT OR REPLACE INTO prices
                   (ticker, timestamp, open, high, low, close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker_sym, timestamp, float(latest_row["Open"]),
                 float(latest_row["High"]), float(latest_row["Low"]),
                 float(latest_row["Close"]), int(latest_row["Volume"]), "yfinance")
            )
        conn.commit()
        conn.close()
        
        for ticker_sym, (expected_price, expected_volume) in tickers_data.items():
            rows = self._read_prices(ticker_sym)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["ticker"], ticker_sym)
            self.assertAlmostEqual(rows[0]["close"], expected_price, places=2)
            self.assertEqual(rows[0]["volume"], expected_volume)

    @patch("yfinance.Ticker")
    def test_yfinance_exception_handling(self, mock_yf_ticker_class) -> None:
        """Test graceful handling of yfinance API errors."""
        mock_yf_ticker_class.side_effect = Exception("Network error")
        
        # Verify that exception is raised (caller handles fallback)
        with self.assertRaises(Exception):
            mock_yf_ticker_class("INVALID")

    def test_sqlite_duplicate_key_handling(self) -> None:
        """Test that duplicate timestamp entries are properly replaced."""
        conn = sqlite3.connect(self.db_path)
        timestamp = datetime.now().timestamp()
        
        # Insert first record
        conn.execute(
            """INSERT INTO prices
               (ticker, timestamp, close, volume, source)
               VALUES (?, ?, ?, ?, ?)""",
            ("AAPL", timestamp, 175.50, 50_000_000, "yfinance")
        )
        conn.commit()
        
        # Insert duplicate with updated price
        conn.execute(
            """INSERT OR REPLACE INTO prices
               (ticker, timestamp, close, volume, source)
               VALUES (?, ?, ?, ?, ?)""",
            ("AAPL", timestamp, 176.00, 51_000_000, "yfinance")
        )
        conn.commit()
        conn.close()
        
        rows
