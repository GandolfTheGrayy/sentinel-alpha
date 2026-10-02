"""
Unit tests for Scout price fetcher module.

This test suite validates the live_prices.py module by:
  - Mocking yfinance.download() responses
  - Verifying correct SQLite writes to the price history table
  - Testing fallback behavior when yfinance fails
  - Asserting data integrity (OHLCV columns, timestamps, ticker consistency)

Part of Sentinel's continuous validation pipeline — ensures price ingestion
remains reliable as market data sources evolve.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# Expected to be importable from sentinel.scout
# In tests, we mock the yfinance layer directly.


@pytest.fixture
def temp_db() -> str:
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date)
        )
    """)
    conn.commit()
    conn.close()
    
    yield db_path
    
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def mock_yfinance_data() -> pd.DataFrame:
    """Generate synthetic yfinance-compatible DataFrame."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    data = pd.DataFrame({
        'Open': [100.0, 101.5, 102.0, 103.2, 104.1],
        'High': [102.0, 103.0, 104.5, 105.1, 106.0],
        'Low': [99.5, 101.0, 101.5, 102.8, 103.5],
        'Close': [101.0, 102.0, 103.0, 104.0, 105.0],
        'Volume': [1000000, 1100000, 900000, 1200000, 1050000],
    }, index=dates)
    data.index.name = 'Date'
    return data


def test_fetch_and_store_prices(temp_db: str, mock_yfinance_data: pd.DataFrame) -> None:
    """Test that prices are correctly fetched and stored in SQLite."""
    with mock.patch('yfinance.download', return_value=mock_yfinance_data):
        from sentinel.scout.live_prices import fetch_and_store_prices
        
        result = fetch_and_store_prices(
            ticker='AAPL',
            db_path=temp_db,
            period='5d'
        )
        
        assert result is True, "fetch_and_store_prices should return True on success"
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM price_history WHERE ticker = ?", ('AAPL',))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 5, f"Expected 5 rows in DB, got {count}"


def test_price_data_integrity(temp_db: str, mock_yfinance_data: pd.DataFrame) -> None:
    """Verify that stored prices match input data exactly."""
    with mock.patch('yfinance.download', return_value=mock_yfinance_data):
        from sentinel.scout.live_prices import fetch_and_store_prices
        
        fetch_and_store_prices(ticker='MSFT', db_path=temp_db, period='5d')
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT date, open, high, low, close, volume
            FROM price_history
            WHERE ticker = 'MSFT'
            ORDER BY date
        """)
        rows = cursor.fetchall()
        conn.close()
        
        assert len(rows) == 5, "Should have 5 price records"
        
        for i, (date_str, open_val, high_val, low_val, close_val, volume_val) in enumerate(rows):
            expected_row = mock_yfinance_data.iloc[i]
            assert float(open_val) == expected_row['Open'], f"Row {i}: Open mismatch"
            assert float(high_val) == expected_row['High'], f"Row {i}: High mismatch"
            assert float(low_val) == expected_row['Low'], f"Row {i}: Low mismatch"
            assert float(close_val) == expected_row['Close'], f"Row {i}: Close mismatch"
            assert int(volume_val) == int(expected_row['Volume']), f"Row {i}: Volume mismatch"


def test_yfinance_failure_fallback(temp_db: str) -> None:
    """Test graceful handling when yfinance.download() raises an exception."""
    with mock.patch('yfinance.download', side_effect=Exception("Network timeout")):
        from sentinel.scout.live_prices import fetch_and_store_prices
        
        result = fetch_and_store_prices(
            ticker='GOOG',
            db_path=temp_db,
            period='5d'
        )
        
        assert result is False, "fetch_and_store_prices should return False on failure"
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM price_history WHERE ticker = ?", ('GOOG',))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 0, "No prices should be stored on failure"


def test_duplicate_prevention(temp_db: str, mock_yfinance_data: pd.DataFrame) -> None:
    """Verify UNIQUE constraint on (ticker, date) prevents duplicate inserts."""
    with mock.patch('yfinance.download', return_value=mock_yfinance_data):
        from sentinel.scout.live_prices import fetch_and_store_prices
        
        fetch_and_store_prices(ticker='TSLA', db_path=temp_db, period='5d')
        fetch_and_store_prices(ticker='TSLA', db_path=temp_db, period='5d')
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM price_history WHERE ticker = ?", ('TSLA',))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 5, "Duplicate inserts should be ignored; still 5 rows"


def test_multiple_tickers_isolated(temp_db: str, mock_yfinance_data: pd.DataFrame) -> None:
    """Test that multiple ticker records remain isolated in the database."""
    with mock.patch('yfinance.download', return_value=mock_yfinance_data):
        from sentinel.scout.live_prices import fetch_and_store_prices
        
        fetch_and_store_prices(ticker='AAPL', db_path=temp_db, period='5d')
        fetch_and_store_prices(ticker='MSFT', db_path=temp_db, period='5d')
        fetch_and_store_prices(ticker='GOOG', db_path=temp_db, period='5d')
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM price_history")
        total = cursor.fetchone()[0]
        
        for ticker in ['
