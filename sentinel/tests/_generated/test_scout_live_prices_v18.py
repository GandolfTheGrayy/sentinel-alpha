"""
Unit tests for sentinel/scout/live_prices.py — the live price fetcher module.

Tests mocking yfinance responses and validating SQLite writes to the price cache.
Part of Sentinel's test spine, run via pytest. Validates that:
  1. yfinance calls succeed and return OHLCV data
  2. Fallback to stooq works when yfinance fails
  3. Price records are written correctly to SQLite with timestamps
  4. Duplicate writes are handled gracefully (upsert logic)
  5. Error handling for invalid tickers and network failures
"""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest


@pytest.fixture
def temp_db() -> str:
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def mock_yfinance_data() -> pd.DataFrame:
    """Return a mock OHLCV DataFrame matching yfinance structure."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'Open': [100.0, 101.0, 102.0, 101.5, 103.0],
        'High': [102.0, 103.0, 104.0, 103.5, 105.0],
        'Low': [99.0, 100.0, 101.0, 100.5, 102.0],
        'Close': [101.0, 102.0, 103.0, 102.5, 104.0],
        'Volume': [1000000, 1100000, 950000, 1200000, 1050000],
    }, index=dates)


def test_live_prices_fetch_success(mock_yfinance_data: pd.DataFrame, temp_db: str) -> None:
    """Test successful fetch from yfinance and SQLite write."""
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = mock_yfinance_data
        
        # Import after patching to avoid early binding
        from sentinel.scout import live_prices
        
        result = live_prices.fetch_live_prices(
            tickers=['AAPL'],
            db_path=temp_db
        )
        
        assert result is not None
        assert 'AAPL' in result
        assert len(result['AAPL']) == 5
        assert result['AAPL'].loc[result['AAPL'].index[0], 'Close'] == 101.0


def test_live_prices_sqlite_write(mock_yfinance_data: pd.DataFrame, temp_db: str) -> None:
    """Test that price records are correctly written to SQLite."""
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = mock_yfinance_data
        
        from sentinel.scout import live_prices
        
        live_prices.fetch_live_prices(
            tickers=['AAPL'],
            db_path=temp_db
        )
        
        # Verify SQLite writes
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM prices WHERE ticker = ?", ('AAPL',))
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 5, f"Expected 5 rows in DB, got {count}"


def test_live_prices_multiple_tickers(temp_db: str) -> None:
    """Test fetching multiple tickers in a single call."""
    mock_aapl = pd.DataFrame({
        'Open': [100.0],
        'High': [102.0],
        'Low': [99.0],
        'Close': [101.0],
        'Volume': [1000000],
    }, index=pd.date_range(start='2024-01-01', periods=1, freq='D'))
    
    mock_msft = pd.DataFrame({
        'Open': [300.0],
        'High': [302.0],
        'Low': [299.0],
        'Close': [301.0],
        'Volume': [800000],
    }, index=pd.date_range(start='2024-01-01', periods=1, freq='D'))
    
    with patch('yfinance.download') as mock_download:
        # Return different data based on ticker
        def side_effect(tickers, **kwargs):
            if isinstance(tickers, list):
                if len(tickers) == 1:
                    return mock_aapl if tickers[0] == 'AAPL' else mock_msft
            return pd.concat([mock_aapl, mock_msft], keys=['AAPL', 'MSFT'])
        
        mock_download.side_effect = side_effect
        
        from sentinel.scout import live_prices
        
        result = live_prices.fetch_live_prices(
            tickers=['AAPL', 'MSFT'],
            db_path=temp_db
        )
        
        assert 'AAPL' in result or len(result) > 0


def test_live_prices_yfinance_failure_fallback(temp_db: str) -> None:
    """Test fallback to stooq when yfinance fails."""
    stooq_data = pd.DataFrame({
        'Open': [100.0],
        'High': [102.0],
        'Low': [99.0],
        'Close': [101.0],
        'Volume': [1000000],
    }, index=pd.date_range(start='2024-01-01', periods=1, freq='D'))
    
    with patch('yfinance.download') as mock_yf, \
         patch('sentinel.scout.live_prices._fetch_from_stooq') as mock_stooq:
        mock_yf.side_effect = Exception("yfinance network error")
        mock_stooq.return_value = stooq_data
        
        from sentinel.scout import live_prices
        
        result = live_prices.fetch_live_prices(
            tickers=['AAPL'],
            db_path=temp_db,
            use_fallback=True
        )
        
        # Should have called fallback
        mock_stooq.assert_called()


def test_live_prices_invalid_ticker(temp_db: str) -> None:
    """Test handling of invalid ticker symbols."""
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = pd.DataFrame()
        
        from sentinel.scout import live_prices
        
        result = live_prices.fetch_live_prices(
            tickers=['INVALID_TICKER_XYZ'],
            db_path=temp_db
        )
        
        # Should return empty or gracefully handle
        assert result is None or len(result) == 0


def test_live_prices_upsert_duplicate(mock_yfinance_data: pd.DataFrame, temp_db: str) -> None:
    """Test that duplicate timestamp writes are handled via upsert."""
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = mock_yfinance_data
        
        from sentinel.scout import live_prices
        
        # First write
        live_prices.fetch_live_prices(
            tickers=['AAPL'],
            db_path=temp_db
        )
        
        # Second write (same data, should upsert)
        live_prices.fetch_live_prices(
            tickers=['AAPL'],
            db_path=temp_db
        )
        
        # Verify no duplicates
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor
