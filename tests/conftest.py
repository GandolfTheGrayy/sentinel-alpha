import os
import sys
import tempfile
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel.config import Settings  # noqa: E402
from sentinel.data.market import MarketData  # noqa: E402
from sentinel.data.synthetic import SyntheticProvider  # noqa: E402
from sentinel.store.db import Database  # noqa: E402

NOW = datetime(2026, 10, 8, 20, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def settings() -> Settings:
    s = Settings()
    s.universe.equities = ["SPY", "QQQ", "AAPL", "MSFT"]
    s.universe.crypto = ["BTC/USD", "ETH/USD"]
    s.data_dir = __import__("pathlib").Path(tempfile.mkdtemp())
    s.sim = True
    return s


@pytest.fixture(scope="session")
def db(settings) -> Database:
    return Database(settings.db_path)


@pytest.fixture(scope="session")
def md(db, settings) -> MarketData:
    prov = SyntheticProvider(clock=lambda: NOW)
    m = MarketData(db, prov, settings.universe.equities, settings.universe.crypto)
    m.backfill(minute_days=25, daily_days=300, now=NOW, log=lambda msg: None)
    return m
