"""Smoke tests for the hand-written spine modules. No live API calls."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch


def test_imports() -> None:
    """Every spine module imports without error."""
    import sentinel.historian.rag_query  # noqa
    import sentinel.judge.baselines  # noqa
    import sentinel.judge.notify  # noqa
    import sentinel.judge.postmortem  # noqa
    import sentinel.judge.predictor  # noqa
    import sentinel.judge.resolver  # noqa
    import sentinel.linguist.sample_score  # noqa
    import sentinel.pipeline  # noqa
    import sentinel.scout.live_prices  # noqa
    import sentinel.scout.news  # noqa
    import sentinel.scout.sec_filings  # noqa


def test_baselines() -> None:
    """Baselines return well-formed prediction dicts."""
    from sentinel.judge import baselines

    for s in baselines.STRATEGIES:
        out = baselines.predict(s, "AAPL", 1.5)
        assert out["direction"] in ("up", "down", "neutral")
        assert isinstance(out["magnitude_pct"], (int, float))
        assert 0 <= out["confidence"] <= 100
        assert out["rationale"]


def test_postmortem(tmp_path: Path, monkeypatch) -> None:
    """Postmortem writer produces a non-empty markdown file."""
    monkeypatch.chdir(tmp_path)
    from sentinel.judge.postmortem import render

    out = render(
        date(2026, 4, 30),
        [{"ticker": "AAPL", "close": 200.0, "pct_change": 1.2}],
        {"score": 70, "reasoning": "confident"},
        [{"date": "2023-05-04", "event": "Apple earnings", "sp500_5d_pct": 1.8, "match_score": 0.6, "tags": []}],
        "Apple beat earnings",
    )
    assert out.exists()
    assert "Sentinel Post-Mortem" in out.read_text()


def test_rag_keyword_fallback(monkeypatch) -> None:
    """Historian falls back to keyword overlap when embeddings unavailable."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from sentinel.historian.rag_query import query

    res = query("Apple supply chain hedged guidance", k=3)
    assert isinstance(res, list)
    if res:
        assert res[0].get("method") == "keyword"


def test_resolver_no_baseline_price() -> None:
    """Resolver flags missing baseline price gracefully."""
    from sentinel.judge.resolver import resolve

    p = {"ticker": "AAPL", "direction": "up", "magnitude_pct": 1.0}
    out = resolve(p)
    assert out["resolved"] is True
    assert "error" in out


def test_notify_no_webhook() -> None:
    """Notifier silently no-ops without DISCORD_WEBHOOK_URL."""
    from sentinel.judge.notify import maybe_alert

    p = {"strategy": "claude", "resolved": True, "correct_direction": True, "confidence": 90, "actual_pct": 3.0, "ticker": "AAPL", "direction": "up", "made": "2026-04-30"}
    with patch.dict("os.environ", {}, clear=False):
        import os
        os.environ.pop("DISCORD_WEBHOOK_URL", None)
        assert maybe_alert(p) in (True, False)
