"""US equity session calendar and timeframe helpers.

Used by the sim broker, the backtester and as a fallback when the live broker clock is
unavailable. Regular session 09:30-16:00 America/New_York; pre 04:00-09:30; post 16:00-20:00.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
UTC = timezone.utc

# NYSE full-day closures (keep a couple of years ahead; the live broker clock is authoritative when available).
HOLIDAYS: set[date] = {
    date(2025, 1, 1), date(2025, 1, 9), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),
    date(2025, 9, 1), date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3),
    date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5),
    date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24),
}
# Early closes at 13:00 ET.
EARLY_CLOSES: set[date] = {
    date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
    date(2026, 11, 27), date(2026, 12, 24),
    date(2027, 11, 26),
}

TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_ny(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(NY)


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in HOLIDAYS


def session_bounds(d: date) -> tuple[datetime, datetime] | None:
    """(open, close) of the regular session on day d in UTC, or None if closed."""
    if not is_trading_day(d):
        return None
    open_ = datetime.combine(d, time(9, 30), tzinfo=NY)
    close = datetime.combine(d, time(13, 0) if d in EARLY_CLOSES else time(16, 0), tzinfo=NY)
    return open_.astimezone(UTC), close.astimezone(UTC)


def session_at(dt: datetime) -> str:
    """'regular' | 'pre' | 'post' | 'closed' for a UTC datetime."""
    ny = to_ny(dt)
    d = ny.date()
    if not is_trading_day(d):
        return "closed"
    t = ny.timetz().replace(tzinfo=None)
    close_t = time(13, 0) if d in EARLY_CLOSES else time(16, 0)
    if time(9, 30) <= t < close_t:
        return "regular"
    if time(4, 0) <= t < time(9, 30):
        return "pre"
    if close_t <= t < time(20, 0):
        return "post"
    return "closed"


def is_regular_session(dt: datetime) -> bool:
    return session_at(dt) == "regular"


def next_open_close(dt: datetime) -> tuple[datetime, datetime]:
    """Next regular-session open and the close that follows the current moment."""
    ny = to_ny(dt)
    d = ny.date()
    for i in range(0, 10):
        day = d + timedelta(days=i)
        b = session_bounds(day)
        if not b:
            continue
        o, c = b
        if dt < o:
            return o, c
        if o <= dt < c:
            # currently open: next open is the following trading day
            for j in range(1, 10):
                nb = session_bounds(day + timedelta(days=j))
                if nb:
                    return nb[0], c
        # after close: continue to next day
    raise RuntimeError("no trading day found within 10 days")


def minutes_since_open(dt: datetime) -> float | None:
    ny = to_ny(dt)
    b = session_bounds(ny.date())
    if not b:
        return None
    return (dt - b[0]).total_seconds() / 60.0


def floor_ts(ts: int, tf: str) -> int:
    """Floor an epoch-seconds timestamp to the start of its timeframe bucket.

    Intraday buckets are aligned to UTC multiples (for 4h this gives 00/04/08... UTC, which
    is the usual crypto convention). Daily bars are aligned to the US session date.
    """
    secs = TF_SECONDS[tf]
    if tf == "1d":
        ny = datetime.fromtimestamp(ts, UTC).astimezone(NY)
        d0 = datetime.combine(ny.date(), time(0, 0), tzinfo=NY)
        return int(d0.timestamp())
    return ts - (ts % secs)


def bar_closed(ts: int, tf: str, now_ts: int) -> bool:
    """True if the bucket containing ts has fully elapsed at now_ts."""
    return floor_ts(ts, tf) + TF_SECONDS[tf] <= now_ts


def parse_schedule(spec: str) -> tuple[int | None, int, int]:
    """'16:35' -> (None, 16, 35); 'Sun 10:00' -> (6, 10, 0). Weekday is Monday=0."""
    parts = spec.strip().split()
    wd: int | None = None
    if len(parts) == 2:
        names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        wd = names.index(parts[0][:3].lower())
        hm = parts[1]
    else:
        hm = parts[0]
    h, m = hm.split(":")
    return wd, int(h), int(m)


def next_fire(spec: str, after: datetime) -> datetime:
    """Next America/New_York wall-clock occurrence of a schedule spec strictly after `after`."""
    wd, h, m = parse_schedule(spec)
    ny = to_ny(after)
    for i in range(0, 8):
        day = ny.date() + timedelta(days=i)
        if wd is not None and day.weekday() != wd:
            continue
        cand = datetime.combine(day, time(h, m), tzinfo=NY)
        if cand > ny:
            return cand.astimezone(UTC)
    raise RuntimeError(f"could not schedule {spec}")
