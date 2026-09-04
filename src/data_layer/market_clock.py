"""A-share trading calendar and session clock (local time, China)."""

from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from src.data_layer.loader import PROJECT_ROOT


TRADING_START = time(9, 30)
TRADING_END = time(15, 0)
REFRESH_SECONDS = 300
TRADE_DATES_CACHE = Path(PROJECT_ROOT) / "data" / "raw" / "_trade_dates.parquet"
TRADE_DATES_TTL = timedelta(days=7)

# Official-ish A-share closed days used when akshare calendar is unavailable.
# Weekends are handled separately; this set is holidays and weekday closures.
CN_CLOSED_DAYS = {
    date(2026, 1, 1),
    date(2026, 1, 2),
    date(2026, 2, 15),
    date(2026, 2, 16),
    date(2026, 2, 17),
    date(2026, 2, 18),
    date(2026, 2, 19),
    date(2026, 2, 20),
    date(2026, 2, 21),
    date(2026, 2, 22),
    date(2026, 2, 23),
    date(2026, 4, 4),
    date(2026, 4, 5),
    date(2026, 4, 6),
    date(2026, 5, 1),
    date(2026, 5, 2),
    date(2026, 5, 3),
    date(2026, 5, 4),
    date(2026, 5, 5),
    date(2026, 6, 19),
    date(2026, 6, 20),
    date(2026, 6, 21),
    date(2026, 9, 25),
    date(2026, 9, 26),
    date(2026, 9, 27),
    date(2026, 10, 1),
    date(2026, 10, 2),
    date(2026, 10, 3),
    date(2026, 10, 4),
    date(2026, 10, 5),
    date(2026, 10, 6),
    date(2026, 10, 7),
}

# Makeup trading days that fall on weekends.
CN_MAKEUP_DAYS = {
    date(2026, 2, 14),
    date(2026, 2, 28),
    date(2026, 10, 10),
}

_TRADE_DATES = None
_TRADE_DATES_LOADED_AT = None


def _now(ts=None):
    if ts is None:
        return pd.Timestamp(datetime.now())
    return pd.Timestamp(ts)


def _load_trade_dates():
    """Load exchange calendar; cache to parquet. Empty set means fallback mode."""
    global _TRADE_DATES, _TRADE_DATES_LOADED_AT
    now = datetime.now()
    if _TRADE_DATES is not None and _TRADE_DATES_LOADED_AT is not None:
        if now - _TRADE_DATES_LOADED_AT < timedelta(hours=12):
            return _TRADE_DATES

    dates = set()
    cache = TRADE_DATES_CACHE
    try:
        if cache.exists():
            mtime = datetime.fromtimestamp(cache.stat().st_mtime)
            if now - mtime < TRADE_DATES_TTL:
                cached = pd.read_parquet(cache)
                col = cached.columns[0]
                dates = {
                    pd.Timestamp(value).date()
                    for value in cached[col].tolist()
                    if pd.notna(value)
                }
                if dates:
                    _TRADE_DATES = dates
                    _TRADE_DATES_LOADED_AT = now
                    return dates
    except Exception:
        dates = set()

    try:
        import akshare as ak

        raw = None
        func = getattr(ak, "tool_trade_date_hist_sina", None)
        if callable(func):
            try:
                raw = func()
            except Exception:
                raw = None
        if isinstance(raw, pd.DataFrame) and not raw.empty:
            col = None
            for candidate in ("trade_date", "日期", "date"):
                if candidate in raw.columns:
                    col = candidate
                    break
            if col is None:
                col = raw.columns[0]
            parsed = pd.to_datetime(raw[col], errors="coerce").dropna()
            dates = {ts.date() for ts in parsed}
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"trade_date": sorted(dates)}).to_parquet(cache, index=False)
            except Exception:
                pass
    except Exception:
        dates = set()

    _TRADE_DATES = dates
    _TRADE_DATES_LOADED_AT = now
    return dates


def is_trading_day(ts=None):
    """True on A-share trading days (excludes weekends and holidays)."""
    day = _now(ts).date()
    calendar = _load_trade_dates()
    if calendar:
        first = min(calendar)
        last = max(calendar)
        if first <= day <= last:
            return day in calendar
    if day in CN_MAKEUP_DAYS:
        return True
    if day.weekday() >= 5:
        return False
    return day not in CN_CLOSED_DAYS


def is_trading_session(ts=None):
    """True during 09:30-15:00 on a trading day (pandas Timestamp comparison)."""
    now = _now(ts)
    if not is_trading_day(now):
        return False
    start = now.normalize() + pd.Timedelta(hours=9, minutes=30)
    end = now.normalize() + pd.Timedelta(hours=15)
    return start <= now <= end


def seconds_until_refresh(updated_at, interval=REFRESH_SECONDS, ts=None):
    """Seconds remaining until the next 5-minute refresh. 0 if stale/missing."""
    now = _now(ts)
    parsed = pd.to_datetime(updated_at, errors="coerce")
    if pd.isna(parsed):
        return 0
    elapsed = (now - parsed).total_seconds()
    remain = int(interval) - int(elapsed)
    return max(remain, 0)
