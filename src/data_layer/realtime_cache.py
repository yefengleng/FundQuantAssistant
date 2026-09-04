"""Realtime estimate cache (TTL ≤ 5 minutes) and 14:00-14:30 snapshot pick."""

import json
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from src.data_layer.loader import PROJECT_ROOT
from src.data_layer.market_clock import REFRESH_SECONDS


REALTIME_PATH = Path(PROJECT_ROOT) / "data" / "realtime_estimates.json"
MONITOR_LOG_PATH = Path(PROJECT_ROOT) / "data" / "logs" / "intraday.log"
SIGNAL_WINDOW_START = time(14, 0)
SIGNAL_WINDOW_END = time(14, 30)
MAX_SNAPSHOTS = 48
COOLDOWN_NOTE = "⚠️ 此为紧急调仓，突破冷却期限制。"
DEADLINE_HINT = "⏰ 建议在今日 15:00 前完成操作，调仓按当日净值确认。"
EMERGENCY_BANNER = "🚨 今日触发紧急调仓信号，请查看详情。"
DEGRADE_CAPTION = "实时估值不可用，基于昨日净值估算"


def log_monitor(event, message):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} [{event}] {message}"
    try:
        from src.data_layer.fetcher import _safe_print

        _safe_print(line)
    except Exception:
        try:
            print(line)
        except Exception:
            pass
    try:
        MONITOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MONITOR_LOG_PATH, "a", encoding="utf-8") as file:
            file.write(line + "\n")
    except Exception:
        pass


def load_realtime_payload():
    try:
        if not REALTIME_PATH.exists():
            return {}
        with open(REALTIME_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _sanitize_items(items):
    safe = {}
    for code, quote in (items or {}).items():
        key = str(code).strip()
        if not key or not isinstance(quote, dict):
            continue
        nav = quote.get("nav_estimate")
        chg = quote.get("change_pct")
        try:
            nav_ok = nav is not None and pd.notna(nav)
        except Exception:
            nav_ok = False
        try:
            chg_ok = chg is not None and pd.notna(chg)
        except Exception:
            chg_ok = False
        row = {
            "nav_estimate": float(nav) if nav_ok else None,
            "change_pct": float(chg) if chg_ok else None,
        }
        gztime = quote.get("gztime")
        if gztime:
            row["gztime"] = str(gztime)
        source = quote.get("source")
        if source:
            row["source"] = str(source)
        safe[key] = row
    return safe


def save_realtime_payload(items, updated_at=None, extra=None):
    now_text = updated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous = load_realtime_payload()
    snapshots = list(previous.get("snapshots") or [])
    snapshots.append({"at": now_text, "items": _sanitize_items(items)})
    if len(snapshots) > MAX_SNAPSHOTS:
        snapshots = snapshots[-MAX_SNAPSHOTS:]
    payload = {
        "updated_at": now_text,
        "items": _sanitize_items(items),
        "snapshots": snapshots,
    }
    if extra:
        payload.update(extra)
    REALTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REALTIME_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return payload


def cache_age_seconds(payload, ts=None):
    updated = (payload or {}).get("updated_at")
    parsed = pd.to_datetime(updated, errors="coerce")
    if pd.isna(parsed):
        return None
    now = pd.Timestamp(ts or datetime.now())
    return float((now - parsed).total_seconds())


def is_cache_stale(payload, ttl=REFRESH_SECONDS, ts=None):
    age = cache_age_seconds(payload, ts=ts)
    if age is None:
        return True
    return age >= float(ttl)


def _in_signal_window(ts):
    clock = pd.Timestamp(ts).to_pydatetime().time()
    return SIGNAL_WINDOW_START <= clock <= SIGNAL_WINDOW_END


def pick_signal_estimates(payload, ts=None):
    """
    Prefer the latest snapshot inside 14:00-14:30 (closest to close).
    Fall back to the latest cache.
    """
    payload = payload or {}
    now = pd.Timestamp(ts or datetime.now())
    today = now.normalize()
    best = None
    for snap in payload.get("snapshots") or []:
        at = pd.to_datetime(snap.get("at"), errors="coerce")
        if pd.isna(at) or at.normalize() != today:
            continue
        if not _in_signal_window(at):
            continue
        items = snap.get("items") or {}
        if not items:
            continue
        if best is None or at >= best[0]:
            best = (at, items)
    if best is not None:
        return {"updated_at": best[0].strftime("%Y-%m-%d %H:%M:%S"), "items": best[1], "picked": "window_14_30"}
    return {
        "updated_at": payload.get("updated_at") or "",
        "items": payload.get("items") or {},
        "picked": "latest",
    }


def quote_is_valid(quote):
    if not isinstance(quote, dict):
        return False
    nav = pd.to_numeric(quote.get("nav_estimate"), errors="coerce")
    chg = pd.to_numeric(quote.get("change_pct"), errors="coerce")
    return pd.notna(nav) or pd.notna(chg)
