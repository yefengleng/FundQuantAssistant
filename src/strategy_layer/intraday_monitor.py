"""Intraday monitor table and emergency-rebalance detection from realtime estimates."""

from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import RETREAT_LIMIT, WEEKLY_SCAN
from src.data_layer.realtime_cache import COOLDOWN_NOTE, quote_is_valid
from src.strategy_layer.signal_generator import _calc_account_drawdown


CRASH_WARN_PCT = -5.0
EMERGENCY_FUND_PCT = -7.0
EMERGENCY_SECTOR_PCT = -5.0
CLEAR_FUND_PCT = -10.0
OPERATION_DEADLINE_HM = "14:50"


def _scan_cfg():
    cfg = WEEKLY_SCAN if isinstance(WEEKLY_SCAN, dict) else {}

    def _float(key, default):
        try:
            return float(cfg.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    return {
        "retreat_limit": _float("retreat_limit", RETREAT_LIMIT),
        "fund_intraday_drop_pct": _float("fund_intraday_drop_pct", EMERGENCY_FUND_PCT),
        "sector_intraday_drop_pct": _float("sector_intraday_drop_pct", EMERGENCY_SECTOR_PCT),
        "crash_warn_pct": _float("crash_warn_pct", CRASH_WARN_PCT),
        "clear_fund_pct": _float("clear_fund_pct", CLEAR_FUND_PCT),
        "reduce_ratio": min(max(_float("reduce_ratio", 0.30), 0.05), 0.90),
        "meltdown_equity_cap": min(max(_float("meltdown_equity_cap", 0.50), 0.10), 0.90),
        "emergency_reduce_ratio": min(max(_float("emergency_reduce_ratio", 0.50), 0.05), 0.95),
    }


def _quote_for(items, code):
    return (items or {}).get(str(code).strip()) or {}


def overlay_holdings_estimates(holdings, estimate_payload):
    """
    Attach 估算净值 / 当日估算涨跌幅 / 当日估算盈亏 / 估算市值.
    If every quote fails, fall back to yesterday NAV (change=0, pnl=0).
    """
    df = holdings.copy() if holdings is not None else pd.DataFrame()
    empty_meta = {
        "ok_count": 0,
        "fail_count": 0,
        "degraded": False,
        "updated_at": "",
        "picked": "",
        "avg_change": None,
        "total_pnl": 0.0,
        "total_est_mv": 0.0,
    }
    if df is None or df.empty:
        return df, empty_meta

    payload = estimate_payload or {}
    items = payload.get("items") or {}
    shares = pd.to_numeric(df.get("持有份额"), errors="coerce")
    last_nav = pd.to_numeric(df.get("最新净值"), errors="coerce")
    sector = df.get("赛道归类")
    if sector is None:
        df["赛道归类"] = "未分类"
    else:
        df["赛道归类"] = sector.fillna("").astype(str).str.strip().replace("", "未分类")

    est_navs = []
    changes = []
    pnls = []
    est_mvs = []
    sources = []
    ok_count = 0
    fail_count = 0
    for idx, row in df.iterrows():
        code = str(row.get("基金代码", "")).strip()
        share = float(shares.at[idx]) if pd.notna(shares.at[idx]) else 0.0
        nav = float(last_nav.at[idx]) if pd.notna(last_nav.at[idx]) else float("nan")
        quote = _quote_for(items, code)
        estimate = pd.to_numeric(quote.get("nav_estimate"), errors="coerce")
        change_pct = pd.to_numeric(quote.get("change_pct"), errors="coerce")
        live = quote_is_valid(quote)
        if live:
            if pd.isna(estimate) and pd.notna(nav) and pd.notna(change_pct):
                estimate = nav * (1.0 + float(change_pct) / 100.0)
            if pd.isna(change_pct) and pd.notna(estimate) and pd.notna(nav) and nav != 0:
                change_pct = (float(estimate) / nav - 1.0) * 100.0
            ok_count += 1
            sources.append(str(quote.get("source") or "realtime"))
        else:
            fail_count += 1
            estimate = nav
            change_pct = 0.0 if pd.notna(nav) else float("nan")
            sources.append("yesterday_nav")
        pnl = float("nan")
        est_mv = float("nan")
        if share > 0 and pd.notna(estimate):
            est_mv = share * float(estimate)
            if pd.notna(nav):
                pnl = share * (float(estimate) - nav)
            elif pd.notna(change_pct):
                pnl = share * float(estimate) * float(change_pct) / (100.0 + float(change_pct)) if float(change_pct) != -100 else 0.0
        est_navs.append(float(estimate) if pd.notna(estimate) else float("nan"))
        changes.append(float(change_pct) if pd.notna(change_pct) else float("nan"))
        pnls.append(float(pnl) if pd.notna(pnl) else float("nan"))
        est_mvs.append(float(est_mv) if pd.notna(est_mv) else float("nan"))

    df["估算净值"] = est_navs
    df["当日估算涨跌幅"] = changes
    df["当日估算盈亏"] = pnls
    df["估算市值"] = est_mvs
    df["估值来源"] = sources
    degraded = ok_count == 0 and fail_count > 0
    weight = pd.to_numeric(df["估算市值"], errors="coerce").fillna(0.0)
    chg = pd.to_numeric(df["当日估算涨跌幅"], errors="coerce")
    mask = (weight > 0) & chg.notna()
    avg_change = float(np.average(chg.loc[mask], weights=weight.loc[mask])) if mask.any() else None
    meta = {
        "ok_count": int(ok_count),
        "fail_count": int(fail_count),
        "degraded": bool(degraded),
        "updated_at": str(payload.get("updated_at") or ""),
        "picked": str(payload.get("picked") or ""),
        "avg_change": avg_change,
        "total_pnl": float(pd.to_numeric(df["当日估算盈亏"], errors="coerce").sum(min_count=1) or 0.0),
        "total_est_mv": float(weight.sum()),
    }
    return df, meta


def build_sector_intraday(monitor_df):
    if monitor_df is None or monitor_df.empty:
        return pd.DataFrame(columns=["赛道", "估算市值", "当日估算涨跌幅", "当日估算盈亏", "基金只数"])
    rows = []
    for sector, group in monitor_df.groupby("赛道归类", sort=False):
        weights = pd.to_numeric(group["估算市值"], errors="coerce").fillna(0.0)
        rets = pd.to_numeric(group["当日估算涨跌幅"], errors="coerce")
        pnl = pd.to_numeric(group["当日估算盈亏"], errors="coerce")
        mask = (weights > 0) & rets.notna()
        avg = float(np.average(rets.loc[mask], weights=weights.loc[mask])) if mask.any() else float("nan")
        rows.append(
            {
                "赛道": str(sector),
                "估算市值": float(weights.sum()),
                "当日估算涨跌幅": avg,
                "当日估算盈亏": float(pnl.sum(min_count=1) or 0.0),
                "基金只数": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("当日估算涨跌幅", ascending=True, na_position="last")
    return out


def calc_intraday_drawdown(holdings, monitor_df=None):
    """Historical peak vs current estimated market value (decimal)."""
    overrides = None
    if monitor_df is not None and not monitor_df.empty and "估算净值" in monitor_df.columns:
        overrides = {}
        for _, row in monitor_df.iterrows():
            code = str(row.get("基金代码", "")).strip()
            nav = pd.to_numeric(row.get("估算净值"), errors="coerce")
            if code and pd.notna(nav) and float(nav) > 0:
                overrides[code] = float(nav)
    return float(_calc_account_drawdown(holdings, nav_overrides=overrides))


def detect_emergency(monitor_df, sector_df, account_drawdown, cfg=None):
    """
    Emergency if any:
    - single fund estimated daily change < fund threshold (default -7%)
    - a sector estimated daily change < sector threshold (default -5%)
    - account drawdown < -18%
    """
    cfg = cfg or _scan_cfg()
    alerts = []
    fund_hits = []
    sector_hits = []
    fund_cut = float(cfg["fund_intraday_drop_pct"])
    sector_cut = float(cfg["sector_intraday_drop_pct"])
    clear_cut = float(cfg["clear_fund_pct"])
    retreat = float(cfg["retreat_limit"])
    crash_warn = float(cfg["crash_warn_pct"])

    if monitor_df is not None and not monitor_df.empty:
        for _, row in monitor_df.iterrows():
            change = pd.to_numeric(row.get("当日估算涨跌幅"), errors="coerce")
            if pd.isna(change):
                continue
            chg = float(change)
            crash = chg <= crash_warn
            emergency = chg <= fund_cut
            if emergency or crash:
                fund_hits.append(
                    {
                        "基金代码": str(row.get("基金代码", "")).strip(),
                        "基金名称": str(row.get("基金名称") or row.get("基金代码") or ""),
                        "赛道归类": str(row.get("赛道归类") or "未分类"),
                        "当日估算涨跌幅": chg,
                        "急跌预警": crash,
                        "紧急": emergency,
                        "清仓": chg <= clear_cut,
                    }
                )
            if emergency:
                alerts.append(
                    f"{row.get('基金名称') or row.get('基金代码')} 当日估算跌幅 {chg:.2f}%"
                )

    if sector_df is not None and not sector_df.empty:
        for _, row in sector_df.iterrows():
            change = pd.to_numeric(row.get("当日估算涨跌幅"), errors="coerce")
            if pd.isna(change):
                continue
            chg = float(change)
            if chg <= sector_cut:
                sector_hits.append(
                    {
                        "赛道": str(row.get("赛道") or ""),
                        "当日估算涨跌幅": chg,
                    }
                )
                alerts.append(f"{row.get('赛道')}赛道当日估算跌幅 {chg:.2f}%")

    melt = pd.notna(account_drawdown) and float(account_drawdown) < retreat
    if melt:
        alerts.append(f"账户总回撤 {float(account_drawdown) * 100:.2f}% 突破 {retreat * 100:.0f}%")

    return {
        "triggered": bool(any(item.get("紧急") for item in fund_hits) or sector_hits or melt),
        "meltdown": bool(melt),
        "alerts": alerts,
        "fund_hits": fund_hits,
        "sector_hits": sector_hits,
        "crash_funds": [item for item in fund_hits if item.get("急跌预警")],
        "account_drawdown": float(account_drawdown or 0.0),
        "cfg": cfg,
    }


def operation_deadline_text(ts=None):
    day = pd.Timestamp(ts or datetime.now()).strftime("%Y-%m-%d")
    return f"{day} {OPERATION_DEADLINE_HM}"


def annotate_emergency_row(reason, bypass_cooldown=True):
    text = str(reason or "").strip()
    if bypass_cooldown and COOLDOWN_NOTE not in text:
        text = f"{text}；{COOLDOWN_NOTE}" if text else COOLDOWN_NOTE
    return text


def build_monitor_bundle(holdings, estimate_payload):
    monitor_df, meta = overlay_holdings_estimates(holdings, estimate_payload)
    sector_df = build_sector_intraday(monitor_df)
    drawdown = calc_intraday_drawdown(holdings, monitor_df)
    emergency = detect_emergency(monitor_df, sector_df, drawdown)
    if monitor_df is not None and not monitor_df.empty:
        monitor_df = monitor_df.sort_values("当日估算涨跌幅", ascending=True, na_position="last")
        crash_codes = {item["基金代码"] for item in emergency.get("crash_funds") or []}
        monitor_df["预警"] = monitor_df["基金代码"].astype(str).str.strip().map(
            lambda code: "⚠️ 急跌预警" if code in crash_codes else ""
        )
    meta["account_drawdown"] = drawdown
    meta["emergency"] = emergency
    return monitor_df, sector_df, meta
