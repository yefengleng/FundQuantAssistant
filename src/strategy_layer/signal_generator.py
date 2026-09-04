import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.data_layer.loader import get_signal_path, load_local_data
from src.factor_layer.portfolio_utils import load_current_holdings
from src.factor_layer.scorer import batch_score_funds

from config.settings import ENABLE_SECTOR_TOP_N, SECTOR_TOP_N, is_sector_top_n_active

from .constraints import (
    INSUFFICIENT_INSTRUCTION,
    KEEP_INSTRUCTION,
    MELTDOWN_INSTRUCTION,
    REDEEM_INSTRUCTION,
    REDUCE_INSTRUCTION,
    apply_equity_cap,
    apply_retreat_meltdown,
    apply_sector_elite,
    apply_sector_limits,
    apply_single_fund_limit,
)
from .cooldown import apply_cooldown
from .filters import check_market_crash, get_market_regime, inspect_market_regime, resolve_strategy_mode


CRASH_NOTE = "市场急跌超12%，暂停新开仓"
INSUFFICIENT_NOTE = "数据不足，暂不操作"
OPEN_INSTRUCTIONS = {"买入", "加仓"}
SECTOR_BUFFER = 0.02
SCORE_EDGE_RATIO = 1.1
CANDIDATE_COLUMNS = [
    "基金代码",
    "基金名称",
    "赛道",
    "综合得分",
    "近60日收益",
    "近60日回撤",
    "当前赛道剩余空间（%）",
    "备注",
]


def _active_holdings(holdings_df):
    """仅保留真实持仓，观察池不进入存量风控。"""
    if holdings_df is None or holdings_df.empty:
        return holdings_df.copy() if holdings_df is not None else pd.DataFrame()
    df = holdings_df.copy()
    shares = pd.to_numeric(df.get("持有份额"), errors="coerce")
    market_value = pd.to_numeric(df.get("持仓市值"), errors="coerce")
    mask = (shares.fillna(0) > 0) & (market_value.fillna(0) > 0)
    return df.loc[mask].reset_index(drop=True)


def _sector_limit(sector, sector_limits=None):
    limits = sector_limits or {}
    other = limits.get("其他", 1.0)
    if sector is None or (isinstance(sector, float) and np.isnan(sector)):
        return other
    return limits.get(str(sector), other)


def _attach_mode_state(df, mode_state):
    payload = df if df is not None else pd.DataFrame()
    payload.attrs["strategy_mode_requested"] = mode_state.get("requested")
    payload.attrs["strategy_mode_effective"] = mode_state.get("effective")
    payload.attrs["strategy_mode_fetch_failed"] = bool(mode_state.get("fetch_failed"))
    payload.attrs["strategy_mode_banner"] = mode_state.get("banner") or ""
    payload.attrs["strategy_mode_profile"] = mode_state.get("profile") or {}
    return payload


def _empty_signal_frame(mode_state):
    empty = pd.DataFrame()
    empty.attrs["meltdown_triggered"] = False
    empty.attrs["crash_filter_triggered"] = False
    empty.attrs["account_return"] = 0.0
    empty.attrs["sector_top_n_sectors"] = 0
    empty.attrs["sector_top_n_funds"] = 0
    empty.attrs["sector_top_n_banner"] = ""
    empty.attrs["sector_elite_note"] = ""
    empty.attrs["sector_elite_cash"] = 0.0
    empty.attrs["operation_frequency"] = "monthly"
    empty.attrs["operation_frequency_banner"] = "📅 当前运行频率：月度调仓"
    return _attach_mode_state(empty, mode_state)


def _calc_account_drawdown(holdings_df, nav_overrides=None):
    """用当前份额与买入日后净值重建账户曲线，计算相对最高点的回撤（小数）。

    nav_overrides: {基金代码: 当前净值}，用于把曲线终点替换为盘中估算净值。
    """
    curves = []
    overrides = nav_overrides or {}
    try:
        asof = pd.Timestamp(datetime.now()).normalize()
        for _, row in holdings_df.iterrows():
            fund_code = str(row.get("基金代码", "")).strip()
            shares = pd.to_numeric(row.get("持有份额"), errors="coerce")
            if not fund_code or pd.isna(shares) or shares <= 0:
                continue
            nav_df = load_local_data(fund_code)
            if nav_df is None or nav_df.empty:
                continue
            nav_df = nav_df.sort_values("date")
            buy_date = pd.to_datetime(row.get("买入日期"), errors="coerce")
            if pd.notna(buy_date):
                nav_df = nav_df[nav_df["date"] >= buy_date]
            if nav_df.empty:
                continue
            series = pd.to_numeric(nav_df.set_index("date")["nav"], errors="coerce") * float(shares)
            series.name = fund_code
            override = pd.to_numeric(overrides.get(fund_code), errors="coerce") if overrides else None
            if override is not None and pd.notna(override) and float(override) > 0:
                last_idx = series.index[-1] if len(series.index) else asof
                stamp = asof if asof >= last_idx else last_idx
                series.loc[stamp] = float(override) * float(shares)
            curves.append(series)
        if not curves:
            return 0.0
        portfolio = pd.concat(curves, axis=1).sort_index().ffill().sum(axis=1)
        portfolio = portfolio[portfolio > 0]
        if portfolio.empty:
            return 0.0
        peak = float(portfolio.max())
        current = float(portfolio.iloc[-1])
        if peak <= 0:
            return 0.0
        return current / peak - 1.0
    except Exception:
        return 0.0


def _finalize_orders(df, total_asset):
    result = df.copy()
    result["最新净值"] = pd.to_numeric(result.get("最新净值"), errors="coerce")
    result["持有份额"] = pd.to_numeric(result.get("持有份额"), errors="coerce")
    result["持仓市值"] = pd.to_numeric(result.get("持仓市值"), errors="coerce")
    result["目标市值"] = pd.to_numeric(result.get("目标市值"), errors="coerce")

    sell_value = (result["持仓市值"] - result["目标市值"]).clip(lower=0)
    nav = result["最新净值"]
    raw_shares = np.where((sell_value > 0) & (nav > 0), sell_value / nav, 0.0)
    result["卖出份额"] = np.ceil(np.nan_to_num(raw_shares, nan=0.0) * 100.0) / 100.0
    result["卖出份额"] = np.minimum(result["卖出份额"].fillna(0.0), result["持有份额"].fillna(0.0))

    redeem_mask = (
        result["指令"].isin({REDEEM_INSTRUCTION, "清仓", "紧急清仓"})
        | (result["目标市值"].fillna(0) <= 1e-8)
        | (result["卖出份额"] >= result["持有份额"].fillna(0) - 1e-8)
    )
    reduce_like = result["指令"].isin(
        {REDUCE_INSTRUCTION, MELTDOWN_INSTRUCTION, REDEEM_INSTRUCTION, "紧急减仓", "清仓", "紧急清仓"}
    )
    full_exit = redeem_mask & reduce_like & (result["持有份额"].fillna(0) > 0)
    to_redeem = full_exit & ~result["指令"].isin({"清仓", "紧急清仓", "紧急减仓"})
    result.loc[to_redeem, "指令"] = REDEEM_INSTRUCTION
    result.loc[full_exit, "目标市值"] = 0.0
    result.loc[full_exit, "卖出份额"] = result.loc[full_exit, "持有份额"]

    if total_asset and total_asset > 0:
        result["目标仓位"] = result["目标市值"] / total_asset
    else:
        result["目标仓位"] = np.nan

    result["建议操作"] = result["指令"]
    return result


def _apply_insufficient_data(df):
    result = df.copy()
    if "综合得分" not in result.columns:
        return result
    insufficient = result["综合得分"].astype(str).eq("数据不足")
    untouched = result["指令"].isin({KEEP_INSTRUCTION, INSUFFICIENT_INSTRUCTION, ""})
    mask = insufficient & untouched
    result.loc[mask, "指令"] = INSUFFICIENT_INSTRUCTION
    result.loc[mask, "目标市值"] = result.loc[mask, "持仓市值"]
    result.loc[mask, "卖出份额"] = 0.0
    result.loc[mask, "操作理由"] = result.loc[mask, "操作理由"].map(
        lambda x: INSUFFICIENT_NOTE if not str(x).strip() else f"{x}；{INSUFFICIENT_NOTE}"
    )
    return result


def generate_trading_signal(year, month, strategy_mode=None, sector_top_n=None):
    """
    总控引擎：熔断 → 总仓位上限 → 赛道上限 → 单基上限 → 急跌过滤 → 冷却期 → 赛道精简 → 输出指令。

    模式只通过 STRATEGY_MODE（或调用方传入的同名参数）统一解析：
    auto 时用 get_market_regime() 覆盖为进攻/防御。
    """
    inspect_market_regime(force=True)
    regime = get_market_regime()
    mode_state = resolve_strategy_mode(strategy_mode, regime=regime)
    profile = mode_state["profile"]

    holdings = _active_holdings(load_current_holdings())
    if holdings.empty:
        return _empty_signal_frame(mode_state)

    scores = batch_score_funds(holdings["基金代码"].tolist())
    merged = holdings.merge(scores, on="基金代码", how="left")
    if "综合得分" not in merged.columns:
        merged["综合得分"] = "数据不足"
    merged["综合得分"] = merged["综合得分"].fillna("数据不足")

    total_asset = float(merged["持仓市值"].sum(min_count=1) or 0.0)
    account_return = _calc_account_drawdown(merged)
    merged["目标市值"] = merged["持仓市值"]
    merged["指令"] = KEEP_INSTRUCTION
    merged["指令来源"] = ""
    merged["操作理由"] = ""

    step1 = apply_retreat_meltdown(
        merged, total_asset, account_return, retreat_limit=profile["RETREAT_LIMIT"]
    )
    meltdown_triggered = bool(step1.attrs.get("meltdown_triggered", False))
    step1 = apply_equity_cap(step1, total_asset, equity_limit=profile["TOTAL_EQUITY_LIMIT"])
    step1 = apply_sector_limits(step1, total_asset, sector_limits=profile["SECTOR_LIMITS"])
    step1 = apply_single_fund_limit(
        step1, total_asset, single_fund_limit=profile["SINGLE_FUND_LIMIT"]
    )

    crash_triggered = bool(check_market_crash(year, month))
    if crash_triggered:
        open_mask = step1["指令"].isin(OPEN_INSTRUCTIONS)
        step1.loc[open_mask, "指令"] = "暂缓"
        step1.loc[open_mask, "操作理由"] = step1.loc[open_mask, "操作理由"].map(
            lambda x: CRASH_NOTE if not str(x).strip() else f"{x}；{CRASH_NOTE}"
        )

    step1 = apply_cooldown(step1)

    top_n_sectors = 0
    top_n_funds = 0
    top_n_banner = ""
    elite_note = ""
    elite_cash = 0.0
    try:
        keep_n = SECTOR_TOP_N if sector_top_n is None else int(sector_top_n)
    except (TypeError, ValueError):
        keep_n = SECTOR_TOP_N
    keep_n = max(1, min(keep_n, 10))
    if ENABLE_SECTOR_TOP_N and is_sector_top_n_active(mode_state.get("effective")):
        step1 = apply_sector_elite(step1, holdings, top_n=keep_n)
        top_n_sectors = int(step1.attrs.get("sector_top_n_sectors", 0) or 0)
        top_n_funds = int(step1.attrs.get("sector_top_n_funds", 0) or 0)
        elite_cash = float(step1.attrs.get("sector_elite_cash", 0.0) or 0.0)
        elite_note = step1.attrs.get("sector_elite_note") or "精简出的资金统一进入货币基金，下月再分配"
        top_n_banner = step1.attrs.get("sector_top_n_banner") or (
            f"✂️ 赛道精简：{top_n_sectors} 个赛道优化，精简 {top_n_funds} 只基金"
        )

    orders = _finalize_orders(step1, total_asset)
    orders = _apply_insufficient_data(orders)
    orders["建议操作"] = orders["指令"]

    keep_reason = orders["指令"].isin({KEEP_INSTRUCTION, INSUFFICIENT_INSTRUCTION, "暂缓"})
    blank_reason = orders["操作理由"].astype(str).str.strip().eq("")
    orders.loc[keep_reason & blank_reason, "操作理由"] = "满足风控约束，维持现有仓位"

    orders.attrs["meltdown_triggered"] = meltdown_triggered
    orders.attrs["crash_filter_triggered"] = crash_triggered
    orders.attrs["account_return"] = account_return
    orders.attrs["hs300_year"] = year
    orders.attrs["hs300_month"] = month
    orders.attrs["sector_top_n_sectors"] = top_n_sectors
    orders.attrs["sector_top_n_funds"] = top_n_funds
    orders.attrs["sector_top_n_banner"] = top_n_banner
    orders.attrs["sector_elite_note"] = elite_note
    orders.attrs["sector_elite_cash"] = elite_cash
    orders.attrs["operation_frequency"] = "monthly"
    orders.attrs["operation_frequency_banner"] = "📅 当前运行频率：月度调仓"
    _attach_mode_state(orders, mode_state)

    try:
        signal_path = get_signal_path()
        os.makedirs(os.path.dirname(signal_path), exist_ok=True)
        save_df = orders.copy()
        save_df["基金代码"] = save_df["基金代码"].astype(str)
        save_df.to_csv(signal_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass

    return orders


def generate_buy_candidates(year, month, top_n=3, strategy_mode=None):
    """
    从观察池筛选买入候选，仅作提示，不改仓位。

    同时满足：赛道剩余空间 > 2%，得分高于同赛道持仓均分 10%，且综合得分为正。
    赛道上限与 generate_trading_signal 共用同一套 STRATEGY_MODE 解析结果。
    """
    mode_state = resolve_strategy_mode(strategy_mode, allow_network=False)
    sector_limits = mode_state["profile"]["SECTOR_LIMITS"]

    empty = pd.DataFrame(columns=CANDIDATE_COLUMNS)
    empty.attrs["year"] = year
    empty.attrs["month"] = month
    empty.attrs["watchlist_failures"] = []
    _attach_mode_state(empty, mode_state)

    try:
        top_n = max(int(top_n), 0)
    except (TypeError, ValueError):
        top_n = 3
    if top_n <= 0:
        return empty

    try:
        from src.factor_layer.scorer import get_unheld_funds_score
    except ImportError:
        import importlib

        import src.factor_layer.scorer as scorer

        scorer = importlib.reload(scorer)
        from src.factor_layer.scorer import get_unheld_funds_score

    try:
        holdings = _active_holdings(load_current_holdings(allow_network=False))
        watch = get_unheld_funds_score(allow_network=False)
    except Exception:
        return empty

    failures = []
    if watch is not None and not watch.empty and "备注" in watch.columns:
        fail_mask = watch["备注"].astype(str).str.contains("数据拉取失败", na=False)
        failures = watch.loc[fail_mask, "基金代码"].astype(str).tolist()
        watch = watch.loc[~fail_mask].copy()
    empty.attrs["watchlist_failures"] = failures

    if watch is None or watch.empty:
        result = empty.copy()
        result.attrs["watchlist_failures"] = failures
        _attach_mode_state(result, mode_state)
        return result

    total_asset = float(holdings["持仓市值"].sum(min_count=1) or 0.0) if not holdings.empty else 0.0
    if total_asset > 0:
        sector_weight = holdings.groupby("赛道归类", dropna=False)["持仓市值"].sum() / total_asset
    else:
        sector_weight = pd.Series(dtype=float)

    held_avg = pd.Series(dtype=float)
    if not holdings.empty:
        held_scores = batch_score_funds(holdings["基金代码"].tolist())
        held_merged = holdings[["基金代码", "赛道归类"]].merge(held_scores, on="基金代码", how="left")
        held_merged["_score"] = pd.to_numeric(held_merged["综合得分"], errors="coerce")
        held_avg = held_merged.groupby("赛道归类", dropna=False)["_score"].mean()

    watch = watch.copy()
    watch["综合得分"] = pd.to_numeric(watch["综合得分"], errors="coerce")
    watch["赛道"] = watch["赛道归类"].fillna("其他") if "赛道归类" in watch.columns else "其他"
    watch["赛道权重上限"] = watch["赛道"].map(lambda s: _sector_limit(s, sector_limits))
    watch["当前赛道占比"] = watch["赛道"].map(lambda s: float(sector_weight.get(s, 0.0) or 0.0))
    watch["当前赛道剩余空间（%）"] = watch["赛道权重上限"] - watch["当前赛道占比"]
    watch["同赛道持仓均分"] = watch["赛道"].map(lambda s: held_avg.get(s, np.nan))

    has_space = watch["当前赛道剩余空间（%）"] > SECTOR_BUFFER
    score_positive = watch["综合得分"] > 0
    beat_held = watch["同赛道持仓均分"].isna() | (
        watch["综合得分"] > watch["同赛道持仓均分"] * SCORE_EDGE_RATIO
    )
    picked = watch.loc[has_space & score_positive & beat_held].copy()
    if picked.empty:
        result = empty.copy()
        result.attrs["watchlist_failures"] = failures
        _attach_mode_state(result, mode_state)
        return result

    picked = picked.sort_values("综合得分", ascending=False, kind="mergesort").head(top_n)
    picked["基金名称"] = picked["基金名称"] if "基金名称" in picked.columns else ""
    picked["近60日收益"] = picked["近60日收益率"] if "近60日收益率" in picked.columns else np.nan
    picked["近60日回撤"] = picked["近60日最大回撤"] if "近60日最大回撤" in picked.columns else np.nan
    picked["备注"] = np.where(
        picked["同赛道持仓均分"].isna(),
        "赛道尚无持仓，且得分为正",
        "得分显著高于现有持仓",
    )

    result = picked[CANDIDATE_COLUMNS].reset_index(drop=True)
    result.attrs["year"] = year
    result.attrs["month"] = month
    result.attrs["watchlist_failures"] = failures
    _attach_mode_state(result, mode_state)
    return result
