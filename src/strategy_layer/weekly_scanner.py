import os
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import RETREAT_LIMIT, WEEKLY_SCAN
from src.data_layer.loader import get_signal_path
from src.data_layer.realtime_cache import (
    COOLDOWN_NOTE,
    load_realtime_payload,
    log_monitor,
    pick_signal_estimates,
)
from src.factor_layer.portfolio_utils import load_current_holdings
from src.strategy_layer.intraday_monitor import (
    annotate_emergency_row,
    build_monitor_bundle,
    operation_deadline_text,
)

from .constraints import (
    KEEP_INSTRUCTION,
    REDEEM_INSTRUCTION,
    _add_source,
    _append_reason,
    _ensure_strategy_cols,
)
from .signal_generator import (
    _active_holdings,
    _finalize_orders,
)


EMERGENCY_INSTRUCTION = "紧急减仓"
CLEAR_INSTRUCTION = "清仓"
WEEKLY_SOURCE = "周调仓"
EMERGENCY_SOURCE = "紧急调仓"
FREQUENCY_BANNER_MONTHLY = "📅 当前运行频率：月度调仓"
FREQUENCY_BANNER_WEEKLY = "📅 当前运行频率：周调仓"


def _scan_cfg():
    cfg = WEEKLY_SCAN if isinstance(WEEKLY_SCAN, dict) else {}
    try:
        weekday = int(cfg.get("run_weekday", 4))
    except (TypeError, ValueError):
        weekday = 4
    try:
        retreat = float(cfg.get("retreat_limit", RETREAT_LIMIT))
    except (TypeError, ValueError):
        retreat = float(RETREAT_LIMIT)
    try:
        reduce_ratio = min(max(float(cfg.get("reduce_ratio", 0.30)), 0.05), 0.90)
    except (TypeError, ValueError):
        reduce_ratio = 0.30
    try:
        emergency_ratio = min(max(float(cfg.get("emergency_reduce_ratio", 0.50)), 0.05), 0.95)
    except (TypeError, ValueError):
        emergency_ratio = 0.50
    try:
        melt_cap = min(max(float(cfg.get("meltdown_equity_cap", 0.50)), 0.10), 0.90)
    except (TypeError, ValueError):
        melt_cap = 0.50
    return {
        "run_weekday": weekday,
        "retreat_limit": retreat,
        "reduce_ratio": reduce_ratio,
        "emergency_reduce_ratio": emergency_ratio,
        "meltdown_equity_cap": melt_cap,
    }


def _cut_emergency(df, idx, reason, target_mv, instruction=EMERGENCY_INSTRUCTION):
    current_target = pd.to_numeric(df.at[idx, "目标市值"], errors="coerce")
    if pd.isna(current_target):
        current_target = pd.to_numeric(df.at[idx, "持仓市值"], errors="coerce")
    if pd.isna(current_target):
        return False
    new_target = min(float(current_target), max(float(target_mv), 0.0))
    if new_target >= float(current_target) - 1e-8:
        return False
    df.at[idx, "目标市值"] = new_target
    df.at[idx, "指令"] = instruction
    df.at[idx, "指令来源"] = _add_source(df.at[idx, "指令来源"], WEEKLY_SOURCE)
    df.at[idx, "指令来源"] = _add_source(df.at[idx, "指令来源"], EMERGENCY_SOURCE)
    df.at[idx, "操作理由"] = annotate_emergency_row(
        _append_reason(df.at[idx, "操作理由"], reason), bypass_cooldown=True
    )
    return True


def _attach_weekly_attrs(df, cfg, as_of, account_return, alerts, changed, extra=None):
    is_friday = as_of.weekday() == int(cfg["run_weekday"])
    alert_n = len(alerts)
    emergency = alert_n > 0 and changed > 0
    if alert_n == 0:
        if is_friday:
            risk_line = "未触发紧急信号，按正式周调仓流程生成月度报告"
        else:
            risk_line = "未触发紧急调仓信号，正式调仓日为每周五"
    else:
        risk_line = f"触发紧急调仓，生成建议 {changed} 只（不受周五限制）"
    banner = f"{FREQUENCY_BANNER_WEEKLY} · {risk_line}"
    df.attrs["operation_frequency"] = "weekly"
    df.attrs["operation_frequency_banner"] = banner
    df.attrs["weekly_is_friday"] = is_friday
    df.attrs["weekly_alerts"] = list(alerts)
    df.attrs["weekly_alert_count"] = alert_n
    df.attrs["weekly_changed_funds"] = int(changed)
    df.attrs["account_return"] = float(account_return or 0.0)
    df.attrs["meltdown_triggered"] = any("总回撤" in item for item in alerts)
    df.attrs["crash_filter_triggered"] = False
    df.attrs["emergency_triggered"] = bool(emergency or alert_n > 0)
    df.attrs["friday_no_emergency"] = bool(is_friday and not (emergency or alert_n > 0))
    df.attrs["operation_deadline"] = operation_deadline_text(as_of)
    df.attrs["cooldown_bypassed"] = bool(emergency or alert_n > 0)
    if extra:
        for key, value in extra.items():
            df.attrs[key] = value
    return df


def run_weekly_scan(as_of=None, estimates=None):
    """
    周调仓：以盘中实时估算净值判断紧急减仓/清仓。

    - 紧急条件（任意交易日立即触发，不受周五和 3 个月冷却期限制）：
      单基当日估算跌幅 < -7%；赛道当日估算跌幅 < -5%；账户回撤 < -18%。
    - 每周五且未触发紧急信号时，由看板走正式月度调仓报告。
    """
    cfg = _scan_cfg()
    now = pd.Timestamp(as_of or datetime.now())
    as_of = now.normalize()
    holdings = _active_holdings(load_current_holdings())
    empty = pd.DataFrame()
    payload = estimates if isinstance(estimates, dict) else pick_signal_estimates(load_realtime_payload(), ts=now)
    extra = {
        "estimate_degraded": False,
        "estimate_updated_at": str((payload or {}).get("updated_at") or ""),
        "estimate_picked": str((payload or {}).get("picked") or ""),
    }
    if holdings.empty:
        return _attach_weekly_attrs(empty, cfg, as_of, 0.0, [], 0, extra)

    monitor_df, _sector_df, monitor_meta = build_monitor_bundle(holdings, payload)
    extra["estimate_degraded"] = bool((monitor_meta or {}).get("degraded"))
    emergency = (monitor_meta or {}).get("emergency") or {}
    account_return = float((monitor_meta or {}).get("account_drawdown") or 0.0)
    alerts = list(emergency.get("alerts") or [])

    work = holdings.merge(
        monitor_df[
            [
                col
                for col in (
                    "基金代码",
                    "估算净值",
                    "当日估算涨跌幅",
                    "当日估算盈亏",
                    "估算市值",
                )
                if col in monitor_df.columns
            ]
        ],
        on="基金代码",
        how="left",
    ) if monitor_df is not None and not monitor_df.empty else holdings.copy()

    if "估算净值" in work.columns:
        est = pd.to_numeric(work["估算净值"], errors="coerce")
        last = pd.to_numeric(work.get("最新净值"), errors="coerce")
        work["最新净值"] = est.where(est.notna() & (est > 0), last)
    if "估算市值" in work.columns:
        est_mv = pd.to_numeric(work["估算市值"], errors="coerce")
        last_mv = pd.to_numeric(work.get("持仓市值"), errors="coerce")
        work["持仓市值"] = est_mv.where(est_mv.notna() & (est_mv > 0), last_mv)

    if "综合得分" not in work.columns:
        work["综合得分"] = "数据不足"
    work["综合得分"] = work["综合得分"].fillna("数据不足")
    if "赛道归类" not in work.columns:
        work["赛道归类"] = "未分类"
    work["赛道归类"] = work["赛道归类"].fillna("未分类")

    total_asset = float(pd.to_numeric(work["持仓市值"], errors="coerce").sum(min_count=1) or 0.0)
    work = _ensure_strategy_cols(work)
    work["目标市值"] = pd.to_numeric(work["持仓市值"], errors="coerce")
    work["指令"] = KEEP_INSTRUCTION
    work["指令来源"] = ""
    work["操作理由"] = ""

    changed_idx = set()
    if emergency.get("triggered"):
        if emergency.get("meltdown"):
            cap_value = cfg["meltdown_equity_cap"] * total_asset if total_asset > 0 else 0.0
            equity = float(work["目标市值"].sum(min_count=1) or 0.0)
            if equity > cap_value + 1e-8 and cap_value > 0:
                scale = cap_value / equity
                reason = (
                    f"紧急调仓：账户总回撤 {account_return * 100:.2f}%，"
                    f"权益仓位压至 {cfg['meltdown_equity_cap'] * 100:.0f}% 以内"
                )
                for idx in work.index:
                    target = float(work.at[idx, "目标市值"]) * scale
                    if _cut_emergency(work, idx, reason, target):
                        changed_idx.add(idx)

        hit_codes = {str(item.get("基金代码", "")).strip() for item in emergency.get("fund_hits") or [] if item.get("紧急")}
        clear_codes = {str(item.get("基金代码", "")).strip() for item in emergency.get("fund_hits") or [] if item.get("清仓")}
        for idx, row in work.iterrows():
            code = str(row.get("基金代码", "")).strip()
            if code not in hit_codes:
                continue
            mv = float(pd.to_numeric(row.get("持仓市值"), errors="coerce") or 0.0)
            chg = pd.to_numeric(row.get("当日估算涨跌幅"), errors="coerce")
            chg_text = f"{float(chg):.2f}%" if pd.notna(chg) else "-"
            if code in clear_codes:
                reason = f"紧急调仓：当日估算跌幅 {chg_text}，建议清仓"
                if _cut_emergency(work, idx, reason, 0.0, instruction=CLEAR_INSTRUCTION):
                    changed_idx.add(idx)
            else:
                reason = f"紧急调仓：当日估算跌幅 {chg_text}，建议减仓"
                if _cut_emergency(work, idx, reason, mv * (1.0 - cfg["emergency_reduce_ratio"])):
                    changed_idx.add(idx)

        sector_names = {str(item.get("赛道", "")).strip() for item in emergency.get("sector_hits") or []}
        for sector in sector_names:
            if not sector:
                continue
            group = work[work["赛道归类"].astype(str) == sector]
            for idx, row in group.iterrows():
                mv = float(pd.to_numeric(row.get("持仓市值"), errors="coerce") or 0.0)
                chg = pd.to_numeric(row.get("当日估算涨跌幅"), errors="coerce")
                chg_text = f"{float(chg):.2f}%" if pd.notna(chg) else "-"
                reason = f"紧急调仓：{sector}赛道当日估算跌幅 {chg_text}，建议减仓"
                if _cut_emergency(work, idx, reason, mv * (1.0 - cfg["reduce_ratio"])):
                    changed_idx.add(idx)

    keep_blank = work["指令"].eq(KEEP_INSTRUCTION) & work["操作理由"].astype(str).str.strip().eq("")
    if emergency.get("triggered"):
        work.loc[keep_blank, "操作理由"] = "紧急调仓未覆盖该基金，维持持仓"
    else:
        work.loc[keep_blank, "操作理由"] = "未触发紧急调仓信号，维持持仓"

    deadline = operation_deadline_text(now)
    work["操作截止时间"] = ""
    sell_like = ~work["指令"].eq(KEEP_INSTRUCTION)
    work.loc[sell_like, "操作截止时间"] = f"{deadline}前"

    orders = _finalize_orders(work, total_asset)
    full_exit = pd.to_numeric(orders.get("卖出份额"), errors="coerce").fillna(0) >= (
        pd.to_numeric(orders.get("持有份额"), errors="coerce").fillna(0) - 1e-8
    )
    keep_mask = orders["指令"].eq(KEEP_INSTRUCTION)
    orders["建议操作"] = np.where(keep_mask, KEEP_INSTRUCTION, np.where(full_exit, CLEAR_INSTRUCTION, "减仓"))
    orders.loc[orders["指令"].eq(CLEAR_INSTRUCTION), "建议操作"] = CLEAR_INSTRUCTION
    orders.loc[orders["指令"].eq(REDEEM_INSTRUCTION) & full_exit, "建议操作"] = CLEAR_INSTRUCTION
    if total_asset > 0:
        orders["当前仓位"] = pd.to_numeric(orders.get("持仓市值"), errors="coerce") / total_asset
    else:
        orders["当前仓位"] = np.nan

    _attach_weekly_attrs(orders, cfg, as_of, account_return, alerts, len(changed_idx), extra)
    log_monitor(
        "周调仓",
        f"生成报告 紧急={bool(orders.attrs.get('emergency_triggered'))} "
        f"周五={bool(orders.attrs.get('weekly_is_friday'))} "
        f"变动={len(changed_idx)} 估值={extra.get('estimate_updated_at') or '-'}",
    )

    try:
        signal_path = get_signal_path()
        os.makedirs(os.path.dirname(signal_path), exist_ok=True)
        save_df = orders.copy()
        save_df["基金代码"] = save_df["基金代码"].astype(str)
        save_df.to_csv(signal_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass
    return orders
