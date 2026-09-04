import json
import os
from datetime import datetime, timedelta

import pandas as pd

from src.data_layer.loader import get_trade_log_path


HARD_RISK_SOURCES = {"赛道超限", "单基超限", "紧急调仓"}
SELL_INSTRUCTIONS = {"减仓", "赎回", "减仓（熔断）"}
EMERGENCY_INSTRUCTIONS = {"紧急减仓", "清仓", "紧急清仓"}
COOLDOWN_DAYS = 90
COOLDOWN_REASON = "冷却期未满（3个月），暂缓卖出"


def _resolve_log_path(trade_log_path):
    if trade_log_path:
        if os.path.isabs(trade_log_path):
            return trade_log_path
        from src.data_layer.loader import PROJECT_ROOT

        return os.path.join(PROJECT_ROOT, trade_log_path)
    return get_trade_log_path()


def _load_sell_records(trade_log_path):
    path = _resolve_log_path(trade_log_path)
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except FileNotFoundError:
        return pd.DataFrame(columns=["fund_code", "sell_date", "shares", "reason"])
    except Exception:
        return pd.DataFrame(columns=["fund_code", "sell_date", "shares", "reason"])

    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not records:
        return pd.DataFrame(columns=["fund_code", "sell_date", "shares", "reason"])

    log_df = pd.DataFrame(records)
    if "fund_code" not in log_df.columns or "sell_date" not in log_df.columns:
        return pd.DataFrame(columns=["fund_code", "sell_date", "shares", "reason"])

    log_df["fund_code"] = log_df["fund_code"].astype(str).str.strip()
    log_df["sell_date"] = pd.to_datetime(log_df["sell_date"], errors="coerce")
    return log_df.dropna(subset=["fund_code", "sell_date"])


def _has_hard_risk(source_text):
    tags = {part for part in str(source_text).split("|") if part}
    return bool(tags & HARD_RISK_SOURCES)


def apply_cooldown(orders_df, trade_log_path=None):
    """
    过去 90 天内卖出过的基金，减仓/赎回降级为持有。
    赛道超限、单基超限属于硬风控，不触发冷却降级。
    紧急调仓（紧急减仓/清仓，或指令来源含「紧急调仓」）不受冷却期限制。
    """
    df = orders_df.copy()
    if df.empty:
        return df

    if "指令" not in df.columns:
        return df
    if "指令来源" not in df.columns:
        df["指令来源"] = ""
    if "操作理由" not in df.columns:
        df["操作理由"] = ""
    if "持仓市值" in df.columns and "目标市值" not in df.columns:
        df["目标市值"] = df["持仓市值"]

    try:
        log_df = _load_sell_records(trade_log_path)
        if log_df.empty:
            return df

        cutoff = datetime.now() - timedelta(days=COOLDOWN_DAYS)
        recent = log_df[log_df["sell_date"] >= cutoff]
        if recent.empty:
            return df

        recent_codes = set(recent["fund_code"].tolist())
        sell_mask = df["指令"].isin(SELL_INSTRUCTIONS)
        emergency_mask = df["指令"].isin(EMERGENCY_INSTRUCTIONS)
        code_mask = df["基金代码"].astype(str).str.strip().isin(recent_codes)
        hard_mask = df["指令来源"].map(_has_hard_risk)
        cooldown_mask = sell_mask & (~emergency_mask) & code_mask & (~hard_mask)
        if not cooldown_mask.any():
            return df

        df.loc[cooldown_mask, "指令"] = "持有"
        if "持仓市值" in df.columns:
            df.loc[cooldown_mask, "目标市值"] = df.loc[cooldown_mask, "持仓市值"]
        df.loc[cooldown_mask, "操作理由"] = df.loc[cooldown_mask, "操作理由"].map(
            lambda x: f"{x}；{COOLDOWN_REASON}" if str(x).strip() else COOLDOWN_REASON
        )
        return df
    except Exception:
        return df
