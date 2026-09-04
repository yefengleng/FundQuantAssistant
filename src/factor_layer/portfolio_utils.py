import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from ..data_layer.loader import (
    COST_COLUMN,
    CUM_PROFIT_COLUMN,
    FUND_POOL_HEADER,
    NOTE_COLUMN,
    _ordered_fund_pool_columns,
    _safe_print,
    ensure_fund_pool_schema,
    get_fund_pool_path,
    get_trade_log_path,
    load_local_data,
)
from .sector_classifier import apply_global_sector_map, auto_tag_fund


HOLDING_COLUMNS = [
    "基金代码",
    "基金名称",
    "赛道归类",
    "持有份额",
    "买入日期",
    COST_COLUMN,
    "最新净值",
    "净值日期",
    "持仓市值",
    "持有收益（元）",
    "持有收益率（%）",
    NOTE_COLUMN,
    CUM_PROFIT_COLUMN,
    "仓位占比",
]


def _latest_nav_row(fund_code):
    nav_df = load_local_data(fund_code)
    if nav_df is None or nav_df.empty:
        return {
            "基金代码": fund_code,
            "最新净值": np.nan,
            "净值日期": pd.NaT,
        }
    last = nav_df.iloc[-1]
    return {
        "基金代码": fund_code,
        "最新净值": last["nav"],
        "净值日期": last["date"],
    }


def load_current_holdings(allow_network=False):
    """
    读取当前账户基金池，用 data/raw 本地净值重算持仓市值、仓位占比与持有收益。

    默认不联网拉净值。空赛道按 SECTOR_CLASSIFY_MODE 自动归类（仅内存补全）。
    """
    pool_path = get_fund_pool_path()
    try:
        ensure_fund_pool_schema(pool_path)
        holdings = pd.read_csv(pool_path, dtype={"基金代码": str})
    except FileNotFoundError:
        return pd.DataFrame(columns=HOLDING_COLUMNS)
    except Exception:
        return pd.DataFrame(columns=HOLDING_COLUMNS)

    if holdings.empty or "基金代码" not in holdings.columns:
        return pd.DataFrame(columns=HOLDING_COLUMNS)

    holdings["基金代码"] = holdings["基金代码"].astype(str).str.strip()
    holdings = holdings[holdings["基金代码"] != ""].copy()

    if "持有份额" in holdings.columns:
        holdings["持有份额"] = pd.to_numeric(holdings["持有份额"], errors="coerce").fillna(0.0)
    else:
        holdings["持有份额"] = 0.0

    if "买入日期" not in holdings.columns:
        holdings["买入日期"] = pd.NaT

    if COST_COLUMN not in holdings.columns:
        holdings[COST_COLUMN] = 0.0
    holdings[COST_COLUMN] = pd.to_numeric(holdings[COST_COLUMN], errors="coerce").fillna(0.0)

    if NOTE_COLUMN not in holdings.columns:
        holdings[NOTE_COLUMN] = ""
    holdings[NOTE_COLUMN] = holdings[NOTE_COLUMN].astype("string").fillna("").replace({"nan": "", "None": "", "<NA>": ""})

    if CUM_PROFIT_COLUMN not in holdings.columns:
        holdings[CUM_PROFIT_COLUMN] = np.nan
    holdings[CUM_PROFIT_COLUMN] = pd.to_numeric(holdings[CUM_PROFIT_COLUMN], errors="coerce")

    if "基金名称" not in holdings.columns:
        holdings["基金名称"] = ""
    if "赛道归类" not in holdings.columns:
        holdings["赛道归类"] = ""
    holdings["赛道归类"] = holdings["赛道归类"].astype("string").fillna("").str.strip()
    holdings["赛道归类"] = holdings["赛道归类"].replace({"nan": "", "None": "", "NaN": "", "<NA>": ""})
    holdings = apply_global_sector_map(holdings)
    holdings["赛道归类"] = holdings["赛道归类"].astype("string").fillna("").str.strip()
    holdings["赛道归类"] = holdings["赛道归类"].replace({"nan": "", "None": "", "NaN": "", "<NA>": ""})
    empty_sector = holdings["赛道归类"].eq("")
    for idx in holdings.index[empty_sector]:
        code = str(holdings.at[idx, "基金代码"]).strip()
        name = str(holdings.at[idx, "基金名称"] or "").strip()
        try:
            tagged = auto_tag_fund(code, name)
        except Exception:
            tagged = {"sector": "其他", "source": "fallback"}
        if isinstance(tagged, str):
            sector = tagged
            source = "name" if tagged and tagged != "其他" else "fallback"
            reason = ""
        else:
            sector = str((tagged or {}).get("sector") or "其他")
            source = str((tagged or {}).get("source") or "fallback")
            reason = str((tagged or {}).get("reason") or "")
        holdings.at[idx, "赛道归类"] = sector
        if source == "global_manual":
            _safe_print(f"🏷️ [全局映射] {code} → {sector}")
        elif source == "name":
            _safe_print(f"🔍 [名称匹配] {code} → {sector}")
        elif source == "holdings":
            extra = f"（依据：{reason}）" if reason else ""
            _safe_print(f"📊 [持仓穿透] {code} → {sector}{extra}")
        else:
            _safe_print(f"⚠️ [兜底] {code} → {sector or '其他'}")

    latest = pd.DataFrame([_latest_nav_row(code) for code in holdings["基金代码"].tolist()])
    holdings = holdings.merge(latest, on="基金代码", how="left")

    holdings["最新净值"] = pd.to_numeric(holdings["最新净值"], errors="coerce")
    holdings["持仓市值"] = holdings["持有份额"] * holdings["最新净值"]
    holdings["持有收益（元）"] = holdings["持仓市值"] - holdings[COST_COLUMN]
    holdings["持有收益率（%）"] = np.where(
        holdings[COST_COLUMN] > 0,
        holdings["持有收益（元）"] / holdings[COST_COLUMN] * 100.0,
        np.nan,
    )
    if ((holdings[COST_COLUMN] <= 0) & (holdings["持有份额"] > 0)).any():
        _safe_print("⚠️ 请先补充持仓成本，否则无法计算收益率")

    try:
        from ..ocr.importer import load_profile_meta

        profile_meta = load_profile_meta()
        csv_cum = holdings[CUM_PROFIT_COLUMN]
        filled = []
        for idx, row in holdings.iterrows():
            if pd.notna(csv_cum.at[idx]):
                filled.append(csv_cum.at[idx])
                continue
            code = str(row.get("基金代码") or "").strip()
            extra = (profile_meta.get("cumulative_profits") or {}).get(code)
            filled.append(np.nan if extra is None else extra)
        holdings[CUM_PROFIT_COLUMN] = pd.to_numeric(pd.Series(filled, index=holdings.index), errors="coerce")
    except Exception:
        pass

    total_assets = holdings["持仓市值"].sum(min_count=1)
    if pd.isna(total_assets) or total_assets == 0:
        holdings["仓位占比"] = np.nan
    else:
        holdings["仓位占比"] = holdings["持仓市值"] / total_assets

    extra_cols = [col for col in holdings.columns if col not in HOLDING_COLUMNS]
    ordered = [col for col in HOLDING_COLUMNS if col in holdings.columns] + extra_cols
    return holdings[ordered].reset_index(drop=True)


def _append_manual_trade_log(fund_code, old_shares, new_shares, reason=""):
    path = get_trade_log_path()
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        payload = {"records": []}
    if not isinstance(payload, dict):
        payload = {"records": []}
    records = payload.get("records")
    if not isinstance(records, list):
        records = []
    records.append(
        {
            "fund_code": str(fund_code).strip(),
            "action": "手动调仓",
            "old_shares": float(old_shares),
            "new_shares": float(new_shares),
            "reason": "" if reason is None else str(reason).strip(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    payload["records"] = records
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def apply_manual_rebalance(fund_code, new_shares, reason=""):
    """把当前账户基金池中指定基金的持有份额改为 new_shares，并写入交易日志。"""
    code = "" if fund_code is None else str(fund_code).strip()
    try:
        shares = float(new_shares)
    except (TypeError, ValueError):
        return False, "新的持有份额无效"
    if not code:
        return False, "基金代码为空"
    if shares < 0:
        return False, "持有份额不能为负数"

    path = get_fund_pool_path()
    ensure_fund_pool_schema(path)
    try:
        pool = pd.read_csv(path, dtype={"基金代码": str})
    except Exception as exc:
        return False, f"读取基金池失败: {exc}"

    if pool.empty or "基金代码" not in pool.columns:
        return False, "基金池为空"
    pool["基金代码"] = pool["基金代码"].astype(str).str.strip()
    mask = pool["基金代码"] == code
    if not mask.any():
        return False, f"未找到基金 {code}"

    if "持有份额" not in pool.columns:
        pool["持有份额"] = 0.0
    old_shares = float(pd.to_numeric(pool.loc[mask, "持有份额"], errors="coerce").fillna(0).iloc[0])
    pool.loc[mask, "持有份额"] = shares
    if COST_COLUMN not in pool.columns:
        pool[COST_COLUMN] = 0.0
    save_cols = _ordered_fund_pool_columns(pool.columns)
    try:
        pool[save_cols].to_csv(path, index=False, encoding="utf-8-sig")
        _append_manual_trade_log(code, old_shares, shares, reason)
    except Exception as exc:
        return False, f"保存调仓失败: {exc}"
    return True, {
        "fund_code": code,
        "old_shares": old_shares,
        "new_shares": shares,
    }


_FUND_CODE_RE = re.compile(r"(\d{6})")


def normalize_fund_code(value):
    text = "" if value is None else str(value).strip()
    match = _FUND_CODE_RE.search(text)
    if match:
        return match.group(1)
    return ""


def add_watch_fund(fund_code):
    """把基金加入当前账户观察池：份额为 0，名称自动查询，赛道自动匹配。"""
    code = normalize_fund_code(fund_code)
    if not code:
        return False, "请输入6位基金代码"

    path = get_fund_pool_path()
    ensure_fund_pool_schema(path)
    try:
        pool = pd.read_csv(path, dtype={"基金代码": str})
    except Exception:
        pool = pd.DataFrame(columns=FUND_POOL_HEADER)
    if pool is None or "基金代码" not in getattr(pool, "columns", []):
        pool = pd.DataFrame(columns=FUND_POOL_HEADER)
    pool["基金代码"] = pool["基金代码"].astype(str).str.strip()
    numeric_cols = {COST_COLUMN, "持有份额", "持仓市值"}
    for column in FUND_POOL_HEADER:
        if column not in pool.columns:
            pool[column] = 0.0 if column in numeric_cols else ""

    if (pool["基金代码"] == code).any():
        existing_name = ""
        if "基金名称" in pool.columns:
            existing_name = str(pool.loc[pool["基金代码"] == code, "基金名称"].iloc[0] or "").strip()
        extra = f" {existing_name}" if existing_name else ""
        return False, f"{code}{extra} 已在当前账户基金池中"

    from ..ocr.fund_matcher import lookup_fund_name_by_code

    name = lookup_fund_name_by_code(code)
    if not name:
        return False, f"未能获取 {code} 的基金名称，请检查代码是否正确"

    sector = ""
    try:
        tagged = auto_tag_fund(code, name, allow_network=True)
        if isinstance(tagged, dict):
            sector = str(tagged.get("sector") or "").strip()
        else:
            sector = str(tagged or "").strip()
    except Exception as exc:
        _safe_print(f"⚠️ 自动匹配赛道失败 {code}: {exc}")
        sector = ""

    new_row = {column: "" for column in pool.columns}
    new_row["基金代码"] = code
    new_row["基金名称"] = name
    new_row["赛道归类"] = sector
    new_row["持有份额"] = 0.0
    new_row["买入日期"] = ""
    new_row[COST_COLUMN] = 0.0
    new_row["持仓市值"] = 0.0
    new_row[NOTE_COLUMN] = ""
    new_row[CUM_PROFIT_COLUMN] = ""
    pool = pd.concat([pool, pd.DataFrame([new_row])], ignore_index=True)
    save_cols = _ordered_fund_pool_columns(pool.columns)
    try:
        pool[save_cols].to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        return False, f"写入基金池失败: {exc}"
    return True, {
        "fund_code": code,
        "fund_name": name,
        "sector": sector,
    }
