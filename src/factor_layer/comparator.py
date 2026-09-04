import numpy as np
import pandas as pd

from ..data_layer.loader import _safe_print, load_local_data
from .portfolio_utils import load_current_holdings
from .scorer import MIN_TRADE_DAYS, batch_score_funds


def _meta_map(holdings_df):
    mapping = {}
    if holdings_df is None or holdings_df.empty or "基金代码" not in holdings_df.columns:
        return mapping
    for _, row in holdings_df.iterrows():
        code = str(row.get("基金代码", "")).strip()
        if not code:
            continue
        name = str(row.get("基金名称") or "").strip() or code
        sector = str(row.get("赛道归类") or "").strip()
        if sector in {"", "nan", "None", "NaN"}:
            sector = "其他"
        mapping[code] = {"基金名称": name, "赛道归类": sector}
    return mapping


def _nav_window(fund_code, days=None):
    length = int(days or MIN_TRADE_DAYS)
    nav_df = load_local_data(fund_code)
    if nav_df is None or nav_df.empty or "nav" not in nav_df.columns:
        return None
    window = nav_df.sort_values("date").copy()
    window["nav"] = pd.to_numeric(window["nav"], errors="coerce")
    window = window.dropna(subset=["nav"]).tail(length)
    if len(window) < length:
        return None
    return window


def _to_float_or_nan(value):
    if value is None or (isinstance(value, str) and value.strip() in {"", "数据不足", "-", "nan", "None"}):
        return np.nan
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return np.nan
    return float(numeric)


def get_fund_comparison_data(fund_codes, window=None):
    """
    汇总对比所需数据，基金数量不设上限。
    无净值或不足 window 个交易日的基金跳过并记日志。
    window 缺省时使用调仓窗口 MIN_TRADE_DAYS。
    """
    days = max(int(window or MIN_TRADE_DAYS), 2)
    if fund_codes is None:
        return {}

    codes = pd.Index(fund_codes).astype(str).str.strip()
    codes = codes[codes.notna() & (codes != "")].unique().tolist()
    if not codes:
        return {}

    meta = _meta_map(load_current_holdings())
    windows = {}
    valid_codes = []
    for code in codes:
        nav_window = _nav_window(code, days=days)
        if nav_window is None:
            _safe_print(f"⚠️ 基金对比跳过 [{code}]: 无净值数据或不足 {days} 个交易日")
            continue
        windows[code] = nav_window
        valid_codes.append(code)

    if not valid_codes:
        return {}

    scores = batch_score_funds(valid_codes, window=days)
    score_map = {}
    if scores is not None and not scores.empty:
        for _, row in scores.iterrows():
            code = str(row.get("基金代码", "")).strip()
            if code:
                score_map[code] = row

    result = {}
    for code in valid_codes:
        nav_window = windows[code]
        series = []
        for _, point in nav_window.iterrows():
            date_val = pd.to_datetime(point.get("date"), errors="coerce")
            series.append(
                {
                    "date": date_val.strftime("%Y-%m-%d") if pd.notna(date_val) else "",
                    "nav": float(point["nav"]),
                }
            )
        row = score_map.get(code, {})
        info = meta.get(code, {})
        result[code] = {
            "基金代码": code,
            "基金名称": info.get("基金名称") or code,
            "赛道归类": info.get("赛道归类") or "其他",
            "近60日净值序列": series,
            "近60日收益率": _to_float_or_nan(row.get("近60日收益率") if len(row) else np.nan),
            "最大回撤": _to_float_or_nan(row.get("近60日最大回撤") if len(row) else np.nan),
            "年化波动率": _to_float_or_nan(row.get("近60日波动率") if len(row) else np.nan),
            "综合得分": _to_float_or_nan(row.get("综合得分") if len(row) else np.nan),
            "最新净值": float(nav_window["nav"].iloc[-1]),
            "窗口天数": days,
        }
    return result
