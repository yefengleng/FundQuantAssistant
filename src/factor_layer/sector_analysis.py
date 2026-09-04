import os
from datetime import datetime

import numpy as np
import pandas as pd

from ..data_layer.loader import RAW_DIR, load_local_data
from .indicators import calc_return
from .scorer import batch_score_funds
from .sector_classifier import (
    auto_tag_fund,
    collect_all_pool_funds,
    list_known_sectors,
    load_global_sector_mapping,
)


MISSING = "数据缺失"
TREND_STRONG = "强势"
TREND_WEAK = "弱势"
TREND_NEUTRAL = "中性"
TREND_STRONG_PCT = 5.0
TREND_WEAK_PCT = -5.0
NAV_WINDOW = 60
TREND_WINDOW = 20


def _effective_sector(code, name, mapping):
    mapped = str((mapping or {}).get(code) or "").strip()
    if mapped:
        return mapped
    tagged = auto_tag_fund(code, name, allow_network=False)
    if isinstance(tagged, str):
        return tagged or "其他"
    return str((tagged or {}).get("sector") or "其他")


def iter_sector_universe():
    """所有账户基金池 + 全局映射中的基金，带有效赛道。"""
    mapping = load_global_sector_mapping()
    funds = collect_all_pool_funds()
    rows = []
    seen = set()
    if funds is not None and not funds.empty:
        for _, row in funds.iterrows():
            code = str(row.get("基金代码") or "").strip()
            if not code or code in seen:
                continue
            seen.add(code)
            name = str(row.get("基金名称") or "").strip()
            rows.append(
                {
                    "基金代码": code,
                    "基金名称": name,
                    "赛道": _effective_sector(code, name, mapping),
                }
            )
    for code, sector in (mapping or {}).items():
        code = str(code or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        rows.append(
            {
                "基金代码": code,
                "基金名称": "",
                "赛道": str(sector or "").strip() or "其他",
            }
        )
    return pd.DataFrame(rows, columns=["基金代码", "基金名称", "赛道"])


def list_sector_fund_codes(sector):
    name = str(sector or "").strip()
    if not name:
        return []
    universe = iter_sector_universe()
    if universe is None or universe.empty:
        return []
    codes = universe.loc[universe["赛道"] == name, "基金代码"]
    return codes.astype(str).str.strip().tolist()


def get_sector_nav_series(sector, window=None):
    """
    该赛道全部基金近 window 个交易日的等权平均净值。
    window 缺省为 NAV_WINDOW（60）。
    返回以日期为索引、含 nav 列的 DataFrame；无数据时为空表。
    """
    days = max(int(window or NAV_WINDOW), 2)
    codes = list_sector_fund_codes(sector)
    series_list = []
    for code in codes:
        nav_df = load_local_data(code)
        if nav_df is None or nav_df.empty or "nav" not in nav_df.columns:
            continue
        work = nav_df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work["nav"] = pd.to_numeric(work["nav"], errors="coerce")
        work = work.dropna(subset=["date", "nav"]).sort_values("date")
        work = work.drop_duplicates(subset=["date"], keep="last")
        if work.empty:
            continue
        item = work.set_index("date")["nav"].astype(float)
        series_list.append(item.rename(code))
    if not series_list:
        empty = pd.DataFrame(columns=["nav"])
        empty.index.name = "date"
        return empty

    aligned = pd.concat(series_list, axis=1).sort_index()
    aligned = aligned.dropna(how="all")
    sliced = aligned.tail(days)
    if sliced.empty:
        empty = pd.DataFrame(columns=["nav"])
        empty.index.name = "date"
        return empty

    normalized = sliced.copy()
    for column in normalized.columns:
        col = normalized[column]
        valid = col.dropna()
        if valid.empty or float(valid.iloc[0]) == 0:
            normalized[column] = np.nan
        else:
            normalized[column] = col / float(valid.iloc[0])
    avg = normalized.mean(axis=1, skipna=True).dropna()
    result = avg.to_frame("nav")
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"
    return result


def get_sector_trend(sector, window=None):
    """返回 (趋势标签, 近 window 日收益率点数)，如 ('强势', 6.2)。"""
    days = max(int(window or NAV_WINDOW), 2)
    series_df = get_sector_nav_series(sector, window=days)
    if series_df is None or series_df.empty or "nav" not in series_df.columns:
        return TREND_NEUTRAL, np.nan
    ret = calc_return(series_df["nav"], window=days)
    if ret is None or pd.isna(ret):
        return TREND_NEUTRAL, np.nan
    value = float(ret)
    if value > TREND_STRONG_PCT:
        return TREND_STRONG, value
    if value < TREND_WEAK_PCT:
        return TREND_WEAK, value
    return TREND_NEUTRAL, value


def get_top_funds_in_sector(sector, top_n=2, scores=None, window=None):
    """返回该赛道综合得分最高的 N 只基金代码。"""
    codes = list_sector_fund_codes(sector)
    if not codes:
        return []
    if scores is None:
        scores = batch_score_funds(codes, window=window)
    if scores is None or scores.empty:
        return []
    work = scores.copy()
    work["基金代码"] = work["基金代码"].astype(str).str.strip()
    work = work[work["基金代码"].isin(codes)]
    work["_score"] = pd.to_numeric(work.get("综合得分"), errors="coerce")
    work = work.dropna(subset=["_score"]).sort_values("_score", ascending=False, kind="mergesort")
    limit = max(int(top_n or 0), 0)
    return work["基金代码"].head(limit).tolist()


def latest_nav_value(fund_code):
    nav_df = load_local_data(fund_code)
    if nav_df is None or nav_df.empty or "nav" not in nav_df.columns:
        return np.nan
    nav = pd.to_numeric(nav_df["nav"], errors="coerce").dropna()
    if nav.empty:
        return np.nan
    return float(nav.iloc[-1])


def get_nav_data_updated_at(fund_codes=None):
    codes = [str(item or "").strip() for item in (fund_codes or []) if str(item or "").strip()]
    latest = None
    if not codes:
        mapping = load_global_sector_mapping()
        codes = list(mapping.keys())
    for code in codes:
        path = os.path.join(RAW_DIR, f"{code}.parquet")
        if not os.path.isfile(path):
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if latest is None or mtime > latest:
            latest = mtime
    if latest is None:
        return None
    return datetime.fromtimestamp(latest)


def refresh_mapped_fund_nav(progress_callback=None, refresh_names=True):
    """只刷新全局映射中的基金净值；可选同步基金名称缓存。"""
    from ..data_layer.loader import update_selected_funds

    mapping = load_global_sector_mapping()
    codes = [str(code).strip() for code in (mapping or {}) if str(code).strip()]
    if refresh_names:
        try:
            from ..ocr.fund_matcher import get_code_name_map

            name_map = get_code_name_map(force=False)
            missing = [code for code in codes if code not in name_map]
            if missing:
                get_code_name_map(force=True)
        except Exception:
            pass
    count = update_selected_funds(codes, progress_callback=progress_callback)
    return {"codes": codes, "updated": count}


def sector_return_nd(sector, window=None):
    days = max(int(window or NAV_WINDOW), 2)
    series_df = get_sector_nav_series(sector, window=days)
    if series_df is None or series_df.empty:
        return np.nan
    return calc_return(series_df["nav"], window=days)


def build_sector_trend_rows(top_n=2, window=None):
    days = max(int(window or NAV_WINDOW), 2)
    universe = iter_sector_universe()
    if universe is None or universe.empty:
        return [], {}
    all_codes = universe["基金代码"].astype(str).str.strip().tolist()
    scores = batch_score_funds(all_codes, window=days)
    sectors = []
    for name in list_known_sectors():
        if name and name not in sectors:
            sectors.append(name)
    for name in universe["赛道"].tolist():
        text = str(name or "").strip()
        if text and text not in sectors:
            sectors.append(text)

    rows = []
    series_map = {}
    for sector in sectors:
        codes = list_sector_fund_codes(sector)
        if not codes:
            continue
        series = get_sector_nav_series(sector, window=days)
        if series is None or series.empty:
            continue
        series_map[sector] = series
        label, ret_nd = get_sector_trend(sector, window=days)
        advice = "建议持有"
        if (ret_nd is not None and not pd.isna(ret_nd) and float(ret_nd) < 0) and label == TREND_WEAK:
            advice = "建议观望/放弃"
        rows.append(
            {
                "赛道": sector,
                "近N日收益": ret_nd,
                "窗口天数": days,
                "趋势": label,
                "建议": advice,
                "推荐基金": get_top_funds_in_sector(sector, top_n=top_n, scores=scores, window=days),
            }
        )
    return rows, series_map
