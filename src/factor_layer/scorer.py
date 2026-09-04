import numpy as np
import pandas as pd

from ..data_layer.loader import get_fund_pool_path, load_local_data
from .indicators import calc_max_drawdown, calc_return, calc_volatility
from .sector_classifier import apply_global_sector_map


try:
    from config.settings import REBALANCE_WINDOW_DAYS

    MIN_TRADE_DAYS = int(REBALANCE_WINDOW_DAYS or 60)
except Exception:
    MIN_TRADE_DAYS = 60
MIN_TRADE_DAYS = max(int(MIN_TRADE_DAYS), 2)
SCORE_COLUMNS = ["基金代码", "近60日收益率", "近60日最大回撤", "近60日波动率", "综合得分"]


def calculate_score(return_60d, mdd_60d):
    """
    修正版综合打分。

    入参与 calc_return / calc_max_drawdown 一致，为百分比点数：
    如 return_60d=5.2 表示 5.2%，mdd_60d=-8.5 表示 -8.5%。
    支持标量或数组，全程向量化，结果保留两位小数。
    """
    ret = np.asarray(return_60d, dtype=float)
    mdd = np.asarray(mdd_60d, dtype=float)
    mdd_abs = np.abs(mdd)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_score = (ret / mdd_abs) * 100.0

    score = np.where(mdd_abs < 0.001, np.where(ret > 0, 100.0, -10.0), ratio_score)
    penalty_mask = (ret < 0) & (mdd_abs > 20)
    score = np.where(penalty_mask, score * 0.5, score)
    score = np.round(score, 2)
    score = np.where(np.isnan(ret) | np.isnan(mdd), np.nan, score)

    if score.ndim == 0:
        return float(score) if not np.isnan(score) else np.nan
    return score


def _score_window_days(window):
    if window is None:
        return int(MIN_TRADE_DAYS)
    try:
        return max(int(window), 2)
    except (TypeError, ValueError):
        return int(MIN_TRADE_DAYS)


def _score_one_fund(fund_code, window=None):
    days = _score_window_days(window)
    nav_df = load_local_data(fund_code)
    if nav_df is None or nav_df.empty or len(nav_df) < days:
        return {
            "基金代码": fund_code,
            "近60日收益率": np.nan,
            "近60日最大回撤": np.nan,
            "近60日波动率": np.nan,
            "综合得分": "数据不足",
        }

    nav_series = nav_df["nav"]
    return_nd = calc_return(nav_series, window=days)
    mdd_nd = calc_max_drawdown(nav_series, window=days)
    vol_nd = calc_volatility(nav_series, window=days)
    score = calculate_score(return_nd, mdd_nd)

    return {
        "基金代码": fund_code,
        "近60日收益率": None if pd.isna(return_nd) else float(return_nd) / 100.0,
        "近60日最大回撤": None if pd.isna(mdd_nd) else float(mdd_nd) / 100.0,
        "近60日波动率": None if pd.isna(vol_nd) else float(vol_nd) / 100.0,
        "综合得分": score,
    }


def batch_score_funds(fund_codes, window=None):
    """
    对基金代码列表批量打分。

    百分比类字段以小数存储（0.052 代表 5.2%）。
    window 缺省时使用 REBALANCE_WINDOW_DAYS（调仓窗口，默认 60）。
    交易日不足窗口长度的基金标记为「数据不足」并跳过打分。
    """
    if fund_codes is None:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    codes = pd.Index(fund_codes).astype(str).str.strip()
    codes = codes[codes.notna() & (codes != "")].unique().tolist()
    if not codes:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    records = [_score_one_fund(code, window=window) for code in codes]
    result = pd.DataFrame(records, columns=SCORE_COLUMNS)

    numeric_score = pd.to_numeric(result["综合得分"], errors="coerce")
    result = result.assign(_rank=numeric_score.fillna(-np.inf))
    result = result.sort_values("_rank", ascending=False, kind="mergesort")
    return result.drop(columns="_rank").reset_index(drop=True)


WATCHLIST_COLUMNS = [
    "基金代码",
    "基金名称",
    "赛道归类",
    "近60日收益率",
    "近60日最大回撤",
    "近60日波动率",
    "综合得分",
    "备注",
]


def _read_fund_pool():
    pool = pd.read_csv(get_fund_pool_path(), dtype={"基金代码": str})
    pool["基金代码"] = pool["基金代码"].astype(str).str.strip()
    pool = pool[pool["基金代码"] != ""].copy()
    if "持有份额" in pool.columns:
        pool["持有份额"] = pd.to_numeric(pool["持有份额"], errors="coerce")
    else:
        pool["持有份额"] = np.nan
    if "持仓市值" in pool.columns:
        pool["持仓市值"] = pd.to_numeric(pool["持仓市值"], errors="coerce")
    else:
        pool["持仓市值"] = np.nan
    return apply_global_sector_map(pool)


def _watchlist_mask(pool):
    no_shares = pool["持有份额"].isna() | (pool["持有份额"] <= 0)
    no_value = pool["持仓市值"].isna() | (pool["持仓市值"] <= 0)
    return no_shares | no_value


def _ensure_watchlist_nav(fund_code, allow_network=False):
    """观察池优先用本地净值；仅 allow_network=True 时才联网补数据。"""
    nav_df = load_local_data(fund_code)
    if nav_df is not None and not nav_df.empty and len(nav_df) >= MIN_TRADE_DAYS:
        return True, ""
    if not allow_network:
        return False, "本地净值不足，请先刷新数据"
    try:
        import time

        from ..data_layer.fetcher import fetch_fund_history
        from ..data_layer.loader import save_local_data

        remote = fetch_fund_history(fund_code)
        if remote is None or remote.empty:
            return False, "数据拉取失败"
        save_local_data(fund_code, remote)
        nav_df = load_local_data(fund_code)
        if nav_df is None or nav_df.empty:
            return False, "数据拉取失败"
        try:
            time.sleep(1.5)
        except Exception:
            pass
        return True, ""
    except Exception:
        return False, "数据拉取失败"


def get_unheld_funds_score(allow_network=False):
    """
    读取基金池中的观察池（持有份额<=0 或持仓市值为空/0），计算得分。

    观察池不参与存量风控。默认只读本地净值；allow_network=True 时才会补拉。
    """
    empty = pd.DataFrame(columns=WATCHLIST_COLUMNS)
    try:
        pool = _read_fund_pool()
    except FileNotFoundError:
        return empty
    except Exception:
        return empty

    if pool.empty:
        return empty

    watch = pool.loc[_watchlist_mask(pool)].copy()
    if watch.empty:
        return empty

    ok_codes = []
    failed_rows = []
    for _, row in watch.iterrows():
        code = str(row["基金代码"]).strip()
        ok, note = _ensure_watchlist_nav(code, allow_network=allow_network)
        if ok:
            ok_codes.append(code)
        else:
            failed_rows.append(
                {
                    "基金代码": code,
                    "基金名称": row.get("基金名称", ""),
                    "赛道归类": row.get("赛道归类", ""),
                    "近60日收益率": np.nan,
                    "近60日最大回撤": np.nan,
                    "近60日波动率": np.nan,
                    "综合得分": np.nan,
                    "备注": note or "数据拉取失败",
                }
            )

    scored = batch_score_funds(ok_codes) if ok_codes else pd.DataFrame(columns=SCORE_COLUMNS)
    meta_cols = [col for col in ["基金代码", "基金名称", "赛道归类"] if col in watch.columns]
    result = watch[meta_cols].merge(scored, on="基金代码", how="left")
    if "备注" not in result.columns:
        result["备注"] = ""
    result["备注"] = result["备注"].fillna("")

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        result = result[~result["基金代码"].isin(failed_df["基金代码"])]
        result = pd.concat([result, failed_df], ignore_index=True, sort=False)

    result["备注"] = result["备注"].replace("", np.nan).fillna("")
    for col in WATCHLIST_COLUMNS:
        if col not in result.columns:
            result[col] = "" if col in {"基金名称", "赛道归类", "备注"} else np.nan
    return result[WATCHLIST_COLUMNS].reset_index(drop=True)


MARKET_SCAN_COLUMNS = [
    "赛道归类",
    "基金代码",
    "基金名称",
    "基金类型",
    "近60日收益率",
    "近60日最大回撤",
    "综合得分",
    "基金规模(亿)",
    "成立日期",
    "赛道内排名",
]
MIN_AUM_YI = 2.0
MIN_AGE_DAYS = 365
MAX_DRAWDOWN_PCT = -25.0
TOP_N_PER_SECTOR = 3
SINA_SCALE_TYPES = ["股票型基金", "混合型基金", "债券型基金", "货币型基金", "QDII基金"]


def _fetch_open_fund_rank():
    """优先按约定调用 fund_em_open_fund_rank，当前 akshare 已更名为 fund_open_fund_rank_em。"""
    import akshare as ak

    try:
        return ak.fund_em_open_fund_rank()
    except (AttributeError, TypeError):
        return ak.fund_open_fund_rank_em(symbol="全部")


def _fetch_fund_type_table():
    import akshare as ak

    names = ak.fund_name_em()
    names["基金代码"] = names["基金代码"].astype(str).str.strip()
    return names[["基金代码", "基金简称", "基金类型"]].drop_duplicates("基金代码", keep="last")


def _fetch_scale_and_inception():
    import time

    import akshare as ak

    frames = []
    for symbol in SINA_SCALE_TYPES:
        try:
            chunk = ak.fund_scale_open_sina(symbol=symbol)
            if chunk is None or chunk.empty:
                continue
            chunk = chunk.copy()
            chunk["基金代码"] = chunk["基金代码"].astype(str).str.strip()
            frames.append(chunk)
        except Exception:
            continue
        try:
            time.sleep(0.4)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=["基金代码", "成立日期", "基金规模(亿)"])
    scale = pd.concat(frames, ignore_index=True).drop_duplicates("基金代码", keep="first")
    shares = pd.to_numeric(scale.get("最近总份额"), errors="coerce")
    nav = pd.to_numeric(scale.get("单位净值"), errors="coerce")
    aum_yi = shares * nav / 1e8
    raised = pd.to_numeric(scale.get("总募集规模"), errors="coerce") / 1e4
    scale["基金规模(亿)"] = aum_yi.where(aum_yi.notna() & (aum_yi > 0), raised)
    scale["成立日期"] = pd.to_datetime(scale.get("成立日期"), errors="coerce")
    return scale[["基金代码", "成立日期", "基金规模(亿)"]]


def scan_market_funds(sector_filter=None):
    """
    全市场开放式基金扫描：打分后按基金类型（赛道）输出每组前 3 名。

    过滤：规模≥2亿、成立满1年、近60日回撤好于 -25%。
    排行接口无独立 60 日字段，近60日收益用「近3月」，回撤用近1周/近1月/近3月最差值。
    """
    empty = pd.DataFrame(columns=MARKET_SCAN_COLUMNS)
    try:
        rank = _fetch_open_fund_rank()
        if rank is None or rank.empty:
            return empty

        rank = rank.copy()
        rank["基金代码"] = rank["基金代码"].astype(str).str.strip()
        if "基金简称" not in rank.columns and "基金名称" in rank.columns:
            rank["基金简称"] = rank["基金名称"]

        ret_pct = pd.to_numeric(rank.get("近3月", rank.get("近60日收益率")), errors="coerce")
        week = pd.to_numeric(rank.get("近1周"), errors="coerce")
        month = pd.to_numeric(rank.get("近1月"), errors="coerce")
        mdd_pct = pd.concat([week, month, ret_pct], axis=1).min(axis=1, skipna=True)
        mdd_pct = mdd_pct.clip(upper=0)
        mdd_for_score = mdd_pct.clip(upper=-1.0)

        rank["综合得分"] = calculate_score(ret_pct.to_numpy(), mdd_for_score.to_numpy())
        rank["近60日收益率"] = ret_pct / 100.0
        rank["近60日最大回撤"] = mdd_pct / 100.0

        types = _fetch_fund_type_table()
        scale = _fetch_scale_and_inception()
        merged = rank.merge(types, on="基金代码", how="left", suffixes=("", "_类型"))
        merged = merged.merge(scale, on="基金代码", how="left")
        merged["基金名称"] = merged["基金简称"] if "基金简称" in merged.columns else merged["基金代码"]
        if "基金简称_类型" in merged.columns:
            merged["基金名称"] = merged["基金名称"].fillna(merged["基金简称_类型"])
        merged["基金类型"] = merged["基金类型"].fillna("未分类")
        merged["赛道归类"] = merged["基金类型"]

        aum = pd.to_numeric(merged["基金规模(亿)"], errors="coerce")
        inception = pd.to_datetime(merged["成立日期"], errors="coerce")
        age_days = (pd.Timestamp.now().normalize() - inception).dt.days
        mdd_ok = pd.to_numeric(merged["近60日最大回撤"], errors="coerce") > (MAX_DRAWDOWN_PCT / 100.0)

        not_money = ~merged["基金类型"].astype(str).str.contains("货币", na=False)
        filtered = merged.loc[
            (aum >= MIN_AUM_YI)
            & (age_days >= MIN_AGE_DAYS)
            & mdd_ok
            & not_money
            & merged["综合得分"].notna()
        ].copy()

        if sector_filter:
            key = str(sector_filter).strip()
            if key:
                mask = filtered["基金类型"].astype(str).str.contains(key, na=False) | filtered[
                    "基金名称"
                ].astype(str).str.contains(key, na=False)
                filtered = filtered.loc[mask]

        if filtered.empty:
            return empty

        filtered = filtered.sort_values("综合得分", ascending=False, kind="mergesort")
        filtered["赛道内排名"] = filtered.groupby("赛道归类", dropna=False).cumcount() + 1
        top = filtered.loc[filtered["赛道内排名"] <= TOP_N_PER_SECTOR].copy()
        top = top.sort_values(["赛道归类", "赛道内排名"], kind="mergesort")
        for col in MARKET_SCAN_COLUMNS:
            if col not in top.columns:
                top[col] = np.nan
        return top[MARKET_SCAN_COLUMNS].reset_index(drop=True)
    except Exception:
        return empty
