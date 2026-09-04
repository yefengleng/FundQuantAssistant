import numpy as np
import pandas as pd
import akshare as ak

from config.settings import AUTO_MODE_RULE, get_strategy_profile, normalize_strategy_mode


CRASH_THRESHOLD = -12.0
_REGIME_DETAIL = None


def _normalize_index_df(raw_df):
    df = raw_df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    rename_map = {
        "日期": "date",
        "date": "date",
        "收盘": "close",
        "收盘价": "close",
        "close": "close",
    }
    df = df.rename(columns=rename_map)
    if "date" not in df.columns or "close" not in df.columns:
        raise KeyError(f"指数数据缺少日期或收盘价列: {list(raw_df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    return df.reset_index(drop=True)


def _fetch_hs300_history():
    try:
        return ak.index_hist_em(symbol="000300")
    except (TypeError, AttributeError):
        return ak.index_zh_a_hist(symbol="000300", period="daily")


def get_hs300_monthly_return(year, month):
    """
    沪深300指定年月的月度涨跌幅（百分比点数，如 -3.2 代表 -3.2%）。
    公式：(月末收盘价 / 月初收盘价 - 1) * 100
    """
    try:
        raw_df = _fetch_hs300_history()
        if raw_df is None or raw_df.empty:
            return np.nan

        hist = _normalize_index_df(raw_df)
        year = int(year)
        month = int(month)
        month_df = hist[(hist["date"].dt.year == year) & (hist["date"].dt.month == month)]
        if month_df.empty or len(month_df) < 1:
            return np.nan

        start_close = float(month_df.iloc[0]["close"])
        end_close = float(month_df.iloc[-1]["close"])
        if start_close == 0:
            return np.nan
        return float((end_close / start_close - 1.0) * 100.0)
    except Exception:
        return np.nan


def check_market_crash(year, month):
    """沪深300单月跌幅 < -12% 时触发急跌过滤器；网络异常或数据不足时不触发。"""
    try:
        monthly_return = get_hs300_monthly_return(year, month)
        if monthly_return is None or pd.isna(monthly_return):
            return False
        return bool(monthly_return < CRASH_THRESHOLD)
    except Exception:
        return False


def _failed_regime_detail(period, index_code):
    return {
        "mode": "defensive",
        "fetch_failed": True,
        "close": np.nan,
        "ma": np.nan,
        "ma_period": period,
        "index_code": index_code,
    }


def inspect_market_regime(force=False):
    """拉取指数并比较最新收盘价与均线；失败时静默降为防御型。"""
    global _REGIME_DETAIL
    if _REGIME_DETAIL is not None and not force:
        return dict(_REGIME_DETAIL)

    rule = AUTO_MODE_RULE if isinstance(AUTO_MODE_RULE, dict) else {}
    try:
        period = int(rule.get("ma_period", 120) or 120)
    except (TypeError, ValueError):
        period = 120
    period = max(period, 1)
    index_code = str(rule.get("index_code") or "000300").strip() or "000300"

    try:
        try:
            raw_df = ak.index_zh_a_hist(symbol=index_code, period="daily")
        except Exception:
            raw_df = None
        if raw_df is None or raw_df.empty:
            try:
                raw_df = _fetch_hs300_history()
            except Exception:
                raw_df = None
        if raw_df is None or raw_df.empty:
            detail = _failed_regime_detail(period, index_code)
            _REGIME_DETAIL = detail
            return dict(detail)

        hist = _normalize_index_df(raw_df)
        window = hist.tail(period)
        if len(window) < period:
            detail = _failed_regime_detail(period, index_code)
            _REGIME_DETAIL = detail
            return dict(detail)

        close = float(window.iloc[-1]["close"])
        ma = float(window["close"].mean())
        if not np.isfinite(close) or not np.isfinite(ma) or ma == 0:
            detail = _failed_regime_detail(period, index_code)
            _REGIME_DETAIL = detail
            return dict(detail)

        mode = "aggressive" if close > ma else "defensive"
        detail = {
            "mode": mode,
            "fetch_failed": False,
            "close": close,
            "ma": ma,
            "ma_period": period,
            "index_code": index_code,
        }
        _REGIME_DETAIL = detail
        return dict(detail)
    except Exception:
        detail = _failed_regime_detail(period, index_code)
        _REGIME_DETAIL = detail
        return dict(detail)


def get_market_regime():
    """
    最新价 > 120 日均线 → aggressive，否则 defensive。
    网络或数据失败时返回 defensive（保守优先）。
    """
    return inspect_market_regime()["mode"]


def _build_mode_banner(requested, effective, regime_detail):
    period = int((regime_detail or {}).get("ma_period") or AUTO_MODE_RULE.get("ma_period", 120) or 120)
    fetch_failed = bool((regime_detail or {}).get("fetch_failed"))
    if requested == "auto" and fetch_failed:
        return "⚠️ 数据拉取失败，默认使用防御型"
    if requested == "auto":
        if effective == "aggressive":
            return f"🚀 自动切换至进攻型（沪深300 > {period}日均线）"
        return f"🛡️ 自动切换至防御型（沪深300 ≤ {period}日均线）"
    if effective == "aggressive":
        return "🚀 当前为进攻型（手动选择）"
    return "🛡️ 当前为防御型（手动选择）"


def resolve_strategy_mode(strategy_mode=None, regime=None, allow_network=True):
    """
    全系统唯一的模式解析入口。

    请求值为 auto 时，用 get_market_regime() 覆盖为进攻或防御；
    其余情况直接采用 STRATEGY_MODE / 传入值。非法值按防御处理。
    allow_network=False 时不拉指数，沿用缓存或回退防御型。
    """
    from config.settings import STRATEGY_MODE as configured_mode

    requested = normalize_strategy_mode(
        strategy_mode if strategy_mode is not None else configured_mode
    )
    if requested == "auto":
        if allow_network:
            regime_detail = inspect_market_regime()
        elif _REGIME_DETAIL is not None:
            regime_detail = dict(_REGIME_DETAIL)
        else:
            regime_detail = {
                "mode": "defensive",
                "fetch_failed": False,
                "close": np.nan,
                "ma": np.nan,
                "ma_period": (AUTO_MODE_RULE or {}).get("ma_period", 120),
                "index_code": (AUTO_MODE_RULE or {}).get("index_code", "000300"),
            }
        if regime in {"defensive", "aggressive"}:
            effective = regime
        else:
            effective = regime_detail.get("mode") or "defensive"
        if effective not in {"defensive", "aggressive"}:
            effective = "defensive"
        fetch_failed = bool(regime_detail.get("fetch_failed"))
        if fetch_failed:
            effective = "defensive"
    else:
        effective = requested
        fetch_failed = False
        regime_detail = {
            "mode": requested,
            "fetch_failed": False,
            "ma_period": (AUTO_MODE_RULE or {}).get("ma_period", 120),
            "index_code": (AUTO_MODE_RULE or {}).get("index_code", "000300"),
        }

    banner = _build_mode_banner(requested, effective, regime_detail)
    return {
        "requested": requested,
        "effective": effective,
        "fetch_failed": fetch_failed,
        "banner": banner,
        "detail": regime_detail,
        "profile": get_strategy_profile(effective),
    }
