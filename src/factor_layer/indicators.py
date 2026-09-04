import numpy as np
import pandas as pd


def _as_nav_series(nav_series):
    """将输入规范为按原顺序排列的净值 Series，去掉无效值。"""
    series = pd.Series(nav_series, copy=True)
    series = pd.to_numeric(series, errors="coerce").dropna()
    return series


def _resolve_window(window, days, default=60):
    if window is not None:
        return max(int(window), 2)
    if days is not None:
        return max(int(days), 2)
    return max(int(default), 2)


def calc_return(nav_series, days=60, window=None):
    """
    计算区间收益率（百分比点数，如 5.2 代表 5.2%）。

    取最近 window（或 days）个交易日；若不足则取全部。
    公式：(最新净值 / 区间首日净值 - 1) * 100
    """
    try:
        series = _as_nav_series(nav_series)
        if series.size < 2:
            return np.nan

        length = _resolve_window(window, days, 60)
        sliced = series.iloc[-length:]
        start_nav = sliced.iloc[0]
        end_nav = sliced.iloc[-1]
        if start_nav == 0 or pd.isna(start_nav) or pd.isna(end_nav):
            return np.nan
        return float((end_nav / start_nav - 1.0) * 100.0)
    except Exception:
        return np.nan


def calc_max_drawdown(nav_series, window=None, days=None):
    """
    计算最大回撤（百分比点数，负值，如 -8.5 代表 -8.5%）。

    若传入 window / days，先截取最近若干交易日再计算。
    回撤 = 当前净值 / 历史累计峰值 - 1，取最小值。
    """
    try:
        series = _as_nav_series(nav_series)
        if series.empty:
            return np.nan
        if window is not None or days is not None:
            length = _resolve_window(window, days, 60)
            series = series.iloc[-length:]
        if series.empty:
            return np.nan

        peak = series.cummax()
        valid_peak = peak.replace(0, np.nan)
        drawdown = series / valid_peak - 1.0
        mdd = drawdown.min()
        if pd.isna(mdd):
            return np.nan
        return float(mdd * 100.0)
    except Exception:
        return np.nan


def calc_volatility(nav_series, days=60, window=None):
    """
    计算年化波动率（百分比点数）。

    最近 window（或 days）个交易日的对数收益率标准差 × sqrt(252)。
    """
    try:
        series = _as_nav_series(nav_series)
        if series.size < 2:
            return np.nan

        length = _resolve_window(window, days, 60)
        log_return = np.log(series / series.shift(1)).dropna()
        if log_return.empty:
            return np.nan

        sliced = log_return.iloc[-length:]
        if sliced.size < 2:
            return np.nan

        annualized = sliced.std(ddof=1) * np.sqrt(252.0) * 100.0
        return float(annualized)
    except Exception:
        return np.nan
