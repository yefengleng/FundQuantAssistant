import numpy as np
import pandas as pd


TRADING_DAYS = 252
DEFAULT_RF = 0.015


def _as_series(equity):
    if isinstance(equity, pd.DataFrame):
        if "strategy" in equity.columns:
            series = equity.set_index(equity["date"] if "date" in equity.columns else equity.index)["strategy"]
        else:
            series = equity.iloc[:, 0]
    else:
        series = pd.Series(equity)
    series = pd.to_numeric(series, errors="coerce").dropna()
    try:
        series.index = pd.to_datetime(series.index, errors="coerce")
        series = series[series.index.notna()].sort_index()
    except Exception:
        pass
    return series


def max_drawdown(equity):
    series = _as_series(equity)
    if series.empty:
        return 0.0
    peak = series.cummax()
    drawdown = series / peak.replace(0, np.nan) - 1.0
    value = float(drawdown.min()) if drawdown.notna().any() else 0.0
    return value if np.isfinite(value) else 0.0


def drawdown_series(equity):
    series = _as_series(equity)
    if series.empty:
        return pd.Series(dtype=float)
    peak = series.cummax()
    return series / peak.replace(0, np.nan) - 1.0


def compute_metrics(equity, benchmark=None, risk_free_rate=DEFAULT_RF):
    """
    由策略净值序列计算绩效。equity 需含 date / strategy，可选 benchmark 列。
    收益率类指标以小数返回（0.12 表示 12%）。
    """
    empty = {
        "total_return": 0.0,
        "annual_return": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "calmar": 0.0,
        "volatility": 0.0,
        "benchmark_return": None,
        "excess_return": None,
        "information_ratio": None,
        "trade_days": 0,
        "years": 0.0,
    }
    if isinstance(equity, pd.DataFrame):
        frame = equity.copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"]).sort_values("date")
        series = pd.to_numeric(frame["strategy"] if "strategy" in frame.columns else frame.iloc[:, 0], errors="coerce")
        dates = pd.to_datetime(frame["date"] if "date" in frame.columns else frame.index, errors="coerce")
        bench_col = None
        if benchmark is None and "benchmark" in frame.columns:
            bench_col = pd.to_numeric(frame["benchmark"], errors="coerce")
        elif benchmark is not None:
            bench_col = pd.to_numeric(benchmark, errors="coerce")
    else:
        series = pd.to_numeric(pd.Series(equity), errors="coerce")
        dates = pd.to_datetime(series.index, errors="coerce")
        bench_col = pd.to_numeric(benchmark, errors="coerce") if benchmark is not None else None

    mask = series.notna() & (series > 0)
    series = series[mask].reset_index(drop=True)
    dates = pd.Series(dates).reset_index(drop=True)
    if len(dates) == len(mask):
        dates = dates[mask.values].reset_index(drop=True)
    else:
        dates = dates.iloc[: len(series)].reset_index(drop=True)
    if bench_col is not None:
        bench_col = pd.Series(bench_col).reset_index(drop=True)
        if len(bench_col) == len(mask):
            bench_col = bench_col[mask.values].reset_index(drop=True)
        else:
            bench_col = bench_col.iloc[: len(series)].reset_index(drop=True)

    if len(series) < 2:
        return empty

    start_value = float(series.iloc[0])
    end_value = float(series.iloc[-1])
    if start_value <= 0:
        return empty

    total_return = end_value / start_value - 1.0
    elapsed_days = (dates.iloc[-1] - dates.iloc[0]).days if dates.notna().all() else max(len(series) - 1, 1)
    years = max(float(elapsed_days) / 365.25, 1.0 / TRADING_DAYS)
    annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0

    daily = series.pct_change().dropna()
    vol = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(daily) > 1 else 0.0
    excess_daily = daily - float(risk_free_rate) / TRADING_DAYS
    daily_vol = float(excess_daily.std(ddof=1) or 0.0) if len(excess_daily) > 1 else 0.0
    if daily_vol > 1e-8:
        sharpe = float(excess_daily.mean() / daily_vol * np.sqrt(TRADING_DAYS))
    else:
        sharpe = 0.0
    mdd = max_drawdown(series)
    calmar = float(annual_return / abs(mdd)) if mdd < 0 else 0.0

    bench_return = None
    excess_return = None
    information_ratio = None
    if bench_col is not None and len(bench_col) >= 2 and float(bench_col.iloc[0] or 0) > 0:
        bench_return = float(bench_col.iloc[-1] / bench_col.iloc[0] - 1.0)
        excess_return = float(total_return) - bench_return
        strat_ret = series.pct_change().dropna().reset_index(drop=True)
        bench_ret = pd.Series(bench_col).pct_change().dropna().reset_index(drop=True)
        n = min(len(strat_ret), len(bench_ret))
        active = strat_ret.iloc[:n] - bench_ret.iloc[:n]
        active_vol = float(active.std(ddof=1) or 0.0) if len(active) > 1 else 0.0
        if active_vol > 1e-8:
            information_ratio = float(active.mean() / active_vol * np.sqrt(TRADING_DAYS))

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": float(mdd),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else 0.0,
        "calmar": float(calmar) if np.isfinite(calmar) else 0.0,
        "volatility": float(vol) if np.isfinite(vol) else 0.0,
        "benchmark_return": bench_return,
        "excess_return": excess_return,
        "information_ratio": information_ratio,
        "trade_days": int(len(series)),
        "years": float(years),
    }
