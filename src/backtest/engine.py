import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from types import SimpleNamespace

import pandas as pd
import requests

from src.data_layer.loader import RAW_DIR, load_local_data, save_local_data

from .metrics import compute_metrics, drawdown_series
from .strategies import (
    BENCHMARK_NAME,
    BUY_FEE_DEFAULT,
    AggressiveStrategy,
    DefensiveStrategy,
    FundCommission,
    FundRebalanceStrategy,
    precompute_scores,
)


INDEX_CACHE = os.path.join(RAW_DIR, "_index_000300.parquet")
WARMUP_DAYS = 400
FETCH_TIMEOUT_SEC = 5
RETRY_SLEEP_SEC = 2
BENCHMARK_UNAVAILABLE_HINT = "⚠️ 沪深300数据获取失败，本次回测未包含基准对比。"


def _safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "gbk"
        print(str(message).encode(encoding, errors="replace").decode(encoding, errors="replace"))


class NavPandasData:
    """延迟导入 backtrader，避免未安装时拖垮模块加载。"""

    @staticmethod
    def create(frame):
        import backtrader as bt

        class _NavPandasData(bt.feeds.PandasData):
            params = (
                ("datetime", None),
                ("open", "close"),
                ("high", "close"),
                ("low", "close"),
                ("close", "close"),
                ("volume", -1),
                ("openinterest", -1),
            )

        return _NavPandasData(dataname=frame)


def _notify(callback, message):
    if not callback:
        return
    try:
        callback(message)
    except Exception:
        pass


def _empty_nav():
    return pd.DataFrame(columns=["date", "nav"])


def load_fund_nav(fund_code, allow_network=True):
    code = str(fund_code or "").strip()
    local = load_local_data(code)
    if local is not None and not local.empty:
        return local[["date", "nav"]].copy()
    if not allow_network:
        return _empty_nav()
    try:
        from src.data_layer.fetcher import fetch_fund_history

        remote = fetch_fund_history(code)
        if remote is not None and not remote.empty:
            save_local_data(code, remote)
            return remote[["date", "nav"]].copy()
    except Exception:
        return _empty_nav()
    return _empty_nav()


def _normalize_index(raw_df):
    if raw_df is None:
        return pd.DataFrame(columns=["date", "close"])
    df = raw_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "close"])
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=["date", "close"])
    df.columns = [str(col).strip() for col in df.columns]
    rename = {
        "日期": "date",
        "date": "date",
        "时间": "date",
        "收盘": "close",
        "收盘价": "close",
        "close": "close",
        "nav": "close",
    }
    df = df.rename(columns=rename)
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex) or str(df.index.name).lower() in {"date", "日期"}:
            df = df.reset_index()
            first = str(df.columns[0])
            if first not in rename and first != "date":
                df = df.rename(columns={first: "date"})
            df = df.rename(columns=rename)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date", "close"])
    if "close" not in df.columns:
        if "nav" in df.columns:
            df["close"] = df["nav"]
        else:
            return pd.DataFrame(columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    return df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")


def _read_index_cache():
    if not os.path.isfile(INDEX_CACHE):
        return None
    try:
        cached = _normalize_index(pd.read_parquet(INDEX_CACHE))
        if cached is not None and not cached.empty:
            return cached
    except Exception as exc:
        _safe_print(f"[WARN] 读取沪深300本地缓存失败：{exc}")
    return None


def _write_index_cache(hist):
    if hist is None or hist.empty:
        return
    try:
        os.makedirs(RAW_DIR, exist_ok=True)
        hist.to_parquet(INDEX_CACHE, index=False)
    except Exception as exc:
        _safe_print(f"[WARN] 写入沪深300本地缓存失败：{exc}")


def parse_benchmark_csv(source):
    """
    解析用户上传的沪深300 CSV。
    要求：日期列（YYYY-MM-DD）+ 收盘价列（close / 收盘价 / 收盘）。
    返回 (DataFrame, None) 或 (None, 错误说明)。
    """
    if source is None:
        return None, "未提供基准 CSV 文件"

    raw_bytes = None
    if hasattr(source, "read"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            raw_bytes = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
        except Exception as exc:
            return None, f"无法读取上传文件：{exc}"
        if isinstance(raw_bytes, str):
            raw_bytes = raw_bytes.encode("utf-8")
    elif isinstance(source, (bytes, bytearray)):
        raw_bytes = bytes(source)
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
        return _validate_benchmark_frame(df)
    else:
        path = os.fspath(source)
        try:
            with open(path, "rb") as handle:
                raw_bytes = handle.read()
        except Exception as exc:
            return None, f"无法读取文件：{exc}"

    if not raw_bytes:
        return None, "上传的 CSV 为空，请重新选择文件"

    last_error = None
    df = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            df = None
    if df is None:
        return None, f"CSV 无法解析（请使用 UTF-8 或 GBK 编码）：{last_error}"
    return _validate_benchmark_frame(df)


def _validate_benchmark_frame(df):
    if df is None or df.empty:
        return None, "CSV 没有有效数据行，请重新上传"

    work = df.copy()
    work.columns = [str(col).strip() for col in work.columns]
    date_aliases = {"date", "日期", "时间", "交易日期"}
    close_aliases = {"close", "收盘价", "收盘", "nav"}
    non_close_names = {
        "open", "high", "low", "volume", "amount",
        "开盘", "开盘价", "最高", "最高价", "最低", "最低价",
        "成交量", "成交额", "涨跌幅", "涨跌额",
    }

    def _norm_name(name):
        return str(name).strip().lower()

    date_col = next((col for col in work.columns if _norm_name(col) in date_aliases or str(col).strip() in date_aliases), None)
    close_col = next((col for col in work.columns if _norm_name(col) in close_aliases or str(col).strip() in close_aliases), None)

    if date_col is None and len(work.columns) >= 1:
        date_col = work.columns[0]
    if close_col is None and len(work.columns) >= 2:
        candidate = work.columns[1]
        if candidate != date_col and _norm_name(candidate) not in non_close_names and str(candidate).strip() not in non_close_names:
            close_col = candidate

    if date_col is None:
        return None, "CSV 缺少日期列。请使用第一列为日期（YYYY-MM-DD），或将列名设为 date / 日期。"
    if close_col is None:
        return None, "CSV 缺少收盘价列。请使用第二列为收盘价，或将列名设为 close / 收盘价。"

    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(work[date_col], errors="coerce"),
            "close": pd.to_numeric(work[close_col], errors="coerce"),
        }
    )
    parsed = parsed.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    parsed = parsed[parsed["close"] > 0]
    if parsed.empty:
        return None, "CSV 中没有有效的日期或收盘价。日期请使用 YYYY-MM-DD，收盘价需为数字。"
    if len(parsed) < 2:
        return None, "有效基准数据不足 2 行，请上传更完整的沪深300历史行情。"
    return parsed.reset_index(drop=True), None


def _run_with_timeout(fetcher):
    original_request = requests.sessions.Session.request

    def _request_with_timeout(self, method, url, *args, **kwargs):
        kwargs.setdefault("timeout", FETCH_TIMEOUT_SEC)
        return original_request(self, method, url, *args, **kwargs)

    requests.sessions.Session.request = _request_with_timeout
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fetcher)
            return future.result(timeout=FETCH_TIMEOUT_SEC)
    finally:
        requests.sessions.Session.request = original_request


def _try_benchmark_source(label, fetcher):
    for attempt in (1, 2):
        suffix = "（重试）" if attempt == 2 else ""
        _safe_print(f"[INFO] 尝试从 {label} 获取沪深300{suffix}...")
        try:
            raw = _run_with_timeout(fetcher)
            hist = _normalize_index(raw)
            if hist is not None and not hist.empty:
                _safe_print(f"[INFO] {label} 成功，共 {len(hist)} 条记录")
                return hist
            _safe_print(f"[WARN] {label} 返回空数据")
        except FutureTimeout:
            _safe_print(f"[WARN] {label} 请求超时（{FETCH_TIMEOUT_SEC}秒）")
        except Exception as exc:
            _safe_print(f"[WARN] {label} 失败：{exc}")
        if attempt == 1:
            _safe_print(f"[INFO] {label} 将在 {RETRY_SLEEP_SEC} 秒后重试一次")
            time.sleep(RETRY_SLEEP_SEC)
    return None


def get_benchmark_data(start_date, end_date):
    """
    按多级降级策略获取沪深300行情。全部失败时返回 None，不抛异常。
    每个数据源超时 5 秒；首次失败等待 2 秒后重试一次。
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    def _index_zh_a_hist():
        import akshare as ak

        return ak.index_zh_a_hist(
            symbol="000300",
            period="daily",
            start_date=start_str,
            end_date=end_str,
        )

    def _index_hist_em():
        import akshare as ak

        fn = getattr(ak, "index_hist_em", None)
        if fn is None:
            raise AttributeError("当前 akshare 版本没有 index_hist_em")
        return fn(symbol="000300")

    def _stock_zh_index_daily():
        import akshare as ak

        return ak.stock_zh_index_daily(symbol="sh000300")

    sources = (
        ("akshare.index_zh_a_hist", _index_zh_a_hist),
        ("akshare.index_hist_em", _index_hist_em),
        ("akshare.stock_zh_index_daily", _stock_zh_index_daily),
    )
    for label, fetcher in sources:
        hist = _try_benchmark_source(label, fetcher)
        if hist is None or hist.empty:
            continue
        _write_index_cache(hist)
        return hist

    _safe_print("[WARN] 所有沪深300数据源均失败，本次将跳过基准对比")
    return None


def load_hs300(allow_network=True, start_date=None, end_date=None):
    cached = _read_index_cache()
    if cached is not None and not cached.empty:
        return cached
    if not allow_network:
        return pd.DataFrame(columns=["date", "close"])
    start = start_date if start_date is not None else "19900101"
    end = end_date if end_date is not None else pd.Timestamp.today()
    hist = get_benchmark_data(start, end)
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["date", "close"])
    return hist


def make_rebalance_dates(calendar, frequency):
    dates = list(calendar)
    if not dates:
        return []
    if str(frequency).strip().lower() == "weekly":
        return [day for day in dates if pd.Timestamp(day).weekday() == 4]
    seen = set()
    result = []
    for day in dates:
        key = (day.year, day.month)
        if key in seen:
            continue
        seen.add(key)
        result.append(day)
    return result


def _to_close_frame(series, start, end):
    frame = pd.DataFrame({"close": pd.to_numeric(series, errors="coerce")})
    frame = frame.loc[start:end].dropna(subset=["close"])
    frame.index = pd.to_datetime(frame.index)
    frame = frame[frame["close"] > 0]
    return frame


def _calendar_from_navs(raw_navs, start, end):
    dates = set()
    for series in (raw_navs or {}).values():
        window = series.loc[start:end]
        for ts in window.index:
            if pd.notna(ts):
                dates.add(pd.Timestamp(ts).normalize())
    return sorted(dates)


def _resolve_hs300(benchmark_df, start, end, allow_network):
    if benchmark_df is not None:
        uploaded = _normalize_index(benchmark_df)
        if uploaded is not None and not uploaded.empty:
            _safe_print(f"[INFO] 使用手动上传的沪深300数据，共 {len(uploaded)} 条记录")
            return uploaded, True
        _safe_print("[WARN] 手动上传的基准数据无效，改为自动获取")

    cached = _read_index_cache()
    if cached is not None and not cached.empty:
        _safe_print(f"[INFO] 使用本地缓存的沪深300数据，共 {len(cached)} 条记录")
        return cached, True

    if not allow_network:
        _safe_print("[WARN] 未允许联网且没有沪深300缓存，跳过基准对比")
        return None, False

    fetched = get_benchmark_data(start, end)
    if fetched is not None and not fetched.empty:
        return fetched, True
    return None, False


def run_backtest(
    fund_codes,
    start_date,
    end_date,
    mode="aggressive",
    frequency="monthly",
    initial_cash=100000.0,
    buy_fee=BUY_FEE_DEFAULT,
    sector_map=None,
    name_map=None,
    allow_network=True,
    progress_callback=None,
    benchmark_df=None,
):
    """
    加载净值（本地优先）、用 Backtrader 跑进攻/保守调仓，返回曲线、交易与指标。
    基准不可用时仍完成策略回测，结果含 benchmark_available=False。
    """
    import backtrader as bt

    codes = []
    for code in fund_codes or []:
        text = str(code).strip()
        if text and text not in codes and text != BENCHMARK_NAME:
            codes.append(text)
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if not codes:
        raise ValueError("请至少选择一只基金")
    if start >= end:
        raise ValueError("结束日期必须晚于开始日期")

    _notify(progress_callback, "正在加载基金净值...")
    raw_navs = {}
    missing = []
    for index, code in enumerate(codes, start=1):
        _notify(progress_callback, f"加载净值 {code}（{index}/{len(codes)}）")
        nav = load_fund_nav(code, allow_network=allow_network)
        if nav is None or nav.empty:
            missing.append(code)
            continue
        series = nav.set_index(pd.to_datetime(nav["date"], errors="coerce"))["nav"]
        series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
        series = series[~series.index.duplicated(keep="last")]
        if not series.empty:
            raw_navs[code] = series
    if not raw_navs:
        raise ValueError("没有可用的基金净值，请先刷新数据")

    _notify(progress_callback, "正在加载沪深300基准...")
    warmup = start - pd.Timedelta(days=WARMUP_DAYS)
    hs300_df, benchmark_available = _resolve_hs300(
        benchmark_df,
        warmup,
        end,
        allow_network=allow_network,
    )
    hs300 = pd.Series(dtype=float)
    if benchmark_available and hs300_df is not None and not hs300_df.empty:
        hs300 = hs300_df.set_index("date")["close"]
        hs300 = pd.to_numeric(hs300, errors="coerce").dropna().sort_index()
        hs300 = hs300[~hs300.index.duplicated(keep="last")]
        if hs300.empty:
            benchmark_available = False

    if benchmark_available:
        calendar = sorted({ts.normalize() for ts in hs300.loc[start:end].index})
    else:
        _safe_print("[INFO] 沪深300不可用，改用基金净值交易日作为回测日历")
        calendar = _calendar_from_navs(raw_navs, start, end)
    calendar = [ts.date() for ts in calendar]
    if len(calendar) < 5:
        raise ValueError("选定区间内交易日过少，请扩大回测区间")

    aligned = {}
    for code, series in raw_navs.items():
        window = series.loc[warmup:end]
        if window.empty:
            continue
        reindexed = window.reindex(pd.DatetimeIndex(pd.to_datetime(calendar))).ffill()
        reindexed = reindexed.loc[start:end]
        if reindexed.dropna().empty:
            continue
        aligned[code] = reindexed

    if not aligned:
        raise ValueError("选定区间内基金净值无法对齐，请调整日期或基金列表")

    bench_aligned = None
    if benchmark_available:
        bench_aligned = hs300.reindex(pd.DatetimeIndex(pd.to_datetime(calendar))).ffill().loc[start:end]
        if bench_aligned.dropna().empty:
            benchmark_available = False
            bench_aligned = None

    rebalance_dates = make_rebalance_dates(calendar, frequency)
    if not rebalance_dates:
        raise ValueError("区间内没有调仓日")

    _notify(progress_callback, "正在预计算调仓日得分...")
    nav_for_score = dict(raw_navs)
    if benchmark_available:
        nav_for_score[BENCHMARK_NAME] = hs300
    score_cache = precompute_scores(nav_for_score, rebalance_dates)

    from src.factor_layer.sector_classifier import load_global_sector_mapping

    merged_sectors = {str(k): (v or "其他") for k, v in (sector_map or {}).items()}
    merged_sectors.update(load_global_sector_mapping())
    ctx = SimpleNamespace(
        rebalance_dates=set(rebalance_dates),
        nav_map=raw_navs,
        score_cache=score_cache,
        sector_map=merged_sectors,
        name_map={str(k): str(v or "") for k, v in (name_map or {}).items()},
        hs300=hs300 if benchmark_available else pd.Series(dtype=float),
    )

    requested = str(mode or "aggressive").strip().lower()
    if requested == "defensive":
        strategy_cls = DefensiveStrategy
    elif requested == "auto":
        strategy_cls = FundRebalanceStrategy
    else:
        strategy_cls = AggressiveStrategy
        requested = "aggressive"

    _notify(progress_callback, "正在运行 Backtrader 回测...")
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(float(initial_cash))
    cerebro.broker.set_coc(True)
    cerebro.broker.addcommissioninfo(FundCommission())

    if benchmark_available and bench_aligned is not None:
        bench_frame = _to_close_frame(bench_aligned, start, end)
        if bench_frame.empty:
            benchmark_available = False
        else:
            cerebro.adddata(NavPandasData.create(bench_frame), name=BENCHMARK_NAME)
    added = []
    for code, series in aligned.items():
        frame = _to_close_frame(series, start, end)
        if frame.empty:
            continue
        cerebro.adddata(NavPandasData.create(frame), name=code)
        added.append(code)
    if not added:
        raise ValueError("没有成功加入回测的基金行情")

    cerebro.addstrategy(
        strategy_cls,
        mode=requested,
        frequency=str(frequency or "monthly"),
        buy_fee=float(buy_fee or 0.0),
        context=ctx,
    )
    result_strats = cerebro.run()
    strat = result_strats[0]

    equity = pd.DataFrame(strat.equity_records)
    if equity.empty:
        raise ValueError("回测没有产生净值记录")
    equity["date"] = pd.to_datetime(equity["date"], errors="coerce")
    equity = equity.dropna(subset=["date"]).sort_values("date").set_index("date")
    equity["drawdown"] = drawdown_series(equity["strategy"])
    equity = equity.reset_index()
    if not benchmark_available and "benchmark" in equity.columns:
        equity = equity.drop(columns=["benchmark"])
    elif "benchmark" in equity.columns:
        equity["benchmark"] = pd.to_numeric(equity["benchmark"], errors="coerce")
        if not equity["benchmark"].notna().any():
            equity = equity.drop(columns=["benchmark"])
            benchmark_available = False
    metrics = compute_metrics(equity)
    trades = pd.DataFrame(strat.trade_records)
    holdings = pd.DataFrame(strat.holding_records)
    if not trades.empty:
        trades["date"] = pd.to_datetime(trades["date"], errors="coerce")
        trades = trades.sort_values("date").reset_index(drop=True)
    if not holdings.empty:
        holdings["date"] = pd.to_datetime(holdings["date"], errors="coerce")
        holdings = holdings.sort_values(["date", "基金代码"]).reset_index(drop=True)

    warnings = []
    if not benchmark_available:
        warnings.append(BENCHMARK_UNAVAILABLE_HINT)
    if missing:
        warnings.append("以下基金没有净值，已跳过：" + "、".join(missing))
    skipped = [code for code in codes if code not in added]
    if skipped:
        warnings.append("以下基金在区间内无法对齐，已跳过：" + "、".join(skipped))
    metrics["trade_count"] = int(len(trades))
    metrics["buy_count"] = int((trades["方向"] == "买入").sum()) if not trades.empty else 0
    metrics["sell_count"] = int((trades["方向"] == "卖出").sum()) if not trades.empty else 0
    metrics["fee_total"] = float(pd.to_numeric(trades.get("费用"), errors="coerce").fillna(0).sum()) if not trades.empty else 0.0
    metrics["end_value"] = float(equity["strategy"].iloc[-1])
    metrics["start_value"] = float(equity["strategy"].iloc[0])
    metrics["final_cash"] = float(equity["cash"].iloc[-1]) if "cash" in equity.columns else 0.0

    return {
        "metrics": metrics,
        "equity": equity,
        "trades": trades,
        "holdings": holdings,
        "mode": requested,
        "frequency": str(frequency or "monthly"),
        "funds": added,
        "warnings": warnings,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "initial_cash": float(initial_cash),
        "benchmark_available": bool(benchmark_available),
    }
