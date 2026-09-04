"""验证基准缺失时回测仍能完成，并输出策略自身绩效指标。

用法：
    python scripts/test_backtest_benchmark.py

建议场景（本脚本已覆盖）：
    1. 注释/短路全部沪深300数据源（模拟断网）→ 回测不中断
    2. 手动 CSV 格式错误 → 给出具体错误
    3. 手动 CSV 正确 → 优先作为基准，benchmark_available=True
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

from src.backtest.engine import parse_benchmark_csv, run_backtest
import src.backtest.engine as engine


def _make_nav(start="2023-01-01", end="2024-12-31", start_nav=1.0, drift=0.0004):
    dates = pd.bdate_range(start, end)
    steps = np.linspace(0, drift * len(dates), len(dates))
    noise = np.sin(np.arange(len(dates)) / 8.0) * 0.002
    nav = start_nav * (1.0 + steps + noise)
    nav = np.maximum(nav, 0.5)
    return pd.DataFrame({"date": dates, "nav": nav})


def _fake_load_fund_nav(fund_code, allow_network=True):
    code = str(fund_code)
    if code.endswith("1"):
        return _make_nav(start_nav=1.0, drift=0.0005)
    return _make_nav(start_nav=1.2, drift=0.0003)


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def test_parse_csv_errors():
    empty, err = parse_benchmark_csv(io.BytesIO(b""))
    _assert(empty is None and err, "空文件应报错")
    print("  空文件：", err)

    bad = "foo,bar\n1,2\n"
    df, err = parse_benchmark_csv(io.BytesIO(bad.encode("utf-8")))
    _assert(df is None and err, "缺少有效日期/收盘价应报错")
    print("  无效列：", err)

    missing_close = "date,open\n2024-01-02,1\n2024-01-03,2\n"
    df, err = parse_benchmark_csv(io.BytesIO(missing_close.encode("utf-8")))
    _assert(df is None and err, "缺少收盘价列应报错")
    print("  缺收盘价：", err)

    only_date = "date\n2024-01-02\n2024-01-03\n"
    df, err = parse_benchmark_csv(io.BytesIO(only_date.encode("utf-8")))
    _assert(df is None and err, "仅有日期列应报错")
    print("  仅日期列：", err)


def test_parse_csv_ok():
    text = "日期,收盘价\n2024-01-02,3200.12\n2024-01-03,3210.5\n2024-01-04,3198.0\n"
    df, err = parse_benchmark_csv(io.BytesIO(text.encode("utf-8")))
    _assert(err is None, err or "合法 CSV 不应报错")
    _assert(list(df.columns) == ["date", "close"], "应标准化为 date/close")
    _assert(len(df) == 3, "应保留 3 行")
    print("  合法 CSV：", len(df), "行")


def test_backtest_without_benchmark():
    engine.load_fund_nav = _fake_load_fund_nav
    engine.get_benchmark_data = lambda start_date, end_date: None
    engine._read_index_cache = lambda: None
    engine._write_index_cache = lambda hist: None

    result = run_backtest(
        ["000001", "000002"],
        "2024-01-02",
        "2024-12-31",
        mode="aggressive",
        frequency="monthly",
        initial_cash=100000.0,
        allow_network=True,
        sector_map={"000001": "新能源", "000002": "医药"},
        name_map={"000001": "测试基金A", "000002": "测试基金B"},
    )
    metrics = result["metrics"]
    _assert(result.get("benchmark_available") is False, "基准应标记为不可用")
    _assert("equity" in result and not result["equity"].empty, "应产出策略净值曲线")
    _assert("benchmark" not in result["equity"].columns, "无基准时不应带沪深300列")
    _assert(any("沪深300数据获取失败" in str(item) for item in result.get("warnings") or []), "应给出基准失败提示")
    for key in ("total_return", "annual_return", "max_drawdown", "sharpe", "calmar", "volatility"):
        _assert(key in metrics, f"缺少绩效字段 {key}")
        _assert(metrics[key] is not None, f"{key} 不应为 None")
        _assert(np.isfinite(float(metrics[key])), f"{key} 应为有限数值")
    _assert(metrics.get("benchmark_return") is None, "无基准时不应计算沪深300收益")
    print("  benchmark_available =", result["benchmark_available"])
    print("  warnings =", result["warnings"])
    print("  总收益率 =", f"{metrics['total_return'] * 100:.2f}%")
    print("  年化收益率 =", f"{metrics['annual_return'] * 100:.2f}%")
    print("  最大回撤 =", f"{metrics['max_drawdown'] * 100:.2f}%")
    print("  夏普比率 =", f"{metrics['sharpe']:.4f}")
    print("  卡玛比率 =", f"{metrics['calmar']:.4f}")
    print("  年化波动 =", f"{metrics['volatility'] * 100:.2f}%")
    print("  交易日数 =", metrics.get("trade_days"))
    return result


def test_backtest_with_uploaded_benchmark():
    engine.load_fund_nav = _fake_load_fund_nav
    def _should_not_fetch(start_date, end_date):
        raise AssertionError("已上传基准时不应再走自动获取")

    engine.get_benchmark_data = _should_not_fetch
    engine._read_index_cache = lambda: None

    dates = pd.bdate_range("2023-01-01", "2024-12-31")
    close = 3000 + np.linspace(0, 200, len(dates))
    bench = pd.DataFrame({"date": dates, "close": close})
    result = run_backtest(
        ["000001"],
        "2024-01-02",
        "2024-06-28",
        mode="defensive",
        frequency="monthly",
        initial_cash=100000.0,
        allow_network=True,
        benchmark_df=bench,
    )
    _assert(result.get("benchmark_available") is True, "上传基准后应可用")
    _assert("benchmark" in result["equity"].columns, "应绘制基准对比列")
    _assert(result["metrics"].get("benchmark_return") is not None, "应计算基准收益")
    print("  上传基准后 benchmark_available =", result["benchmark_available"])
    print("  沪深300收益 =", f"{result['metrics']['benchmark_return'] * 100:.2f}%")
    print("  超额收益 =", f"{result['metrics']['excess_return'] * 100:.2f}%")


def main():
    print("== 1. CSV 校验 ==")
    test_parse_csv_errors()
    test_parse_csv_ok()
    print("\n== 2. 全部数据源失败时回测仍完成 ==")
    test_backtest_without_benchmark()
    print("\n== 3. 手动上传基准优先使用 ==")
    test_backtest_with_uploaded_benchmark()
    print("\n全部验证通过。")
    print("手动断网验证：可临时把 get_benchmark_data 内三个数据源注释掉，再在看板运行回测。")


if __name__ == "__main__":
    main()
