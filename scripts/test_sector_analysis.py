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

from src.factor_layer import sector_analysis as sa


def _nav_frame(start, stop, periods=60, tail_start=None, tail_stop=None):
    dates = pd.date_range("2025-01-02", periods=periods, freq="B")
    if tail_start is None:
        nav = np.linspace(start, stop, periods)
    else:
        nav = np.full(periods, float(start), dtype=float)
        nav[-20:] = np.linspace(float(tail_start), float(tail_stop), 20)
    return pd.DataFrame({"date": dates, "nav": nav, "change_pct": 0.0})


def test_equal_weight_series_and_trend():
    strong = _nav_frame(1.0, 1.0, tail_start=1.0, tail_stop=1.08)
    weak = _nav_frame(1.0, 1.0, tail_start=1.0, tail_stop=0.92)
    lookup = {"000001": strong, "000002": weak}

    original_list = sa.list_sector_fund_codes
    original_load = sa.load_local_data
    sa.list_sector_fund_codes = lambda sector: ["000001"] if sector == "半导体" else ["000002"]
    sa.load_local_data = lambda code: lookup[code]
    try:
        series = sa.get_sector_nav_series("半导体")
        assert not series.empty
        assert series.index.name == "date"
        assert abs(float(series["nav"].iloc[0]) - 1.0) < 1e-9
        label, ret = sa.get_sector_trend("半导体", window=20)
        assert label == sa.TREND_STRONG
        assert ret > 5
        weak_label, weak_ret = sa.get_sector_trend("新能源", window=20)
        assert weak_label == sa.TREND_WEAK
        assert weak_ret < -5
        print("ok nav series and trend")
    finally:
        sa.list_sector_fund_codes = original_list
        sa.load_local_data = original_load


def test_top_funds_and_watch_advice():
    scores = pd.DataFrame(
        {
            "基金代码": ["000001", "000002", "000003"],
            "近60日收益率": [0.08, 0.03, -0.02],
            "近60日最大回撤": [-0.05, -0.04, -0.06],
            "近60日波动率": [0.1, 0.1, 0.1],
            "综合得分": [20.0, 12.0, 5.0],
        }
    )
    original_list = sa.list_sector_fund_codes
    original_score = sa.batch_score_funds
    sa.list_sector_fund_codes = lambda sector: ["000001", "000002", "000003"]
    sa.batch_score_funds = lambda codes, window=None: scores
    try:
        tops = sa.get_top_funds_in_sector("半导体", top_n=2, scores=scores)
        assert tops == ["000001", "000002"]
        print("ok top funds")
    finally:
        sa.list_sector_fund_codes = original_list
        sa.batch_score_funds = original_score


def test_equal_weight_two_funds():
    up = _nav_frame(1.0, 1.10)
    down = _nav_frame(1.0, 0.90)
    lookup = {"A": up, "B": down}
    original_list = sa.list_sector_fund_codes
    original_load = sa.load_local_data
    sa.list_sector_fund_codes = lambda sector: ["A", "B"]
    sa.load_local_data = lambda code: lookup[code]
    try:
        series = sa.get_sector_nav_series("混合")
        # equal-weight of +10% and -10% stays near 1
        assert abs(float(series["nav"].iloc[-1]) - 1.0) < 1e-6
        label, ret = sa.get_sector_trend("混合", window=20)
        assert label == sa.TREND_NEUTRAL
        print("ok equal weight")
    finally:
        sa.list_sector_fund_codes = original_list
        sa.load_local_data = original_load


def test_display_window_slice():
    long_nav = _nav_frame(1.0, 1.20, periods=120)
    original_list = sa.list_sector_fund_codes
    original_load = sa.load_local_data
    sa.list_sector_fund_codes = lambda sector: ["000001"]
    sa.load_local_data = lambda code: long_nav
    try:
        series20 = sa.get_sector_nav_series("半导体", window=20)
        series120 = sa.get_sector_nav_series("半导体", window=120)
        assert len(series20) == 20
        assert len(series120) == 120
        label, ret = sa.get_sector_trend("半导体", window=20)
        assert label == sa.TREND_NEUTRAL
        assert ret is not None
        print("ok display window slice")
    finally:
        sa.list_sector_fund_codes = original_list
        sa.load_local_data = original_load


def main():
    test_equal_weight_series_and_trend()
    test_top_funds_and_watch_advice()
    test_equal_weight_two_funds()
    test_display_window_slice()
    print("all sector analysis tests passed")


if __name__ == "__main__":
    raise SystemExit(main())
