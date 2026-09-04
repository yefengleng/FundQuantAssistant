import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

from src.data_layer.market_clock import is_trading_day, is_trading_session, seconds_until_refresh
from src.data_layer.realtime_cache import is_cache_stale, pick_signal_estimates
from src.strategy_layer.cooldown import apply_cooldown
from src.strategy_layer.intraday_monitor import (
    build_sector_intraday,
    detect_emergency,
    overlay_holdings_estimates,
)


def _holdings():
    return pd.DataFrame(
        {
            "基金代码": ["000001", "000002", "000003"],
            "基金名称": ["基金A", "基金B", "基金C"],
            "赛道归类": ["新能源", "新能源", "消费"],
            "持有份额": [1000.0, 2000.0, 1500.0],
            "最新净值": [1.00, 2.00, 1.50],
            "持仓市值": [1000.0, 4000.0, 2250.0],
        }
    )


def test_trading_session():
    saturday = pd.Timestamp("2026-09-05 10:30:00")
    assert is_trading_day(saturday) is False
    assert is_trading_session(saturday) is False

    weekday_open = pd.Timestamp("2026-09-04 10:30:00")
    weekday_close = pd.Timestamp("2026-09-04 16:00:00")
    weekday_early = pd.Timestamp("2026-09-04 09:00:00")
    assert is_trading_day(weekday_open) is True
    assert is_trading_session(weekday_open) is True
    assert is_trading_session(weekday_close) is False
    assert is_trading_session(weekday_early) is False
    assert is_trading_session(pd.Timestamp("2026-09-04 09:30:00")) is True
    assert is_trading_session(pd.Timestamp("2026-09-04 15:00:00")) is True
    print("OK trading session")


def test_cache_ttl():
    now = pd.Timestamp("2026-09-04 10:30:00")
    fresh = {"updated_at": "2026-09-04 10:26:00"}
    stale = {"updated_at": "2026-09-04 10:24:00"}
    assert is_cache_stale(fresh, ttl=300, ts=now) is False
    assert is_cache_stale(stale, ttl=300, ts=now) is True
    assert is_cache_stale({}, ttl=300, ts=now) is True
    remain = seconds_until_refresh("2026-09-04 10:26:00", interval=300, ts=now)
    assert remain == 60
    print("OK cache ttl")


def test_overlay_and_emergency():
    holdings = _holdings()
    payload = {
        "updated_at": "2026-09-04 14:10:00",
        "items": {
            "000001": {"nav_estimate": 0.92, "change_pct": -8.0},
            "000002": {"nav_estimate": 1.88, "change_pct": -6.0},
            "000003": {"nav_estimate": 1.52, "change_pct": 1.3},
        },
    }
    frame, meta = overlay_holdings_estimates(holdings, payload)
    assert meta["degraded"] is False
    assert meta["ok_count"] == 3
    assert float(frame.loc[frame["基金代码"] == "000001", "当日估算涨跌幅"].iloc[0]) == -8.0
    pnl_a = float(frame.loc[frame["基金代码"] == "000001", "当日估算盈亏"].iloc[0])
    assert abs(pnl_a - (1000.0 * (0.92 - 1.00))) < 1e-8

    sector = build_sector_intraday(frame)
    energy = sector.loc[sector["赛道"] == "新能源", "当日估算涨跌幅"].iloc[0]
    assert energy < -5.0

    emergency = detect_emergency(frame, sector, account_drawdown=-0.05)
    assert emergency["triggered"] is True
    assert any(item.get("紧急") for item in emergency["fund_hits"])
    assert any(item.get("赛道") == "新能源" for item in emergency["sector_hits"])

    melt = detect_emergency(frame, sector, account_drawdown=-0.19)
    assert melt["meltdown"] is True
    print("OK overlay/emergency")


def test_degrade_yesterday_nav():
    holdings = _holdings()
    frame, meta = overlay_holdings_estimates(holdings, {"items": {}})
    assert meta["degraded"] is True
    assert (pd.to_numeric(frame["当日估算涨跌幅"], errors="coerce").fillna(0) == 0).all()
    assert abs(float(frame["估算净值"].iloc[0]) - 1.00) < 1e-8
    print("OK degrade")


def test_signal_window_pick():
    payload = {
        "updated_at": "2026-09-04 14:50:00",
        "items": {"000001": {"nav_estimate": 1.1, "change_pct": 1.0}},
        "snapshots": [
            {
                "at": "2026-09-04 14:12:00",
                "items": {"000001": {"nav_estimate": 1.05, "change_pct": 0.4}},
            },
            {
                "at": "2026-09-04 14:28:00",
                "items": {"000001": {"nav_estimate": 1.06, "change_pct": 0.5}},
            },
        ],
    }
    picked = pick_signal_estimates(payload, ts=pd.Timestamp("2026-09-04 14:50:00"))
    assert picked["picked"] == "window_14_30"
    assert picked["items"]["000001"]["nav_estimate"] == 1.06
    print("OK signal window")


def test_cooldown_skips_emergency():
    orders = pd.DataFrame(
        {
            "基金代码": ["000001", "000002"],
            "指令": ["紧急减仓", "减仓"],
            "指令来源": ["紧急调仓", "月度调仓"],
            "操作理由": ["急跌", "常规"],
            "持仓市值": [1000.0, 2000.0],
            "目标市值": [500.0, 1000.0],
        }
    )
    log_path = ROOT / "data" / "logs" / "_test_trade_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        '{"records":[{"fund_code":"000001","sell_date":"2026-08-01","shares":10,"reason":"x"},'
        '{"fund_code":"000002","sell_date":"2026-08-01","shares":10,"reason":"x"}]}',
        encoding="utf-8",
    )
    out = apply_cooldown(orders, trade_log_path=str(log_path))
    assert out.loc[out["基金代码"] == "000001", "指令"].iloc[0] == "紧急减仓"
    assert out.loc[out["基金代码"] == "000002", "指令"].iloc[0] == "持有"
    log_path.unlink(missing_ok=True)
    print("OK cooldown")


def main():
    test_trading_session()
    test_cache_ttl()
    test_overlay_and_emergency()
    test_degrade_yesterday_nav()
    test_signal_window_pick()
    test_cooldown_skips_emergency()
    print("\n全部通过")


if __name__ == "__main__":
    main()
