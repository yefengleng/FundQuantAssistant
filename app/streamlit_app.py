import json
import re
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DISPLAY_WINDOW_CHOICES,
    DISPLAY_WINDOW_DAYS,
    OPERATION_FREQUENCY,
    SECTOR_TOP_N,
    STRATEGY_MODE,
    get_strategy_profile,
    normalize_display_window,
    normalize_operation_frequency,
    normalize_strategy_mode,
)
from src.data_layer.loader import (
    PROJECT_ROOT,
    can_delete_account,
    create_account,
    delete_account,
    ensure_profiles_initialized,
    get_app_meta_path,
    get_current_account_name,
    get_signal_path,
    get_trade_log_path,
    list_accounts,
    load_local_data,
    set_current_account,
    update_all_funds,
)
from src.data_layer.market_clock import (
    REFRESH_SECONDS,
    is_trading_session,
    seconds_until_refresh,
)
from src.data_layer.realtime_cache import (
    DEADLINE_HINT,
    DEGRADE_CAPTION,
    EMERGENCY_BANNER,
    REALTIME_PATH,
    is_cache_stale,
    load_realtime_payload,
    log_monitor,
    save_realtime_payload,
)
from src.factor_layer.comparator import get_fund_comparison_data
from src.factor_layer.portfolio_utils import apply_manual_rebalance, load_current_holdings
from src.factor_layer.scorer import batch_score_funds


REALTIME_PATH = Path(PROJECT_ROOT) / "data" / "realtime_estimates.json"

KEEP_SET = {"保留", "持有", "暂缓"}
REDUCE_SET = {"减仓", "减仓（熔断）", "紧急减仓"}
REDEEM_SET = {"赎回", "清仓", "紧急清仓"}
WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
SETTINGS_PATH = ROOT / "config" / "settings.py"
MODE_RADIO_OPTIONS = ["防御型", "进攻型", "自动识别"]
MODE_BY_LABEL = {"防御型": "defensive", "进攻型": "aggressive", "自动识别": "auto"}
LABEL_BY_MODE = {v: k for k, v in MODE_BY_LABEL.items()}
FREQ_OPTIONS = ["月度调仓", "周调仓"]
FREQ_BY_LABEL = {"月度调仓": "monthly", "周调仓": "weekly", "周度监控": "weekly"}
LABEL_BY_FREQ = {"monthly": "月度调仓", "weekly": "周调仓"}
PNL_RED = "#ff3333"
PNL_GREEN = "#33cc33"
PNL_ZERO = "#111111"
PNL_DISPLAY_COLUMNS = (
    "持有收益（元）",
    "持有收益率（%）",
    "昨日收益（元）",
    "累计收益",
    "近60日收益",
    "收益率",
)

st.set_page_config(
    page_title="基金量化辅助系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _safe_render(label, func):
    try:
        func()
    except Exception as exc:
        st.error(f"「{label}」加载失败：{exc}")
        with st.expander("错误详情", expanded=False):
            st.exception(exc)


def _inject_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .hero-title { font-size: 1.8rem; font-weight: 750; margin-bottom: 0.2rem; }
        .hero-sub { color: #64748b; margin-bottom: 1rem; }
        .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 16px 18px 14px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        }
        .metric-label { color: #64748b; font-size: 0.88rem; margin-bottom: 6px; }
        .metric-value { font-size: 1.7rem; font-weight: 750; color: #0f172a; line-height: 1.2; }
        .metric-hint { color: #94a3b8; font-size: 0.8rem; margin-top: 6px; }
        .metric-card-wide {
            margin-bottom: 14px;
            border: 2px solid #fde68a;
            background: linear-gradient(180deg, #fffbeb 0%, #ffffff 100%);
        }
        .metric-tag {
            display: inline-block;
            margin-left: 8px;
            padding: 2px 8px;
            border-radius: 999px;
            background: #fef3c7;
            color: #b45309;
            font-size: 0.75rem;
            font-weight: 700;
            vertical-align: middle;
        }
        .metric-value-profit { color: #ff3333; }
        .metric-value-loss { color: #33cc33; }
        .metric-value-flat { color: #111111; }
        .alert-banner {
            background: #fef2f2;
            border: 2px solid #dc2626;
            color: #b91c1c;
            border-radius: 14px;
            padding: 14px 18px;
            font-size: 1.35rem;
            font-weight: 800;
            margin: 8px 0 16px;
        }
        .mode-banner {
            font-size: 1.7rem;
            font-weight: 800;
            border-radius: 14px;
            padding: 16px 18px;
            margin: 4px 0 16px;
            line-height: 1.35;
        }
        .mode-banner-aggressive {
            background: #fff7ed;
            border: 2px solid #f97316;
            color: #9a3412;
        }
        .mode-banner-defensive {
            background: #eff6ff;
            border: 2px solid #3b82f6;
            color: #1e3a8a;
        }
        .mode-banner-fail {
            background: #fefce8;
            border: 2px solid #eab308;
            color: #854d0e;
        }
        .trim-banner {
            font-size: 1.25rem;
            font-weight: 750;
            border-radius: 14px;
            padding: 12px 18px;
            margin: 0 0 16px;
            line-height: 1.35;
            background: #f5f3ff;
            border: 2px solid #8b5cf6;
            color: #5b21b6;
        }
        .freq-banner {
            font-size: 1.35rem;
            font-weight: 800;
            border-radius: 14px;
            padding: 14px 18px;
            margin: 4px 0 12px;
            line-height: 1.35;
        }
        .freq-banner-monthly {
            background: #ecfeff;
            border: 2px solid #06b6d4;
            color: #155e75;
        }
        .freq-banner-weekly {
            background: #fff1f2;
            border: 2px solid #f43f5e;
            color: #9f1239;
        }
        .list-banner {
            border-radius: 14px 14px 0 0;
            padding: 12px 16px;
            font-weight: 750;
            font-size: 1.05rem;
        }
        .list-keep { background: #dcfce7; border: 2px solid #22c55e; color: #166534; }
        .list-reduce { background: #ffedd5; border: 2px solid #f97316; color: #9a3412; }
        .list-redeem { background: #fee2e2; border: 2px solid #ef4444; color: #991b1b; }
        .list-body {
            border-radius: 0 0 14px 14px;
            padding: 8px 12px 12px;
            margin-bottom: 18px;
        }
        .list-body-keep { border: 2px solid #22c55e; border-top: 0; background: #f0fdf4; }
        .list-body-reduce { border: 2px solid #f97316; border-top: 0; background: #fff7ed; }
        .list-body-redeem { border: 2px solid #ef4444; border-top: 0; background: #fef2f2; }
        .refresh-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            padding: 10px 14px;
            margin: 0 0 12px;
            color: #334155;
            font-weight: 650;
        }
        .deadline-banner {
            background: #fff7ed;
            border: 2px solid #f97316;
            color: #9a3412;
            border-radius: 14px;
            padding: 12px 18px;
            font-size: 1.15rem;
            font-weight: 800;
            margin: 4px 0 14px;
        }
        .degrade-caption {
            background: #fefce8;
            border: 1px solid #eab308;
            color: #854d0e;
            border-radius: 10px;
            padding: 8px 12px;
            margin: 0 0 12px;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_signed_money(value):
    if value is None or pd.isna(value):
        return "-"
    number = float(value)
    if number > 0:
        return f"+{number:,.2f}"
    return f"{number:,.2f}"


def fmt_money(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.2f}"


def fmt_pct(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def fmt_score(value):
    if value is None or pd.isna(value):
        return "-"
    text = str(value).strip()
    if text in {"数据不足", "持有", "数据拉取失败"}:
        return text
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return text


def fmt_shares(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.2f}"


def fmt_return_pct(value):
    """持有收益率已是百分数，不再乘 100。"""
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.2f}%"


def holdings_missing_cost(df):
    if df is None or df.empty:
        return False
    if "持仓成本（元）" not in df.columns:
        return True
    cost = pd.to_numeric(df["持仓成本（元）"], errors="coerce").fillna(0)
    shares = pd.to_numeric(df.get("持有份额"), errors="coerce").fillna(0)
    market_value = pd.to_numeric(df.get("持仓市值"), errors="coerce").fillna(0)
    active = (shares > 0) & (market_value > 0)
    if not active.any():
        return False
    return bool((cost[active] <= 0).any())


def calc_cumulative_return(df):
    empty = {"pnl": None, "rate": None, "cost": 0.0, "missing_cost": False}
    if df is None or df.empty:
        return empty
    cost = pd.to_numeric(df.get("持仓成本（元）"), errors="coerce").fillna(0)
    pnl = pd.to_numeric(df.get("持有收益（元）"), errors="coerce")
    total_cost = float(cost.sum())
    if pnl.notna().any():
        total_pnl = float(pnl.sum(min_count=1))
    else:
        total_pnl = None
    rate = None
    if total_cost > 0 and total_pnl is not None:
        rate = total_pnl / total_cost
    return {
        "pnl": total_pnl,
        "rate": rate,
        "cost": total_cost,
        "missing_cost": holdings_missing_cost(df),
    }


def estimate_redeem_rate(buy_date, as_of=None):
    """按持有期粗估赎回费率：<7日 1.5%，<30日 0.5%，<1年 0.5%，满1年 0%。"""
    as_of = pd.Timestamp(as_of or datetime.now()).normalize()
    buy = pd.to_datetime(buy_date, errors="coerce")
    if pd.isna(buy):
        return 0.015
    hold_days = (as_of - buy.normalize()).days
    if hold_days < 7:
        return 0.015
    if hold_days < 365:
        return 0.005
    return 0.0


def persist_strategy_mode(mode):
    """把侧边栏选择写回 config/settings.py 的 STRATEGY_MODE，并同步内存中的配置。"""
    import config.settings as settings_mod

    mode = normalize_strategy_mode(mode)
    try:
        text = SETTINGS_PATH.read_text(encoding="utf-8")
        pattern = r"^STRATEGY_MODE\s*=\s*['\"][^'\"]*['\"]"
        replacement = f'STRATEGY_MODE = "{mode}"'
        if re.search(pattern, text, flags=re.M):
            new_text = re.sub(pattern, replacement, text, count=1, flags=re.M)
        else:
            new_text = replacement + "\n" + text
        if new_text != text:
            SETTINGS_PATH.write_text(new_text, encoding="utf-8")
        settings_mod.STRATEGY_MODE = mode
    except Exception:
        settings_mod.STRATEGY_MODE = mode
    return mode


def persist_operation_frequency(frequency):
    """把运行模式写回 config/settings.py 的 OPERATION_FREQUENCY。"""
    import config.settings as settings_mod

    frequency = normalize_operation_frequency(frequency)
    try:
        text = SETTINGS_PATH.read_text(encoding="utf-8")
        pattern = r"^OPERATION_FREQUENCY\s*=\s*['\"][^'\"]*['\"]"
        replacement = f'OPERATION_FREQUENCY = "{frequency}"'
        if re.search(pattern, text, flags=re.M):
            new_text = re.sub(pattern, replacement, text, count=1, flags=re.M)
        else:
            new_text = replacement + "\n" + text
        if new_text != text:
            SETTINGS_PATH.write_text(new_text, encoding="utf-8")
        settings_mod.OPERATION_FREQUENCY = frequency
    except Exception:
        settings_mod.OPERATION_FREQUENCY = frequency
    return frequency


def render_mode_banner(meta):
    banner = str((meta or {}).get("strategy_mode_banner") or "").strip()
    if not banner:
        return
    css = "mode-banner-defensive"
    if (meta or {}).get("strategy_mode_fetch_failed") or banner.startswith("⚠️"):
        css = "mode-banner-fail"
    elif (meta or {}).get("strategy_mode_effective") == "aggressive" or "进攻" in banner:
        css = "mode-banner-aggressive"
    st.markdown(f'<div class="mode-banner {css}">{banner}</div>', unsafe_allow_html=True)


def render_freq_banner(meta, fallback_freq="monthly"):
    banner = str((meta or {}).get("operation_frequency_banner") or "").strip()
    if not banner:
        frequency = normalize_operation_frequency(
            (meta or {}).get("operation_frequency") or fallback_freq
        )
        banner = "📅 当前运行频率：周调仓" if frequency == "weekly" else "📅 当前运行频率：月度调仓"
    css = "freq-banner-weekly" if ("周调仓" in banner or "周度" in banner) else "freq-banner-monthly"
    st.markdown(f'<div class="freq-banner {css}">{banner}</div>', unsafe_allow_html=True)


def render_trim_banner(meta):
    banner = str((meta or {}).get("sector_top_n_banner") or "").strip()
    if not banner:
        return
    st.markdown(f'<div class="trim-banner">{banner}</div>', unsafe_allow_html=True)
    note = str((meta or {}).get("sector_elite_note") or "").strip()
    cash = (meta or {}).get("sector_elite_cash")
    if note:
        if cash not in (None, "", 0, 0.0):
            try:
                st.caption(f"{note}（约 {float(cash):,.2f} 元）")
            except (TypeError, ValueError):
                st.caption(note)
        else:
            st.caption(note)


def signal_file():
    return Path(get_signal_path())


def meta_file():
    return Path(get_app_meta_path())


def trade_log_file():
    return Path(get_trade_log_path())


def load_meta():
    try:
        with open(meta_file(), "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def load_realtime_estimates():
    return load_realtime_payload()


def save_realtime_estimates(items, updated_at=None):
    return save_realtime_payload(items, updated_at=updated_at)


def _sanitize_estimate_items(estimates):
    safe_items = {}
    for code, quote in (estimates or {}).items():
        nav_est = None if not isinstance(quote, dict) else quote.get("nav_estimate")
        chg = None if not isinstance(quote, dict) else quote.get("change_pct")
        try:
            nav_ok = nav_est is not None and pd.notna(nav_est)
        except Exception:
            nav_ok = False
        try:
            chg_ok = chg is not None and pd.notna(chg)
        except Exception:
            chg_ok = False
        row = {
            "nav_estimate": float(nav_est) if nav_ok else None,
            "change_pct": float(chg) if chg_ok else None,
        }
        if isinstance(quote, dict):
            if quote.get("gztime"):
                row["gztime"] = str(quote.get("gztime"))
            if quote.get("source"):
                row["source"] = str(quote.get("source"))
        safe_items[str(code).strip()] = row
    return safe_items


def refresh_realtime_quotes(holdings, reason="manual"):
    """Only fetch realtime estimates (no historical NAV)."""
    from src.data_layer.fetcher import fetch_realtime_estimates
    from src.data_layer.realtime_cache import quote_is_valid

    codes = []
    if holdings is not None and not holdings.empty and "基金代码" in holdings.columns:
        codes = holdings["基金代码"].astype(str).str.strip().tolist()
    estimates = fetch_realtime_estimates(codes)
    safe_items = _sanitize_estimate_items(estimates)
    ok_count = sum(1 for quote in safe_items.values() if quote_is_valid(quote))
    fail_count = max(len(safe_items) - ok_count, 0)
    degraded = bool(safe_items) and ok_count == 0
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = save_realtime_estimates(safe_items, updated_at=now_text)
    payload["degraded"] = degraded
    save_meta({"last_realtime": now_text, "last_realtime_reason": reason, "realtime_degraded": degraded})
    log_monitor("日监控", f"{reason} 成功 {ok_count} 失败 {fail_count} 降级={degraded}")
    return payload


def render_intraday_refresh_bar(in_session, payload, force_clicked):
    remain = seconds_until_refresh((payload or {}).get("updated_at")) if in_session else None
    if in_session:
        label = f"🔄 下次刷新：{int(remain or 0)}秒后（交易时段每5分钟自动更新）"
    else:
        label = "🔄 非交易时段，自动刷新已暂停（交易日 09:30-15:00 每5分钟更新）"
    st.markdown(f'<div class="refresh-bar">{label}</div>', unsafe_allow_html=True)
    return force_clicked


def _style_monitor_change(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number < 0:
        return "color: #dc2626; font-weight: 700"
    if number > 0:
        return "color: #16a34a; font-weight: 700"
    return "color: #64748b"


def render_daily_monitor_tab(monitor_df, sector_df, monitor_meta):
    meta = monitor_meta or {}
    emergency = meta.get("emergency") or {}
    if monitor_df is None or monitor_df.empty:
        st.info("暂无持仓，无法展示日监控。")
        return

    if meta.get("degraded"):
        st.markdown(f'<div class="degrade-caption">{DEGRADE_CAPTION}</div>', unsafe_allow_html=True)

    drawdown = float(meta.get("account_drawdown") or 0.0)
    if emergency.get("meltdown") or drawdown < -0.18:
        st.markdown(
            f'<div class="alert-banner">⚠️ 熔断警告：账户估算回撤 {fmt_pct(drawdown)}，已突破 -18%。</div>',
            unsafe_allow_html=True,
        )
    if emergency.get("triggered"):
        st.markdown(f'<div class="alert-banner">{EMERGENCY_BANNER}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("账户估算回撤", fmt_pct(drawdown))
    c2.metric("当日估算盈亏", fmt_signed_money(meta.get("total_pnl")))
    avg = meta.get("avg_change")
    c3.metric("持仓加权估算涨跌", "-" if avg is None or pd.isna(avg) else f"{float(avg):+.2f}%")
    c4.metric("估值成功", f"{int(meta.get('ok_count') or 0)} / {int(meta.get('ok_count') or 0) + int(meta.get('fail_count') or 0)}")
    updated = meta.get("updated_at") or ""
    if updated:
        st.caption(f"估值时间 {updated}" + (" · 已取 14:00-14:30 快照" if meta.get("picked") == "window_14_30" else ""))

    view = pd.DataFrame(
        {
            "基金代码": monitor_df.get("基金代码"),
            "基金名称": monitor_df.get("基金名称"),
            "赛道": monitor_df.get("赛道归类"),
            "估算净值": pd.to_numeric(monitor_df.get("估算净值"), errors="coerce").map(
                lambda x: "-" if pd.isna(x) else f"{float(x):.4f}"
            ),
            "当日估算涨跌幅": pd.to_numeric(monitor_df.get("当日估算涨跌幅"), errors="coerce"),
            "当日估算盈亏": pd.to_numeric(monitor_df.get("当日估算盈亏"), errors="coerce"),
            "估算市值": pd.to_numeric(monitor_df.get("估算市值"), errors="coerce").map(fmt_money),
            "预警": monitor_df.get("预警"),
        }
    )
    numeric_chg = pd.to_numeric(view["当日估算涨跌幅"], errors="coerce")
    min_idx = int(numeric_chg.idxmin()) if numeric_chg.notna().any() else None
    max_idx = int(numeric_chg.idxmax()) if numeric_chg.notna().any() else None

    def _highlight(row):
        styles = [""] * len(row)
        warn = str(row.get("预警") or "")
        chg = pd.to_numeric(row.get("当日估算涨跌幅"), errors="coerce")
        if "急跌" in warn or (pd.notna(chg) and float(chg) <= -5):
            styles = ["background-color: #fee2e2"] * len(row)
        if row.name == min_idx:
            styles = ["background-color: #fecaca"] * len(row)
        if row.name == max_idx:
            styles = ["background-color: #dcfce7"] * len(row)
        return styles

    show = view.copy()
    show["当日估算涨跌幅"] = pd.to_numeric(show["当日估算涨跌幅"], errors="coerce").map(
        lambda x: "-" if pd.isna(x) else f"{float(x):+.2f}%"
    )
    show["当日估算盈亏"] = pd.to_numeric(view["当日估算盈亏"], errors="coerce").map(fmt_signed_money)
    styler = show.style.apply(_highlight, axis=1)

    def _color_from_view(col_name):
        raw = pd.to_numeric(view[col_name], errors="coerce")

        def _apply(_series):
            colors = []
            for idx in show.index:
                colors.append(_style_monitor_change(raw.at[idx] if idx in raw.index else None))
            return colors

        return _apply

    styler = styler.apply(_color_from_view("当日估算涨跌幅"), subset=["当日估算涨跌幅"])
    styler = styler.apply(_color_from_view("当日估算盈亏"), subset=["当日估算盈亏"])
    st.dataframe(styler, width="stretch", hide_index=True)

    st.markdown("#### 赛道汇总")
    if sector_df is None or sector_df.empty:
        st.caption("暂无赛道汇总。")
        return
    sector_view = pd.DataFrame(
        {
            "赛道": sector_df.get("赛道"),
            "基金只数": sector_df.get("基金只数"),
            "估算市值": pd.to_numeric(sector_df.get("估算市值"), errors="coerce").map(fmt_money),
            "当日估算涨跌幅": pd.to_numeric(sector_df.get("当日估算涨跌幅"), errors="coerce"),
            "当日估算盈亏": pd.to_numeric(sector_df.get("当日估算盈亏"), errors="coerce").map(fmt_signed_money),
        }
    )
    sector_raw_chg = pd.to_numeric(sector_view["当日估算涨跌幅"], errors="coerce")
    sector_show = sector_view.copy()
    sector_show["当日估算涨跌幅"] = sector_raw_chg.map(lambda x: "-" if pd.isna(x) else f"{float(x):+.2f}%")

    def _color_sector(series):
        return [_style_monitor_change(sector_raw_chg.at[idx] if idx in sector_raw_chg.index else None) for idx in series.index]

    st.dataframe(
        sector_show.style.apply(_color_sector, subset=["当日估算涨跌幅"]),
        width="stretch",
        hide_index=True,
    )


def calc_realtime_pnl(holdings, estimate_payload):
    """持仓份额 ×（实时估值 - 最新净值），估值缺失时用估算涨跌幅。"""
    empty = {
        "pnl": None,
        "ok_count": 0,
        "fail_count": 0,
        "updated_at": "",
        "avg_change": None,
    }
    if holdings is None or holdings.empty:
        return empty
    items = (estimate_payload or {}).get("items") or {}
    pnl_total = 0.0
    weight_change = 0.0
    weight_sum = 0.0
    ok_count = 0
    fail_count = 0
    shares = pd.to_numeric(holdings.get("持有份额"), errors="coerce")
    last_nav = pd.to_numeric(holdings.get("最新净值"), errors="coerce")
    for idx, row in holdings.iterrows():
        code = str(row.get("基金代码", "")).strip()
        share = float(shares.at[idx]) if pd.notna(shares.at[idx]) else 0.0
        nav = float(last_nav.at[idx]) if pd.notna(last_nav.at[idx]) else float("nan")
        if share <= 0:
            continue
        quote = items.get(code) or {}
        try:
            estimate = pd.to_numeric(quote.get("nav_estimate"), errors="coerce")
        except Exception:
            estimate = float("nan")
        try:
            change_pct = pd.to_numeric(quote.get("change_pct"), errors="coerce")
        except Exception:
            change_pct = float("nan")
        fund_pnl = None
        if pd.notna(estimate) and pd.notna(nav):
            fund_pnl = share * (float(estimate) - nav)
        elif pd.notna(change_pct) and pd.notna(nav):
            fund_pnl = share * nav * float(change_pct) / 100.0
        if fund_pnl is None:
            fail_count += 1
            continue
        pnl_total += fund_pnl
        ok_count += 1
        market_value = share * nav if pd.notna(nav) else 0.0
        if pd.notna(change_pct) and market_value > 0:
            weight_change += float(change_pct) * market_value
            weight_sum += market_value
    if ok_count == 0:
        empty["fail_count"] = fail_count
        empty["updated_at"] = str((estimate_payload or {}).get("updated_at") or "")
        return empty
    return {
        "pnl": pnl_total,
        "ok_count": ok_count,
        "fail_count": fail_count,
        "updated_at": str((estimate_payload or {}).get("updated_at") or ""),
        "avg_change": (weight_change / weight_sum) if weight_sum > 0 else None,
    }


def save_meta(updates):
    meta = load_meta()
    meta.update(updates)
    path = meta_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


def file_time_text(path, fallback="暂无"):
    try:
        path = Path(path)
        if path.exists():
            return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return fallback


@st.cache_data(show_spinner=False)
def cached_load_holdings(account):
    df = load_current_holdings()
    if df.empty:
        return df
    df["基金代码"] = df["基金代码"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def cached_score_funds(fund_codes, window=None):
    return batch_score_funds(list(fund_codes), window=window)


@st.cache_data(show_spinner=False)
def cached_load_nav(fund_code):
    return load_local_data(fund_code)


@st.cache_data(show_spinner="正在准备基金对比数据...")
def cached_fund_comparison(fund_codes, window=None):
    return get_fund_comparison_data(list(fund_codes), window=window)


@st.cache_data(show_spinner="正在计算赛道趋势...")
def cached_sector_trend_rows(window, top_n=2, _color_version=2):
    from src.factor_layer.sector_analysis import build_sector_trend_rows

    return build_sector_trend_rows(top_n=top_n, window=window)


def _load_signal_generator():
    """Streamlit 热更新会缓存旧模块，先刷新 scorer / filters，再刷新 signal_generator。"""
    import importlib
    import sys

    def _reload(mod_name):
        module = sys.modules.get(mod_name) or importlib.import_module(mod_name)
        return importlib.reload(module)

    try:
        scorer = importlib.import_module("src.factor_layer.scorer")
        if not hasattr(scorer, "get_unheld_funds_score"):
            scorer = _reload("src.factor_layer.scorer")

        filters = importlib.import_module("src.strategy_layer.filters")
        if not hasattr(filters, "get_market_regime"):
            filters = _reload("src.strategy_layer.filters")

        constraints = importlib.import_module("src.strategy_layer.constraints")
        if not hasattr(constraints, "apply_sector_elite"):
            constraints = _reload("src.strategy_layer.constraints")

        sig = importlib.import_module("src.strategy_layer.signal_generator")
        needs_reload = not hasattr(sig, "generate_buy_candidates")
        try:
            varnames = sig.generate_trading_signal.__code__.co_varnames
            if "strategy_mode" not in varnames or "sector_top_n" not in varnames:
                needs_reload = True
        except Exception:
            needs_reload = True
        if needs_reload:
            sig = _reload("src.strategy_layer.signal_generator")
        return sig
    except Exception:
        _reload("src.factor_layer.scorer")
        _reload("src.strategy_layer.filters")
        _reload("src.strategy_layer.constraints")
        return _reload("src.strategy_layer.signal_generator")


def _load_weekly_scanner():
    import importlib

    import src.strategy_layer.weekly_scanner as weekly

    if not hasattr(weekly, "run_weekly_scan"):
        weekly = importlib.reload(weekly)
    return weekly


@st.cache_data(show_spinner=False)
def cached_account_drawdown(holdings):
    if holdings is None or holdings.empty:
        return 0.0
    return float(_load_signal_generator()._calc_account_drawdown(holdings))


@st.cache_data(show_spinner=False)
def cached_read_signal_csv(account):
    path = signal_file()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"基金代码": str})
    df["基金代码"] = df["基金代码"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner="正在生成调仓报告，请稍候...")
def cached_generate_trading_signal(year, month, nonce, strategy_mode, sector_top_n, account):
    generate_trading_signal = _load_signal_generator().generate_trading_signal

    orders = generate_trading_signal(
        year, month, strategy_mode=strategy_mode, sector_top_n=sector_top_n
    )
    meta = {
        "meltdown_triggered": bool(orders.attrs.get("meltdown_triggered", False)),
        "crash_filter_triggered": bool(orders.attrs.get("crash_filter_triggered", False)),
        "account_return": float(orders.attrs.get("account_return", 0.0) or 0.0),
        "nonce": nonce,
        "strategy_mode_banner": orders.attrs.get("strategy_mode_banner", ""),
        "strategy_mode_requested": orders.attrs.get("strategy_mode_requested"),
        "strategy_mode_effective": orders.attrs.get("strategy_mode_effective"),
        "strategy_mode_fetch_failed": bool(orders.attrs.get("strategy_mode_fetch_failed")),
        "sector_top_n_banner": orders.attrs.get("sector_top_n_banner", ""),
        "sector_top_n_sectors": int(orders.attrs.get("sector_top_n_sectors", 0) or 0),
        "sector_top_n_funds": int(orders.attrs.get("sector_top_n_funds", 0) or 0),
        "sector_elite_note": orders.attrs.get("sector_elite_note", ""),
        "sector_elite_cash": float(orders.attrs.get("sector_elite_cash", 0.0) or 0.0),
        "operation_frequency": orders.attrs.get("operation_frequency", "monthly"),
        "operation_frequency_banner": orders.attrs.get("operation_frequency_banner", "📅 当前运行频率：月度调仓"),
    }
    if not orders.empty:
        orders = orders.copy()
        orders["基金代码"] = orders["基金代码"].astype(str).str.strip()
    return orders, meta


@st.cache_data(show_spinner="正在运行周调仓扫描...")
def cached_run_weekly_scan(nonce, account):
    orders = _load_weekly_scanner().run_weekly_scan()
    meta = {
        "meltdown_triggered": bool(orders.attrs.get("meltdown_triggered", False)),
        "crash_filter_triggered": False,
        "account_return": float(orders.attrs.get("account_return", 0.0) or 0.0),
        "nonce": nonce,
        "strategy_mode_banner": orders.attrs.get("strategy_mode_banner", ""),
        "operation_frequency": "weekly",
        "operation_frequency_banner": orders.attrs.get("operation_frequency_banner", "📅 当前运行频率：周调仓"),
        "weekly_alert_count": int(orders.attrs.get("weekly_alert_count", 0) or 0),
        "weekly_changed_funds": int(orders.attrs.get("weekly_changed_funds", 0) or 0),
        "weekly_is_friday": bool(orders.attrs.get("weekly_is_friday", False)),
        "weekly_alerts": list(orders.attrs.get("weekly_alerts", []) or []),
        "emergency_triggered": bool(orders.attrs.get("emergency_triggered", False)),
        "friday_no_emergency": bool(orders.attrs.get("friday_no_emergency", False)),
        "operation_deadline": orders.attrs.get("operation_deadline", ""),
        "estimate_degraded": bool(orders.attrs.get("estimate_degraded", False)),
        "cooldown_bypassed": bool(orders.attrs.get("cooldown_bypassed", False)),
        "sector_top_n_banner": "",
        "sector_elite_note": "",
        "sector_elite_cash": 0.0,
    }
    if not orders.empty:
        orders = orders.copy()
        orders["基金代码"] = orders["基金代码"].astype(str).str.strip()
    return orders, meta


@st.cache_data(show_spinner="正在筛选买入候选...")
def cached_buy_candidates(year, month, top_n, strategy_mode, account):
    generate_buy_candidates = _load_signal_generator().generate_buy_candidates

    df = generate_buy_candidates(year, month, top_n=int(top_n), strategy_mode=strategy_mode)
    failures = list(df.attrs.get("watchlist_failures", [])) if df is not None else []
    if df is not None and not df.empty:
        df = df.copy()
        df["基金代码"] = df["基金代码"].astype(str).str.strip()
    return df, failures


def active_holdings_only(df):
    if df is None or df.empty:
        return df
    shares = pd.to_numeric(df.get("持有份额"), errors="coerce")
    market_value = pd.to_numeric(df.get("持仓市值"), errors="coerce")
    return df.loc[(shares.fillna(0) > 0) & (market_value.fillna(0) > 0)].copy()


def merge_holdings_scores(holdings):
    if holdings.empty:
        return holdings
    scores = cached_score_funds(tuple(holdings["基金代码"].tolist()))
    merged = holdings.merge(scores, on="基金代码", how="left")
    if "综合得分" in merged.columns:
        merged["综合得分"] = merged["综合得分"].fillna("数据不足")
    return merged


def append_trade_log(orders):
    if orders is None or orders.empty:
        return
    action_col = "建议操作" if "建议操作" in orders.columns else "指令"
    sell_df = orders[orders[action_col].isin(REDUCE_SET | REDEEM_SET)].copy()
    if "卖出份额" in sell_df.columns:
        sell_df["卖出份额"] = pd.to_numeric(sell_df["卖出份额"], errors="coerce")
        sell_df = sell_df[sell_df["卖出份额"] > 0]
    if sell_df.empty:
        return

    path = trade_log_file()
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        else:
            payload = {"records": []}
    except Exception:
        payload = {"records": []}

    records = payload.get("records", []) if isinstance(payload, dict) else []
    today = datetime.now().strftime("%Y-%m-%d")
    sell_codes = set(sell_df["基金代码"].astype(str).str.strip())
    kept = [
        item
        for item in records
        if not (
            str(item.get("fund_code", "")).strip() in sell_codes
            and str(item.get("sell_date", ""))[:10] == today
        )
    ]
    for _, row in sell_df.iterrows():
        kept.append(
            {
                "fund_code": str(row["基金代码"]).strip(),
                "sell_date": today,
                "shares": float(row.get("卖出份额") or 0.0),
                "reason": str(row.get("指令来源") or row.get("操作理由") or row.get(action_col) or ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"records": kept}, file, ensure_ascii=False, indent=2)


def add_fee_columns(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    nav = pd.to_numeric(out.get("最新净值"), errors="coerce")
    shares = pd.to_numeric(out.get("卖出份额"), errors="coerce")
    out["预估卖出金额"] = shares * nav
    if "买入日期" in out.columns:
        out["预估赎回费率"] = out["买入日期"].map(estimate_redeem_rate)
    else:
        out["预估赎回费率"] = 0.015
    out["预估赎回费"] = out["预估卖出金额"] * out["预估赎回费率"]
    return out


def _parse_signed_display(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(",", "").replace("%", "").replace("＋", "+")
    if text in {"", "-", "—", "–", "None", "nan"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def pnl_color_css(value, *, cost=False):
    """正收益红、负收益绿；cost=True 时作为费用项固定红色。"""
    if cost:
        number = _parse_signed_display(value)
        if number is None:
            return ""
        return f"color: {PNL_RED}; font-weight: 650;"
    number = _parse_signed_display(value)
    if number is None:
        return ""
    if number > 0:
        return f"color: {PNL_RED}; font-weight: 650;"
    if number < 0:
        return f"color: {PNL_GREEN}; font-weight: 650;"
    return f"color: {PNL_ZERO};"


def _styler_applymap(styled, func, subset):
    if hasattr(styled, "applymap"):
        return styled.applymap(func, subset=subset)
    return styled.map(func, subset=subset)


def pnl_bar_colors(values):
    colors = []
    for value in values:
        number = _parse_signed_display(value)
        if number is None:
            colors.append("#94a3b8")
        elif number > 0:
            colors.append(PNL_RED)
        elif number < 0:
            colors.append(PNL_GREEN)
        else:
            colors.append("#64748b")
    return colors


def style_insufficient(df, subset=None, pnl_df=None):
    if df is None or df.empty:
        return df

    def highlight(row):
        score = str(row.get("综合得分", ""))
        action = str(row.get("建议操作", row.get("指令", "")))
        reason = str(row.get("操作理由", ""))
        gray = score == "数据不足" or action == "数据不足" or "数据不足" in reason
        color = "color: #94a3b8;" if gray else ""
        return [color] * len(row)

    try:
        styled = df.style.apply(highlight, axis=1)
    except Exception:
        return df

    pnl_cols = [col for col in PNL_DISPLAY_COLUMNS if col in df.columns]
    try:
        if pnl_cols:
            styled = _styler_applymap(styled, pnl_color_css, pnl_cols)
        if "预估赎回费" in df.columns:
            styled = _styler_applymap(styled, lambda value: pnl_color_css(value, cost=True), ["预估赎回费"])
    except Exception:
        return styled
    return styled


def csv_bytes(df):
    buffer = StringIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue().encode("utf-8-sig")


def _metric_card_html(label, value, hint, value_class=""):
    klass = f"metric-value {value_class}".strip()
    return f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="{klass}">{value}</div>
                    <div class="metric-hint">{hint}</div>
                </div>
                """


def render_metric_cards(
    total_asset,
    equity_ratio,
    drawdown,
    top_name,
    top_score,
    equity_limit,
    retreat_limit,
    total_pnl=None,
    total_return=None,
    missing_cost=False,
):
    drawdown_class = "metric-value-flat"
    if drawdown is not None and not (isinstance(drawdown, float) and pd.isna(drawdown)):
        if float(drawdown) > 0:
            drawdown_class = "metric-value-profit"
        elif float(drawdown) < 0:
            drawdown_class = "metric-value-loss"
    cards = [
        ("总资产（元）", fmt_money(total_asset), "按最新净值估算", ""),
        ("权益占比（%）", fmt_pct(equity_ratio), f"上限 {fmt_pct(equity_limit)}", ""),
        ("账户总回撤（%）", fmt_pct(drawdown), f"熔断线 {fmt_pct(retreat_limit)}", drawdown_class),
        ("本月得分最高基金", top_name or "-", f"综合得分 {fmt_score(top_score)}", ""),
    ]
    cols = st.columns(4)
    for col, (label, value, hint, value_class) in zip(cols, cards):
        with col:
            st.markdown(_metric_card_html(label, value, hint, value_class), unsafe_allow_html=True)

    if total_pnl is None or (isinstance(total_pnl, float) and pd.isna(total_pnl)):
        pnl_text = "-"
        pnl_class = "metric-value-flat"
    else:
        pnl_text = fmt_signed_money(total_pnl)
        if float(total_pnl) > 0:
            pnl_class = "metric-value-profit"
        elif float(total_pnl) < 0:
            pnl_class = "metric-value-loss"
        else:
            pnl_class = "metric-value-flat"

    if total_return is None or (isinstance(total_return, float) and pd.isna(total_return)):
        rate_text = "-"
        rate_class = "metric-value-flat"
    else:
        rate_text = fmt_pct(total_return)
        if float(total_return) > 0:
            rate_class = "metric-value-profit"
        elif float(total_return) < 0:
            rate_class = "metric-value-loss"
        else:
            rate_class = "metric-value-flat"

    cost_hint = "请先补充持仓成本，否则无法计算收益率" if missing_cost else "持有收益合计 / 持仓成本合计"
    extra = [
        ("累计总收益（元）", pnl_text, "全部基金持有收益之和", pnl_class),
        ("累计收益率（%）", rate_text, cost_hint, rate_class),
    ]
    extra_cols = st.columns(2)
    for col, (label, value, hint, value_class) in zip(extra_cols, extra):
        with col:
            st.markdown(_metric_card_html(label, value, hint, value_class), unsafe_allow_html=True)


def render_realtime_pnl_card(pnl_info):
    pnl = (pnl_info or {}).get("pnl")
    if pnl is None or (isinstance(pnl, float) and pd.isna(pnl)):
        value_text = "-"
        value_class = "metric-value-flat"
    else:
        value_text = fmt_signed_money(pnl)
        if float(pnl) > 0:
            value_class = "metric-value-profit"
        elif float(pnl) < 0:
            value_class = "metric-value-loss"
        else:
            value_class = "metric-value-flat"

    avg_change = (pnl_info or {}).get("avg_change")
    updated_at = (pnl_info or {}).get("updated_at") or ""
    ok_count = int((pnl_info or {}).get("ok_count") or 0)
    fail_count = int((pnl_info or {}).get("fail_count") or 0)
    hint_parts = ["（实时估算）非正式净值，仅供盘中参考"]
    if avg_change is not None and pd.notna(avg_change):
        hint_parts.append(f"持仓加权估算涨跌 {float(avg_change):+.2f}%")
    if updated_at:
        hint_parts.append(f"估值时间 {updated_at}")
    if ok_count or fail_count:
        hint_parts.append(f"成功 {ok_count} 只" + (f"，失败 {fail_count} 只" if fail_count else ""))
    hint = " · ".join(hint_parts)

    st.markdown(
        f"""
        <div class="metric-card metric-card-wide">
            <div class="metric-label">今日预计盈亏（实时）<span class="metric-tag">（实时估算）</span></div>
            <div class="metric-value {value_class}">{value_text}</div>
            <div class="metric-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _yesterday_profit_map():
    try:
        from src.ocr.importer import load_profile_meta

        payload = load_profile_meta() or {}
        raw = payload.get("yesterday_profits") or {}
        return {str(code).strip(): value for code, value in raw.items()}
    except Exception:
        return {}


def holdings_display(df):
    if df.empty:
        return df
    yesterday_map = _yesterday_profit_map()
    codes = df.get("基金代码")
    yesterday = None
    if codes is not None:
        yesterday = codes.astype(str).str.strip().map(yesterday_map)
    if "昨日收益（元）" in df.columns:
        yesterday = pd.to_numeric(df["昨日收益（元）"], errors="coerce")
    view = pd.DataFrame(
        {
            "基金代码": df.get("基金代码"),
            "基金名称": df.get("基金名称"),
            "赛道": df.get("赛道归类"),
            "持有份额": pd.to_numeric(df.get("持有份额"), errors="coerce").map(fmt_shares),
            "最新净值": pd.to_numeric(df.get("最新净值"), errors="coerce").map(
                lambda x: "-" if pd.isna(x) else f"{float(x):.4f}"
            ),
            "持仓成本（元）": pd.to_numeric(df.get("持仓成本（元）"), errors="coerce").map(fmt_money),
            "市值": pd.to_numeric(df.get("持仓市值"), errors="coerce").map(fmt_money),
            "持有收益（元）": pd.to_numeric(df.get("持有收益（元）"), errors="coerce").map(fmt_signed_money),
            "持有收益率（%）": pd.to_numeric(df.get("持有收益率（%）"), errors="coerce").map(fmt_return_pct),
            "昨日收益（元）": pd.to_numeric(yesterday, errors="coerce").map(fmt_signed_money)
            if yesterday is not None
            else "-",
            "备注": df.get("备注") if "备注" in df.columns else "",
            "累计收益": pd.to_numeric(df.get("累计收益（元）"), errors="coerce").map(fmt_signed_money)
            if "累计收益（元）" in df.columns
            else "-",
            "占比": pd.to_numeric(df.get("仓位占比"), errors="coerce").map(fmt_pct),
            "近60日收益": pd.to_numeric(df.get("近60日收益率"), errors="coerce").map(fmt_pct),
            "回撤": pd.to_numeric(df.get("近60日最大回撤"), errors="coerce").map(fmt_pct),
            "综合得分": df.get("综合得分").map(fmt_score) if "综合得分" in df.columns else "-",
        }
    )
    if "操作理由" in df.columns:
        view["操作理由"] = df["操作理由"]
    if "建议操作" in df.columns:
        view["建议操作"] = df["建议操作"]
    return view


def order_display(df):
    if df is None or df.empty:
        return pd.DataFrame()
    fee_df = add_fee_columns(df)
    view = pd.DataFrame(
        {
            "基金代码": fee_df.get("基金代码"),
            "基金名称": fee_df.get("基金名称"),
            "赛道": fee_df.get("赛道归类"),
            "当前市值": pd.to_numeric(fee_df.get("持仓市值"), errors="coerce").map(fmt_money),
            "当前仓位": pd.to_numeric(fee_df.get("当前仓位"), errors="coerce").map(fmt_pct)
            if "当前仓位" in fee_df.columns
            else None,
            "建议操作": fee_df.get("建议操作", fee_df.get("指令")),
            "卖出份额": pd.to_numeric(fee_df.get("卖出份额"), errors="coerce").map(fmt_shares),
            "预估卖出金额": pd.to_numeric(fee_df.get("预估卖出金额"), errors="coerce").map(fmt_money),
            "预估赎回费": pd.to_numeric(fee_df.get("预估赎回费"), errors="coerce").map(fmt_money),
            "赎回费率": pd.to_numeric(fee_df.get("预估赎回费率"), errors="coerce").map(fmt_pct),
            "操作截止时间": fee_df.get("操作截止时间") if "操作截止时间" in fee_df.columns else None,
            "操作理由": fee_df.get("操作理由"),
            "综合得分": fee_df.get("综合得分").map(fmt_score) if "综合得分" in fee_df.columns else None,
        }
    )
    return view.dropna(axis=1, how="all")


def render_order_list(title, css_name, df, download_name, download_key):
    st.markdown(f'<div class="list-banner list-{css_name}">{title}</div>', unsafe_allow_html=True)
    body_open = f'<div class="list-body list-body-{css_name}">'
    st.markdown(body_open, unsafe_allow_html=True)
    if df is None or df.empty:
        st.caption("本次无需该项操作。")
    else:
        view = order_display(df)
        st.dataframe(style_insufficient(view), width="stretch", hide_index=True)
        export_df = add_fee_columns(df)
        st.download_button(
            label=f"⬇️ 下载{title} CSV",
            data=csv_bytes(export_df),
            file_name=download_name,
            mime="text/csv",
            key=download_key,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def rebalance_orders_state_key(account):
    return f"rebalance_orders_{account}"


def save_rebalance_orders(account, orders, frequency, extra=None):
    payload = {
        "orders": pd.DataFrame() if orders is None else orders.copy(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "frequency": normalize_operation_frequency(frequency),
    }
    if extra:
        payload.update(extra)
    st.session_state[rebalance_orders_state_key(account)] = payload


def render_rebalance_orders_tab(account, has_holdings, live_emergency=False):
    if live_emergency:
        st.markdown(f'<div class="alert-banner">{EMERGENCY_BANNER}</div>', unsafe_allow_html=True)
        st.caption("自动刷新只更新估值，不会自动生成调仓建议。请点击侧边栏「生成调仓报告」。")

    if not has_holdings:
        st.info("当前账户无持仓，无法生成调仓指令。")
        return

    payload = st.session_state.get(rebalance_orders_state_key(account))
    if not payload:
        st.info("暂无调仓指令，请先点击侧边栏“生成调仓报告”。")
        return

    generated_at = payload.get("generated_at") or "未知"
    st.markdown(f"📅 报告生成时间：{generated_at}")

    orders = payload.get("orders")
    if orders is None:
        orders = pd.DataFrame()
    report_freq = normalize_operation_frequency(payload.get("frequency") or "monthly")
    emergency_report = bool(payload.get("emergency_triggered"))
    if report_freq == "weekly" or emergency_report:
        st.markdown("### 周调仓建议")
        st.markdown(f'<div class="deadline-banner">{DEADLINE_HINT}</div>', unsafe_allow_html=True)
    else:
        st.markdown("### 调仓指令")

    action_col = "建议操作" if "建议操作" in orders.columns else "指令"
    keep_df = orders[orders[action_col].isin(KEEP_SET)] if not orders.empty else orders
    reduce_df = orders[orders[action_col].isin(REDUCE_SET)] if not orders.empty else orders
    redeem_df = orders[orders[action_col].isin(REDEEM_SET)] if not orders.empty else orders
    suffix = str(account)

    render_order_list("✅ 保留清单", "keep", keep_df, "keep_list.csv", f"dl_keep_{suffix}")
    reduce_title = "🚨 紧急减仓建议" if (report_freq == "weekly" or emergency_report) else "⚠️ 减仓清单"
    render_order_list(reduce_title, "reduce", reduce_df, "reduce_list.csv", f"dl_reduce_{suffix}")
    redeem_title = "❌ 清仓 / 赎回清单" if emergency_report else "❌ 赎回清单"
    render_order_list(redeem_title, "redeem", redeem_df, "redeem_list.csv", f"dl_redeem_{suffix}")


def plot_sector_scores(df):
    plot_df = df.copy()
    plot_df["得分数值"] = pd.to_numeric(plot_df["综合得分"], errors="coerce")
    plot_df = plot_df.dropna(subset=["得分数值"])
    if plot_df.empty:
        st.info("暂无可绘制的得分数据（可能全部为数据不足）。")
        return
    plot_df["显示名称"] = plot_df["基金名称"].fillna(plot_df["基金代码"])
    plot_df = plot_df.sort_values(["赛道归类", "得分数值"], ascending=[True, True])
    plot_df["涨跌"] = np.where(plot_df["得分数值"] > 0, "正", np.where(plot_df["得分数值"] < 0, "负", "平"))
    fig = px.bar(
        plot_df,
        x="得分数值",
        y="显示名称",
        color="涨跌",
        facet_row="赛道归类" if plot_df["赛道归类"].nunique() > 1 else None,
        orientation="h",
        color_discrete_map={"正": PNL_RED, "负": PNL_GREEN, "平": "#64748b"},
        labels={"得分数值": "综合得分", "显示名称": ""},
        title="各赛道基金综合得分",
    )
    fig.update_layout(height=max(360, 80 * len(plot_df) + 80), showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


def plot_nav_compare(holdings, selected_codes):
    fig = go.Figure()
    for code in selected_codes:
        nav_df = cached_load_nav(code)
        if nav_df is None or nav_df.empty:
            continue
        window = nav_df.sort_values("date").tail(60)
        if window.empty or float(window.iloc[0]["nav"]) == 0:
            continue
        name_row = holdings.loc[holdings["基金代码"] == code]
        name = name_row.iloc[0]["基金名称"] if not name_row.empty else code
        base = float(window.iloc[0]["nav"])
        fig.add_trace(
            go.Scatter(
                x=window["date"],
                y=window["nav"] / base * 100.0,
                mode="lines",
                name=f"{code} {name}",
                line=dict(width=2.4),
            )
        )
    if not fig.data:
        st.info("所选基金暂无足够净值数据。")
        return
    fig.update_layout(
        title="近60日净值走势（起点=100）",
        xaxis_title="日期",
        yaxis_title="归一化净值",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, width="stretch")


def _fmt_nav(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def _comparison_label(code, name):
    name = str(name or "").strip() or str(code)
    return f"{code} - {name}"


def _comparison_option(code, name, sector):
    sector = str(sector or "").strip()
    if sector in {"", "nan", "None", "NaN"}:
        sector = "未分类"
    return f"[{sector}] {code} - {str(name or '').strip() or code}"


def _display_sector(value):
    text = "" if value is None else str(value).strip()
    if text in {"", "nan", "None", "NaN", "<NA>"}:
        return "未分类"
    return text


def pool_counts(df):
    if df is None or df.empty:
        return 0, 0, 0
    shares = pd.to_numeric(df.get("持有份额"), errors="coerce").fillna(0)
    market_value = pd.to_numeric(df.get("持仓市值"), errors="coerce").fillna(0)
    held = int(((shares > 0) & (market_value > 0)).sum())
    total = int(len(df))
    return total, held, total - held


def _plot_normalized_nav(compare_data, window=None):
    days = normalize_display_window(window)
    fig = go.Figure()
    for code, item in compare_data.items():
        series = item.get("近60日净值序列") or []
        dates = [point.get("date") for point in series]
        navs = [pd.to_numeric(point.get("nav"), errors="coerce") for point in series]
        pairs = [(d, n) for d, n in zip(dates, navs) if d and pd.notna(n)]
        if len(pairs) < 2 or float(pairs[0][1]) == 0:
            continue
        base = float(pairs[0][1])
        name = _comparison_label(code, item.get("基金名称"))
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in pairs],
                y=[float(p[1]) / base * 100.0 for p in pairs],
                mode="lines",
                name=name,
                line=dict(width=2.2),
            )
        )
    if not fig.data:
        st.info("所选基金暂无足够净值数据，无法绘制走势。")
        return
    count = len(fig.data)
    many = count > 10
    legend_rows = 1 if not many else (count + 4) // 5
    legend_space = 0 if not many else 24 * legend_rows
    fig.update_layout(
        title=f"近{days}日归一化净值走势（起点=100）",
        xaxis_title="日期",
        yaxis_title="归一化净值",
        hovermode="x unified",
        height=420 + (40 if many else 0) + legend_space,
        margin=dict(l=10, r=10, t=56 + legend_space, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            xanchor="left",
            font=dict(size=10 if many else 12),
            itemsizing="constant",
            entrywidthmode="pixels",
            entrywidth=160 if many else 220,
            tracegroupgap=4,
        ),
    )
    st.plotly_chart(fig, width="stretch")


def _plot_metric_bars(compare_data, window=None):
    days = normalize_display_window(window)
    rows = []
    for code, item in compare_data.items():
        rows.append(
            {
                "基金": _comparison_label(code, item.get("基金名称")),
                "收益率": pd.to_numeric(item.get("近60日收益率"), errors="coerce"),
                "最大回撤": pd.to_numeric(item.get("最大回撤"), errors="coerce"),
                "综合得分": pd.to_numeric(item.get("综合得分"), errors="coerce"),
            }
        )
    plot_df = pd.DataFrame(rows)
    count = len(plot_df)
    many = count > 10
    specs = [
        ("收益率", f"近{days}日收益率（%）", True, False),
        ("最大回撤", f"近{days}日最大回撤（绝对值，%）", True, True),
        ("综合得分", "综合得分", False, False),
    ]
    slots = [st.container(), st.container(), st.container()] if many else st.columns(3)
    bar_height = 360 if not many else max(420, 28 * count + 140)
    tickangle = -35 if not many else -60
    tick_size = 11 if not many else 9
    bargap = 0.25 if not many else max(0.04, min(0.18, 1.2 / max(count, 1)))
    for slot, (field, title, as_pct, abs_val) in zip(slots, specs):
        with slot:
            raw = pd.to_numeric(plot_df[field], errors="coerce")
            series = raw.abs() if abs_val else raw
            if as_pct:
                series = series * 100.0
            chart_df = plot_df.assign(指标=series, _signed=raw).dropna(subset=["指标"])
            if chart_df.empty:
                st.info(f"暂无{title}数据。")
                continue
            chart_df["涨跌"] = np.where(
                chart_df["_signed"] > 0, "正", np.where(chart_df["_signed"] < 0, "负", "平")
            )
            fig = px.bar(
                chart_df,
                x="基金",
                y="指标",
                color="涨跌",
                color_discrete_map={"正": PNL_RED, "负": PNL_GREEN, "平": "#64748b"},
                title=title,
                labels={"指标": title, "基金": ""},
            )
            fig.update_layout(
                showlegend=False,
                height=bar_height,
                bargap=bargap,
                margin=dict(l=10, r=10, t=50, b=110 if many else 80),
                xaxis=dict(tickangle=tickangle, tickfont=dict(size=tick_size), automargin=True),
            )
            st.plotly_chart(fig, width="stretch")


def _render_comparison_table(compare_data, window=None):
    days = normalize_display_window(window)
    rows = []
    for code, item in compare_data.items():
        rows.append(
            {
                "基金代码": code,
                "基金名称": item.get("基金名称") or "-",
                "赛道": item.get("赛道归类") or "-",
                f"近{days}日收益率": fmt_pct(item.get("近60日收益率")),
                f"近{days}日最大回撤": fmt_pct(item.get("最大回撤")),
                "波动率": fmt_pct(item.get("年化波动率")),
                "综合得分": fmt_score(item.get("综合得分")),
                "最新净值": _fmt_nav(item.get("最新净值")),
            }
        )
    view = pd.DataFrame(rows)
    st.dataframe(style_insufficient(view), width="stretch", hide_index=True)


def render_fund_comparison_tab(holdings):
    window = render_display_window_selector("display_window_compare")
    st.caption("选择任意数量的基金进行对比，可跨赛道。走势图已归一化，便于比较相对表现。建议至少选择2只基金。")
    if holdings is None or holdings.empty:
        st.info("暂无基金可对比。请先填写当前账户的 fund_pool.csv，并点击「刷新数据」。")
        return

    pool = holdings.copy()
    pool["基金代码"] = pool["基金代码"].astype(str).str.strip()
    pool["基金名称"] = pool.get("基金名称", pd.Series(dtype=str)).fillna("").astype(str)
    if "赛道归类" not in pool.columns:
        pool["赛道归类"] = ""
    pool["赛道归类"] = pool["赛道归类"].map(_display_sector)

    mode = st.radio(
        "对比方式",
        ["赛道内对比", "自由对比"],
        horizontal=True,
        key="fund_compare_mode",
    )

    selected_codes = []
    if mode == "赛道内对比":
        sectors = sorted(pool["赛道归类"].dropna().unique().tolist())
        if not sectors:
            st.info("暂无赛道可对比。")
            return
        counts = pool.groupby("赛道归类")["基金代码"].nunique()
        default_sector = next((s for s in sectors if int(counts.get(s, 0)) >= 2), sectors[0])
        sector = st.selectbox(
            "选择赛道",
            sectors,
            index=sectors.index(default_sector),
            key="fund_compare_sector",
        )
        selected_codes = pool.loc[pool["赛道归类"] == sector, "基金代码"].tolist()
        if len(selected_codes) < 2:
            st.warning("当前赛道可对比基金不足 2 只，无法对比。")
            return
    else:
        options = [
            _comparison_option(code, name, sector)
            for code, name, sector in zip(
                pool["基金代码"].tolist(),
                pool["基金名称"].tolist(),
                pool["赛道归类"].tolist(),
            )
        ]
        code_map = dict(zip(options, pool["基金代码"].tolist()))
        default_n = min(3, len(options))
        selected_labels = st.multiselect(
            "选择基金",
            options=options,
            default=options[:default_n],
            key="fund_compare_free",
            help="可选择任意数量，建议至少 2 只。",
        )
        selected_codes = [code_map[item] for item in selected_labels]
        if len(selected_codes) < 2:
            st.warning("建议至少选择 2 只基金进行对比。")
            return

    compare_data = cached_fund_comparison(tuple(selected_codes), window)
    skipped = [code for code in selected_codes if code not in compare_data]
    if skipped:
        st.caption("已跳过数据不足的基金：" + "、".join(skipped))
    if len(compare_data) < 2:
        st.warning("有效对比基金不足 2 只，无法对比。")
        return

    _plot_normalized_nav(compare_data, window=window)
    _plot_metric_bars(compare_data, window=window)
    _render_comparison_table(compare_data, window=window)


def sync_current_account():
    ensure_profiles_initialized()
    accounts = list_accounts()
    session_name = st.session_state.get("current_account")
    if session_name in accounts:
        if session_name != get_current_account_name():
            set_current_account(session_name)
        return session_name
    current = get_current_account_name()
    st.session_state.current_account = current
    return current


def apply_account_switch(name):
    """切换账户：只改当前账户并清缓存，不调用 update_all_funds，不联网。"""
    set_current_account(name)
    st.session_state.current_account = name
    st.session_state.show_new_account = False
    st.session_state.confirm_delete_account = False
    st.session_state.signal_nonce = int(st.session_state.get("signal_nonce") or 0) + 1
    st.cache_data.clear()
    st.rerun()


def render_sector_overview_tab(pool):
    if pool is None or pool.empty:
        st.info("当前基金池为空。请先在账户的 fund_pool.csv 中添加基金。")
        return
    board = merge_holdings_scores(pool.copy())
    board["展示赛道"] = board.get("赛道归类", pd.Series(dtype=str)).map(_display_sector)
    order = [name for name in board["展示赛道"].value_counts().index.tolist() if name != "未分类"]
    if (board["展示赛道"] == "未分类").any():
        order.append("未分类")
    covered = board.loc[board["展示赛道"] != "未分类", "展示赛道"].nunique()
    st.caption(f"覆盖 {int(covered)} 个赛道，共 {len(board)} 只基金（含观察池）")
    for sector in order:
        group = board.loc[board["展示赛道"] == sector].copy()
        title = f"{sector}（{len(group)}）"
        with st.expander(title, expanded=False):
            rows = []
            for _, row in group.iterrows():
                rows.append(
                    {
                        "基金": f"{row.get('基金代码')} - {row.get('基金名称')}",
                        "当前市值": fmt_money(row.get("持仓市值")),
                        "持有份额": fmt_shares(row.get("持有份额")),
                        "综合得分": fmt_score(row.get("综合得分")),
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _refresh_after_mapping_change():
    st.cache_data.clear()
    st.rerun()


def _remember_auto_added_sectors(names):
    added = st.session_state.setdefault("sector_auto_added", [])
    for name in names or []:
        text = str(name or "").strip()
        if text and text not in added:
            added.append(text)


def _show_auto_added_banners():
    from src.factor_layer.sector_classifier import DEFAULT_NEW_SECTOR_LIMIT

    added = st.session_state.get("sector_auto_added") or []
    pct = int(round(float(DEFAULT_NEW_SECTOR_LIMIT) * 100))
    for name in added:
        st.warning(f"检测到新赛道「{name}」，将自动添加，默认上限为 {pct}%。可在「赛道配置」中调整上限。")


def _sector_is_new(name):
    from src.factor_layer.sector_classifier import AUTO_MATCH_LABEL, CUSTOM_SECTOR_LABEL, get_sector_limits

    text = str(name or "").strip()
    if not text or text in {AUTO_MATCH_LABEL, CUSTOM_SECTOR_LABEL}:
        return False
    return text not in get_sector_limits()


def _render_sector_limits_panel():
    from src.factor_layer.sector_classifier import (
        DEFAULT_NEW_SECTOR_LIMIT,
        delete_sector_limit,
        funds_mapped_to_sector,
        get_sector_limits,
        set_sector_limit,
        sync_mapped_sectors_into_limits,
    )

    st.markdown("#### 赛道配置")
    st.caption("在此可以管理所有赛道及其仓位上限。新增赛道会自动添加，您可随时调整上限。")
    try:
        sync_mapped_sectors_into_limits()
    except Exception:
        pass
    limits = get_sector_limits()
    if not limits:
        st.info("尚未配置赛道上限。")
    else:
        header = st.columns([2.2, 1.4, 1.2])
        header[0].caption("赛道名称")
        header[1].caption("上限 %")
        header[2].caption("")
        for name, limit in limits.items():
            saved_pct = int(round(float(limit) * 100))
            cols = st.columns([2.2, 1.4, 1.2])
            cols[0].write(name)
            widget_key = f"slim_{name}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = saved_pct
            new_pct = cols[1].number_input(
                "上限",
                min_value=0,
                max_value=100,
                step=1,
                key=widget_key,
                label_visibility="collapsed",
            )
            if int(new_pct) != saved_pct:
                set_sector_limit(name, int(new_pct) / 100.0)
                st.rerun()
            if cols[2].button("删除", key=f"sdel_{name}"):
                st.session_state[f"sdel_ask_{name}"] = True
            if st.session_state.get(f"sdel_ask_{name}"):
                mapped = funds_mapped_to_sector(name)
                if mapped:
                    st.warning(
                        f"「{name}」仍有 {len(mapped)} 只基金映射。"
                        "请先在映射总览中移除，或确认一并删除这些映射。"
                    )
                    drop = st.checkbox("一并删除这些基金的映射", key=f"sdel_drop_{name}")
                    confirm = st.checkbox("我确认删除该赛道", key=f"sdel_ok_{name}")
                    if st.button("确认删除", type="primary", key=f"sdel_go_{name}", disabled=not (drop and confirm)):
                        delete_sector_limit(name, drop_mappings=True)
                        st.session_state.pop(f"sdel_ask_{name}", None)
                        st.session_state.pop(f"slim_{name}", None)
                        _refresh_after_mapping_change()
                else:
                    st.warning(f"即将删除赛道「{name}」。")
                    confirm = st.checkbox("我确认删除该赛道", key=f"sdel_ok_{name}")
                    if st.button("确认删除", type="primary", key=f"sdel_go_{name}", disabled=not confirm):
                        delete_sector_limit(name, drop_mappings=False)
                        st.session_state.pop(f"sdel_ask_{name}", None)
                        st.session_state.pop(f"slim_{name}", None)
                        st.rerun()

    add_cols = st.columns([2.2, 1.4, 1.2])
    new_name = add_cols[0].text_input("新赛道名称", key="sadd_name", placeholder="例如：机器人")
    default_pct = int(round(float(DEFAULT_NEW_SECTOR_LIMIT) * 100))
    new_pct = add_cols[1].number_input(
        "新赛道上限 %",
        min_value=0,
        max_value=100,
        step=1,
        value=default_pct,
        key="sadd_pct",
    )
    if add_cols[2].button("添加赛道", key="sadd_go"):
        name = str(new_name or "").strip()
        if not name:
            st.error("请输入赛道名称。")
        elif name in get_sector_limits():
            st.error(f"赛道「{name}」已存在。")
        else:
            set_sector_limit(name, int(new_pct) / 100.0)
            st.success(f"已添加赛道「{name}」，上限 {int(new_pct)}%。")
            st.rerun()


def _bump_sector_editor():
    st.session_state.sector_editor_rev = int(st.session_state.get("sector_editor_rev") or 0) + 1


def _commit_sector_import(pairs):
    from src.factor_layer.sector_classifier import apply_batch_sector_import
    from src.ocr.fund_matcher import remember_user_mappings

    report = apply_batch_sector_import(pairs, allow_network=True)
    try:
        entries = [
            (ident, code, name)
            for ident, code, name in (report.get("remembered") or [])
            if ident and code
        ]
        if entries:
            remember_user_mappings(entries)
    except Exception:
        pass
    if report.get("new_sectors"):
        _remember_auto_added_sectors(report["new_sectors"])
    st.session_state.sector_import_report = report
    st.session_state.pop("sector_map_preview", None)
    _bump_sector_editor()
    st.cache_data.clear()
    return report


def _sync_display_window(widget_key):
    st.session_state.display_window_days = normalize_display_window(st.session_state.get(widget_key))


def render_display_window_selector(widget_key):
    """两个 Tab 共用 display_window_days；控件 key 必须互不相同。"""
    options = list(DISPLAY_WINDOW_CHOICES)
    if "display_window_days" not in st.session_state:
        st.session_state.display_window_days = normalize_display_window(DISPLAY_WINDOW_DAYS)
    shared = normalize_display_window(st.session_state.display_window_days)
    st.session_state.display_window_days = shared
    st.session_state[widget_key] = shared
    pick_col, hint_col = st.columns([1.6, 3.4])
    with pick_col:
        chosen = st.selectbox(
            "📊 展示时间窗口",
            options,
            format_func=lambda days: f"{int(days)} 日",
            key=widget_key,
            on_change=_sync_display_window,
            args=(widget_key,),
        )
    with hint_col:
        st.caption("💡 此选择仅影响图表和表格展示，调仓决策仍基于固定周期。")
    chosen = normalize_display_window(chosen)
    st.session_state.display_window_days = chosen
    return chosen


def _fmt_sector_metric(value, kind="pct"):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "数据缺失"
    text = str(value).strip()
    if text in {"", "-", "数据不足", "数据缺失", "nan", "None"}:
        return "数据缺失"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "数据缺失"
    if kind == "nav":
        return f"{number:.4f}"
    if kind == "score":
        return f"{number:.2f}"
    return f"{number:.2f}"


def _render_sector_refresh_bar(mapped_codes):
    try:
        from streamlit_autorefresh import st_autorefresh
    except Exception:
        st_autorefresh = None

    left, mid, right = st.columns([2.2, 1.3, 2.5])
    auto = left.checkbox("自动刷新（每5分钟）", key="sector_auto_refresh_on")
    if auto:
        if st_autorefresh is not None:
            st_autorefresh(interval=5 * 60 * 1000, key="sector_auto_refresh_tick")
        else:
            left.caption("未安装 streamlit-autorefresh，自动刷新不可用。请执行 pip install streamlit-autorefresh")
    refresh_clicked = mid.button("立即刷新", key="sector_nav_refresh_now")
    from src.factor_layer.sector_analysis import get_nav_data_updated_at

    stamp = st.session_state.get("sector_nav_refreshed_at")
    if not stamp:
        updated = get_nav_data_updated_at(mapped_codes)
        stamp = updated.strftime("%Y-%m-%d %H:%M:%S") if updated else "暂无本地净值"
    right.markdown(
        f"<div style='text-align:right;color:#64748b;font-size:0.85rem;padding-top:0.45rem;'>数据更新于 {stamp}</div>",
        unsafe_allow_html=True,
    )
    return refresh_clicked


def _render_sector_overview_editor(funds, mapping, window=None):
    from src.factor_layer.sector_classifier import (
        AUTO_MATCH_LABEL,
        apply_global_sector_edits,
        list_known_sectors,
    )
    from src.ocr.fund_matcher import get_code_name_map

    days = normalize_display_window(window)
    ret_col = f"近{days}日收益率（%）"
    mdd_col = f"近{days}日最大回撤（%）"

    if funds is None or funds.empty:
        st.info("所有账户的基金池都为空。可先在各账户添加基金，或通过下方批量导入写入映射。")
        return

    sectors = list_known_sectors()
    select_options = [AUTO_MATCH_LABEL] + [item for item in sectors if item != AUTO_MATCH_LABEL]
    for sector in mapping.values():
        text = str(sector or "").strip()
        if text and text not in select_options:
            select_options.append(text)

    codes = []
    pool_names = {}
    for _, row in funds.iterrows():
        code = str(row.get("基金代码") or "").strip()
        if not code:
            continue
        codes.append(code)
        pool_names[code] = str(row.get("基金名称") or "").strip()
    if not codes:
        st.info("没有可展示的基金。")
        return

    try:
        name_map = get_code_name_map(force=False)
    except Exception:
        name_map = {}
    scores = cached_score_funds(tuple(codes), days)
    score_map = {}
    if scores is not None and not scores.empty:
        for _, row in scores.iterrows():
            code = str(row.get("基金代码") or "").strip()
            if code:
                score_map[code] = row

    rows = []
    for code in codes:
        nav_df = cached_load_nav(code)
        has_nav = nav_df is not None and not nav_df.empty and "nav" in nav_df.columns
        latest = np.nan
        if has_nav:
            nav_vals = pd.to_numeric(nav_df["nav"], errors="coerce").dropna()
            if not nav_vals.empty:
                latest = float(nav_vals.iloc[-1])
        scored = score_map.get(code)
        ret = mdd = score = None
        if scored is not None:
            ret = pd.to_numeric(scored.get("近60日收益率"), errors="coerce")
            mdd = pd.to_numeric(scored.get("近60日最大回撤"), errors="coerce")
            score = scored.get("综合得分")
        name = str(name_map.get(code) or pool_names.get(code) or "").strip()
        mapped = mapping.get(code)
        rows.append(
            {
                "基金代码": code,
                "基金名称": name or "数据缺失",
                "最新净值": _fmt_sector_metric(latest, "nav") if has_nav else "数据缺失",
                ret_col: _fmt_sector_metric(
                    None if pd.isna(ret) else float(ret) * 100.0,
                    "pct",
                ),
                mdd_col: _fmt_sector_metric(
                    None if pd.isna(mdd) else float(mdd) * 100.0,
                    "pct",
                ),
                "综合得分": _fmt_sector_metric(score, "score"),
                "设定赛道": mapped if mapped else AUTO_MATCH_LABEL,
            }
        )

    view = pd.DataFrame(rows)
    rev = int(st.session_state.get("sector_editor_rev") or 0)
    edited = st.data_editor(
        view,
        hide_index=True,
        width="stretch",
        height=min(560, 52 + 36 * min(len(view), 14)),
        num_rows="fixed",
        disabled=["基金代码", "基金名称", "最新净值", ret_col, mdd_col, "综合得分"],
        column_config={
            "设定赛道": st.column_config.SelectboxColumn(
                "设定赛道",
                options=select_options,
                required=True,
            )
        },
        key=f"sector_overview_editor_{rev}_{days}",
    )
    st.caption("修改「设定赛道」后立即保存。选「（自动匹配）」即移除手动映射。本地无净值时显示「数据缺失」，可点上方「立即刷新」补数。")

    updates = {}
    for _, row in edited.iterrows():
        code = str(row.get("基金代码") or "").strip()
        if not code:
            continue
        chosen = str(row.get("设定赛道") or "").strip() or AUTO_MATCH_LABEL
        current = mapping.get(code) or AUTO_MATCH_LABEL
        if chosen != current:
            updates[code] = "" if chosen == AUTO_MATCH_LABEL else chosen
    if not updates:
        return
    result = apply_global_sector_edits(updates)
    if result.get("new_sectors"):
        _remember_auto_added_sectors(result["new_sectors"])
    _bump_sector_editor()
    _refresh_after_mapping_change()


def _render_sector_trend_panel(window=None):
    from src.factor_layer.sector_analysis import (
        TREND_NEUTRAL,
        TREND_STRONG,
        TREND_WEAK,
    )

    days = normalize_display_window(window)
    trend_label_color = {
        TREND_STRONG: "#ff0000",
        TREND_WEAK: "#00cc00",
        TREND_NEUTRAL: "#888888",
    }
    ret_pos_color = "#ff0000"
    ret_neg_color = "#00cc00"
    ret_zero_color = "#888888"

    def _signed_return_label(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None, ret_zero_color
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None, ret_zero_color
        if number > 0:
            return f"+{number:.2f}%", ret_pos_color
        if number < 0:
            return f"{number:.2f}%", ret_neg_color
        return f"{number:.2f}%", ret_zero_color

    with st.expander(f"赛道趋势分析（近{days}日等权净值）", expanded=False):
        with st.spinner("正在计算赛道趋势..."):
            rows, series_map = cached_sector_trend_rows(days, 2)
        if not rows:
            st.info("暂无足够净值绘制赛道趋势。请先为映射基金点击「立即刷新」。")
            return

        fig = go.Figure()
        endpoints = []
        for item in rows:
            sector = item["赛道"]
            series = series_map.get(sector)
            if series is None or series.empty or "nav" not in series.columns:
                continue
            nav = pd.to_numeric(series["nav"], errors="coerce").dropna()
            if nav.empty or float(nav.iloc[0]) == 0:
                continue
            trend = item.get("趋势") or TREND_NEUTRAL
            y_values = (nav / float(nav.iloc[0]) * 100.0)
            fig.add_trace(
                go.Scatter(
                    x=list(nav.index),
                    y=y_values.tolist(),
                    mode="lines",
                    name=f"{sector}（{trend}）",
                    line=dict(width=2.4),
                )
            )
            endpoints.append(
                {
                    "sector": sector,
                    "trend": trend,
                    "ret": item.get("近N日收益"),
                    "x": nav.index[-1],
                    "y": float(y_values.iloc[-1]),
                }
            )
        if not fig.data:
            st.info("赛道净值序列不足，无法绘图。")
            return

        ranked = sorted(endpoints, key=lambda row: row["y"])
        for index, point in enumerate(ranked):
            stagger = (index - (len(ranked) - 1) / 2.0) * 10
            ret_text, ret_color = _signed_return_label(point["ret"])
            if ret_text:
                fig.add_annotation(
                    x=point["x"],
                    y=point["y"],
                    text=ret_text,
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    xshift=8,
                    yshift=4 + stagger,
                    font_color=ret_color,
                    font_size=12,
                )
            fig.add_annotation(
                x=point["x"],
                y=point["y"],
                text=str(point["trend"]),
                showarrow=False,
                xanchor="left",
                yanchor="top",
                xshift=8,
                yshift=-2 + stagger,
                font_color=trend_label_color.get(point["trend"], ret_zero_color),
                font_size=11,
            )

        fig.update_layout(
            title=f"各赛道等权平均净值（近{days}日，起点=100）",
            xaxis_title="日期",
            yaxis_title="归一化净值",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=420,
            margin=dict(l=10, r=96, t=60, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        scores = cached_score_funds(
            tuple(code for item in rows for code in (item.get("推荐基金") or [])),
            days,
        )
        score_lookup = {}
        if scores is not None and not scores.empty:
            for _, row in scores.iterrows():
                code = str(row.get("基金代码") or "").strip()
                score_lookup[code] = row.get("综合得分")

        table_rows = []
        for item in rows:
            tops = item.get("推荐基金") or []
            hold_bits = []
            for code in tops:
                hold_bits.append(f"{code}（{fmt_score(score_lookup.get(code))}）")
            hold_text = "建议持有 " + "、".join(hold_bits) if hold_bits else "-"
            ret_nd = item.get("近N日收益")
            table_rows.append(
                {
                    "赛道": item.get("赛道"),
                    f"近{days}日收益（%）": _fmt_sector_metric(ret_nd, "pct"),
                    "趋势": item.get("趋势"),
                    "建议": item.get("建议"),
                    "推荐基金": hold_text,
                }
            )
        view = pd.DataFrame(table_rows)
        ret_col = f"近{days}日收益（%）"

        def _trend_css(value):
            text = str(value or "")
            if text == TREND_STRONG:
                return "color: #ff0000; font-weight: 700;"
            if text == TREND_WEAK:
                return "color: #00cc00; font-weight: 700;"
            if text == TREND_NEUTRAL:
                return "color: #888888; font-weight: 650;"
            return ""

        def _ret_css(value):
            number = _parse_signed_display(value)
            if number is None:
                return ""
            if number > 0:
                return "color: #ff0000; font-weight: 650;"
            if number < 0:
                return "color: #00cc00; font-weight: 650;"
            return "color: #888888;"

        styled = view.style
        styled = _styler_applymap(styled, _trend_css, ["趋势"])
        if ret_col in view.columns:
            styled = _styler_applymap(styled, _ret_css, [ret_col])
        st.dataframe(styled, width="stretch", hide_index=True)
        st.caption(
            f"强势：近{days}日收益 > 5%；弱势：近{days}日收益 < -5%。"
            f"近{days}日收益为负且弱势时建议观望/放弃。"
        )


def _render_sector_import_report():
    from src.factor_layer.sector_classifier import DEFAULT_NEW_SECTOR_LIMIT

    report = st.session_state.get("sector_import_report")
    if not report:
        return
    imported = int(report.get("imported") or 0)
    diff_rows = list(report.get("diff_rows") or [])
    duplicates = list(report.get("duplicates") or [])
    new_sectors = list(report.get("new_sectors") or [])
    skipped = list(report.get("skipped") or [])
    close_col, _ = st.columns([1, 4])
    with close_col:
        if st.button("关闭报告", key="sector_import_report_close"):
            st.session_state.sector_import_report = None
            st.rerun()

    st.markdown(
        f"📊 已导入 {imported} 条映射，其中 {len(diff_rows)} 条与系统自动识别不同（已按您的输入为准）。"
    )
    if new_sectors:
        pct = int(round(float(DEFAULT_NEW_SECTOR_LIMIT) * 100))
        shown = "、".join(f"「{name}」" for name in new_sectors)
        st.info(f"已自动添加新赛道{shown}，默认上限 {pct}%。可在上方「赛道配置」中调整。")
    if duplicates:
        st.warning("同一只基金在本次导入中出现多次，已保留最后一次映射。")
        st.dataframe(pd.DataFrame(duplicates), width="stretch", hide_index=True)
    if skipped:
        st.warning(f"有 {len(skipped)} 行未能识别，已跳过。")
        st.dataframe(pd.DataFrame(skipped), width="stretch", hide_index=True)
    if diff_rows:
        st.markdown("⚠️ 以下映射与系统自动识别结果不同（已按您输入为准）：")
        st.dataframe(
            pd.DataFrame(diff_rows, columns=["基金名称", "您输入的赛道", "系统自动识别的赛道"]),
            width="stretch",
            hide_index=True,
        )
    elif imported:
        st.caption("本次导入的赛道与系统自动识别一致，无需额外核对。")


def render_sector_manage_tab():
    try:
        from src.factor_layer.sector_classifier import (
            clear_global_sector_mapping,
            collect_all_pool_funds,
            export_global_mapping_df,
            load_global_sector_mapping,
            parse_sector_csv_text,
        )
    except Exception as exc:
        st.error(f"赛道管理模块导入失败：{exc}")
        st.caption("可运行 `python scripts/diagnose_dashboard.py` 检查语法与依赖。")
        return

    st.caption("在此处可以全局设定基金与赛道的对应关系，所有账户共享此映射。手动映射将覆盖系统的自动匹配结果。")
    window = render_display_window_selector("display_window_sector")
    _show_auto_added_banners()

    mapping = load_global_sector_mapping()
    mapped_codes = [str(code).strip() for code in mapping if str(code).strip()]
    refresh_clicked = _render_sector_refresh_bar(mapped_codes)
    if refresh_clicked:
        if not mapped_codes:
            st.warning("当前没有手动映射的基金。请先设定赛道或批量导入后再刷新净值。")
        else:
            from src.factor_layer.sector_analysis import refresh_mapped_fund_nav

            progress = st.progress(0, text="正在刷新映射基金净值...")

            def _on_mapped_progress(index, total, fund_code):
                ratio = float(index) / float(total or 1)
                progress.progress(min(ratio, 1.0), text=f"正在更新 {fund_code}（{index}/{total}）")

            with st.spinner("正在刷新映射基金净值与名称缓存..."):
                refresh_mapped_fund_nav(progress_callback=_on_mapped_progress, refresh_names=True)
            progress.progress(1.0, text="刷新完成")
            st.session_state.sector_nav_refreshed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.cache_data.clear()
            _bump_sector_editor()
            st.rerun()

    st.markdown(f"**已手动映射 {len(mapping)} 只基金**")
    _render_sector_limits_panel()

    funds = collect_all_pool_funds()
    extra_rows = []
    seen = set(funds["基金代码"].astype(str)) if funds is not None and not funds.empty else set()
    for code in mapping:
        if code not in seen:
            extra_rows.append({"基金代码": code, "基金名称": ""})
    if extra_rows:
        funds = pd.concat([funds, pd.DataFrame(extra_rows)], ignore_index=True)

    st.markdown("#### 映射总览")
    _render_sector_overview_editor(funds, mapping, window=window)
    _render_sector_trend_panel(window=window)

    st.markdown("#### 批量导入")
    st.caption("💡 导入后映射将直接生效，无需逐条确认。系统会生成差异报告供您参考。")
    st.caption("第一列可输入基金代码（6位数字）或基金名称，系统将自动识别并匹配。")
    shot_col, paste_col = st.columns(2)
    with shot_col:
        st.markdown("##### 截图导入")
        st.caption("上传含「基金名称/基金代码」和「赛道」两列的截图。识别结果会自动判断是代码还是名称。")
        st.caption("示例：`012349,半导体` 或 `华夏国证半导体芯片ETF联接C,半导体`。")
        shot_file = st.file_uploader("上传赛道截图", type=["png", "jpg", "jpeg"], key="sector_map_shot_file")
        if st.button("识别并导入", key="sector_map_shot_import"):
            try:
                from src.ocr.ocr_engine import extract_sector_pairs, models_ready
            except Exception as exc:
                st.error(f"OCR 模块导入失败：{exc}")
            else:
                path = _save_upload(shot_file, "sector_map")
                if not path:
                    st.error("请先上传截图。")
                else:
                    hint = "正在识别并写入映射..." if models_ready() else "首次使用正在下载模型，请稍候..."
                    with st.spinner(hint):
                        parsed = extract_sector_pairs(image_path=path)
                    if not parsed:
                        st.error("未识别到基金名称与赛道，请确保截图包含这两列。")
                    else:
                        with st.spinner("正在写入映射..."):
                            _commit_sector_import(parsed)
                        st.rerun()
    with paste_col:
        st.markdown("##### 粘贴导入")
        st.caption("每行格式：`基金标识,赛道`。第一列可以是 6 位代码或基金名称，支持逗号或制表符分隔。")
        paste_text = st.text_area(
            "CSV文本",
            height=180,
            placeholder=(
                "基金标识,赛道\n"
                "012349,半导体\n"
                "华夏国证半导体芯片ETF联接C,半导体\n"
                "易方达蓝筹精选混合,消费"
            ),
            key="sector_map_paste_text",
            label_visibility="collapsed",
        )
        if st.button("解析并导入", key="sector_map_paste_import"):
            with st.spinner("正在解析并写入映射..."):
                parsed, error = parse_sector_csv_text(paste_text)
                if error:
                    report_error = error
                    report = None
                else:
                    report_error = None
                    report = _commit_sector_import(parsed)
            if report_error:
                st.error(report_error)
            elif report is not None:
                st.rerun()

    st.markdown("#### 导出与重置")
    export_df = export_global_mapping_df()
    down_col, reset_col = st.columns(2)
    with down_col:
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "导出映射",
            data=csv_bytes,
            file_name="global_fund_sector_mapping.csv",
            mime="text/csv",
            disabled=export_df.empty,
            key="sector_map_export",
        )
        if export_df.empty:
            st.caption("当前没有手动映射可导出。")
    with reset_col:
        if st.button("重置所有映射", key="sector_map_reset_ask"):
            st.session_state.sector_map_reset_confirm = True
        if st.session_state.get("sector_map_reset_confirm"):
            st.warning(
                "将清空 global_fund_sector_mapping.json 中的用户手动映射。"
                "关键词自动匹配规则（sector_mapping.json）不受影响。"
            )
            if st.checkbox("我确认清空所有手动映射", key="sector_map_reset_check"):
                if st.button("确认清空", type="primary", key="sector_map_reset_go"):
                    clear_global_sector_mapping()
                    st.session_state.sector_map_reset_confirm = False
                    st.session_state.sector_import_report = None
                    _bump_sector_editor()
                    st.success("已清空全部手动映射，自动匹配规则仍保留。")
                    _refresh_after_mapping_change()

    if st.session_state.get("sector_import_report"):
        st.markdown("#### 差异报告")
        _render_sector_import_report()


def open_rebalance_dialog(code, name, old_shares):
    title = f"调仓 - {name or code}"

    @st.dialog(title)
    def _dialog():
        current = float(old_shares or 0.0)
        st.caption(f"当前持有份额：{current:,.2f}")
        new_shares = st.number_input(
            "新的持有份额",
            min_value=0.0,
            value=float(current),
            step=0.01,
            format="%.2f",
            key=f"rebalance_shares_{code}",
        )
        reason = st.text_input("操作备注", placeholder="选填", key=f"rebalance_reason_{code}")
        big_cut = current > 0 and float(new_shares) < current * 0.5
        confirmed = True
        if big_cut:
            st.warning("减仓幅度较大，请确认")
            confirmed = st.checkbox("我已确认减仓幅度较大", key=f"rebalance_confirm_{code}")
        if st.button("确认调仓", type="primary", disabled=not confirmed, key=f"rebalance_submit_{code}"):
            ok, result = apply_manual_rebalance(code, new_shares, reason)
            if not ok:
                st.error(result)
                return
            st.cache_data.clear()
            st.session_state.pop("rebalance_fund", None)
            st.success(f"已将 {code} 份额由 {result['old_shares']:,.2f} 调整为 {result['new_shares']:,.2f}")
            st.rerun()

    _dialog()


def render_add_watch_fund_panel(account, pool=None):
    flash_key = f"watch_fund_flash_{account}"
    flash = st.session_state.pop(flash_key, None)
    if flash:
        st.success(flash)

    st.markdown("##### 添加观察基金")
    st.caption("输入 6 位基金代码，系统会自动查询名称、匹配赛道，并以 0 份额写入当前账户观察池。")
    with st.form(key=f"add_watch_fund_form_{account}", clear_on_submit=True):
        cols = st.columns([3.2, 1.2])
        code_input = cols[0].text_input(
            "基金代码",
            placeholder="例如 012349",
            max_chars=10,
        )
        submitted = cols[1].form_submit_button("添加观察基金", type="primary")
    if submitted:
        from src.factor_layer.portfolio_utils import add_watch_fund

        with st.spinner("正在获取基金名称并写入观察池..."):
            ok, result = add_watch_fund(code_input)
        if ok:
            sector_text = _display_sector(result.get("sector"))
            st.session_state[flash_key] = (
                f"已添加观察基金 {result['fund_code']} {result['fund_name']}（赛道：{sector_text}）"
            )
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(result)

    if pool is None or pool.empty or "基金代码" not in pool.columns:
        return
    shares = pd.to_numeric(pool.get("持有份额"), errors="coerce").fillna(0)
    values = pd.to_numeric(pool.get("持仓市值"), errors="coerce").fillna(0)
    watch = pool.loc[(shares <= 0) | (values <= 0)].copy()
    if watch.empty:
        return
    sector_series = (
        watch["赛道归类"] if "赛道归类" in watch.columns else pd.Series("", index=watch.index)
    )
    name_series = watch["基金名称"] if "基金名称" in watch.columns else pd.Series("", index=watch.index)
    view = pd.DataFrame(
        {
            "基金代码": watch["基金代码"].astype(str),
            "基金名称": name_series,
            "赛道": sector_series.map(_display_sector),
            "持有份额": pd.to_numeric(watch.get("持有份额"), errors="coerce").fillna(0),
        }
    )
    st.caption(f"观察池 {len(view)} 只（份额为 0，不计入持仓市值）")
    st.dataframe(view, width="stretch", hide_index=True)


def render_holdings_tab(board, account, pool=None):
    render_add_watch_fund_panel(account, pool)
    if board is None or board.empty:
        st.info("暂无持仓。可在上方添加观察基金，或点击「刷新数据」更新净值后再查看持仓。")
        return
    if holdings_missing_cost(board):
        st.warning("请先补充持仓成本，否则无法计算收益率")

    view = holdings_display(board)
    st.dataframe(
        style_insufficient(view, pnl_df=board),
        width="stretch",
        hide_index=True,
    )

    st.markdown("##### 在线调仓")
    st.caption("点击右侧「调仓」修改持有份额，将写入当前账户基金池并追加 trade_log。")
    for _, row in board.iterrows():
        code = str(row.get("基金代码") or "").strip()
        name = str(row.get("基金名称") or "").strip() or code
        shares = float(pd.to_numeric(row.get("持有份额"), errors="coerce") or 0.0)
        c1, c2, c3, c4 = st.columns([1.4, 3.6, 1.6, 1.0])
        c1.write(code)
        c2.write(name)
        c3.write(f"{shares:,.2f} 份")
        if c4.button("调仓", key=f"rebalance_btn_{account}_{code}", width="stretch"):
            st.session_state.rebalance_fund = {
                "code": code,
                "name": name,
                "shares": shares,
            }
            open_rebalance_dialog(code, name, shares)


def _save_upload(uploaded, prefix):
    if uploaded is None:
        return ""
    folder = Path(PROJECT_ROOT) / "data" / "ocr_tmp"
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix or ".png"
    path = folder / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    path.write_bytes(uploaded.getbuffer())
    return str(path)


def _attach_matches(rows):
    from src.ocr.fund_matcher import match_fund_code

    enriched = []
    for row in rows or []:
        item = dict(row)
        matched = match_fund_code(item.get("fund_name")) or {}
        item["match"] = matched
        item["match_tier"] = matched.get("match_tier") or "fail"
        item["fail_reason"] = matched.get("fail_reason") or ""
        item["fund_code"] = matched.get("fund_code") or ""
        item["match_score"] = float(matched.get("confidence") or matched.get("score") or 0.0)
        item["include"] = True
        enriched.append(item)
    return enriched


def _style_ocr_preview(view, source_rows):
    if view is None or view.empty:
        return view
    colors = []
    for row in source_rows:
        tier = str(row.get("match_tier") or "")
        score = float(row.get("match_score") or row.get("confidence") or 0.0)
        if tier == "fail" or score < 0.75:
            colors.append("#fecaca")
        elif tier == "partial" or score < 0.90:
            colors.append("#fef08a")
        else:
            colors.append("")
    if not any(colors):
        return view

    def highlight(_row):
        idx = _row.name
        fill = colors[idx] if idx < len(colors) else ""
        return [f"background-color: {fill};" if fill else ""] * len(_row)

    try:
        return view.style.apply(highlight, axis=1)
    except Exception:
        return view


def _match_tier_of(row):
    tier = str((row.get("match") or {}).get("match_tier") or row.get("match_tier") or "")
    if tier in {"high", "partial", "fail"}:
        return tier
    score = float(row.get("match_score") or (row.get("match") or {}).get("confidence") or 0.0)
    if score > 0.90:
        return "high"
    if score >= 0.75:
        return "partial"
    return "fail"


def _render_match_selector(row, key_prefix, index):
    from src.ocr.fund_matcher import search_fund_names

    matched = row.get("match") or {}
    candidates = matched.get("candidates") or []
    name = str(row.get("fund_name") or "未知基金")
    current = str(row.get("fund_code") or "").strip()
    score = float(matched.get("confidence") or row.get("match_score") or 0.0)
    tier = _match_tier_of(row)
    matched_name = str(matched.get("matched_name") or "")
    reason = str(matched.get("fail_reason") or "")

    if tier == "high" and current:
        st.markdown(
            f"<div style='background:#dcfce7;border:1px solid #86efac;border-radius:8px;padding:8px 10px;margin-bottom:6px;'>"
            f"✅ {name} → <b>{current}</b> {matched_name}（{score:.0%}）</div>",
            unsafe_allow_html=True,
        )
        return current

    if tier == "partial":
        st.markdown(
            f"<div style='background:#fef9c3;border:1px solid #facc15;border-radius:8px;padding:8px 10px;margin-bottom:6px;'>",
            unsafe_allow_html=True,
        )
        st.caption(f"⚠️ {name} 匹配待确认（{score:.0%}）{(' · ' + matched_name) if matched_name else ''}")
    else:
        st.markdown(
            f"<div style='background:#fee2e2;border:1px solid #f87171;border-radius:8px;padding:8px 10px;margin-bottom:6px;'>",
            unsafe_allow_html=True,
        )
        st.caption(f"❌ {name} 未自动匹配" + (f"（{reason}）" if reason else "") + "，请搜索或手动输入代码")

    options = []
    labels = []
    for item in candidates:
        code = str(item.get("fund_code") or "").strip()
        cand_name = str(item.get("fund_name") or "").strip()
        cand_score = float(item.get("score") or 0.0)
        if not code or code in options:
            continue
        options.append(code)
        labels.append(f"{code} - {cand_name}（{cand_score:.0%}）")
    if tier == "fail":
        options.append("__search__")
        labels.append("搜索基金名称…")
    options.append("__manual__")
    labels.append("手动输入代码")
    default_idx = options.index(current) if current in options else 0
    chosen = st.selectbox(
        "匹配基金代码",
        options=options,
        index=min(default_idx, len(options) - 1) if options else 0,
        format_func=lambda value: labels[options.index(value)] if value in options else value,
        key=f"{key_prefix}_code_{index}",
        label_visibility="collapsed",
    )
    picked_name = ""
    if chosen == "__search__":
        query = st.text_input("搜索基金名称或代码", key=f"{key_prefix}_search_{index}", placeholder="输入名称关键字，如 人工智能")
        hits = search_fund_names(query, limit=12) if str(query or "").strip() else []
        if hits:
            hit_codes = [str(item.get("fund_code") or "") for item in hits]
            hit_labels = [f"{item.get('fund_code')} - {item.get('fund_name')}" for item in hits]
            selected = st.selectbox(
                "搜索结果",
                options=hit_codes,
                format_func=lambda value: hit_labels[hit_codes.index(value)] if value in hit_codes else value,
                key=f"{key_prefix}_search_pick_{index}",
            )
            picked_name = next((item.get("fund_name") for item in hits if item.get("fund_code") == selected), "")
            chosen = selected
        else:
            chosen = ""
            if str(query or "").strip():
                st.caption("没有搜到，请改关键字或手动输入代码。")
    if chosen == "__manual__":
        typed = st.text_input(
            "手动基金代码",
            value=current if current and current not in options else "",
            key=f"{key_prefix}_manual_{index}",
            placeholder="6位基金代码",
        )
        chosen = str(typed or "").strip()
    st.markdown("</div>", unsafe_allow_html=True)
    return str(chosen or "").strip()


def _recognize_image(uploaded, kind):
    try:
        from src.ocr.ocr_engine import (
            extract_holdings_full,
            extract_text_from_image,
            extract_transactions,
            models_ready,
        )
        from src.ocr.importer import build_holdings_import_preview
    except Exception as exc:
        return [], f"OCR 模块导入失败：{exc}"

    path = _save_upload(uploaded, kind)
    if not path:
        return [], "请先上传截图。"
    hint = "正在识别..." if models_ready() else "首次使用正在下载模型，请稍候..."
    with st.spinner(hint):
        if kind == "holdings":
            parsed = extract_holdings_full(image_path=path)
            if not parsed:
                return [], "识别失败，请确保截图清晰且包含完整信息"
            return build_holdings_import_preview(parsed, allow_network=True), ""
        texts = extract_text_from_image(path)
        if not texts:
            return [], "识别失败，请确保截图清晰且包含完整信息"
        parsed = extract_transactions(texts)
        if not parsed:
            return [], "识别失败，请确保截图清晰且包含完整信息"
        return _attach_matches(parsed), ""


def _load_holdings_preview(extracted, account):
    from src.ocr.importer import build_holdings_import_preview

    with st.spinner("正在匹配基金代码并查询净值..."):
        rows = build_holdings_import_preview(extracted, allow_network=True)
    st.session_state.ocr_holdings_preview = rows
    st.session_state.ocr_holdings_original = [dict(row) for row in rows]
    st.session_state.ocr_hold_editor_nonce = int(st.session_state.get("ocr_hold_editor_nonce") or 0) + 1
    st.session_state.pop(f"ocr_csv_need_map_{account}", None)


def _render_csv_paste_backup(account):
    from src.ocr.importer import CSV_MAP_FIELDS, parse_holdings_csv

    map_key = f"ocr_csv_need_map_{account}"
    with st.expander("📋 备用方案：粘贴AI识别的CSV数据", expanded=False):
        st.info("如果OCR识别不准，建议使用此备用方案。")
        st.markdown(
            """
**如何获取 CSV**
1. 截一张支付宝持仓表格图（尽量包含表头）。
2. 发给豆包 / 通义千问，使用类似指令：

> 请把这张支付宝持仓截图转成 CSV，第一行必须是表头，列名为：  
> 基金名称,持有金额(元),占比(%),昨日收益(元),持有收益(元),持有收益率(%),累计收益(元),备注  
> 只输出 CSV，不要解释。

3. 把生成的文本粘贴到下方，点「解析并导入」。

**CSV 格式示例**
```
基金名称,持有金额(元),占比(%),昨日收益(元),持有收益(元),持有收益率(%),累计收益(元),备注
东方人工智能主题混合C,201579.85,12.34,+106.02,-30123.45,-13.22,1357.80,定投
```
第一行可以是中文或英文列名（如 `fund_name`）。若没有表头，解析后可手动指定列对应关系。
            """
        )
        csv_text = st.text_area(
            "粘贴 CSV 文本",
            height=220,
            placeholder=(
                "基金名称,持有金额(元),占比(%),昨日收益(元),持有收益(元),持有收益率(%),累计收益(元),备注\n"
                "东方人工智能主题混合C,201579.85,12.34,+106.02,-30123.45,-13.22,1357.80,定投"
            ),
            key=f"ocr_hold_csv_{account}",
        )
        pending = st.session_state.get(map_key)
        if pending:
            st.warning(pending.get("error") or "未能自动识别列名，请指定每列对应的字段。")
            options = ["（不使用）"] + list(pending.get("columns") or [])
            guess = pending.get("guess") or {}
            chosen_map = {}
            cols = st.columns(3)
            for index, (field, label) in enumerate(CSV_MAP_FIELDS):
                default = guess.get(field) if guess.get(field) in options else "（不使用）"
                default_idx = options.index(default) if default in options else 0
                picked = cols[index % 3].selectbox(
                    label,
                    options=options,
                    index=default_idx,
                    key=f"ocr_csv_colmap_{account}_{field}",
                )
                if picked and picked != "（不使用）":
                    chosen_map[field] = picked
            if st.button("按指定列解析", type="primary", key=f"ocr_csv_colmap_apply_{account}"):
                extracted, error, meta = parse_holdings_csv(csv_text, column_map=chosen_map)
                if error:
                    st.error(error)
                else:
                    _load_holdings_preview(extracted, account)
                    st.rerun()
        if st.button("解析并导入", type="primary", key=f"ocr_hold_csv_preview_{account}"):
            extracted, error, meta = parse_holdings_csv(csv_text)
            if error and meta and meta.get("need_column_map"):
                st.session_state[map_key] = {
                    "error": error,
                    "columns": meta.get("columns") or [],
                    "guess": meta.get("guess") or {},
                }
                st.rerun()
            elif error:
                st.error(error)
                st.session_state.ocr_holdings_preview = []
                st.session_state.ocr_holdings_original = []
            else:
                _load_holdings_preview(extracted, account)
                st.rerun()


def render_ocr_import_tab(account):
    try:
        from src.ocr.importer import apply_holdings_import, apply_transaction_import, has_undo_snapshot, restore_undo_snapshot
    except Exception as exc:
        st.error(f"截图导入模块导入失败：{exc}")
        st.caption("可运行 `python scripts/diagnose_dashboard.py` 检查语法与依赖。Python 3.14 请使用 RapidOCR，不必安装 PaddleOCR。")
        return

    st.caption("从支付宝持仓或交易记录截图识别数据。识别结果需你核对并确认后才会写入当前账户，不会自动覆盖。")
    undo_col, _ = st.columns([1, 3])
    with undo_col:
        if st.button("↩️ 撤销上次导入", disabled=not has_undo_snapshot(), key=f"ocr_undo_{account}"):
            ok, message = restore_undo_snapshot()
            if ok:
                st.cache_data.clear()
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    st.markdown("#### 持仓导入")
    st.caption("💡 如果基金名称未自动匹配成功，请从下拉列表中选择或手动输入代码。您的选择将被记录，下次自动匹配。")
    with st.expander("截图要求（避免字段错位）", expanded=False):
        st.markdown(
            "请上传清晰、尽量水平且完整的支付宝持仓表格截图，建议包含这些表头："
            "**基金名称、持有金额、昨日收益、持有收益、累计收益、占比、持有收益率**。"
            "表头齐全时系统会按列位置对齐；若识别不到表头，会按金额大小和正负号/百分号做启发式分配，请务必在预览里核对。"
        )
    hold_file = st.file_uploader("上传持仓截图", type=["png", "jpg", "jpeg"], key=f"ocr_hold_file_{account}")
    if st.button("识别并预览", key=f"ocr_hold_preview_{account}"):
        rows, error = _recognize_image(hold_file, "holdings")
        if error:
            st.error(error)
            st.session_state.ocr_holdings_preview = []
            st.session_state.ocr_holdings_original = []
        else:
            st.session_state.ocr_holdings_preview = rows
            st.session_state.ocr_holdings_original = [dict(row) for row in rows]
            st.session_state.ocr_hold_editor_nonce = int(st.session_state.get("ocr_hold_editor_nonce") or 0) + 1

    _render_csv_paste_backup(account)

    preview = st.session_state.get("ocr_holdings_preview") or []
    if preview:
        from src.ocr.importer import REMAP_FIELDS, remap_preview_columns, validate_holdings_preview

        st.markdown("##### 预览与修正")
        if any(row.get("needs_review") or row.get("parse_mode") == "heuristic" for row in preview):
            st.warning("部分行未能按表头对齐，已启用启发式分配。字段数可能不完整，请手动修正后再导入。")
        if any(_match_tier_of(row) != "high" for row in preview):
            st.warning("有基金未高置信匹配，请在下方「基金代码匹配」中确认或搜索。")
        if any(row.get("rate_diff") is not None and abs(float(row.get("rate_diff") or 0)) > 0.3 for row in preview):
            st.info("截图中的持有收益率可能与系统按「市值-成本」计算的结果略有差异，导入后以系统计算为准。")

        problem_rows = [row for row in preview if row.get("issues") or row.get("needs_review")]
        if problem_rows:
            st.error("以下行存在匹配失败或数据异常，可取消勾选「导入」跳过，或在表格中改完再确认。")
            issue_view = pd.DataFrame(
                {
                    "基金名称": [row.get("fund_name") for row in problem_rows],
                    "持有金额": [row.get("hold_amount") for row in problem_rows],
                    "昨日收益": [row.get("yesterday_profit") for row in problem_rows],
                    "问题": [row.get("issue_text") or "；".join(row.get("issues") or []) or "待核对" for row in problem_rows],
                }
            )

            def _mark_issue_amount(value):
                number = _parse_signed_display(value)
                if number is not None and number < 0:
                    return "background-color: #fecaca; color: #991b1b;"
                return "background-color: #fee2e2;"

            styled = issue_view.style
            styled = _styler_applymap(styled, _mark_issue_amount, ["持有金额"])
            st.dataframe(styled, width="stretch", hide_index=True)

        st.caption("若整列字段对错了，把「当前列」改成它实际代表的字段，然后点应用。")
        remap_cols = st.columns(len(REMAP_FIELDS))
        field_labels = [label for _, label in REMAP_FIELDS]
        key_by_label = {label: key for key, label in REMAP_FIELDS}
        mapping = {}
        for index, (field_key, label) in enumerate(REMAP_FIELDS):
            chosen = remap_cols[index].selectbox(
                f"{label} ←",
                options=field_labels,
                index=index,
                key=f"ocr_hold_remap_{account}_{field_key}",
            )
            mapping[field_key] = key_by_label[chosen]
        if st.button("应用列映射", key=f"ocr_hold_remap_apply_{account}"):
            source_rows = st.session_state.get("ocr_holdings_original") or preview
            st.session_state.ocr_holdings_preview = remap_preview_columns(source_rows, mapping)
            st.session_state.ocr_hold_editor_nonce = int(st.session_state.get("ocr_hold_editor_nonce") or 0) + 1
            st.rerun()

        with st.expander("逐行指定字段（适合个别行错位）", expanded=any(row.get("needs_review") for row in preview)):
            st.caption("每一行可以把当前数值重新指定到「持有金额 / 昨日收益」等字段。改完后点「应用本行映射」。")
            for index, row in enumerate(preview):
                title = str(row.get("fund_name") or f"第 {index + 1} 行")
                issues = row.get("issue_text") or ""
                if issues:
                    title = f"⚠️ {title} — {issues}"
                with st.expander(title, expanded=bool(row.get("needs_review"))):
                    st.caption(str(row.get("raw_text") or "")[:240])
                    row_map = {}
                    row_cols = st.columns(len(REMAP_FIELDS))
                    for col_idx, (field_key, label) in enumerate(REMAP_FIELDS):
                        chosen = row_cols[col_idx].selectbox(
                            label,
                            options=field_labels,
                            index=col_idx,
                            key=f"ocr_row_map_{account}_{index}_{field_key}",
                        )
                        row_map[field_key] = key_by_label[chosen]
                    if st.button("应用本行映射", key=f"ocr_row_map_apply_{account}_{index}"):
                        updated = list(preview)
                        updated[index] = remap_preview_columns([row], row_map)[0]
                        st.session_state.ocr_holdings_preview = updated
                        st.session_state.ocr_hold_editor_nonce = int(st.session_state.get("ocr_hold_editor_nonce") or 0) + 1
                        st.rerun()

        editor_rows = []
        for row in preview:
            tier = _match_tier_of(row)
            score = float(row.get("match_score") or 0.0)
            if tier == "high":
                match_text = f"✅ {score:.0%}"
            elif tier == "partial":
                match_text = f"⚠️ {score:.0%}"
            else:
                match_text = "❌ 未匹配"
            block_import = any(
                token in (row.get("issue_text") or "")
                for token in ("基金代码未找到", "持有金额为负数", "持有金额格式错误", "持有金额缺失", "持有金额无效")
            )
            editor_rows.append(
                {
                    "导入": bool(row.get("include", True)) and not block_import,
                    "基金名称": row.get("fund_name") or "",
                    "基金代码": row.get("fund_code") or "",
                    "匹配": match_text,
                    "问题": row.get("issue_text") or "",
                    "最新净值": row.get("nav"),
                    "持有金额": row.get("hold_amount"),
                    "计算份额": row.get("shares"),
                    "计算成本": row.get("cost"),
                    "昨日收益": row.get("yesterday_profit"),
                    "持有收益": row.get("hold_profit"),
                    "累计收益": row.get("cumulative_profit"),
                    "占比": row.get("weight_pct"),
                    "持有收益率": row.get("hold_return_rate"),
                    "备注": row.get("remark") or "",
                    "解析": row.get("parse_mode") or "",
                }
            )
        edited = st.data_editor(
            pd.DataFrame(editor_rows),
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            key=f"ocr_hold_editor_{account}_{st.session_state.get('ocr_hold_editor_nonce') or 0}",
            column_config={
                "导入": st.column_config.CheckboxColumn("导入"),
                "基金代码": st.column_config.TextColumn("基金代码"),
                "匹配": st.column_config.TextColumn("匹配", disabled=True),
                "问题": st.column_config.TextColumn("问题", disabled=True),
                "最新净值": st.column_config.NumberColumn("最新净值", format="%.4f", min_value=0.0),
                "持有金额": st.column_config.NumberColumn("持有金额", format="%.2f"),
                "计算份额": st.column_config.NumberColumn("计算份额", format="%.2f", min_value=0.0),
                "计算成本": st.column_config.NumberColumn("计算成本", format="%.2f"),
                "昨日收益": st.column_config.NumberColumn("昨日收益", format="%.2f"),
                "持有收益": st.column_config.NumberColumn("持有收益", format="%.2f"),
                "累计收益": st.column_config.NumberColumn("累计收益", format="%.2f"),
                "占比": st.column_config.NumberColumn("占比", format="%.2f"),
                "持有收益率": st.column_config.NumberColumn("持有收益率", format="%.2f"),
            },
        )
        st.markdown("##### 基金代码匹配")
        st.caption("💡 如果基金名称未自动匹配成功，请从下拉列表中选择或手动输入代码。您的选择将被记录，下次自动匹配。")
        code_overrides = {}
        for index, row in enumerate(preview):
            code_overrides[str(row.get("fund_name") or "")] = _render_match_selector(row, f"ocr_hold_{account}", index)

        check_rows = []
        for _, row in edited.iterrows():
            check_rows.append(
                {
                    "include": bool(row.get("导入")),
                    "fund_name": row.get("基金名称"),
                    "fund_code": str(row.get("基金代码") or "").strip(),
                    "hold_amount": row.get("持有金额"),
                    "yesterday_profit": row.get("昨日收益"),
                    "hold_profit": row.get("持有收益"),
                    "weight_pct": row.get("占比"),
                    "issues": [row.get("问题")] if row.get("问题") else [],
                }
            )
        for message in validate_holdings_preview(check_rows):
            st.warning(message)

        unmatched = [row for _, row in edited.iterrows() if bool(row.get("导入")) and not str(row.get("基金代码") or "").strip()]
        if unmatched:
            st.warning("仍有基金未选择代码，导入时将跳过这些行。")

        if st.button("确认导入", type="primary", key=f"ocr_hold_commit_{account}"):
            confirmed = []
            auto_skipped = 0
            for _, row in edited.iterrows():
                try:
                    nav = float(row.get("最新净值") or 0.0)
                except (TypeError, ValueError):
                    nav = 0.0
                try:
                    hold_amount = float(row.get("持有金额"))
                except (TypeError, ValueError):
                    hold_amount = None
                try:
                    hold_profit = float(row.get("持有收益") or 0.0)
                except (TypeError, ValueError):
                    hold_profit = 0.0
                try:
                    shares = float(row.get("计算份额") or 0.0)
                except (TypeError, ValueError):
                    shares = 0.0
                if shares <= 0 and nav > 0 and hold_amount and hold_amount > 0:
                    shares = hold_amount / nav
                try:
                    cost = float(row.get("计算成本") or 0.0)
                except (TypeError, ValueError):
                    cost = (hold_amount or 0.0) - hold_profit
                code = str(row.get("基金代码") or "").strip() or str(code_overrides.get(str(row.get("基金名称") or ""), "") or "").strip()
                include = bool(row.get("导入")) and bool(code) and (nav > 0 or shares > 0)
                if include and (hold_amount is None or hold_amount < 0):
                    include = False
                    auto_skipped += 1
                confirmed.append(
                    {
                        "include": include,
                        "fund_code": code,
                        "fund_name": str(row.get("基金名称") or ""),
                        "shares": shares,
                        "hold_amount": hold_amount,
                        "cost": cost,
                        "hold_profit": hold_profit,
                        "cumulative_profit": row.get("累计收益"),
                        "yesterday_profit": row.get("昨日收益"),
                        "weight_pct": row.get("占比"),
                        "remark": str(row.get("备注") or ""),
                    }
                )
            from src.ocr.fund_matcher import remember_user_mappings

            remember_user_mappings(
                [
                    (item["fund_name"], item["fund_code"], "")
                    for item in confirmed
                    if item.get("include") and item.get("fund_code") and item.get("fund_name")
                ]
            )
            ok, result = apply_holdings_import(confirmed)
            if ok:
                st.cache_data.clear()
                st.session_state.ocr_holdings_preview = []
                st.session_state.ocr_holdings_original = []
                extra = f"；另有 {auto_skipped} 行因持有金额异常已跳过" if auto_skipped else ""
                st.success(
                    f"已导入：新增 {result['added']} 只，更新 {result['updated']} 只，跳过 {result['skipped']} 只{extra}。"
                )
                st.rerun()

    st.markdown("#### 调仓记录导入")
    tx_file = st.file_uploader("上传交易记录截图", type=["png", "jpg", "jpeg"], key=f"ocr_tx_file_{account}")
    if st.button("识别并预览", key=f"ocr_tx_preview_{account}"):
        rows, error = _recognize_image(tx_file, "transactions")
        if error:
            st.error(error)
            st.session_state.ocr_tx_preview = []
        else:
            st.session_state.ocr_tx_preview = rows
    preview = st.session_state.get("ocr_tx_preview") or []
    if preview:
        view = pd.DataFrame(
            {
                "操作": [row.get("action") for row in preview],
                "基金名称": [row.get("fund_name") for row in preview],
                "匹配代码": [row.get("fund_code") or "待选择" for row in preview],
                "份额": [row.get("shares") for row in preview],
                "日期": [row.get("date") or "-" for row in preview],
                "置信度": [row.get("confidence") for row in preview],
            }
        )
        st.dataframe(_style_ocr_preview(view, preview), width="stretch", hide_index=True)
        if any(_match_tier_of(row) != "high" for row in preview):
            st.warning("有交易未高置信匹配，请在下方确认代码后再导入。")
        st.caption("💡 如果基金名称未自动匹配成功，请从下拉列表中选择或手动输入代码。您的选择将被记录，下次自动匹配。")
        confirmed = []
        for index, row in enumerate(preview):
            c1, c2, c3 = st.columns([0.5, 1.2, 2.8])
            include = c1.checkbox("导入", value=bool(row.get("include", True)), key=f"ocr_tx_inc_{account}_{index}")
            action = c2.selectbox(
                "操作",
                ["买入", "卖出"],
                index=0 if row.get("action") != "卖出" else 1,
                key=f"ocr_tx_act_{account}_{index}",
                label_visibility="collapsed",
            )
            c3.write(row.get("fund_name") or "-")
            code = _render_match_selector(row, f"ocr_tx_{account}", index)
            s1, s2 = st.columns(2)
            shares = s1.number_input("交易份额", min_value=0.0, value=float(row.get("shares") or 0.0), step=0.01, key=f"ocr_tx_shares_{account}_{index}")
            date_text = s2.text_input("交易日期", value=str(row.get("date") or ""), key=f"ocr_tx_date_{account}_{index}")
            confirmed.append(
                {
                    "include": include,
                    "fund_code": code,
                    "fund_name": row.get("fund_name") or "",
                    "action": action,
                    "shares": shares,
                    "date": date_text,
                }
            )
        if st.button("确认导入交易", type="primary", key=f"ocr_tx_commit_{account}"):
            from src.ocr.fund_matcher import remember_user_mappings

            remember_user_mappings(
                [
                    (item["fund_name"], item["fund_code"], "")
                    for item in confirmed
                    if item.get("include") and item.get("fund_code") and item.get("fund_name")
                ]
            )
            ok, result = apply_transaction_import(confirmed)
            if ok:
                st.cache_data.clear()
                st.session_state.ocr_tx_preview = []
                extra = "；".join(result.get("warnings") or [])
                st.success(f"已导入 {result['applied']} 条交易，跳过 {result['skipped']} 条。" + (f" {extra}" if extra else ""))
                st.rerun()


def _fmt_bt_pct(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _fmt_bt_num(value, digits=2):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def render_backtest_tab(pool, account):
    try:
        from src.backtest.engine import parse_benchmark_csv, run_backtest
        from src.backtest.visualizer import plot_drawdown_curve, plot_equity_curve
    except Exception as exc:
        st.error(f"回测模块导入失败：{exc}")
        st.caption("请确认已安装 `backtrader`，或运行 `python scripts/diagnose_dashboard.py`。")
        return

    st.caption("用历史净值回放进攻/保守调仓规则，并尽量与沪深300对比。基准获取失败时仍会完成策略回测。结果仅供研究，不会改写当前持仓。")
    with st.expander("使用说明与回测局限", expanded=False):
        st.markdown(
            """
**指标含义**
- **总收益率**：回测结束净值相对期初资金的涨跌幅。
- **年化收益率**：把总收益折算到一年，便于比较不同长度的区间。
- **最大回撤**：净值相对历史高点的最大跌幅，衡量最差体验。
- **夏普比率**：超额收益（相对约 1.5% 无风险利率）除以波动，越高越好。
- **卡玛比率**：年化收益 / 最大回撤绝对值。
- **超额收益**：策略总收益减去沪深300总收益。

**策略怎么调仓**
- 在每月首个交易日（或每周五）用当日及之前 60 日净值打分，再套用与实盘相同的总仓位、赛道、单基上限和熔断规则。
- **进攻型**仓位更高，并启用赛道精简（每赛道保留得分前 N）；**保守型**降低仓位与集中度。
- 申购费按设定费率扣除；赎回费按持有天数：**<7 天 1.5%、<1 年 0.5%、≥1 年 0.25%**。

**局限性（请务必阅读）**
- 回测默认按当日净值成交，忽略基金申购确认 T+1、赎回到账延迟和限额。
- 得分与调仓只用已公布净值，但真实交易无法预知当日净值。
- 未模拟大额赎回冲击、暂停申赎、分红再投资细节和销售服务费差异。
- 历史表现不能代表未来，样本外或换一组基金可能完全不同。
            """
        )

    if pool is None or pool.empty or "基金代码" not in pool.columns:
        st.info("当前账户基金池为空，请先在持仓或截图导入中加入基金。")
        return

    work = pool.copy()
    work["基金代码"] = work["基金代码"].astype(str).str.strip()
    work = work[work["基金代码"] != ""].copy()
    if work.empty:
        st.info("当前账户基金池为空，请先加入基金。")
        return
    if "基金名称" not in work.columns:
        work["基金名称"] = ""
    if "赛道归类" not in work.columns:
        work["赛道归类"] = "其他"
    shares = pd.to_numeric(work.get("持有份额"), errors="coerce").fillna(0)
    values = pd.to_numeric(work.get("持仓市值"), errors="coerce").fillna(0)
    held_mask = (shares > 0) & (values > 0)
    labels = (work["基金代码"] + "  " + work["基金名称"].fillna("")).tolist()
    code_of = dict(zip(labels, work["基金代码"].tolist()))
    held_labels = [label for label, held in zip(labels, held_mask.tolist()) if held]
    default_labels = held_labels or labels[: min(8, len(labels))]

    today = datetime.now().date()
    default_start = (pd.Timestamp(today) - pd.DateOffset(years=3)).date()
    c1, c2, c3 = st.columns(3)
    selected_labels = c1.multiselect(
        "回测基金",
        options=labels,
        default=default_labels,
        help="建议选择同一账户内、净值较完整的基金。数量过多会增加耗时。",
    )
    start_date = c2.date_input("开始日期", value=default_start, max_value=today, key=f"bt_start_{account}")
    end_date = c3.date_input("结束日期", value=today, max_value=today, key=f"bt_end_{account}")

    r1, r2, r3, r4 = st.columns(4)
    mode_label = r1.radio("策略模式", ["进攻型", "保守型", "自动识别"], horizontal=True, key=f"bt_mode_{account}")
    freq_label = r2.selectbox("调仓频率", ["月度调仓", "周调仓"], key=f"bt_freq_{account}")
    initial_cash = r3.number_input("初始资金（元）", min_value=1000.0, value=100000.0, step=1000.0, key=f"bt_cash_{account}")
    buy_fee_pct = r4.number_input("申购费率（%）", min_value=0.0, max_value=1.5, value=0.15, step=0.05, key=f"bt_fee_{account}")

    mode_key = {"进攻型": "aggressive", "保守型": "defensive", "自动识别": "auto"}[mode_label]
    freq_key = "weekly" if freq_label == "周调仓" else "monthly"

    uploaded_bench = st.file_uploader(
        "手动上传基准数据（可选 CSV）",
        type=["csv"],
        key=f"bt_bench_upload_{account}",
        help="上传后优先使用该文件作为沪深300基准，不再自动联网获取。",
    )
    st.caption(
        "自动获取失败时可手动上传沪深300历史行情。"
        "导出示例：打开东方财富网 → 搜索「沪深300」或代码 000300 → 进入行情/K线页面 → 导出历史数据为 CSV。"
        "文件格式：第一列为日期（YYYY-MM-DD），第二列为收盘价（列名可为 close 或 收盘价）。"
    )
    upload_error_key = f"bt_bench_upload_error_{account}"
    if uploaded_bench is None:
        st.session_state.pop(upload_error_key, None)
    else:
        preview_df, preview_err = parse_benchmark_csv(uploaded_bench)
        if preview_err:
            st.session_state[upload_error_key] = preview_err
            st.error(preview_err)
            st.info("请修正 CSV 后重新上传。常见问题：缺少日期列、缺少收盘价列、日期不是 YYYY-MM-DD。")
        else:
            st.session_state.pop(upload_error_key, None)
            st.caption(f"已识别基准数据 {len(preview_df)} 条，区间 {preview_df['date'].min().date()} ~ {preview_df['date'].max().date()}。")

    run_clicked = st.button("🚀 运行回测", type="primary", key=f"bt_run_{account}")
    if run_clicked:
        selected_codes = [code_of[item] for item in selected_labels if item in code_of]
        if not selected_codes:
            st.error("请至少选择一只基金。")
        elif start_date >= end_date:
            st.error("结束日期必须晚于开始日期。")
        else:
            benchmark_df = None
            csv_error = None
            if uploaded_bench is not None:
                benchmark_df, csv_error = parse_benchmark_csv(uploaded_bench)
            if csv_error:
                st.error(csv_error)
                st.info("基准 CSV 格式不正确，已停止本次回测。请修正后重新上传，或清空文件改用自动获取。")
            else:
                sector_map = dict(zip(work["基金代码"].astype(str), work["赛道归类"].fillna("其他").astype(str)))
                name_map = dict(zip(work["基金代码"].astype(str), work["基金名称"].fillna("").astype(str)))
                status = st.empty()

                def _on_progress(message):
                    status.caption(str(message))

                try:
                    with st.spinner("回测进行中，请稍候..."):
                        result = run_backtest(
                            selected_codes,
                            start_date,
                            end_date,
                            mode=mode_key,
                            frequency=freq_key,
                            initial_cash=float(initial_cash),
                            buy_fee=float(buy_fee_pct) / 100.0,
                            sector_map=sector_map,
                            name_map=name_map,
                            allow_network=True,
                            progress_callback=_on_progress,
                            benchmark_df=benchmark_df,
                        )
                    st.session_state[f"backtest_result_{account}"] = result
                    status.empty()
                    if result.get("benchmark_available"):
                        st.success("回测完成。")
                    else:
                        st.success("回测完成（未包含沪深300基准对比）。")
                except Exception as exc:
                    status.empty()
                    st.error(f"回测失败：{exc}")

    result = st.session_state.get(f"backtest_result_{account}")
    if not result:
        st.info("配置参数后点击「运行回测」。首次若缺本地净值，会自动联网拉取并缓存。")
        return

    for warning in result.get("warnings") or []:
        st.warning(warning)
    if result.get("benchmark_available") is False:
        st.info("本次只展示策略自身净值与绩效指标；夏普比率、最大回撤等不受基准缺失影响。")

    metrics = result.get("metrics") or {}
    signed_cards = [
        ("总收益率", _fmt_bt_pct(metrics.get("total_return")), metrics.get("total_return")),
        ("年化收益率", _fmt_bt_pct(metrics.get("annual_return")), metrics.get("annual_return")),
        ("最大回撤", _fmt_bt_pct(metrics.get("max_drawdown")), metrics.get("max_drawdown")),
    ]
    if result.get("benchmark_available"):
        signed_cards.extend(
            [
                ("沪深300收益", _fmt_bt_pct(metrics.get("benchmark_return")), metrics.get("benchmark_return")),
                ("超额收益", _fmt_bt_pct(metrics.get("excess_return")), metrics.get("excess_return")),
            ]
        )
    signed_cards.append(("夏普比率", _fmt_bt_num(metrics.get("sharpe")), None))
    cols = st.columns(len(signed_cards))
    for col, (label, text, number) in zip(cols, signed_cards):
        with col:
            css = "metric-value-flat"
            if number is not None and not (isinstance(number, float) and pd.isna(number)):
                if float(number) > 0:
                    css = "metric-value-profit"
                elif float(number) < 0:
                    css = "metric-value-loss"
            st.markdown(
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value {css}' style='font-size:1.35rem;'>{text}</div>",
                unsafe_allow_html=True,
            )

    extra1, extra2, extra3, extra4 = st.columns(4)
    extra1.caption(f"卡玛比率：{_fmt_bt_num(metrics.get('calmar'))}")
    extra2.caption(f"年化波动：{_fmt_bt_pct(metrics.get('volatility'))}")
    extra3.caption(f"交易笔数：{int(metrics.get('trade_count') or 0)}（买 {int(metrics.get('buy_count') or 0)} / 卖 {int(metrics.get('sell_count') or 0)}）")
    extra4.caption(f"费用合计：{_fmt_bt_num(metrics.get('fee_total'))} 元")

    equity = result.get("equity")
    if equity is not None and not equity.empty:
        st.plotly_chart(plot_equity_curve(equity), width="stretch")
        st.plotly_chart(plot_drawdown_curve(equity), width="stretch")

    trades = result.get("trades")
    st.markdown("##### 交易记录")
    if trades is None or trades.empty:
        st.caption("区间内没有触发调仓成交。")
    else:
        trade_view = trades.copy()
        if "date" in trade_view.columns:
            trade_view["date"] = pd.to_datetime(trade_view["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for column in ("份额", "净值", "金额", "费用", "费率"):
            if column in trade_view.columns:
                trade_view[column] = pd.to_numeric(trade_view[column], errors="coerce")
        st.dataframe(
            trade_view.rename(
                columns={
                    "date": "日期",
                    "方向": "方向",
                    "费率": "费率",
                    "持有天数": "持有天数",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    holdings = result.get("holdings")
    st.markdown("##### 每日持仓明细")
    if holdings is None or holdings.empty:
        st.caption("没有持仓快照。")
        return
    hold_view = holdings.copy()
    if "date" in hold_view.columns:
        hold_view["日期"] = pd.to_datetime(hold_view["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        hold_view = hold_view.drop(columns=["date"])
    if "仓位占比" in hold_view.columns:
        hold_view["仓位占比"] = pd.to_numeric(hold_view["仓位占比"], errors="coerce")
    st.dataframe(hold_view, width="stretch", hide_index=True, height=360)
    st.caption(
        f"区间 {result.get('start')} ~ {result.get('end')}，"
        f"模式 { {'aggressive': '进攻型', 'defensive': '保守型', 'auto': '自动识别'}.get(result.get('mode'), result.get('mode')) }，"
        f"基金 {len(result.get('funds') or [])} 只。"
    )


def render_account_sidebar(current_account, pool=None):
    st.markdown(f"**📁 当前账户：{current_account}**")
    total, held, watch = pool_counts(pool)
    st.caption(f"📊 基金池：{total} 只（持仓 {held} 只，观察 {watch} 只）")
    accounts = list_accounts()
    if current_account not in accounts:
        accounts = [current_account] + accounts
    chosen = st.selectbox(
        "切换账户",
        accounts,
        index=accounts.index(current_account),
        help="切换后只读本地净值重算市值，不会联网。点「刷新数据」才会拉取最新净值。",
    )
    if chosen != current_account:
        apply_account_switch(chosen)
    st.caption("切换账户仅读取 data/raw 本地净值，不会调用 AkShare。")

    add_col, del_col = st.columns(2)
    with add_col:
        if st.button("➕ 新增账户", width="stretch", key="btn_add_account"):
            st.session_state.show_new_account = True
            st.session_state.confirm_delete_account = False
    with del_col:
        delete_ok = can_delete_account(current_account)
        if st.button("🗑️ 删除当前账户", width="stretch", disabled=not delete_ok, key="btn_del_account"):
            st.session_state.confirm_delete_account = True
            st.session_state.show_new_account = False
        if not delete_ok:
            st.caption("默认账户或仅剩一个账户时不可删除。")

    if st.session_state.get("show_new_account"):
        new_name = st.text_input("新账户名", key="new_account_name", placeholder="仅中文、英文、数字和下划线")
        confirm, cancel = st.columns(2)
        with confirm:
            if st.button("确认创建", width="stretch", key="btn_confirm_create_account"):
                ok, result = create_account(new_name)
                if ok:
                    apply_account_switch(result)
                else:
                    st.error(result)
        with cancel:
            if st.button("取消", width="stretch", key="btn_cancel_create_account"):
                st.session_state.show_new_account = False
                st.rerun()

    if st.session_state.get("confirm_delete_account"):
        st.warning(f"确认删除账户「{current_account}」？该账户的持仓和交易日志将无法恢复。")
        yes, no = st.columns(2)
        with yes:
            if st.button("确认删除", width="stretch", key="btn_confirm_delete_account"):
                ok, result = delete_account(current_account)
                if ok:
                    apply_account_switch(result)
                else:
                    st.error(result)
                    st.session_state.confirm_delete_account = False
        with no:
            if st.button("我再想想", width="stretch", key="btn_cancel_delete_account"):
                st.session_state.confirm_delete_account = False
                st.rerun()
    st.markdown("---")


def main():
    try:
        _run_dashboard()
    except Exception as exc:
        st.error("看板加载失败。请重启 Streamlit 后刷新浏览器；仍失败可运行 scripts/diagnose_dashboard.py。")
        st.exception(exc)


def _run_dashboard():
    _inject_css()
    today = datetime.now()
    if "signal_nonce" not in st.session_state:
        st.session_state.signal_nonce = 0
    current_account = sync_current_account()
    pool = cached_load_holdings(current_account)
    holdings = active_holdings_only(pool)
    board = merge_holdings_scores(holdings)

    in_session = is_trading_session()
    if in_session:
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=int(REFRESH_SECONDS) * 1000, key="intraday_auto_refresh")
        except Exception:
            pass

    bar_left, bar_right = st.columns([5.2, 1.1])
    with bar_right:
        force_realtime = st.button("立即刷新", width="stretch", key="intraday_refresh_now")
    realtime_payload = load_realtime_estimates()
    should_fetch = bool(force_realtime) or (in_session and is_cache_stale(realtime_payload))
    if should_fetch:
        reason = "immediate" if force_realtime else "auto"
        try:
            with st.spinner("正在更新实时估值（不拉取历史净值）..."):
                realtime_payload = refresh_realtime_quotes(holdings, reason=reason)
        except Exception as exc:
            st.error(f"实时估值刷新失败：{exc}")
            log_monitor("日监控", f"{reason} 失败 {exc}")
        else:
            if force_realtime:
                st.rerun()
    with bar_left:
        render_intraday_refresh_bar(in_session, realtime_payload, force_realtime)

    from src.strategy_layer.intraday_monitor import build_monitor_bundle

    monitor_df, sector_df, monitor_meta = build_monitor_bundle(board, realtime_payload)
    live_emergency = bool(((monitor_meta or {}).get("emergency") or {}).get("triggered"))
    live_drawdown = float((monitor_meta or {}).get("account_drawdown") or 0.0)

    st.markdown(
        f'<div class="hero-title">📊 {current_account}的基金组合</div>',
        unsafe_allow_html=True,
    )

    configured_freq = normalize_operation_frequency(OPERATION_FREQUENCY)
    freq_label = LABEL_BY_FREQ.get(configured_freq, "月度调仓")
    if freq_label not in FREQ_OPTIONS:
        freq_label = FREQ_OPTIONS[0]
    if configured_freq == "weekly":
        st.markdown(
            '<div class="hero-sub">周调仓：每周五生成正式报告；盘中触发紧急信号可当日执行。自动刷新只更新估值，不会自动生成调仓建议。</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="hero-sub">月度调仓：建议每月第一个交易日先刷新净值，再生成调仓报告。</div>',
            unsafe_allow_html=True,
        )

    with st.sidebar:
        render_account_sidebar(current_account, pool)
        st.markdown("### 控制台")
        st.write(f"📅 {today.strftime('%Y-%m-%d')} {WEEKDAYS[today.weekday()]}")
        if live_emergency:
            st.markdown(f'<div class="alert-banner">{EMERGENCY_BANNER}</div>', unsafe_allow_html=True)
        chosen_freq_label = st.selectbox(
            "运行模式",
            FREQ_OPTIONS,
            index=FREQ_OPTIONS.index(freq_label),
            help="月度调仓执行完整风控再平衡；周调仓按盘中估算净值扫描紧急减仓，周五未触发则生成正式报告。",
        )
        selected_freq = FREQ_BY_LABEL[chosen_freq_label]
        if selected_freq != configured_freq:
            persist_operation_frequency(selected_freq)
            configured_freq = selected_freq
        if selected_freq == "weekly":
            st.caption("每周五为正式调仓日。紧急信号不受周五限制，当日即可生成建议。自动刷新不会生成报告。")
        else:
            st.caption("建议在每月第一个交易日运行一次。")

        configured_mode = normalize_strategy_mode(STRATEGY_MODE)
        current_label = LABEL_BY_MODE.get(configured_mode, "自动识别")
        chosen_label = st.radio(
            "策略模式",
            MODE_RADIO_OPTIONS,
            index=MODE_RADIO_OPTIONS.index(current_label),
            horizontal=True,
            help="防御型收紧仓位上限；进攻型使用较高上限；自动识别按沪深300与120日均线切换。",
        )
        selected_mode = MODE_BY_LABEL[chosen_label]
        if selected_mode != configured_mode:
            persist_strategy_mode(selected_mode)

        refresh_clicked = st.button("🔄 刷新数据", width="stretch", type="primary")
        report_clicked = st.button("📈 生成调仓报告", width="stretch")
        st.caption("「刷新数据」会拉取历史净值；顶部「立即刷新」和自动刷新只更新实时估值。")
        try:
            sector_keep_default = max(int(SECTOR_TOP_N), 1)
        except (TypeError, ValueError):
            sector_keep_default = 3
        sector_keep_n = st.number_input(
            "每个赛道保留数量",
            min_value=1,
            max_value=10,
            value=min(max(sector_keep_default, 1), 10),
            step=1,
            help="每个赛道只保留得分最高的 N 只。精简赎回的资金转入货币基金，下月再分配。",
        )
        top_n = st.number_input("候选数量", min_value=1, max_value=10, value=3, step=1)

        meta = load_meta()
        last_refresh = meta.get("last_refresh") or file_time_text(Path(PROJECT_ROOT) / "data" / "raw")
        last_report = meta.get("last_report") or file_time_text(signal_file())
        st.markdown("---")
        st.write(f"上次数据更新：{last_refresh}")
        st.write(f"上次实时估值：{meta.get('last_realtime') or '暂无（点击刷新）'}")
        st.write(f"上次报告生成：{last_report}")

        if refresh_clicked:
            progress = st.progress(0, text="准备拉取最新净值...")
            status = st.empty()

            def _on_progress(index, total, fund_code):
                ratio = min(index / max(total, 1), 1.0)
                progress.progress(ratio, text=f"正在更新 {fund_code}（{index}/{total}）")
                status.caption(f"已完成 {index}/{total}")

            try:
                update_all_funds(progress_callback=_on_progress)
                progress.progress(1.0, text="净值更新完成，正在拉取实时估值...")
                status.caption("正在更新实时估值（实时估算）")
                fresh_holdings = active_holdings_only(load_current_holdings())
                refresh_realtime_quotes(fresh_holdings, reason="full_refresh")
                now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                progress.progress(1.0, text="净值与实时估值更新完成")
                save_meta({"last_refresh": now_text})
                st.cache_data.clear()
                st.success("数据刷新完成，已同步实时估值（实时估算）。")
                st.rerun()
            except Exception as exc:
                st.error(f"刷新失败：{exc}")

        if report_clicked:
            held_now = active_holdings_only(pool)
            if held_now is None or held_now.empty:
                st.warning("当前账户无持仓，无法生成调仓指令。")
            else:
                st.session_state.signal_nonce += 1
                try:
                    friday_monthly = False
                    if selected_freq == "weekly":
                        orders, signal_meta = cached_run_weekly_scan(
                            st.session_state.signal_nonce, current_account
                        )
                        if signal_meta.get("emergency_triggered"):
                            success_text = "紧急调仓建议已生成，请到「📋 周调仓」查看。请于今日 14:50 前完成操作。"
                        elif signal_meta.get("friday_no_emergency") or signal_meta.get("weekly_is_friday"):
                            orders, monthly_meta = cached_generate_trading_signal(
                                today.year,
                                today.month,
                                st.session_state.signal_nonce,
                                selected_mode,
                                int(sector_keep_n),
                                current_account,
                            )
                            friday_monthly = True
                            signal_meta = dict(monthly_meta)
                            signal_meta["operation_frequency"] = "weekly"
                            signal_meta["operation_frequency_banner"] = (
                                "📅 当前运行频率：周调仓 · 周五未触发紧急信号，已生成正式调仓报告"
                            )
                            signal_meta["emergency_triggered"] = False
                            signal_meta["friday_no_emergency"] = True
                            signal_meta["weekly_is_friday"] = True
                            success_text = "周五未触发紧急信号，已生成正式周调仓报告，请到「📋 周调仓」查看。"
                        else:
                            success_text = "未触发紧急调仓信号。正式调仓日为每周五，当前报告为维持持仓。请到「📋 周调仓」查看。"
                    else:
                        orders, signal_meta = cached_generate_trading_signal(
                            today.year,
                            today.month,
                            st.session_state.signal_nonce,
                            selected_mode,
                            int(sector_keep_n),
                            current_account,
                        )
                        success_text = "调仓报告已生成，请到「📋 周调仓」查看。减仓/赎回指令已写入交易日志。"
                    if selected_freq != "weekly" or friday_monthly:
                        append_trade_log(orders)
                    save_rebalance_orders(
                        current_account,
                        orders,
                        selected_freq,
                        extra={
                            "emergency_triggered": bool(signal_meta.get("emergency_triggered")),
                            "operation_deadline": signal_meta.get("operation_deadline", ""),
                        },
                    )
                    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_monitor("周调仓", f"用户生成报告 {report_time} 紧急={bool(signal_meta.get('emergency_triggered'))}")
                    save_meta(
                        {
                            "last_report": report_time,
                            "meltdown_triggered": signal_meta.get("meltdown_triggered", False),
                            "crash_filter_triggered": signal_meta.get("crash_filter_triggered", False),
                            "account_return": signal_meta.get("account_return", 0.0),
                            "strategy_mode_banner": signal_meta.get("strategy_mode_banner", ""),
                            "strategy_mode_requested": signal_meta.get("strategy_mode_requested"),
                            "strategy_mode_effective": signal_meta.get("strategy_mode_effective"),
                            "strategy_mode_fetch_failed": signal_meta.get("strategy_mode_fetch_failed", False),
                            "sector_top_n_banner": signal_meta.get("sector_top_n_banner", ""),
                            "sector_top_n_sectors": signal_meta.get("sector_top_n_sectors", 0),
                            "sector_top_n_funds": signal_meta.get("sector_top_n_funds", 0),
                            "sector_elite_note": signal_meta.get("sector_elite_note", ""),
                            "sector_elite_cash": signal_meta.get("sector_elite_cash", 0.0),
                            "operation_frequency": signal_meta.get("operation_frequency", selected_freq),
                            "operation_frequency_banner": signal_meta.get("operation_frequency_banner", ""),
                            "weekly_alert_count": signal_meta.get("weekly_alert_count", 0),
                            "weekly_changed_funds": signal_meta.get("weekly_changed_funds", 0),
                            "weekly_alerts": signal_meta.get("weekly_alerts", []),
                            "emergency_triggered": bool(signal_meta.get("emergency_triggered")),
                        }
                    )
                    st.cache_data.clear()
                    st.success(success_text)
                    st.rerun()
                except Exception as exc:
                    st.error(f"生成报告失败：{exc}")

    meta = load_meta()

    effective_mode = meta.get("strategy_mode_effective")
    if effective_mode not in {"defensive", "aggressive"}:
        effective_mode = selected_mode if selected_mode != "auto" else "defensive"
    profile = get_strategy_profile(effective_mode)
    equity_limit = profile["TOTAL_EQUITY_LIMIT"]
    retreat_limit = profile["RETREAT_LIMIT"]

    total_asset = float(board["持仓市值"].sum(min_count=1) or 0.0) if not board.empty else 0.0
    equity_ratio = 1.0 if total_asset > 0 else 0.0
    try:
        drawdown = cached_account_drawdown(board) if not board.empty else float(meta.get("account_return") or 0.0)
    except Exception as exc:
        st.warning(f"账户回撤计算失败，已跳过：{exc}")
        drawdown = float(meta.get("account_return") or 0.0)

    top_name, top_score = "-", None
    if not board.empty and "综合得分" in board.columns:
        numeric_score = pd.to_numeric(board["综合得分"], errors="coerce")
        if numeric_score.notna().any():
            top_idx = numeric_score.idxmax()
            top_name = str(board.at[top_idx, "基金名称"])
            top_score = board.at[top_idx, "综合得分"]

    render_freq_banner(meta, fallback_freq=selected_freq)
    render_mode_banner(meta)
    render_trim_banner(meta)

    warnings = []
    if live_emergency:
        warnings.append(EMERGENCY_BANNER)
    if live_drawdown < -0.18:
        warnings.append(f"⚠️ 熔断警告：账户估算回撤 {fmt_pct(live_drawdown)}，已突破 -18%。")
    if meta.get("meltdown_triggered"):
        warnings.append("⚠️ 熔断已触发：账户回撤低于阈值，请优先执行减仓。")
    if meta.get("crash_filter_triggered"):
        warnings.append("⚠️ 市场急跌过滤器生效：沪深300单月跌幅超过 12%，暂停新开仓。")
    if drawdown < retreat_limit:
        warnings.append(f"⚠️ 当前账户回撤 {fmt_pct(drawdown)}，已低于熔断线 {fmt_pct(retreat_limit)}。")
    for item in meta.get("weekly_alerts") or []:
        warnings.append(f"🚨 周调仓：{item}")
    for text in warnings:
        st.markdown(f'<div class="alert-banner">{text}</div>', unsafe_allow_html=True)

    if (monitor_meta or {}).get("degraded"):
        st.markdown(f'<div class="degrade-caption">{DEGRADE_CAPTION}</div>', unsafe_allow_html=True)

    render_realtime_pnl_card(calc_realtime_pnl(board, realtime_payload))
    cumulative = calc_cumulative_return(board)
    if cumulative.get("missing_cost"):
        st.warning("请先补充持仓成本，否则无法计算收益率")
    render_metric_cards(
        total_asset,
        equity_ratio,
        drawdown,
        top_name,
        top_score,
        equity_limit,
        retreat_limit,
        total_pnl=cumulative.get("pnl"),
        total_return=cumulative.get("rate"),
        missing_cost=cumulative.get("missing_cost"),
    )
    st.write("")

    tab_mon, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
        [
            "📡 日监控",
            "📋 持仓明细",
            "📋 周调仓",
            "🏁 赛道分组排名",
            "📉 净值走势对比",
            "📊 基金对比",
            "📂 赛道总览",
            "🏷️ 赛道管理",
            "🌟 买入候选",
            "📈 策略回测",
            "📸 截图导入",
        ]
    )
    with tab_mon:
        _safe_render(
            "日监控",
            lambda: render_daily_monitor_tab(monitor_df, sector_df, monitor_meta),
        )

    with tab1:
        _safe_render("持仓明细", lambda: render_holdings_tab(board, current_account, pool))

    with tab2:
        _safe_render(
            "周调仓",
            lambda: render_rebalance_orders_tab(
                current_account,
                has_holdings=not holdings.empty,
                live_emergency=live_emergency,
            ),
        )

    with tab3:
        def _tab_sector_rank():
            if board.empty:
                st.info("暂无持仓，无法绘制赛道排名。")
            else:
                plot_sector_scores(board)

        _safe_render("赛道分组排名", _tab_sector_rank)

    with tab4:
        def _tab_nav_compare():
            if board.empty:
                st.info("暂无持仓，无法对比净值走势。")
                return
            labels = (board["基金代码"] + "  " + board["基金名称"].fillna("")).tolist()
            code_map = dict(zip(labels, board["基金代码"].tolist()))
            min_select = 1 if len(labels) < 2 else 2
            max_select = min(4, len(labels))
            default_n = min(max(min_select, 2), max_select)
            selected_labels = st.multiselect(
                "选择 2-4 只基金进行近60日净值对比（起点归一化为 100）",
                options=labels,
                default=labels[:default_n],
                max_selections=4,
            )
            selected_codes = [code_map[item] for item in selected_labels]
            if len(labels) >= 2 and len(selected_codes) < 2:
                st.warning("请至少选择 2 只基金。")
            elif selected_codes:
                plot_nav_compare(board, selected_codes)

        _safe_render("净值走势对比", _tab_nav_compare)

    with tab5:
        _safe_render("基金对比", lambda: render_fund_comparison_tab(pool))

    with tab6:
        _safe_render("赛道总览", lambda: render_sector_overview_tab(pool))

    with tab7:
        _safe_render("赛道管理", render_sector_manage_tab)

    with tab8:
        def _tab_buy_candidates():
            st.caption(
                "仅当赛道有剩余空间且候选基金表现优于现有持仓时，才会出现在此清单。"
                "实际买入前请结合自身现金流决策。"
            )
            candidates, failures = cached_buy_candidates(
                today.year, today.month, int(top_n), selected_mode, current_account
            )
            if failures:
                st.warning("以下观察池基金数据拉取失败，已跳过：" + "、".join(failures))
            if candidates is None or candidates.empty:
                st.info("本月暂无符合条件的买入候选，继续持有现金。")
                return
            view = pd.DataFrame(
                {
                    "基金代码": candidates.get("基金代码"),
                    "基金名称": candidates.get("基金名称"),
                    "赛道": candidates.get("赛道"),
                    "综合得分": candidates.get("综合得分").map(fmt_score),
                    "近60日收益": pd.to_numeric(candidates.get("近60日收益"), errors="coerce").map(fmt_pct),
                    "近60日回撤": pd.to_numeric(candidates.get("近60日回撤"), errors="coerce").map(fmt_pct),
                    "赛道剩余空间": pd.to_numeric(
                        candidates.get("当前赛道剩余空间（%）"), errors="coerce"
                    ).map(fmt_pct),
                    "备注": candidates.get("备注"),
                }
            )
            st.dataframe(style_insufficient(view), width="stretch", hide_index=True)
            st.caption("买入候选只作参考，不会自动改仓。确认后请到支付宝手动下单。")

        _safe_render("买入候选", _tab_buy_candidates)

    with tab9:
        _safe_render("策略回测", lambda: render_backtest_tab(pool, current_account))

    with tab10:
        _safe_render("截图导入", lambda: render_ocr_import_tab(current_account))


if __name__ == "__main__":
    main()
