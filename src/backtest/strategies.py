from datetime import date, datetime, timedelta

import backtrader as bt
import numpy as np
import pandas as pd

from config.settings import (
    AUTO_MODE_RULE,
    SECTOR_TOP_N,
    get_strategy_profile,
    is_sector_top_n_active,
    normalize_strategy_mode,
)
from src.factor_layer.indicators import calc_max_drawdown, calc_return
from src.factor_layer.scorer import MIN_TRADE_DAYS, calculate_score
from src.strategy_layer.constraints import apply_equity_cap, apply_retreat_meltdown, apply_single_fund_limit
from src.strategy_layer.filters import CRASH_THRESHOLD
from src.strategy_layer.cooldown import COOLDOWN_DAYS


BENCHMARK_NAME = "000300"
MIN_TRADE_VALUE = 10.0
BUY_FEE_DEFAULT = 0.0015
REDEEM_FEE_LT_7D = 0.015
REDEEM_FEE_LT_1Y = 0.005
REDEEM_FEE_GE_1Y = 0.0025


def redemption_fee_rate(holding_days):
    try:
        days = int(holding_days)
    except (TypeError, ValueError):
        days = 0
    if days < 7:
        return REDEEM_FEE_LT_7D
    if days < 365:
        return REDEEM_FEE_LT_1Y
    return REDEEM_FEE_GE_1Y


def _to_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def score_nav_asof(nav_series, asof):
    """用 asof 当日及之前的净值打分，避免未来函数。"""
    series = pd.to_numeric(nav_series, errors="coerce").dropna()
    if series.empty:
        return None
    asof_ts = pd.Timestamp(asof)
    hist = series.loc[:asof_ts]
    if len(hist) < MIN_TRADE_DAYS:
        return None
    window = hist.iloc[-MIN_TRADE_DAYS:]
    ret = calc_return(window, days=MIN_TRADE_DAYS)
    mdd = calc_max_drawdown(window)
    score = calculate_score(ret, mdd)
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return None
    return {
        "综合得分": float(score),
        "近60日收益率": None if pd.isna(ret) else float(ret) / 100.0,
        "近60日最大回撤": None if pd.isna(mdd) else float(mdd) / 100.0,
    }


def precompute_scores(nav_map, rebalance_dates):
    cache = {}
    for asof in rebalance_dates:
        day = {}
        for code, series in (nav_map or {}).items():
            if str(code) == BENCHMARK_NAME:
                continue
            scored = score_nav_asof(series, asof)
            if scored:
                day[str(code)] = scored
        cache[_to_date(asof)] = day
    return cache


def resolve_mode_asof(requested, hs300, asof):
    requested = normalize_strategy_mode(requested)
    if requested != "auto":
        return requested
    series = pd.to_numeric(hs300, errors="coerce").dropna() if hs300 is not None else pd.Series(dtype=float)
    if series.empty:
        return "defensive"
    try:
        period = int((AUTO_MODE_RULE or {}).get("ma_period", 120) or 120)
    except (TypeError, ValueError):
        period = 120
    hist = series.loc[: pd.Timestamp(asof)]
    if len(hist) < period:
        return "defensive"
    close = float(hist.iloc[-1])
    ma = float(hist.iloc[-period:].mean())
    if not np.isfinite(close) or not np.isfinite(ma) or ma == 0:
        return "defensive"
    return "aggressive" if close > ma else "defensive"


def hs300_month_return(hs300, asof):
    series = pd.to_numeric(hs300, errors="coerce").dropna() if hs300 is not None else pd.Series(dtype=float)
    if series.empty:
        return np.nan
    asof_ts = pd.Timestamp(asof)
    month = series.loc[(series.index.year == asof_ts.year) & (series.index.month == asof_ts.month) & (series.index <= asof_ts)]
    if len(month) < 2 or float(month.iloc[0]) == 0:
        return np.nan
    return float(month.iloc[-1] / month.iloc[0] - 1.0) * 100.0


def _clip_sector_limits(df, total_asset, sector_limits):
    work = df.copy()
    if work.empty or total_asset <= 0:
        return work
    if "赛道归类" not in work.columns:
        work["赛道归类"] = "其他"
    work["赛道归类"] = work["赛道归类"].fillna("其他").astype(str)
    other = (sector_limits or {}).get("其他", 1.0)
    for sector, group in work.groupby("赛道归类", sort=False):
        limit = (sector_limits or {}).get(sector, other)
        cap_value = float(limit) * total_asset
        sector_sum = float(group["目标市值"].sum())
        if sector_sum <= cap_value + 1e-8:
            continue
        excess = sector_sum - cap_value
        ranked = group.copy()
        ranked["_score"] = pd.to_numeric(ranked.get("综合得分"), errors="coerce")
        sellable = ranked.sort_values("_score", ascending=True, na_position="first", kind="mergesort")
        remaining = excess
        for idx in sellable.index:
            if remaining <= 1e-8:
                break
            current = float(work.at[idx, "目标市值"])
            if current <= 1e-8:
                continue
            cut = min(current, remaining)
            work.at[idx, "目标市值"] = current - cut
            remaining -= cut
            work.at[idx, "指令来源"] = "赛道超限"
    return work


def _keep_sector_elite(scored_rows, top_n):
    if not scored_rows:
        return []
    frame = pd.DataFrame(scored_rows)
    kept = []
    for _, group in frame.groupby("赛道归类", sort=False):
        ranked = group.sort_values("综合得分", ascending=False, kind="mergesort")
        kept.extend(ranked.head(int(top_n)).to_dict("records"))
    return kept


def build_target_values(
    asof,
    nav_map,
    current_values,
    total_asset,
    peak_value,
    requested_mode,
    hs300,
    sector_map,
    score_row,
    last_sell_dates,
    buy_blocked=None,
):
    """
    按现有 strategy_layer 约束生成目标市值。
    在可选基金中等权分配至总仓位上限，再套用熔断/赛道/单基/精简。
    """
    asof = _to_date(asof)
    effective = resolve_mode_asof(requested_mode, hs300, asof)
    profile = get_strategy_profile(effective)
    scores = score_row or {}
    sector_map = sector_map or {}
    current_values = {str(k): float(v or 0.0) for k, v in (current_values or {}).items()}

    scored_rows = []
    for code, payload in scores.items():
        score = payload.get("综合得分") if isinstance(payload, dict) else payload
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(score):
            continue
        scored_rows.append(
            {
                "基金代码": str(code),
                "综合得分": score,
                "赛道归类": sector_map.get(str(code)) or "其他",
            }
        )

    if is_sector_top_n_active(effective):
        scored_rows = _keep_sector_elite(scored_rows, SECTOR_TOP_N)

    selected = [row["基金代码"] for row in scored_rows if row.get("综合得分", 0) > 0]
    if not selected:
        selected = [row["基金代码"] for row in scored_rows[: max(1, min(3, len(scored_rows)))]]

    equity_budget = float(total_asset) * float(profile["TOTAL_EQUITY_LIMIT"])
    crash = False
    month_ret = hs300_month_return(hs300, asof)
    if month_ret is not None and np.isfinite(month_ret) and month_ret < CRASH_THRESHOLD:
        crash = True
        current_equity = sum(current_values.values())
        equity_budget = min(equity_budget, current_equity)

    drawdown = 0.0
    if peak_value and peak_value > 0 and total_asset > 0:
        drawdown = float(total_asset) / float(peak_value) - 1.0

    universe = sorted(set(list(current_values) + selected + list(nav_map or {})))
    universe = [code for code in universe if str(code) != BENCHMARK_NAME]
    equal = equity_budget / len(selected) if selected else 0.0
    records = []
    for code in universe:
        current = float(current_values.get(code, 0.0) or 0.0)
        target = equal if code in selected else 0.0
        score_payload = scores.get(code) or {}
        records.append(
            {
                "基金代码": code,
                "持仓市值": current,
                "目标市值": target,
                "综合得分": score_payload.get("综合得分", np.nan) if isinstance(score_payload, dict) else score_payload,
                "赛道归类": sector_map.get(code) or "其他",
                "指令": "保留",
                "指令来源": "",
                "操作理由": "",
            }
        )
    holdings = pd.DataFrame(records)
    if holdings.empty:
        return {}, effective, crash

    holdings = apply_retreat_meltdown(
        holdings, total_asset, drawdown, retreat_limit=profile["RETREAT_LIMIT"]
    )
    holdings = apply_equity_cap(holdings, total_asset, equity_limit=profile["TOTAL_EQUITY_LIMIT"])
    holdings = _clip_sector_limits(holdings, total_asset, profile["SECTOR_LIMITS"])
    holdings = apply_single_fund_limit(holdings, total_asset, single_fund_limit=profile["SINGLE_FUND_LIMIT"])

    if crash:
        over = holdings["目标市值"] > holdings["持仓市值"] + MIN_TRADE_VALUE
        holdings.loc[over, "目标市值"] = holdings.loc[over, "持仓市值"]

    cutoff = asof - timedelta(days=COOLDOWN_DAYS) if asof else None
    if cutoff and last_sell_dates:
        for idx, row in holdings.iterrows():
            code = str(row["基金代码"])
            last_sell = last_sell_dates.get(code)
            if not last_sell or last_sell < cutoff:
                continue
            if float(row["目标市值"]) >= float(row["持仓市值"]) - MIN_TRADE_VALUE:
                continue
            source = str(row.get("指令来源") or "")
            if any(tag in source for tag in ("赛道超限", "单基超限", "熔断")):
                continue
            holdings.at[idx, "目标市值"] = row["持仓市值"]

    blocked = set(buy_blocked or [])
    if blocked:
        newbie = holdings["持仓市值"].fillna(0) <= MIN_TRADE_VALUE
        holdings.loc[newbie & holdings["基金代码"].isin(blocked), "目标市值"] = 0.0

    targets = {
        str(row["基金代码"]): max(0.0, float(row["目标市值"]))
        for _, row in holdings.iterrows()
    }
    return targets, effective, crash


class FundCommission(bt.CommInfoBase):
    """允许基金按净值成交小数份额；买卖费率在策略里按持有期另行扣除。"""

    params = (
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
        ("commission", 0.0),
    )

    def getsize(self, price, cash):
        if not price:
            return 0.0
        return cash / price


class FundRebalanceStrategy(bt.Strategy):
    params = dict(
        mode="aggressive",
        frequency="monthly",
        buy_fee=BUY_FEE_DEFAULT,
        context=None,
    )

    def __init__(self):
        ctx = self.p.context
        self.ctx = ctx
        self.lots = {}
        self.last_sell_dates = {}
        self.equity_records = []
        self.trade_records = []
        self.holding_records = []
        self.peak_value = 0.0
        self.rebalance_set = set(getattr(ctx, "rebalance_dates", set()) or [])
        self.nav_map = getattr(ctx, "nav_map", {}) or {}
        self.score_cache = getattr(ctx, "score_cache", {}) or {}
        self.sector_map = getattr(ctx, "sector_map", {}) or {}
        self.name_map = getattr(ctx, "name_map", {}) or {}
        self.hs300 = getattr(ctx, "hs300", None)
        self.fund_datas = [data for data in self.datas if data._name != BENCHMARK_NAME]

    def _asof(self):
        return self.datetime.date(0)

    def _portfolio_value(self):
        return float(self.broker.getvalue())

    def _current_values(self):
        values = {}
        for data in self.fund_datas:
            price = float(data.close[0]) if len(data) and data.close[0] == data.close[0] else 0.0
            size = float(self.getposition(data).size or 0.0)
            values[data._name] = size * price if price > 0 else 0.0
        return values

    def _consume_lots(self, code, shares, sell_date, price):
        remaining = abs(float(shares))
        lots = self.lots.setdefault(code, [])
        fee = 0.0
        hold_days = 0
        used = 0.0
        while remaining > 1e-8 and lots:
            lot = lots[0]
            take = min(float(lot["shares"]), remaining)
            days = (sell_date - lot["date"]).days if lot.get("date") else 0
            fee += take * price * redemption_fee_rate(days)
            hold_days = max(hold_days, days)
            lot["shares"] -= take
            remaining -= take
            used += take
            if lot["shares"] <= 1e-8:
                lots.pop(0)
        if remaining > 1e-8:
            fee += remaining * price * redemption_fee_rate(0)
            used += remaining
        return fee, hold_days

    def _record_day(self, asof, crash, effective):
        total = self._portfolio_value()
        cash = float(self.broker.getcash())
        self.peak_value = max(self.peak_value, total)
        bench = None
        for data in self.datas:
            if data._name == BENCHMARK_NAME and len(data):
                bench = float(data.close[0])
                break
        self.equity_records.append(
            {
                "date": asof,
                "strategy": total,
                "cash": cash,
                "position_value": total - cash,
                "benchmark": bench,
                "mode": effective,
                "crash": crash,
            }
        )
        for data in self.fund_datas:
            size = float(self.getposition(data).size or 0.0)
            if size <= 1e-8:
                continue
            price = float(data.close[0]) if data.close[0] == data.close[0] else 0.0
            value = size * price
            self.holding_records.append(
                {
                    "date": asof,
                    "基金代码": data._name,
                    "基金名称": self.name_map.get(data._name, ""),
                    "份额": size,
                    "净值": price,
                    "市值": value,
                    "仓位占比": (value / total) if total else 0.0,
                }
            )

    def next(self):
        asof = self._asof()
        total = self._portfolio_value()
        if total > self.peak_value:
            self.peak_value = total
        crash = False
        effective = resolve_mode_asof(self.p.mode, self.hs300, asof)
        if asof in self.rebalance_set:
            crash, effective = self._rebalance(asof)
        self._record_day(asof, crash, effective)

    def _rebalance(self, asof):
        total = self._portfolio_value()
        current = self._current_values()
        scores = self.score_cache.get(asof) or {}
        blocked = {
            data._name
            for data in self.fund_datas
            if (not len(data)) or data.close[0] != data.close[0] or float(data.close[0]) <= 0
        }
        targets, effective, crash = build_target_values(
            asof=asof,
            nav_map=self.nav_map,
            current_values=current,
            total_asset=total,
            peak_value=self.peak_value or total,
            requested_mode=self.p.mode,
            hs300=self.hs300,
            sector_map=self.sector_map,
            score_row=scores,
            last_sell_dates=self.last_sell_dates,
            buy_blocked=blocked,
        )
        pending = []
        for data in self.fund_datas:
            price = float(data.close[0]) if len(data) and data.close[0] == data.close[0] else 0.0
            if price <= 0:
                continue
            target = float(targets.get(data._name, 0.0) or 0.0)
            current_size = float(self.getposition(data).size or 0.0)
            delta_value = target - current_size * price
            if abs(delta_value) < MIN_TRADE_VALUE:
                continue
            pending.append((data, price, current_size, delta_value))
        for data, price, current_size, delta_value in pending:
            if delta_value >= 0:
                continue
            shares = min(current_size, abs(delta_value) / price)
            if shares * price >= MIN_TRADE_VALUE:
                self.sell(data=data, size=shares)
        reserved = 0.0
        cash = float(self.broker.getcash())
        for data, price, current_size, delta_value in pending:
            if delta_value <= 0:
                continue
            affordable = min(delta_value, max(0.0, (cash - reserved) / (1.0 + float(self.p.buy_fee))))
            shares = affordable / price
            if shares * price < MIN_TRADE_VALUE:
                continue
            reserved += affordable * (1.0 + float(self.p.buy_fee))
            self.buy(data=data, size=shares)
        return crash, effective

    def notify_order(self, order):
        if order.status != order.Completed:
            return
        data = order.data
        code = data._name
        asof = bt.num2date(order.executed.dt).date()
        price = float(order.executed.price)
        size = abs(float(order.executed.size))
        value = size * price
        name = self.name_map.get(code, "")
        if order.isbuy():
            fee = value * float(self.p.buy_fee)
            if fee:
                self.broker.add_cash(-fee)
            self.lots.setdefault(code, []).append({"date": asof, "shares": size, "price": price})
            self.trade_records.append(
                {
                    "date": asof,
                    "基金代码": code,
                    "基金名称": name,
                    "方向": "买入",
                    "份额": size,
                    "净值": price,
                    "金额": value,
                    "费用": fee,
                    "持有天数": 0,
                    "费率": float(self.p.buy_fee),
                }
            )
            return
        fee, hold_days = self._consume_lots(code, size, asof, price)
        if fee:
            self.broker.add_cash(-fee)
        self.last_sell_dates[code] = asof
        self.trade_records.append(
            {
                "date": asof,
                "基金代码": code,
                "基金名称": name,
                "方向": "卖出",
                "份额": size,
                "净值": price,
                "金额": value,
                "费用": fee,
                "持有天数": hold_days,
                "费率": (fee / value) if value else 0.0,
            }
        )


class AggressiveStrategy(FundRebalanceStrategy):
    params = dict(mode="aggressive")


class DefensiveStrategy(FundRebalanceStrategy):
    params = dict(mode="defensive")
