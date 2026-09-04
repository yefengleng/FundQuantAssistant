import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import RETREAT_LIMIT, SINGLE_FUND_LIMIT, TOTAL_EQUITY_LIMIT
from src.data_layer.loader import get_fund_pool_path
from src.factor_layer.sector_classifier import apply_global_sector_map, auto_tag_fund, get_sector_limits


MELTDOWN_EQUITY_CAP = 0.50
KEEP_INSTRUCTION = "保留"
REDUCE_INSTRUCTION = "减仓"
MELTDOWN_INSTRUCTION = "减仓（熔断）"
REDEEM_INSTRUCTION = "赎回"
INSUFFICIENT_INSTRUCTION = "持有"
PROTECTED_INSTRUCTIONS = {REDEEM_INSTRUCTION, REDUCE_INSTRUCTION, MELTDOWN_INSTRUCTION}
SECTOR_TOP_N_SOURCE = "赛道精简"
ELITE_CASH_NOTE = "精简出的资金统一进入货币基金，下月再分配"


def _elite_banner(sectors, funds):
    return f"✂️ 赛道精简：{int(sectors)} 个赛道优化，精简 {int(funds)} 只基金"


def _ensure_strategy_cols(holdings_df):
    df = holdings_df.copy()
    if "持仓市值" not in df.columns:
        df["持仓市值"] = np.nan
    if "目标市值" not in df.columns:
        df["目标市值"] = pd.to_numeric(df["持仓市值"], errors="coerce")
    else:
        df["目标市值"] = pd.to_numeric(df["目标市值"], errors="coerce")
    if "指令" not in df.columns:
        df["指令"] = KEEP_INSTRUCTION
    df["指令"] = df["指令"].fillna(KEEP_INSTRUCTION).astype(str)
    if "指令来源" not in df.columns:
        df["指令来源"] = ""
    df["指令来源"] = df["指令来源"].fillna("").astype(str)
    if "操作理由" not in df.columns:
        df["操作理由"] = ""
    df["操作理由"] = df["操作理由"].fillna("").astype(str)
    return df.reset_index(drop=True)


def _add_source(current, tag):
    parts = [p for p in str(current).split("|") if p]
    if tag not in parts:
        parts.append(tag)
    return "|".join(parts)


def _append_reason(current, text):
    current = "" if current is None or (isinstance(current, float) and np.isnan(current)) else str(current)
    if not current:
        return text
    if text in current:
        return current
    return f"{current}；{text}"


def _pct_text(value):
    return f"{float(value) * 100:.0f}%"


def apply_equity_cap(holdings_df, total_asset, equity_limit=None):
    """总权益仓位超过上限时，等比例降仓，不挑基金。上限由 STRATEGY_MODE 对应画像传入。"""
    df = _ensure_strategy_cols(holdings_df)
    try:
        limit = TOTAL_EQUITY_LIMIT if equity_limit is None else float(equity_limit)
        total_asset = float(total_asset)
        equity = float(df["目标市值"].sum(min_count=1))
        if total_asset <= 0 or pd.isna(equity) or equity <= 0:
            return df
        if equity / total_asset <= limit + 1e-12:
            return df

        scale = (limit * total_asset) / equity
        new_target = df["目标市值"] * scale
        reduced = new_target < df["目标市值"] - 1e-8
        df.loc[reduced, "目标市值"] = new_target.loc[reduced]
        reason = f"总权益仓位超过{_pct_text(limit)}，等比例降仓至上限"
        keep_meltdown = reduced & df["指令"].eq(MELTDOWN_INSTRUCTION)
        df.loc[reduced & ~keep_meltdown, "指令"] = REDUCE_INSTRUCTION
        df.loc[reduced, "指令来源"] = df.loc[reduced, "指令来源"].map(lambda x: _add_source(x, "总仓位上限"))
        df.loc[reduced, "操作理由"] = df.loc[reduced, "操作理由"].map(lambda x: _append_reason(x, reason))
        return df
    except Exception:
        return df


def apply_sector_limits(holdings_df, total_asset, sector_limits=None):
    """赛道市值超限时，按综合得分从低到高依次减仓，直至回到上限内。"""
    df = _ensure_strategy_cols(holdings_df)
    try:
        limits = get_sector_limits() if sector_limits is None else sector_limits
        total_asset = float(total_asset)
        if total_asset <= 0:
            return df

        pool = pd.read_csv(get_fund_pool_path(), dtype={"基金代码": str})
        pool = apply_global_sector_map(pool)
        if "基金代码" in pool.columns and "赛道归类" in pool.columns:
            sector_map = (
                pool.assign(基金代码=pool["基金代码"].astype(str).str.strip())
                [["基金代码", "赛道归类"]]
                .drop_duplicates("基金代码", keep="last")
            )
            if "赛道归类" in df.columns:
                df = df.drop(columns=["赛道归类"])
            df = df.merge(sector_map, on="基金代码", how="left")
        df = apply_global_sector_map(df)
        if "赛道归类" not in df.columns:
            df["赛道归类"] = ""
        blank = df["赛道归类"].isna() | df["赛道归类"].astype(str).str.strip().isin(
            ["", "nan", "None", "NaN", "<NA>"]
        )
        for idx in df.index[blank]:
            code = str(df.at[idx, "基金代码"]).strip()
            name = ""
            if "基金名称" in df.columns:
                name = str(df.at[idx, "基金名称"] or "").strip()
            try:
                tagged = auto_tag_fund(code, name, allow_network=False)
                sector = tagged.get("sector") if isinstance(tagged, dict) else tagged
            except Exception:
                sector = "其他"
            df.at[idx, "赛道归类"] = sector or "其他"
        df["赛道归类"] = df["赛道归类"].fillna("其他")

        other_limit = limits.get("其他", 1.0)
        for sector, group in df.groupby("赛道归类", sort=False):
            limit = limits.get(sector, other_limit)
            cap_value = limit * total_asset
            sector_sum = float(group["目标市值"].sum())
            if sector_sum <= cap_value + 1e-8:
                continue

            excess = sector_sum - cap_value
            ranked = group.copy()
            ranked["_score"] = pd.to_numeric(ranked["综合得分"], errors="coerce") if "综合得分" in ranked.columns else np.nan
            sellable = ranked[ranked["_score"].notna()].sort_values("_score", ascending=True, kind="mergesort")
            if sellable.empty:
                continue

            remaining = excess
            reason = f"{sector}赛道仓位超过{_pct_text(limit)}，优先减仓得分最低品种"
            for idx in sellable.index:
                if remaining <= 1e-8:
                    break
                current_mv = float(df.at[idx, "目标市值"])
                if current_mv <= 1e-8:
                    continue
                cut = min(current_mv, remaining)
                new_mv = current_mv - cut
                df.at[idx, "目标市值"] = 0.0 if new_mv <= 1e-8 else new_mv
                remaining -= cut
                if df.at[idx, "目标市值"] <= 1e-8:
                    df.at[idx, "指令"] = REDEEM_INSTRUCTION
                elif df.at[idx, "指令"] != MELTDOWN_INSTRUCTION:
                    df.at[idx, "指令"] = REDUCE_INSTRUCTION
                df.at[idx, "指令来源"] = _add_source(df.at[idx, "指令来源"], "赛道超限")
                df.at[idx, "操作理由"] = _append_reason(df.at[idx, "操作理由"], reason)
        return df
    except Exception:
        return df


def apply_single_fund_limit(holdings_df, total_asset, single_fund_limit=None):
    """单只基金市值超过上限时减仓至上限以内。"""
    df = _ensure_strategy_cols(holdings_df)
    try:
        fund_limit = SINGLE_FUND_LIMIT if single_fund_limit is None else float(single_fund_limit)
        total_asset = float(total_asset)
        if total_asset <= 0:
            return df

        cap_value = fund_limit * total_asset
        over = df["目标市值"] > cap_value + 1e-8
        if not over.any():
            return df

        df.loc[over, "目标市值"] = cap_value
        keep_meltdown = over & df["指令"].eq(MELTDOWN_INSTRUCTION)
        df.loc[over & ~keep_meltdown, "指令"] = REDUCE_INSTRUCTION
        reason = f"单基金仓位超过{_pct_text(fund_limit)}，减仓至上限以内"
        df.loc[over, "指令来源"] = df.loc[over, "指令来源"].map(lambda x: _add_source(x, "单基超限"))
        df.loc[over, "操作理由"] = df.loc[over, "操作理由"].map(lambda x: _append_reason(x, reason))
        return df
    except Exception:
        return df


def apply_retreat_meltdown(holdings_df, total_asset, account_return, retreat_limit=None):
    """账户回撤低于熔断线时，权益仓位强制压到 50% 以内。"""
    df = _ensure_strategy_cols(holdings_df)
    try:
        limit = RETREAT_LIMIT if retreat_limit is None else float(retreat_limit)
        total_asset = float(total_asset)
        account_return = float(account_return)
        if pd.isna(account_return) or account_return >= limit:
            return df
        if total_asset <= 0:
            return df

        equity = float(df["目标市值"].sum(min_count=1))
        if pd.notna(equity) and equity > 0 and equity / total_asset > MELTDOWN_EQUITY_CAP + 1e-12:
            scale = (MELTDOWN_EQUITY_CAP * total_asset) / equity
            df["目标市值"] = df["目标市值"] * scale

        reason = f"账户回撤触发熔断（低于{_pct_text(limit)}），权益仓位降至{_pct_text(MELTDOWN_EQUITY_CAP)}以内"
        df["指令"] = MELTDOWN_INSTRUCTION
        df["指令来源"] = df["指令来源"].map(lambda x: _add_source(x, "熔断"))
        df["操作理由"] = df["操作理由"].map(lambda x: _append_reason(x, reason))
        df.attrs["meltdown_triggered"] = True
        return df
    except Exception:
        return df


def apply_sector_elite(orders_df, holdings_df, top_n=3):
    """
    每个赛道只保留综合得分最高的前 top_n 只，其余尚未被风控标记的改为赎回。

    已是减仓/赎回（含熔断减仓）的基金不改指令，避免过度操作。
    精简资金视为转入货币基金，下月再分配。
    """
    df = _ensure_strategy_cols(orders_df if orders_df is not None else pd.DataFrame())
    df.attrs["sector_top_n_sectors"] = 0
    df.attrs["sector_top_n_funds"] = 0
    df.attrs["sector_elite_cash"] = 0.0
    df.attrs["sector_top_n_banner"] = _elite_banner(0, 0)
    df.attrs["sector_elite_note"] = ELITE_CASH_NOTE

    try:
        n = int(top_n)
    except (TypeError, ValueError):
        n = 3
    if n <= 0 or df.empty:
        return df

    try:
        work = df.copy()
        if "基金代码" in work.columns:
            work["基金代码"] = work["基金代码"].astype(str).str.strip()

        if holdings_df is not None and not holdings_df.empty and "基金代码" in holdings_df.columns:
            extra = holdings_df.copy()
            extra["基金代码"] = extra["基金代码"].astype(str).str.strip()
            keep_cols = [col for col in ["基金代码", "赛道归类", "综合得分"] if col in extra.columns]
            extra = extra[keep_cols].drop_duplicates("基金代码", keep="last")
            if "赛道归类" in extra.columns:
                sector_map = extra.set_index("基金代码")["赛道归类"]
                if "赛道归类" not in work.columns:
                    work["赛道归类"] = work["基金代码"].map(sector_map)
                else:
                    blank = work["赛道归类"].isna() | work["赛道归类"].astype(str).str.strip().isin(
                        ["", "nan", "None", "NaN"]
                    )
                    work.loc[blank, "赛道归类"] = work.loc[blank, "基金代码"].map(sector_map)
            if "综合得分" in extra.columns and "综合得分" not in work.columns:
                work["综合得分"] = work["基金代码"].map(extra.set_index("基金代码")["综合得分"])

        work = apply_global_sector_map(work)

        if "赛道归类" not in work.columns:
            work["赛道归类"] = "其他"
        work["赛道归类"] = work["赛道归类"].fillna("其他").astype(str)
        work.loc[work["赛道归类"].str.strip().isin(["", "nan", "None", "NaN"]), "赛道归类"] = "其他"

        if "综合得分" not in work.columns:
            work["综合得分"] = np.nan
        rank_score = pd.to_numeric(work["综合得分"], errors="coerce")
        insufficient = work["综合得分"].astype(str).str.strip().eq("数据不足")
        rank_score = rank_score.mask(insufficient, np.nan)
        work["_rank_score"] = rank_score

        reason = f"赛道精简：保留前{n}名，优胜劣汰"
        trimmed_sectors = set()
        trimmed_count = 0
        trimmed_cash = 0.0

        for sector, group in work.groupby("赛道归类", sort=False):
            if len(group) <= n:
                continue
            ranked = group.sort_values(
                "_rank_score", ascending=False, na_position="last", kind="mergesort"
            )
            for idx in ranked.index[n:]:
                current = str(work.at[idx, "指令"]).strip()
                if current in PROTECTED_INSTRUCTIONS:
                    continue
                mv = pd.to_numeric(work.at[idx, "持仓市值"], errors="coerce")
                if pd.notna(mv):
                    trimmed_cash += float(mv)
                work.at[idx, "指令"] = REDEEM_INSTRUCTION
                work.at[idx, "目标市值"] = 0.0
                work.at[idx, "指令来源"] = _add_source(work.at[idx, "指令来源"], SECTOR_TOP_N_SOURCE)
                work.at[idx, "操作理由"] = _append_reason(work.at[idx, "操作理由"], reason)
                trimmed_count += 1
                trimmed_sectors.add(str(sector))

        work = work.drop(columns=["_rank_score"], errors="ignore")
        work.attrs["sector_top_n_sectors"] = len(trimmed_sectors)
        work.attrs["sector_top_n_funds"] = int(trimmed_count)
        work.attrs["sector_elite_cash"] = float(trimmed_cash)
        work.attrs["sector_top_n_banner"] = _elite_banner(len(trimmed_sectors), trimmed_count)
        work.attrs["sector_elite_note"] = ELITE_CASH_NOTE
        return work
    except Exception:
        df.attrs["sector_top_n_sectors"] = 0
        df.attrs["sector_top_n_funds"] = 0
        df.attrs["sector_elite_cash"] = 0.0
        df.attrs["sector_top_n_banner"] = _elite_banner(0, 0)
        df.attrs["sector_elite_note"] = ELITE_CASH_NOTE
        return df


def apply_sector_top_n(orders_df, holdings_df, top_n=3):
    """兼容旧名，逻辑与 apply_sector_elite 相同。"""
    return apply_sector_elite(orders_df, holdings_df, top_n=top_n)
