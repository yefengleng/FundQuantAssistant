import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

from src.factor_layer import batch_score_funds, load_current_holdings


def _fmt_pct(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _fmt_money(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.2f}"


def _fmt_score(value):
    if value is None or pd.isna(value):
        return "-"
    if value == "数据不足":
        return "数据不足"
    return f"{float(value):.2f}"


def _print_holdings(holdings):
    total_assets = holdings["持仓市值"].sum(min_count=1)
    print("=== 当前持仓概览 ===")
    print(f"基金只数: {len(holdings)}")
    print(f"总资产: {_fmt_money(total_assets)}")
    print("")

    view = holdings.copy()
    view["最新净值"] = view["最新净值"].map(lambda x: "-" if pd.isna(x) else f"{float(x):.4f}")
    view["持仓市值"] = view["持仓市值"].map(_fmt_money)
    view["仓位占比"] = view["仓位占比"].map(_fmt_pct)
    cols = [col for col in ["基金代码", "基金名称", "赛道归类", "持有份额", "买入日期", "最新净值", "持仓市值", "仓位占比"] if col in view.columns]
    print(view[cols].to_string(index=False))


def _print_score_table(score_df, title):
    print(title)
    if score_df.empty:
        print("暂无得分数据")
        return

    view = score_df.copy()
    for col in ["近60日收益率", "近60日最大回撤", "近60日波动率"]:
        if col in view.columns:
            view[col] = view[col].map(_fmt_pct)
    if "综合得分" in view.columns:
        view["综合得分"] = view["综合得分"].map(_fmt_score)
    print(view.to_string(index=False))


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.unicode.east_asian_width", True)

    holdings = load_current_holdings()
    _print_holdings(holdings)

    fund_codes = holdings["基金代码"].tolist() if not holdings.empty else []
    scores = batch_score_funds(fund_codes)

    print("\n=== 综合得分排名 ===")
    ranked = scores.merge(
        holdings[["基金代码", "基金名称", "赛道归类"]],
        on="基金代码",
        how="left",
    )
    show_cols = ["基金代码", "基金名称", "赛道归类", "近60日收益率", "近60日最大回撤", "近60日波动率", "综合得分"]
    show_cols = [col for col in show_cols if col in ranked.columns]
    _print_score_table(ranked[show_cols], "")

    print("\n=== 分赛道得分排序 ===")
    if ranked.empty or "赛道归类" not in ranked.columns:
        print("无法按赛道分组")
        return

    for sector, group in ranked.groupby("赛道归类", sort=True, dropna=False):
        sector_name = "未分类" if pd.isna(sector) else sector
        _print_score_table(group[show_cols], f"\n[{sector_name}]")


if __name__ == "__main__":
    main()
