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

from src.strategy_layer import generate_buy_candidates, generate_trading_signal


KEEP_SET = {"保留", "持有", "暂缓"}
REDUCE_SET = {"减仓", "减仓（熔断）"}
REDEEM_SET = {"赎回"}


def _fmt_pct(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def _fmt_money(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.2f}"


def _fmt_shares(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.2f}"


def _view(df):
    if df.empty:
        return df
    out = pd.DataFrame(
        {
            "基金名称": df.get("基金名称"),
            "赛道": df.get("赛道归类"),
            "当前市值": df.get("持仓市值").map(_fmt_money) if "持仓市值" in df.columns else "-",
            "建议操作": df.get("建议操作"),
            "卖出份额": df.get("卖出份额").map(_fmt_shares) if "卖出份额" in df.columns else "-",
            "操作理由": df.get("操作理由"),
        }
    )
    if "目标仓位" in df.columns:
        out["目标仓位"] = df["目标仓位"].map(_fmt_pct)
    return out


def _print_list(title, df):
    print(f"\n{title}")
    if df.empty:
        print("（空）")
        return
    print(_view(df).to_string(index=False))


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    pd.set_option("display.unicode.east_asian_width", True)

    orders = generate_trading_signal(2026, 8)

    banner = orders.attrs.get("strategy_mode_banner")
    if banner:
        print(banner)
    print(
        f"策略模式: {orders.attrs.get('strategy_mode_requested')} → "
        f"{orders.attrs.get('strategy_mode_effective')}"
    )

    if orders.attrs.get("meltdown_triggered"):
        print("⚠️  熔断已触发")
        print(f"账户回撤: {_fmt_pct(orders.attrs.get('account_return'))}")
    if orders.attrs.get("crash_filter_triggered"):
        print("⚠️  市场急跌过滤器生效")

    if not orders.attrs.get("meltdown_triggered") and not orders.attrs.get("crash_filter_triggered"):
        print("未触发熔断，未触发市场急跌过滤器")
        print(f"账户回撤: {_fmt_pct(orders.attrs.get('account_return'))}")

    keep_df = orders[orders["建议操作"].isin(KEEP_SET)] if not orders.empty else orders
    reduce_df = orders[orders["建议操作"].isin(REDUCE_SET)] if not orders.empty else orders
    redeem_df = orders[orders["建议操作"].isin(REDEEM_SET)] if not orders.empty else orders

    _print_list("=== ✅ 保留清单 ===", keep_df)
    _print_list("=== ⚠️ 减仓清单 ===", reduce_df)
    _print_list("=== ❌ 赎回清单 ===", redeem_df)

    print(f"\n指令已保存: {ROOT / 'data' / 'latest_signal.csv'}")

    candidates = generate_buy_candidates(2026, 8, top_n=3)
    print("\n🌟 本月候选买入基金（建议关注）：")
    if candidates is None or candidates.empty:
        print("本月暂无符合条件的买入候选，继续持有现金。")
    else:
        print("| 基金代码 | 基金名称 | 赛道 | 得分 | 赛道剩余空间 | 备注 |")
        print("|----------|---------|------|------|-------------|------|")
        for _, row in candidates.iterrows():
            score = f"{float(row['综合得分']):.1f}" if pd.notna(row.get("综合得分")) else "-"
            space = _fmt_pct(row.get("当前赛道剩余空间（%）"))
            note = row.get("备注") or "得分显著高于现有持仓"
            print(
                f"| {row.get('基金代码', '')} | {row.get('基金名称', '')} | {row.get('赛道', '')} | "
                f"{score} | {space} | {note} |"
            )

    print("\n=== 全市场扫描结果 ===")
    try:
        from src.factor_layer.scorer import scan_market_funds

        scan_df = scan_market_funds()
    except Exception as exc:
        print(f"全市场扫描失败: {exc}")
        return

    if scan_df is None or scan_df.empty:
        print("暂无符合规模/年限/回撤条件的扫描结果。")
        return

    print("筛选：规模≥2亿，成立满1年，近60日回撤好于-25%；每个基金类型取综合得分前3。")
    print("说明：排行接口无独立60日字段，收益用近3月，回撤用近1周/近1月/近3月最差值。")
    for sector, group in scan_df.groupby("赛道归类", sort=True, dropna=False):
        print(f"\n[{sector if pd.notna(sector) else '未分类'}]")
        view = group.copy()
        view["近60日收益率"] = view["近60日收益率"].map(_fmt_pct)
        view["近60日最大回撤"] = view["近60日最大回撤"].map(_fmt_pct)
        view["综合得分"] = view["综合得分"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "-")
        view["基金规模(亿)"] = view["基金规模(亿)"].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "-")
        cols = ["赛道内排名", "基金代码", "基金名称", "综合得分", "近60日收益率", "近60日最大回撤", "基金规模(亿)"]
        print(view[cols].to_string(index=False))


if __name__ == "__main__":
    main()
