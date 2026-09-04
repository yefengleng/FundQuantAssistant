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

from src.data_layer.fetcher import fetch_fund_history
from src.data_layer.loader import get_fund_pool_path, load_local_data, update_all_funds


def main():
    pool_path = Path(get_fund_pool_path())
    pool = pd.read_csv(pool_path, dtype={"基金代码": str})
    fund_code = str(pool.loc[0, "基金代码"]).strip()

    print(f"=== 1) 在线拉取 {fund_code} ===")
    history = fetch_fund_history(fund_code)
    if history.empty:
        print("拉取结果为空，请检查网络或基金代码。")
        return

    print(f"共 {len(history)} 条净值记录")
    print("最近 5 个交易日：")
    print(history.tail(5).to_string(index=False))

    print(f"\n=== 2) 增量落盘到 data/raw/ ===")
    update_all_funds()

    print(f"\n=== 3) 读取本地 Parquet {fund_code}.parquet ===")
    local_df = load_local_data(fund_code)
    print(local_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
