import glob
import json
import os
import re
import shutil
import sys
import time

import pandas as pd


def _safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "gbk"
        print(str(message).encode(encoding, errors="replace").decode(encoding, errors="replace"))


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROFILES_DIR = os.path.join(PROJECT_ROOT, "config", "profiles")
DEFAULT_ACCOUNT_NAME = "默认账户"
CURRENT_ACCOUNT_FILE = os.path.join(PROFILES_DIR, ".current")
LEGACY_FUND_POOL_PATH = os.path.join(PROJECT_ROOT, "config", "fund_pool.csv")
LEGACY_TRADE_LOG_PATH = os.path.join(PROJECT_ROOT, "data", "trade_log.json")
LEGACY_SIGNAL_PATH = os.path.join(PROJECT_ROOT, "data", "latest_signal.csv")
LEGACY_META_PATH = os.path.join(PROJECT_ROOT, "data", "app_meta.json")
FUND_POOL_HEADER = [
    "基金代码",
    "基金名称",
    "赛道归类",
    "持有份额",
    "买入日期",
    "持仓成本（元）",
    "持仓市值",
    "备注",
    "累计收益（元）",
]
COST_COLUMN = "持仓成本（元）"
NOTE_COLUMN = "备注"
CUM_PROFIT_COLUMN = "累计收益（元）"
CUM_PROFIT_ALIASES = ("累计收益（元）", "累计收益(元)")
EMPTY_TRADE_LOG = {"records": []}
ACCOUNT_NAME_RE = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9_]+$")
WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
EMPTY_NAV_COLUMNS = ["date", "nav", "change_pct"]


class _LazyAccountPath(os.PathLike):
    """调用时再解析，避免账户切换后仍指向旧路径。"""

    def __init__(self, getter):
        self._getter = getter

    def __fspath__(self):
        return self._getter()

    def __str__(self):
        return self._getter()

    def __repr__(self):
        return f"{self._getter()!r}"


def validate_account_name(name):
    text = "" if name is None else str(name).strip()
    if not text:
        return False, "账户名不能为空"
    if len(text) > 32:
        return False, "账户名最多 32 个字符"
    if not ACCOUNT_NAME_RE.fullmatch(text):
        return False, "账户名仅允许中文、英文、数字和下划线"
    if text.upper() in WINDOWS_RESERVED:
        return False, "账户名不可用"
    return True, text


def _account_dir(name):
    return os.path.join(PROFILES_DIR, name)


def _write_current_account(name):
    os.makedirs(PROFILES_DIR, exist_ok=True)
    with open(CURRENT_ACCOUNT_FILE, "w", encoding="utf-8") as file:
        file.write(str(name).strip())


def _read_current_account():
    try:
        with open(CURRENT_ACCOUNT_FILE, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception:
        return ""


def _write_fund_pool_template(path):
    pd.DataFrame(columns=FUND_POOL_HEADER).to_csv(path, index=False, encoding="utf-8-sig")


def _ordered_fund_pool_columns(columns):
    extras = [col for col in columns if col not in FUND_POOL_HEADER]
    return [col for col in FUND_POOL_HEADER if col in columns] + extras


def ensure_fund_pool_schema(path):
    """读取基金池 CSV，补齐成本、备注、累计收益等列并回写。"""
    if not path or not os.path.isfile(path):
        return False
    try:
        df = pd.read_csv(path, dtype={"基金代码": str})
    except Exception:
        return False

    changed = False
    for alias in CUM_PROFIT_ALIASES:
        if alias in df.columns and alias != CUM_PROFIT_COLUMN:
            df = df.rename(columns={alias: CUM_PROFIT_COLUMN})
            changed = True
            break

    if COST_COLUMN not in df.columns:
        if "买入日期" in df.columns:
            insert_at = list(df.columns).index("买入日期") + 1
            df.insert(insert_at, COST_COLUMN, 0.0)
        else:
            df[COST_COLUMN] = 0.0
        changed = True

    if NOTE_COLUMN not in df.columns:
        df[NOTE_COLUMN] = ""
        changed = True
    if CUM_PROFIT_COLUMN not in df.columns:
        df[CUM_PROFIT_COLUMN] = ""
        changed = True

    cost = pd.to_numeric(df[COST_COLUMN], errors="coerce")
    if cost.isna().any():
        df[COST_COLUMN] = cost.fillna(0.0)
        changed = True
    else:
        df[COST_COLUMN] = cost

    ordered = _ordered_fund_pool_columns(df.columns)
    if list(df.columns) != ordered:
        df = df[ordered]
        changed = True

    if not changed:
        return False
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        _safe_print(f"⚠️ 已补齐基金池列：{path}")
        return True
    except Exception as exc:
        _safe_print(f"⚠️ 回写基金池列失败: {exc}")
        return False


def ensure_all_fund_pool_schemas():
    if not os.path.isdir(PROFILES_DIR):
        return
    try:
        names = os.listdir(PROFILES_DIR)
    except OSError:
        return
    for item in names:
        if item.startswith("."):
            continue
        path = os.path.join(PROFILES_DIR, item, "fund_pool.csv")
        if os.path.isfile(path):
            ensure_fund_pool_schema(path)


def _write_empty_trade_log(path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(EMPTY_TRADE_LOG, file, ensure_ascii=False, indent=2)


def _migrate_file(src, dest):
    if not src or not os.path.isfile(src):
        return False
    if os.path.isfile(dest):
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(src, dest)
    try:
        os.remove(src)
    except OSError:
        pass
    return True


def _ensure_account_files(name, migrate_legacy=False):
    folder = _account_dir(name)
    os.makedirs(folder, exist_ok=True)
    pool_path = os.path.join(folder, "fund_pool.csv")
    log_path = os.path.join(folder, "trade_log.json")
    if migrate_legacy:
        _migrate_file(LEGACY_FUND_POOL_PATH, pool_path)
        _migrate_file(LEGACY_TRADE_LOG_PATH, log_path)
        _migrate_file(LEGACY_SIGNAL_PATH, os.path.join(folder, "latest_signal.csv"))
        _migrate_file(LEGACY_META_PATH, os.path.join(folder, "app_meta.json"))
    if not os.path.isfile(pool_path):
        _write_fund_pool_template(pool_path)
    if not os.path.isfile(log_path):
        _write_empty_trade_log(log_path)
    return folder


def list_accounts():
    ensure_profiles_initialized()
    names = []
    try:
        for item in os.listdir(PROFILES_DIR):
            if item.startswith("."):
                continue
            if os.path.isdir(_account_dir(item)):
                names.append(item)
    except FileNotFoundError:
        names = []
    names.sort()
    if DEFAULT_ACCOUNT_NAME in names:
        names.remove(DEFAULT_ACCOUNT_NAME)
        names.insert(0, DEFAULT_ACCOUNT_NAME)
    return names


def get_current_account_name():
    ensure_profiles_initialized()
    accounts = list_accounts()
    saved = _read_current_account()
    if saved in accounts:
        return saved
    if DEFAULT_ACCOUNT_NAME in accounts:
        _write_current_account(DEFAULT_ACCOUNT_NAME)
        return DEFAULT_ACCOUNT_NAME
    if accounts:
        _write_current_account(accounts[0])
        return accounts[0]
    _ensure_account_files(DEFAULT_ACCOUNT_NAME, migrate_legacy=True)
    _write_current_account(DEFAULT_ACCOUNT_NAME)
    return DEFAULT_ACCOUNT_NAME


def set_current_account(name):
    ok, text = validate_account_name(name)
    if not ok:
        raise ValueError(text)
    if text not in list_accounts():
        raise FileNotFoundError(f"账户不存在: {text}")
    _write_current_account(text)
    return text


def get_current_profile_path():
    """返回当前账户文件夹路径，例如 config/profiles/默认账户。"""
    name = get_current_account_name()
    folder = _ensure_account_files(name, migrate_legacy=False)
    return folder


def get_fund_pool_path():
    return os.path.join(get_current_profile_path(), "fund_pool.csv")


def get_trade_log_path():
    return os.path.join(get_current_profile_path(), "trade_log.json")


def get_signal_path():
    return os.path.join(get_current_profile_path(), "latest_signal.csv")


def get_app_meta_path():
    return os.path.join(get_current_profile_path(), "app_meta.json")


def get_profile_meta_path():
    return os.path.join(get_current_profile_path(), "profile_meta.json")


def can_delete_account(name=None):
    current = name or get_current_account_name()
    accounts = list_accounts()
    if len(accounts) <= 1:
        return False
    if current == DEFAULT_ACCOUNT_NAME:
        return False
    return current in accounts


def create_account(name):
    ok, text = validate_account_name(name)
    if not ok:
        return False, text
    ensure_profiles_initialized()
    folder = _account_dir(text)
    if os.path.isdir(folder):
        return False, f"账户已存在: {text}"
    _ensure_account_files(text, migrate_legacy=False)
    set_current_account(text)
    return True, text


def delete_account(name=None):
    current = name or get_current_account_name()
    if not can_delete_account(current):
        return False, "默认账户或仅剩一个账户时不能删除"
    folder = _account_dir(current)
    try:
        shutil.rmtree(folder)
    except OSError as exc:
        return False, f"删除失败: {exc}"
    remaining = list_accounts()
    fallback = DEFAULT_ACCOUNT_NAME if DEFAULT_ACCOUNT_NAME in remaining else (remaining[0] if remaining else DEFAULT_ACCOUNT_NAME)
    if fallback not in remaining:
        _ensure_account_files(fallback, migrate_legacy=False)
    _write_current_account(fallback)
    return True, fallback


def ensure_profiles_initialized():
    """创建 profiles 目录；若无账户则建立默认账户并迁移旧文件。"""
    os.makedirs(PROFILES_DIR, exist_ok=True)
    has_account = False
    try:
        for item in os.listdir(PROFILES_DIR):
            if not item.startswith(".") and os.path.isdir(_account_dir(item)):
                has_account = True
                break
    except FileNotFoundError:
        has_account = False
    if not has_account:
        _ensure_account_files(DEFAULT_ACCOUNT_NAME, migrate_legacy=True)
        _write_current_account(DEFAULT_ACCOUNT_NAME)
        ensure_all_fund_pool_schemas()
        return
    if not os.path.isdir(_account_dir(DEFAULT_ACCOUNT_NAME)):
        if os.path.isfile(LEGACY_FUND_POOL_PATH) or os.path.isfile(LEGACY_TRADE_LOG_PATH):
            _ensure_account_files(DEFAULT_ACCOUNT_NAME, migrate_legacy=True)
    saved = _read_current_account()
    accounts = []
    for item in os.listdir(PROFILES_DIR):
        if not item.startswith(".") and os.path.isdir(_account_dir(item)):
            accounts.append(item)
    if saved not in accounts:
        fallback = DEFAULT_ACCOUNT_NAME if DEFAULT_ACCOUNT_NAME in accounts else (accounts[0] if accounts else DEFAULT_ACCOUNT_NAME)
        if fallback not in accounts:
            _ensure_account_files(fallback, migrate_legacy=True)
        _write_current_account(fallback)
    ensure_all_fund_pool_schemas()


FUND_POOL_PATH = _LazyAccountPath(get_fund_pool_path)


def _empty_nav_df():
    return pd.DataFrame(columns=EMPTY_NAV_COLUMNS)


def _ensure_raw_dir():
    try:
        os.makedirs(RAW_DIR, exist_ok=True)
    except OSError as exc:
        _safe_print(f"⚠️ 创建目录失败 {RAW_DIR}: {exc}")
        raise


def load_local_data(fund_code):
    """读取 data/raw/{fund_code}.parquet；不存在或读取失败时返回空 DataFrame。"""
    try:
        _ensure_raw_dir()
        pattern = os.path.join(RAW_DIR, f"{fund_code}.parquet")
        matches = glob.glob(pattern)
        if not matches:
            return _empty_nav_df()

        df = pd.read_parquet(matches[0])
        if df is None or df.empty:
            return _empty_nav_df()

        missing = [col for col in EMPTY_NAV_COLUMNS if col not in df.columns]
        if missing:
            _safe_print(f"⚠️ 本地文件字段缺失 [{fund_code}]: {missing}")
            return _empty_nav_df()

        df = df[EMPTY_NAV_COLUMNS].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
        df = df.dropna(subset=["date", "nav"]).sort_values("date", ascending=True)
        return df.reset_index(drop=True)
    except FileNotFoundError:
        return _empty_nav_df()
    except Exception as exc:
        _safe_print(f"⚠️ 读取本地数据失败 [{fund_code}]: {exc}")
        return _empty_nav_df()


def save_local_data(fund_code, df):
    """将净值 DataFrame 覆盖写入 data/raw/{fund_code}.parquet。"""
    try:
        if df is None or df.empty:
            _safe_print(f"⚠️ 跳过保存 [{fund_code}]：传入数据为空")
            return

        _ensure_raw_dir()
        output_path = os.path.join(RAW_DIR, f"{fund_code}.parquet")
        save_df = df.copy()
        save_df["date"] = pd.to_datetime(save_df["date"], errors="coerce")
        save_df = save_df.dropna(subset=["date"]).sort_values("date", ascending=True)
        save_df.to_parquet(output_path, index=False)
    except Exception as exc:
        _safe_print(f"⚠️ 保存本地数据失败 [{fund_code}]: {exc}")


def _merge_incremental(local_df, remote_df):
    """本地已有数据时，仅在远程存在更新日期时合并，并按 date 去重保留最新一条。"""
    local_max_date = local_df["date"].max()
    newer_mask = remote_df["date"] > local_max_date
    if not newer_mask.any():
        return local_df.copy(), False

    merged = pd.concat([local_df, remote_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="last")
    merged = merged.sort_values("date", ascending=True).reset_index(drop=True)
    return merged, True


def update_all_funds(progress_callback=None):
    """读取基金池，增量更新全部基金净值到 data/raw/。结束后清理过期赛道缓存。"""
    try:
        try:
            pool = pd.read_csv(get_fund_pool_path(), dtype={"基金代码": str})
        except FileNotFoundError:
            _safe_print(f"❌ 未找到基金池文件: {get_fund_pool_path()}")
            return
        except Exception as exc:
            _safe_print(f"❌ 读取基金池失败: {exc}")
            return

        if pool.empty or "基金代码" not in pool.columns:
            _safe_print("❌ 基金池为空或缺少「基金代码」列")
            return

        fund_codes = (
            pool["基金代码"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .tolist()
        )
        if not fund_codes:
            _safe_print("❌ 基金池中没有有效的基金代码")
            return

        update_selected_funds(fund_codes, progress_callback=progress_callback)
    finally:
        try:
            from ..factor_layer.sector_classifier import cleanup_sector_cache
            cleanup_sector_cache(30)
        except Exception as exc:
            _safe_print(f"⚠️ 清理赛道缓存失败: {exc}")


def update_selected_funds(fund_codes, progress_callback=None):
    """增量更新指定基金净值到 data/raw/，不遍历整个基金池。"""
    from . import fetcher

    codes = []
    seen = set()
    for item in fund_codes or []:
        code = str(item or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    if not codes:
        _safe_print("⚠️ 没有可更新的基金代码")
        return 0

    total = len(codes)
    updated = 0
    for index, fund_code in enumerate(codes, start=1):
        if progress_callback is not None:
            try:
                progress_callback(index, total, fund_code)
            except Exception:
                pass
        try:
            local_df = load_local_data(fund_code)
            remote_df = fetcher.fetch_fund_history(fund_code)

            if remote_df.empty:
                _safe_print(f"⚠️ 网络数据为空，保留本地数据 [{fund_code}]")
            elif local_df.empty:
                save_local_data(fund_code, remote_df)
                updated += 1
            else:
                merged_df, has_new = _merge_incremental(local_df, remote_df)
                if has_new:
                    save_local_data(fund_code, merged_df)
                    updated += 1

            _safe_print(f"✅ 已更新 {fund_code}")
        except Exception as exc:
            _safe_print(f"❌ 更新失败 [{fund_code}]: {exc}")

        try:
            time.sleep(1.5)
        except Exception as exc:
            _safe_print(f"⚠️ 节流等待异常 [{fund_code}]: {exc}")
    return updated


try:
    ensure_profiles_initialized()
except Exception as exc:
    _safe_print(f"⚠️ 账户初始化失败: {exc}")
