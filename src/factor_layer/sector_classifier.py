import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd

from ..data_layer.loader import PROJECT_ROOT, _safe_print


SECTOR_MAPPING_PATH = os.path.join(PROJECT_ROOT, "config", "sector_mapping.json")
GLOBAL_MAPPING_PATH = os.path.join(PROJECT_ROOT, "config", "global_fund_sector_mapping.json")
GLOBAL_SECTOR_LIMITS_PATH = os.path.join(PROJECT_ROOT, "config", "global_sector_limits.json")
SECTOR_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "sector_cache")
AUTO_MATCH_LABEL = "（自动匹配）"
CUSTOM_SECTOR_LABEL = "（自定义赛道…）"
DEFAULT_NEW_SECTOR_LIMIT = 0.10
HOLDINGS_TIMEOUT_SEC = 5
STALE_AFTER_MONTHS = 4
CACHE_MAX_AGE_DAYS = 30
INDUSTRY_COLUMNS = ["序号", "行业类别", "占净值比例", "市值", "截止时间"]

_FALLBACK_MAPPING = {
    "半导体": ["半导体", "芯片", "集成电路", "人工智能"],
    "CPO": ["CPO", "光通信", "通信"],
    "新能源": ["新能源", "光伏", "储能"],
    "美股": ["美股", "纳斯达克", "QDII"],
    "其他": [],
}

# 名称命中这些词时视为高置信，不再穿透持仓。
STRONG_KEYWORDS = frozenset({
    "半导体", "芯片", "集成电路", "纳斯达克", "新能源", "光伏",
    "CPO", "光通信", "5G通信", "标普", "美股", "QDII",
    "军工", "创新药", "CXO", "港创新药", "商业航天", "算力租赁",
    "白酒", "存储芯片", "MLCC指数", "PCB", "固态电池", "锂矿",
    "先锋半导体", "恒生科技", "美国成长", "可控核聚变", "脑机接口",
})

_INDUSTRY_ALIASES = {
    "制造业": ("制造业",),
    "信息传输、软件和信息技术服务业": ("信息传输、软件和信息技术服务业", "信息传输", "软件和信息技术"),
    "科学研究和技术服务业": ("科学研究和技术服务业", "科学研究"),
    "金融业": ("金融业",),
    "采矿业": ("采矿业",),
}

_INDUSTRY_SHORT = {
    "制造业": "制造业",
    "信息传输、软件和信息技术服务业": "信息传输业",
    "科学研究和技术服务业": "科研服务业",
    "金融业": "金融业",
    "采矿业": "采矿业",
}

_MEMORY_CACHE = {}
_GLOBAL_MAPPING_CACHE = {"mtime": None, "data": None}


def load_sector_mapping():
    try:
        with open(SECTOR_MAPPING_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict) and payload:
            return {str(key): list(value or []) for key, value in payload.items()}
    except Exception:
        pass
    return {key: list(value) for key, value in _FALLBACK_MAPPING.items()}


def _keyword_pairs():
    """关键词按长度优先，同长度时优先落到同名赛道。"""
    pairs = []
    for sector, keywords in load_sector_mapping().items():
        if sector == "其他":
            continue
        pairs.append((sector, sector))
        for word in keywords or []:
            text = str(word).strip()
            if text:
                pairs.append((text, sector))
    pairs.sort(key=lambda item: (len(item[0]), item[0] == item[1]), reverse=True)
    return pairs


SECTOR_NAMES = tuple(load_sector_mapping().keys())


def invalidate_global_mapping_cache():
    _GLOBAL_MAPPING_CACHE["mtime"] = None
    _GLOBAL_MAPPING_CACHE["data"] = None


def load_global_sector_mapping():
    """读取全局基金-赛道映射。按文件 mtime 缓存，避免每次调用都读盘。"""
    path = GLOBAL_MAPPING_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        invalidate_global_mapping_cache()
        return {}
    cached = _GLOBAL_MAPPING_CACHE.get("data")
    if _GLOBAL_MAPPING_CACHE.get("mtime") == mtime and cached is not None:
        return dict(cached)
    data = {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            for key, value in payload.items():
                code = str(key).strip()
                sector = str(value or "").strip()
                if code and sector and sector != AUTO_MATCH_LABEL:
                    data[code] = sector
    except Exception:
        data = {}
    _GLOBAL_MAPPING_CACHE["mtime"] = mtime
    _GLOBAL_MAPPING_CACHE["data"] = data
    return dict(data)


def save_global_sector_mapping(mapping):
    data = {}
    for key, value in (mapping or {}).items():
        code = str(key).strip()
        sector = str(value or "").strip()
        if code and sector and sector != AUTO_MATCH_LABEL:
            data[code] = sector
    data = {key: data[key] for key in sorted(data)}
    folder = os.path.dirname(GLOBAL_MAPPING_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(GLOBAL_MAPPING_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    try:
        _GLOBAL_MAPPING_CACHE["mtime"] = os.path.getmtime(GLOBAL_MAPPING_PATH)
    except OSError:
        _GLOBAL_MAPPING_CACHE["mtime"] = None
    _GLOBAL_MAPPING_CACHE["data"] = data
    return dict(data)


def _fallback_sector_limits():
    try:
        from config.settings import SECTOR_LIMITS

        payload = {
            str(key).strip(): float(value)
            for key, value in (SECTOR_LIMITS or {}).items()
            if str(key).strip()
        }
        if payload:
            return payload
    except Exception:
        pass
    return {"半导体": 0.25, "CPO": 0.12, "新能源": 0.10, "美股": 0.13, "其他": 0.08}


def _normalize_sector_limits(payload):
    data = {}
    if not isinstance(payload, dict):
        return data
    for key, value in payload.items():
        name = str(key).strip()
        if not name or name in {AUTO_MATCH_LABEL, CUSTOM_SECTOR_LABEL}:
            continue
        try:
            limit = float(value)
        except (TypeError, ValueError):
            continue
        if limit < 0:
            limit = 0.0
        elif limit > 1.0 and limit <= 100.0:
            limit = limit / 100.0
        if limit > 1.0:
            limit = 1.0
        data[name] = round(limit, 4)
    return data


def _ordered_sector_limits(data):
    ordered = {}
    for key, value in (data or {}).items():
        if key != "其他":
            ordered[key] = value
    if "其他" in (data or {}):
        ordered["其他"] = data["其他"]
    return ordered


def save_sector_limits(limits):
    data = _ordered_sector_limits(_normalize_sector_limits(limits))
    folder = os.path.dirname(GLOBAL_SECTOR_LIMITS_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(GLOBAL_SECTOR_LIMITS_PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return dict(data)


def ensure_sector_limits_file():
    if os.path.isfile(GLOBAL_SECTOR_LIMITS_PATH):
        return GLOBAL_SECTOR_LIMITS_PATH
    save_sector_limits(_fallback_sector_limits())
    return GLOBAL_SECTOR_LIMITS_PATH


def _sync_mapped_sectors(data):
    changed = False
    for sector in load_global_sector_mapping().values():
        name = str(sector or "").strip()
        if not name or name in {AUTO_MATCH_LABEL, CUSTOM_SECTOR_LABEL}:
            continue
        if name not in data:
            data[name] = DEFAULT_NEW_SECTOR_LIMIT
            changed = True
    return data, changed


def get_sector_limits():
    """每次读取 config/global_sector_limits.json。文件不存在则从 settings.SECTOR_LIMITS 迁移。"""
    if not os.path.isfile(GLOBAL_SECTOR_LIMITS_PATH):
        ensure_sector_limits_file()
    data = {}
    try:
        with open(GLOBAL_SECTOR_LIMITS_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        data = _normalize_sector_limits(payload)
    except Exception:
        data = {}
    if not data:
        data = dict(_fallback_sector_limits())
        save_sector_limits(data)
    return dict(data)


def sync_mapped_sectors_into_limits():
    """把已有全局映射中的赛道补进上限配置（缺省 10%）。仅在赛道管理页调用。"""
    data = get_sector_limits()
    data, changed = _sync_mapped_sectors(data)
    if changed:
        data = save_sector_limits(data)
    return dict(data)


def set_sector_limit(name, limit):
    sector = str(name or "").strip()
    if not sector or sector in {AUTO_MATCH_LABEL, CUSTOM_SECTOR_LABEL}:
        return get_sector_limits()
    limits = get_sector_limits()
    parsed = _normalize_sector_limits({sector: limit})
    if sector not in parsed:
        return limits
    limits[sector] = parsed[sector]
    return save_sector_limits(limits)


def ensure_sector_limit(name, default=None):
    """若赛道不在上限配置中则写入，默认上限 10%。返回 (limits, added)。"""
    sector = str(name or "").strip()
    limits = get_sector_limits()
    if not sector or sector in {AUTO_MATCH_LABEL, CUSTOM_SECTOR_LABEL}:
        return limits, False
    if sector in limits:
        return limits, False
    cap = DEFAULT_NEW_SECTOR_LIMIT if default is None else default
    limits[sector] = cap
    return save_sector_limits(limits), True


def funds_mapped_to_sector(name):
    sector = str(name or "").strip()
    if not sector:
        return []
    return [
        code
        for code, mapped in load_global_sector_mapping().items()
        if str(mapped).strip() == sector
    ]


def delete_sector_limit(name, drop_mappings=False):
    """
    删除赛道上限。若仍有基金映射且 drop_mappings=False，不删除并返回这些代码。
    返回 {"ok", "mapped", "limits"}。
    """
    sector = str(name or "").strip()
    mapped = funds_mapped_to_sector(sector)
    if mapped and not drop_mappings:
        return {"ok": False, "mapped": mapped, "limits": get_sector_limits()}
    if drop_mappings and mapped:
        mapping = load_global_sector_mapping()
        for code in mapped:
            mapping.pop(code, None)
            _MEMORY_CACHE.pop(code, None)
        save_global_sector_mapping(mapping)
    limits = get_sector_limits()
    if sector in limits:
        limits.pop(sector, None)
        save_sector_limits(limits)
    return {"ok": True, "mapped": mapped, "limits": get_sector_limits()}


def set_global_sector(fund_code, sector):
    code = str(fund_code or "").strip()
    name = str(sector or "").strip()
    if not code:
        return load_global_sector_mapping()
    mapping = load_global_sector_mapping()
    if not name or name == AUTO_MATCH_LABEL or name == CUSTOM_SECTOR_LABEL:
        mapping.pop(code, None)
    else:
        mapping[code] = name
        ensure_sector_limit(name)
    _MEMORY_CACHE.pop(code, None)
    return save_global_sector_mapping(mapping)


def remove_global_sector(fund_code):
    return set_global_sector(fund_code, "")


def merge_global_sector_mapping(updates):
    mapping = load_global_sector_mapping()
    for key, value in (updates or {}).items():
        code = str(key).strip()
        sector = str(value or "").strip()
        if not code or not sector or sector in {AUTO_MATCH_LABEL, CUSTOM_SECTOR_LABEL}:
            continue
        mapping[code] = sector
        ensure_sector_limit(sector)
        _MEMORY_CACHE.pop(code, None)
    return save_global_sector_mapping(mapping)


def apply_global_sector_edits(updates):
    """一次写入多条编辑。值为空或「（自动匹配）」表示删除手动映射。"""
    mapping = load_global_sector_mapping()
    changed = False
    new_sectors = []
    for key, value in (updates or {}).items():
        code = str(key).strip()
        if not code:
            continue
        sector = str(value or "").strip()
        if not sector or sector in {AUTO_MATCH_LABEL, CUSTOM_SECTOR_LABEL}:
            if code in mapping:
                mapping.pop(code, None)
                _MEMORY_CACHE.pop(code, None)
                changed = True
            continue
        if mapping.get(code) != sector:
            mapping[code] = sector
            _MEMORY_CACHE.pop(code, None)
            changed = True
        _, added = ensure_sector_limit(sector)
        if added:
            new_sectors.append(sector)
    if changed:
        save_global_sector_mapping(mapping)
    return {
        "changed": changed,
        "new_sectors": new_sectors,
        "mapping": load_global_sector_mapping(),
    }


def _resolve_import_code(resolved):
    payload = dict(resolved or {})
    status = str(payload.get("status") or "")
    code = str(payload.get("fund_code") or payload.get("value") or "").strip()
    if status == "ok" and code:
        return code, payload
    if status == "candidates":
        for item in payload.get("candidates") or []:
            cand = str(item.get("fund_code") or "").strip()
            if cand:
                payload["fund_code"] = cand
                payload["matched_name"] = str(item.get("fund_name") or payload.get("matched_name") or "")
                return cand, payload
    if isinstance(payload.get("value"), str) and str(payload.get("value")).strip():
        code = str(payload.get("value")).strip()
        if code:
            return code, payload
    return "", payload


def apply_batch_sector_import(pairs, resolver=None, allow_network=True):
    """
    批量解析并直接写入全局映射。同一基金多次出现时保留最后一次。
    返回差异报告数据，供页面一次性展示。
    """
    if resolver is None:
        from ..ocr.fund_matcher import resolve_fund_identifier

        resolver = resolve_fund_identifier

    written = {}
    seen_count = {}
    skipped = []
    existing_limits = set(get_sector_limits())

    for pair in pairs or []:
        item = dict(pair or {})
        identifier = str(
            item.get("identifier") or item.get("fund_name") or item.get("fund_code") or ""
        ).strip()
        sector = str(item.get("sector") or "").strip()
        if sector in {AUTO_MATCH_LABEL, CUSTOM_SECTOR_LABEL}:
            sector = ""
        if not identifier or not sector:
            skipped.append({"identifier": identifier or "（空）", "reason": "缺少基金标识或赛道"})
            continue
        try:
            resolved, payload = _resolve_import_code(resolver(identifier, allow_network=allow_network))
        except TypeError:
            resolved, payload = _resolve_import_code(resolver(identifier))
        except Exception as exc:
            skipped.append({"identifier": identifier, "reason": f"识别失败：{exc}"})
            continue
        if not resolved:
            reason = str(payload.get("fail_reason") or "未能识别基金")
            skipped.append({"identifier": identifier, "reason": reason})
            continue
        display_name = str(
            payload.get("matched_name") or item.get("fund_name") or identifier
        ).strip()
        seen_count[resolved] = seen_count.get(resolved, 0) + 1
        written[resolved] = {
            "fund_code": resolved,
            "fund_name": display_name,
            "identifier": identifier,
            "sector": sector,
        }

    duplicates = []
    for code, count in seen_count.items():
        if count > 1 and code in written:
            row = written[code]
            duplicates.append(
                {
                    "基金名称": row.get("fund_name") or code,
                    "基金代码": code,
                    "保留赛道": row.get("sector"),
                    "出现次数": count,
                }
            )

    updates = {code: row["sector"] for code, row in written.items()}
    if updates:
        merge_global_sector_mapping(updates)

    diff_rows = []
    for code, row in written.items():
        auto = auto_tag_fund(
            code,
            row.get("fund_name") or row.get("identifier"),
            allow_network=False,
            ignore_global=True,
        )
        auto_sector = str((auto or {}).get("sector") or "其他") if isinstance(auto, dict) else str(auto or "其他")
        user_sector = str(row.get("sector") or "")
        if user_sector and user_sector != auto_sector:
            diff_rows.append(
                {
                    "基金名称": row.get("fund_name") or code,
                    "您输入的赛道": user_sector,
                    "系统自动识别的赛道": auto_sector,
                }
            )

    new_sectors = []
    for sector in dict.fromkeys(row["sector"] for row in written.values()):
        if sector and sector not in existing_limits:
            new_sectors.append(sector)

    remembered = [
        (row.get("identifier") or "", code, row.get("fund_name") or "")
        for code, row in written.items()
    ]
    return {
        "imported": len(written),
        "diff_rows": diff_rows,
        "duplicates": duplicates,
        "new_sectors": new_sectors,
        "skipped": skipped,
        "remembered": remembered,
    }


def clear_global_sector_mapping():
    mapped = list(load_global_sector_mapping().keys())
    for code in mapped:
        _MEMORY_CACHE.pop(code, None)
    return save_global_sector_mapping({})


def apply_global_sector_map(df):
    """用全局映射覆盖 DataFrame 的「赛道归类」列；不改动未映射基金。"""
    if df is None or getattr(df, "empty", True) or "基金代码" not in df.columns:
        return df
    mapping = load_global_sector_mapping()
    if not mapping:
        return df
    work = df.copy()
    if "赛道归类" not in work.columns:
        work["赛道归类"] = ""
    codes = work["基金代码"].astype(str).str.strip()
    overlay = codes.map(mapping)
    mask = overlay.notna() & overlay.astype(str).str.strip().ne("")
    if mask.any():
        work.loc[mask, "赛道归类"] = overlay.loc[mask].astype(str)
    return work


def list_known_sectors():
    """下拉选项：上限配置中的赛道优先，再并入关键词表与已有映射。"""
    ordered = []
    for key in get_sector_limits():
        text = str(key).strip()
        if text and text not in ordered:
            ordered.append(text)
    for key in load_sector_mapping().keys():
        text = str(key).strip()
        if text and text not in ordered:
            ordered.append(text)
    for sector in load_global_sector_mapping().values():
        text = str(sector or "").strip()
        if text and text not in ordered and text not in {AUTO_MATCH_LABEL, CUSTOM_SECTOR_LABEL}:
            ordered.append(text)
    if "其他" in ordered:
        ordered = [item for item in ordered if item != "其他"] + ["其他"]
    return ordered


def resolve_sector_label(text):
    """把 OCR/粘贴中的赛道或关键词规整为正式赛道名。"""
    value = "" if text is None else str(text).strip()
    if not value or value in {AUTO_MATCH_LABEL, "赛道", "赛道名称", "行业"}:
        return ""
    known = list_known_sectors()
    if value in known:
        return value
    pairs = []
    for sector, keywords in load_sector_mapping().items():
        pairs.append((str(sector), str(sector)))
        for word in keywords or []:
            word = str(word).strip()
            if word:
                pairs.append((word, str(sector)))
    pairs.sort(key=lambda item: len(item[0]), reverse=True)
    for word, sector in pairs:
        if not word:
            continue
        if value == word or word in value:
            return sector
    return ""


def format_sector_source(source):
    text = str(source or "").strip()
    if text == "global_manual":
        return "手动映射"
    if text in {"name", "holdings", "fallback"}:
        return f"自动匹配（{text}）"
    return "自动匹配（fallback）"


def collect_all_pool_funds():
    """合并所有账户 fund_pool.csv，按基金代码去重。"""
    from ..data_layer.loader import PROFILES_DIR, list_accounts

    rows = []
    seen = set()
    for account in list_accounts():
        path = os.path.join(PROFILES_DIR, account, "fund_pool.csv")
        if not os.path.isfile(path):
            continue
        try:
            frame = pd.read_csv(path, dtype={"基金代码": str})
        except Exception:
            continue
        if frame is None or frame.empty or "基金代码" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            code = str(row.get("基金代码") or "").strip()
            if not code or code in seen:
                continue
            seen.add(code)
            name = str(row.get("基金名称") or "").strip()
            if name.lower() in {"nan", "none", "<na>"}:
                name = ""
            rows.append({"基金代码": code, "基金名称": name})
    return pd.DataFrame(rows, columns=["基金代码", "基金名称"])


def export_global_mapping_df():
    mapping = load_global_sector_mapping()
    funds = collect_all_pool_funds()
    name_map = {}
    if funds is not None and not funds.empty:
        name_map = dict(
            zip(funds["基金代码"].astype(str), funds["基金名称"].fillna("").astype(str))
        )
    rows = []
    for code, sector in mapping.items():
        rows.append({"基金代码": code, "基金名称": name_map.get(code, ""), "赛道": sector})
    return pd.DataFrame(rows, columns=["基金代码", "基金名称", "赛道"])


def parse_sector_csv_text(text):
    """解析「基金标识,赛道」文本。第一列可以是基金代码或基金名称。"""
    raw = "" if text is None else str(text)
    if not raw.strip():
        return [], "请粘贴 CSV 文本，每行格式为：基金标识,赛道"
    rows = []
    errors = []
    header_names = {"基金名称", "名称", "基金代码", "基金标识", "代码"}
    header_sectors = {"赛道", "赛道名称", "行业"}
    for index, line in enumerate(raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")):
        stripped = line.strip()
        if not stripped:
            continue
        if "\t" in stripped and "," not in stripped.split("\t")[0]:
            parts = [part.strip() for part in stripped.split("\t")]
        else:
            parts = [part.strip().strip('"') for part in stripped.split(",")]
        if len(parts) < 2:
            errors.append(f"第 {index + 1} 行无法解析")
            continue
        ident, sector_text = parts[0], parts[1]
        if ident in header_names or sector_text in header_sectors:
            continue
        if not ident or not sector_text:
            errors.append(f"第 {index + 1} 行缺少基金标识或赛道")
            continue
        sector = resolve_sector_label(sector_text) or sector_text
        rows.append(
            {
                "identifier": ident,
                "fund_name": ident,
                "sector": sector,
                "raw_sector": sector_text,
            }
        )
    if not rows:
        extra = ("：" + "；".join(errors[:3])) if errors else ""
        return [], f"没有解析到有效的 基金标识,赛道 行{extra}"
    return rows, ""


def _classify_mode():
    try:
        from config.settings import SECTOR_CLASSIFY_MODE
        mode = str(SECTOR_CLASSIFY_MODE or "").strip().lower()
        if mode in {"name_only", "hybrid"}:
            return mode
    except Exception:
        pass
    return "hybrid"


def classify_by_keyword(fund_name):
    """从基金名称匹配赛道关键词；无法判断时返回 None。"""
    matched = _match_name(fund_name)
    return None if matched is None else matched[0]


def _match_name(fund_name):
    """返回 (赛道, 命中关键词)，未命中则为 None。"""
    name = "" if fund_name is None else str(fund_name).strip()
    if not name or name.lower() in {"nan", "none"}:
        return None
    lower = name.lower()
    for word, sector in _keyword_pairs():
        if word.lower() in lower or word in name:
            return sector, word
    return None


def _is_strong_keyword(keyword):
    text = "" if keyword is None else str(keyword).strip()
    return bool(text) and text in STRONG_KEYWORDS


def _empty_industry_df():
    return pd.DataFrame(columns=INDUSTRY_COLUMNS)


def _latest_quarter(alloc_df):
    if alloc_df is None or alloc_df.empty or "截止时间" not in alloc_df.columns:
        return alloc_df if alloc_df is not None else _empty_industry_df()
    df = alloc_df.copy()
    df["截止时间"] = pd.to_datetime(df["截止时间"], errors="coerce")
    df = df.dropna(subset=["截止时间"])
    if df.empty:
        return _empty_industry_df()
    latest = df["截止时间"].max()
    return df.loc[df["截止时间"] == latest].copy()


def _call_industry_api(fund_code, year):
    import akshare as ak
    return ak.fund_portfolio_industry_allocation_em(symbol=fund_code, date=str(year))


def fetch_holdings_industry(fund_code):
    """
    拉取基金最新一期季报行业配置。失败或超时（5 秒）返回空表。

    东财接口按年份查询；优先当年，再回退到上年和 2024，并截取最新截止时间。
    """
    code = "" if fund_code is None else str(fund_code).strip()
    if not code:
        return _empty_industry_df()

    this_year = datetime.now().year
    years = []
    for year in (this_year, this_year - 1, 2024):
        if year not in years:
            years.append(year)

    for year in years:
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_call_industry_api, code, year)
                chunk = future.result(timeout=HOLDINGS_TIMEOUT_SEC)
        except Exception:
            continue
        if chunk is None or chunk.empty:
            continue
        latest = _latest_quarter(chunk)
        if latest is not None and not latest.empty:
            return latest
        return chunk
    return _empty_industry_df()


def _to_percent_series(values):
    series = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if series.empty:
        return series
    if float(series.max()) <= 1.0:
        return series * 100.0
    return series


def _industry_weight(industry_df, category):
    if industry_df is None or industry_df.empty or "行业类别" not in industry_df.columns:
        return 0.0, ""
    aliases = _INDUSTRY_ALIASES.get(category, (category,))
    weights = _to_percent_series(industry_df.get("占净值比例", 0))
    labels = industry_df["行业类别"].astype(str)
    matched = pd.Series(False, index=industry_df.index)
    for alias in aliases:
        if alias == "制造业":
            matched = matched | labels.str.contains("制造业", na=False)
        elif alias == "信息传输":
            matched = matched | (
                labels.str.contains("信息传输", na=False)
                & ~labels.str.contains("制造业", na=False)
            )
        elif alias == "软件和信息技术":
            matched = matched | labels.str.contains("软件和信息技术", na=False)
        else:
            matched = matched | labels.str.contains(alias, na=False)
    if not matched.any():
        return 0.0, ""
    weight = float(weights.loc[matched].sum())
    sample = str(labels.loc[matched].iloc[0])
    return weight, sample


def _name_has_any(fund_name, words):
    name = "" if fund_name is None else str(fund_name)
    return any(word in name for word in words)


def _fmt_pct(value):
    if value is None or pd.isna(value):
        return "0%"
    number = float(value)
    if abs(number - round(number)) < 0.05:
        return f"{int(round(number))}%"
    return f"{number:.1f}%"


def _report_date(industry_df):
    if industry_df is None or industry_df.empty or "截止时间" not in industry_df.columns:
        return None
    dates = pd.to_datetime(industry_df["截止时间"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().to_pydatetime()


def _is_report_stale(report_date, today=None):
    if report_date is None:
        return True
    today = today or datetime.now()
    months = (today.year - report_date.year) * 12 + (today.month - report_date.month)
    if months > STALE_AFTER_MONTHS:
        return True
    if months == STALE_AFTER_MONTHS and today.day > report_date.day:
        return True
    return False


def map_industry_to_sector(industry_df, fund_name):
    """
    按季报行业占比 + 基金名称判断赛道。

    返回 {"sector": 赛道, "reason": 依据, "stale": bool, "report_date": datetime|None}。
    """
    latest = _latest_quarter(industry_df) if industry_df is not None else _empty_industry_df()
    report_date = _report_date(latest)
    stale = _is_report_stale(report_date)
    empty_result = {
        "sector": "其他",
        "reason": "",
        "stale": stale,
        "report_date": report_date,
    }
    if latest is None or latest.empty:
        return empty_result

    manufacturing, _ = _industry_weight(latest, "制造业")
    it_weight, _ = _industry_weight(latest, "信息传输、软件和信息技术服务业")
    research, _ = _industry_weight(latest, "科学研究和技术服务业")
    finance, _ = _industry_weight(latest, "金融业")
    mining, _ = _industry_weight(latest, "采矿业")

    def _result(sector, category, weight):
        short = _INDUSTRY_SHORT.get(category, category)
        return {
            "sector": sector,
            "reason": f"{short} {_fmt_pct(weight)}",
            "stale": stale,
            "report_date": report_date,
        }

    if manufacturing > 30:
        if _name_has_any(fund_name, ("半导体", "芯片", "电子", "集成电路")):
            return _result("半导体", "制造业", manufacturing)
        if _name_has_any(fund_name, ("新能源", "光伏", "电池", "汽车")):
            return _result("新能源", "制造业", manufacturing)
        if _name_has_any(fund_name, ("医疗", "医药", "健康")):
            return _result("医药医疗", "制造业", manufacturing)
        return _result("其他", "制造业", manufacturing)

    if it_weight > 20:
        if _name_has_any(fund_name, ("通信", "5G", "AI", "CPO")):
            return _result("CPO", "信息传输、软件和信息技术服务业", it_weight)
        if _name_has_any(fund_name, ("半导体", "芯片")):
            return _result("半导体", "信息传输、软件和信息技术服务业", it_weight)
        return _result("CPO", "信息传输、软件和信息技术服务业", it_weight)

    if research > 15:
        return _result("医药医疗", "科学研究和技术服务业", research)
    if finance > 30:
        return _result("金融", "金融业", finance)
    if mining > 20:
        return _result("周期", "采矿业", mining)
    return empty_result


def _cache_path(fund_code):
    return os.path.join(SECTOR_CACHE_DIR, f"{fund_code}.json")


def _ensure_cache_dir():
    os.makedirs(SECTOR_CACHE_DIR, exist_ok=True)


def _today_text():
    return datetime.now().strftime("%Y-%m-%d")


def _public_result(payload):
    result = {
        "sector": str(payload.get("sector") or "其他"),
        "source": str(payload.get("source") or "fallback"),
    }
    if payload.get("reason"):
        result["reason"] = str(payload["reason"])
    return result


def _load_disk_cache(fund_code):
    path = _cache_path(fund_code)
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict) or not payload.get("sector"):
            return None
        return payload
    except Exception:
        return None


def _should_reuse_cache(payload):
    if not payload:
        return False
    if str(payload.get("update_time") or "") == _today_text():
        return True
    return payload.get("stale") is not True


def _save_disk_cache(fund_code, result, stale=False, report_date=None):
    if not fund_code:
        return
    payload = {
        "sector": result.get("sector") or "其他",
        "source": result.get("source") or "fallback",
        "update_time": _today_text(),
        "stale": bool(stale),
    }
    if result.get("reason"):
        payload["reason"] = result["reason"]
    if report_date is not None:
        try:
            payload["report_date"] = pd.Timestamp(report_date).strftime("%Y-%m-%d")
        except Exception:
            pass
    try:
        _ensure_cache_dir()
        with open(_cache_path(fund_code), "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _memory_get(fund_code):
    item = _MEMORY_CACHE.get(fund_code)
    if not item:
        return None
    if item.get("update_time") != _today_text():
        _MEMORY_CACHE.pop(fund_code, None)
        return None
    return _public_result(item)


def _memory_set(fund_code, result, stale=False):
    if not fund_code:
        return
    _MEMORY_CACHE[fund_code] = {
        "sector": result.get("sector") or "其他",
        "source": result.get("source") or "fallback",
        "reason": result.get("reason") or "",
        "update_time": _today_text(),
        "stale": bool(stale),
    }


def cleanup_sector_cache(max_age_days=CACHE_MAX_AGE_DAYS):
    """删除 data/sector_cache 中超过 max_age_days 天的缓存文件。"""
    if not os.path.isdir(SECTOR_CACHE_DIR):
        return 0
    cutoff = datetime.now().timestamp() - float(max_age_days) * 86400
    removed = 0
    try:
        names = os.listdir(SECTOR_CACHE_DIR)
    except OSError:
        return 0
    for name in names:
        if not name.endswith(".json"):
            continue
        path = os.path.join(SECTOR_CACHE_DIR, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            continue
    if removed:
        _safe_print(f"🧹 已清理过期赛道缓存 {removed} 个")
    return removed


def auto_tag_fund(fund_code=None, fund_name=None, allow_network=None, ignore_global=False):
    """
    混合模式自动归类。返回 {"sector": 赛道名, "source": "global_manual|name|holdings|fallback"}。

    优先使用 config/global_fund_sector_mapping.json 中的手动映射。
    兼容旧调用 auto_tag_fund(fund_name) 或 auto_tag_fund(fund_code, fund_name)。
    ignore_global=True 时跳过手动映射，只返回关键词/持仓/兜底结果。
    """
    if fund_name is None and fund_code is not None:
        name = str(fund_code).strip()
        code = ""
    else:
        code = "" if fund_code is None else str(fund_code).strip()
        name = "" if fund_name is None else str(fund_name).strip()

    if code:
        if not ignore_global:
            global_sector = load_global_sector_mapping().get(code)
            if global_sector:
                return {"sector": global_sector, "source": "global_manual"}
        cached_memory = _memory_get(code)
        if cached_memory:
            return cached_memory

    matched = _match_name(name)
    if matched and matched[0] != "其他" and _is_strong_keyword(matched[1]):
        result = {"sector": matched[0], "source": "name"}
        _memory_set(code, result, stale=False)
        _save_disk_cache(code, result, stale=False)
        return result

    mode = _classify_mode()
    use_holdings = mode == "hybrid" and bool(code)

    if use_holdings:
        disk = _load_disk_cache(code)
        if _should_reuse_cache(disk) and str(disk.get("source") or "") != "global_manual":
            result = _public_result(disk)
            _memory_set(code, result, stale=bool(disk.get("stale")))
            return result

        if allow_network is not False:
            industry_df = fetch_holdings_industry(code)
            if industry_df is None or industry_df.empty:
                _safe_print("⚠️ 无法获取持仓数据，归入「其他」")
                result = {"sector": "其他", "source": "fallback"}
                _memory_set(code, result, stale=True)
                _save_disk_cache(code, result, stale=True)
                return result

            mapped = map_industry_to_sector(industry_df, name)
            result = {
                "sector": mapped.get("sector") or "其他",
                "source": "holdings",
            }
            if mapped.get("reason"):
                result["reason"] = mapped["reason"]
            stale = bool(mapped.get("stale"))
            _memory_set(code, result, stale=stale)
            _save_disk_cache(code, result, stale=stale, report_date=mapped.get("report_date"))
            return result

    if matched and matched[0] != "其他":
        result = {"sector": matched[0], "source": "name"}
        _memory_set(code, result, stale=False)
        _save_disk_cache(code, result, stale=False)
        return result

    result = {"sector": "其他", "source": "fallback"}
    _memory_set(code, result, stale=False)
    _save_disk_cache(code, result, stale=False)
    return result
