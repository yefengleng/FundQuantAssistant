import json
import os
import re
import shutil
from datetime import datetime
from io import StringIO

import pandas as pd

from ..data_layer.loader import (
    COST_COLUMN,
    CUM_PROFIT_COLUMN,
    FUND_POOL_HEADER,
    NOTE_COLUMN,
    _ordered_fund_pool_columns,
    _safe_print,
    ensure_fund_pool_schema,
    get_current_profile_path,
    get_fund_pool_path,
    get_profile_meta_path,
    get_trade_log_path,
    load_local_data,
    save_local_data,
)
from ..factor_layer.sector_classifier import auto_tag_fund


UNDO_DIR_NAME = ".ocr_undo"


def _undo_dir():
    return os.path.join(get_current_profile_path(), UNDO_DIR_NAME)


def _read_pool():
    path = get_fund_pool_path()
    ensure_fund_pool_schema(path)
    try:
        pool = pd.read_csv(path, dtype={"基金代码": str})
    except Exception:
        return pd.DataFrame(columns=FUND_POOL_HEADER)
    if "基金代码" in pool.columns:
        pool["基金代码"] = pool["基金代码"].astype(str).str.strip()
    return pool


def _write_pool(pool):
    path = get_fund_pool_path()
    if "基金代码" in pool.columns:
        pool = pool.copy()
        pool["基金代码"] = pool["基金代码"].astype(str).str.strip()
    columns = _ordered_fund_pool_columns(pool.columns)
    pool[columns].to_csv(path, index=False, encoding="utf-8-sig")


def _read_trade_log():
    path = get_trade_log_path()
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        payload = {"records": []}
    if not isinstance(payload, dict):
        payload = {"records": []}
    records = payload.get("records")
    if not isinstance(records, list):
        records = []
    payload["records"] = records
    return payload


def _write_trade_log(payload):
    path = get_trade_log_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_profile_meta():
    path = get_profile_meta_path()
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict):
            payload.setdefault("yesterday_profits", {})
            payload.setdefault("cumulative_profits", {})
            return payload
    except Exception:
        pass
    return {"yesterday_profits": {}, "cumulative_profits": {}}


def save_profile_meta(payload):
    path = get_profile_meta_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, dict) else {}
    data.setdefault("yesterday_profits", {})
    data.setdefault("cumulative_profits", {})
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return data


def create_undo_snapshot():
    folder = _undo_dir()
    os.makedirs(folder, exist_ok=True)
    pool_src = get_fund_pool_path()
    log_src = get_trade_log_path()
    meta_src = get_profile_meta_path()
    if os.path.isfile(pool_src):
        shutil.copy2(pool_src, os.path.join(folder, "fund_pool.csv"))
    if os.path.isfile(log_src):
        shutil.copy2(log_src, os.path.join(folder, "trade_log.json"))
    if os.path.isfile(meta_src):
        shutil.copy2(meta_src, os.path.join(folder, "profile_meta.json"))
    meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "account_path": get_current_profile_path(),
    }
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
    return meta


def has_undo_snapshot():
    folder = _undo_dir()
    return os.path.isfile(os.path.join(folder, "fund_pool.csv"))


def restore_undo_snapshot():
    folder = _undo_dir()
    pool_bak = os.path.join(folder, "fund_pool.csv")
    log_bak = os.path.join(folder, "trade_log.json")
    if not os.path.isfile(pool_bak):
        return False, "没有可撤销的导入记录"
    try:
        shutil.copy2(pool_bak, get_fund_pool_path())
        if os.path.isfile(log_bak):
            shutil.copy2(log_bak, get_trade_log_path())
        meta_bak = os.path.join(folder, "profile_meta.json")
        if os.path.isfile(meta_bak):
            shutil.copy2(meta_bak, get_profile_meta_path())
        return True, "已回滚到导入前状态"
    except Exception as exc:
        return False, f"撤销失败: {exc}"


def _tag_sector(code, name):
    try:
        tagged = auto_tag_fund(code, name, allow_network=False)
    except Exception:
        tagged = {}
    if isinstance(tagged, dict):
        if tagged.get("source") == "global_manual":
            return ""
        return str(tagged.get("sector") or "")
    return str(tagged or "")


def _ensure_row_columns(pool):
    for column in FUND_POOL_HEADER:
        if column not in pool.columns:
            pool[column] = 0.0 if column in {COST_COLUMN, "持有份额", "持仓市值", CUM_PROFIT_COLUMN} else ""
    return pool


CSV_COLUMN_ALIASES = {
    "fund_name": ("基金名称", "基金简称", "名称", "fund_name", "fundname", "name"),
    "fund_code": ("基金代码", "代码", "fund_code", "fundcode", "code"),
    "hold_amount": (
        "持有金额(元)", "持有金额（元）", "持有金额", "持有市值", "当前市值", "市值",
        "hold_amount", "market_value", "amount",
    ),
    "weight_pct": ("占比(%)", "占比（%）", "仓位占比", "占比", "weight_pct", "weight", "pct"),
    "yesterday_profit": (
        "昨日收益(元)", "昨日收益（元）", "昨日收益", "日收益", "日盈亏",
        "yesterday_profit", "daily_pnl",
    ),
    "hold_profit": ("持有收益(元)", "持有收益（元）", "持有收益", "hold_profit", "pnl"),
    "hold_return_rate": ("持有收益率(%)", "持有收益率（%）", "持有收益率", "收益率", "hold_return_rate", "return_rate"),
    "cumulative_profit": ("累计收益(元)", "累计收益（元）", "累计收益", "cumulative_profit"),
    "remark": ("备注", "标签", "remark", "note"),
}


def _normalize_header(name):
    return re.sub(r"\s+", "", str(name or "").strip())


def _csv_block(text):
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    for index, line in enumerate(lines):
        compact = line.replace(" ", "").replace("\u3000", "")
        if "基金名称" in compact or "基金代码" in compact or "fund_name" in compact.lower() or "fund_code" in compact.lower():
            return "\n".join(lines[index:]).strip()
    return str(text or "").strip()


def _read_markdown_table(text):
    lines = [line.rstrip() for line in str(text or "").splitlines() if str(line).strip()]
    pipe_lines = [line for line in lines if "|" in line]
    if len(pipe_lines) < 2:
        return None
    rows = []
    for line in pipe_lines:
        if re.match(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells:
            rows.append(cells)
    if len(rows) < 2:
        return None
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(rows[1:], columns=rows[0], dtype=str)


def _read_csv_text(text):
    raw = _csv_block(text).lstrip("\ufeff")
    if not raw:
        return None, "请先粘贴CSV文本。"
    if raw.count("，") > raw.count(","):
        raw = raw.replace("，", ",")
    frame = _read_markdown_table(raw)
    if frame is not None and not frame.empty:
        return frame, ""
    last_error = None
    for kwargs in (
        {"sep": ",", "engine": "python"},
        {"sep": "\t", "engine": "python"},
        {"sep": ";", "engine": "python"},
    ):
        try:
            candidate = pd.read_csv(StringIO(raw), dtype=str, keep_default_na=False, **kwargs)
        except Exception as exc:
            last_error = exc
            continue
        if candidate is None or candidate.empty:
            continue
        if candidate.shape[1] < 2:
            last_error = last_error or "未能按分隔符拆出多列"
            continue
        return candidate, ""
    return None, f"CSV解析失败: {last_error or '没有有效数据行'}"


def _map_csv_columns(columns):
    mapping = {}
    used = set()
    normalized = [(_normalize_header(column), column) for column in columns]
    for field, aliases in CSV_COLUMN_ALIASES.items():
        for alias in aliases:
            alias_key = _normalize_header(alias)
            for compact, original in normalized:
                if original in used or not compact:
                    continue
                if compact != alias_key and alias_key not in compact:
                    continue
                if field == "hold_profit" and "率" in compact:
                    continue
                if field == "fund_name" and "代码" in compact:
                    continue
                mapping[field] = original
                used.add(original)
                break
            if field in mapping:
                break
    return mapping


def _parse_csv_number(value, percent=False):
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text in {"-", "—", "–", "nan", "None", "NaN"}:
        return None
    from .ocr_engine import parse_percent, parse_signed_number

    if percent or "%" in text:
        parsed = parse_percent(text)
        if parsed is not None:
            return parsed
    return parse_signed_number(text)


def parse_holdings_csv(text, column_map=None):
    """把粘贴的CSV/TSV/Markdown表格解析为 extract_holdings_full 同结构的记录。

    返回 (rows, error, meta)。meta 在需要手动指定列映射时含 columns/guess。
    """
    frame, error = _read_csv_text(text)
    if error:
        return [], error, None
    columns = dict(column_map or {}) or _map_csv_columns(frame.columns)
    if "fund_name" not in columns and "fund_code" not in columns:
        return [], "未识别到「基金名称」或「基金代码」列，请手动选择列对应关系。", {
            "need_column_map": True,
            "columns": [str(col) for col in frame.columns],
            "guess": columns,
        }

    def _cell(raw, field):
        column = columns.get(field)
        if not column:
            return ""
        return raw.get(column, "")

    from .ocr_engine import _flag_anomalous_holdings

    rows = []
    skipped = 0
    for _, raw in frame.iterrows():
        name = str(_cell(raw, "fund_name") or "").strip()
        code = str(_cell(raw, "fund_code") or "").strip()
        if (not name or name.lower() in {"nan", "none"}) and (not code or code.lower() in {"nan", "none"}):
            skipped += 1
            continue
        hold_amount = _parse_csv_number(_cell(raw, "hold_amount"))
        hold_profit = _parse_csv_number(_cell(raw, "hold_profit"))
        record = {
            "fund_name": name,
            "fund_code": code,
            "hold_amount": hold_amount,
            "yesterday_profit": _parse_csv_number(_cell(raw, "yesterday_profit")),
            "hold_profit": hold_profit,
            "hold_return_rate": _parse_csv_number(_cell(raw, "hold_return_rate"), percent=True),
            "cumulative_profit": _parse_csv_number(_cell(raw, "cumulative_profit")),
            "weight_pct": _parse_csv_number(_cell(raw, "weight_pct"), percent=True),
            "remark": str(_cell(raw, "remark") or "").strip(),
            "confidence": 1.0,
            "raw_text": ",".join(str(raw.get(col, "") or "") for col in frame.columns),
            "parse_mode": "csv",
            "needs_review": hold_amount is None or hold_amount <= 0 or not name,
            "issues": [],
        }
        _flag_anomalous_holdings(record)
        rows.append(record)
    if not rows:
        return [], "没有解析到有效基金行，请确认CSV包含数据。", None
    return rows, "", {"skipped": skipped, "columns": columns}


def resolve_latest_nav(fund_code, allow_network=True):
    code = "" if fund_code is None else str(fund_code).strip()
    if not code:
        return None, None
    local = load_local_data(code)
    if local is not None and not local.empty:
        last = local.iloc[-1]
        try:
            return float(last["nav"]), last.get("date")
        except (TypeError, ValueError):
            pass
    if not allow_network:
        return None, None
    try:
        from ..data_layer.fetcher import fetch_fund_history

        remote = fetch_fund_history(code)
        if remote is not None and not remote.empty:
            save_local_data(code, remote)
            last = remote.iloc[-1]
            return float(last["nav"]), last.get("date")
    except Exception:
        return None, None
    return None, None


def build_holdings_import_preview(extracted_rows, allow_network=True):
    from .fund_matcher import list_fund_candidates, match_fund_code

    preview = []
    for row in extracted_rows or []:
        item = dict(row)
        provided_code = str(item.get("fund_code") or "").strip()
        if provided_code.lower() in {"", "nan", "none", "nat"}:
            provided_code = ""
        matched = match_fund_code(item.get("fund_name")) or {}
        candidates = matched.get("candidates") or list_fund_candidates(item.get("fund_name"))
        item["match"] = matched
        item["match"]["candidates"] = candidates
        item["match_tier"] = matched.get("match_tier") or "fail"
        item["fail_reason"] = matched.get("fail_reason") or ""
        if provided_code:
            item["fund_code"] = provided_code
            item["match_score"] = 1.0 if str(matched.get("fund_code") or "") == provided_code else 0.95
            item["match_tier"] = "high"
        elif matched.get("match_tier") in {"high", "partial"} and matched.get("fund_code"):
            item["fund_code"] = matched.get("fund_code") or ""
            item["match_score"] = float(matched.get("confidence") or 0.0)
        else:
            item["fund_code"] = ""
            item["match_score"] = float(candidates[0]["score"]) if candidates else 0.0
        nav, nav_date = resolve_latest_nav(item.get("fund_code"), allow_network=allow_network)
        item["nav"] = nav
        item["nav_date"] = str(nav_date)[:10] if nav_date is not None and str(nav_date) not in {"NaT", "nan"} else ""
        hold_amount = item.get("hold_amount")
        hold_profit = item.get("hold_profit")
        try:
            hold_amount = None if hold_amount is None else float(hold_amount)
        except (TypeError, ValueError):
            hold_amount = None
        try:
            hold_profit = None if hold_profit is None else float(hold_profit)
        except (TypeError, ValueError):
            hold_profit = None
        item["hold_amount"] = hold_amount
        item["hold_profit"] = hold_profit
        if hold_amount is not None and nav and float(nav) > 0:
            item["shares"] = hold_amount / float(nav)
        else:
            item["shares"] = None
        if hold_amount is not None and hold_profit is not None:
            item["cost"] = hold_amount - hold_profit
        else:
            item["cost"] = None
        ocr_rate = item.get("hold_return_rate")
        try:
            ocr_rate = None if ocr_rate is None else float(ocr_rate)
        except (TypeError, ValueError):
            ocr_rate = None
        item["hold_return_rate"] = ocr_rate
        system_rate = None
        if item["cost"] and float(item["cost"]) != 0 and hold_profit is not None:
            system_rate = hold_profit / float(item["cost"]) * 100.0
        item["system_return_rate"] = system_rate
        item["rate_diff"] = None
        if ocr_rate is not None and system_rate is not None:
            item["rate_diff"] = system_rate - ocr_rate
        item["include"] = True
        issues = [str(item_text) for item_text in (item.get("issues") or []) if item_text]
        if not item["fund_code"]:
            issues.append("基金代码未找到")
        elif nav is None:
            issues.append("缺少最新净值，请手动填写或跳过")
        if item.get("needs_review") and "字段可能错位，请在预览中核对" not in issues:
            if item.get("parse_mode") == "heuristic":
                issues.append("字段可能错位，请在预览中核对")
        seen = []
        for text in issues:
            if text not in seen:
                seen.append(text)
        item["issues"] = seen
        item["issue_text"] = "；".join(seen)
        item["skip_reason"] = item["issue_text"]
        if seen:
            item["needs_review"] = True
            _safe_print(f"[WARN] 持仓导入 {item.get('fund_name') or item.get('fund_code')}: {item['issue_text']}")
        preview.append(item)
    return preview


REMAP_FIELDS = (
    ("hold_amount", "持有金额"),
    ("yesterday_profit", "昨日收益"),
    ("hold_profit", "持有收益"),
    ("cumulative_profit", "累计收益"),
    ("weight_pct", "占比"),
    ("hold_return_rate", "持有收益率"),
)

CSV_MAP_FIELDS = (
    ("fund_name", "基金名称"),
    ("fund_code", "基金代码（可选）"),
    ("hold_amount", "持有金额"),
    ("weight_pct", "占比"),
    ("yesterday_profit", "昨日收益"),
    ("hold_profit", "持有收益"),
    ("hold_return_rate", "持有收益率"),
    ("cumulative_profit", "累计收益"),
    ("remark", "备注"),
)


def remap_preview_columns(rows, mapping):
    """mapping: 目标字段 -> 来源字段。按原始值重填，避免连环覆盖。"""
    remapped = []
    for row in rows or []:
        original = dict(row)
        item = dict(row)
        for dest, source in (mapping or {}).items():
            if dest == source:
                continue
            item[dest] = original.get(source)
        hold_amount = item.get("hold_amount")
        hold_profit = item.get("hold_profit")
        nav = item.get("nav")
        try:
            hold_amount = None if hold_amount in {None, ""} else float(hold_amount)
        except (TypeError, ValueError):
            hold_amount = None
        try:
            hold_profit = None if hold_profit in {None, ""} else float(hold_profit)
        except (TypeError, ValueError):
            hold_profit = None
        try:
            nav = None if nav in {None, ""} else float(nav)
        except (TypeError, ValueError):
            nav = None
        item["hold_amount"] = hold_amount
        item["hold_profit"] = hold_profit
        item["nav"] = nav
        if hold_amount is not None and nav and nav > 0:
            item["shares"] = hold_amount / nav
        if hold_amount is not None and hold_profit is not None:
            item["cost"] = hold_amount - hold_profit
        remapped.append(item)
    return remapped


def validate_holdings_preview(rows):
    warnings = []
    weights = []
    for row in rows or []:
        if row.get("include") is False:
            continue
        name = row.get("fund_name") or row.get("fund_code") or "未命名"
        issues = list(row.get("issues") or [])
        try:
            amount = float(row.get("hold_amount"))
        except (TypeError, ValueError):
            amount = None
        try:
            yesterday = float(row.get("yesterday_profit"))
        except (TypeError, ValueError):
            yesterday = None
        try:
            hold_profit = float(row.get("hold_profit"))
        except (TypeError, ValueError):
            hold_profit = None
        if amount is None:
            warnings.append(f"{name}：持有金额格式错误或为空。")
        elif amount < 0:
            warnings.append(f"{name}：持有金额为负数，请修改后再导入。")
        elif amount == 0:
            warnings.append(f"{name}：持有金额无效。")
        if row.get("yesterday_profit") not in {None, ""} and yesterday is None:
            warnings.append(f"{name}：昨日收益不是有效数字。")
        if row.get("hold_profit") not in {None, ""} and hold_profit is None:
            warnings.append(f"{name}：持有收益不是有效数字。")
        if amount is not None and yesterday is not None and amount < abs(yesterday):
            warnings.append(f"{name}：持有金额不大于昨日收益，请确认是否字段错位。")
        try:
            weight = float(row.get("weight_pct"))
        except (TypeError, ValueError):
            weight = None
        if weight is not None:
            weights.append(weight)
        for issue in issues:
            text = f"{name}：{issue}"
            if text not in warnings:
                warnings.append(text)
    if weights:
        total = sum(weights)
        if abs(total - 100.0) > 5.0:
            warnings.append(f"占比合计 {total:.2f}%，通常应接近 100%，请核对截图是否完整。")
    return warnings


def apply_holdings_import(rows):
    """按用户确认结果合并持仓：已存在则覆盖份额/市值/成本，不覆盖买入日期和赛道。"""
    create_undo_snapshot()
    pool = _ensure_row_columns(_read_pool())
    profile_meta = load_profile_meta()
    yesterday_map = profile_meta.setdefault("yesterday_profits", {})
    cumulative_map = profile_meta.setdefault("cumulative_profits", {})
    added = 0
    updated = 0
    skipped = 0
    for row in rows or []:
        if not row.get("include", True):
            skipped += 1
            continue
        code = str(row.get("fund_code") or "").strip()
        name = str(row.get("fund_name") or "").strip()
        if not code:
            skipped += 1
            continue
        try:
            shares = float(row.get("shares") if row.get("shares") is not None else 0.0)
        except (TypeError, ValueError):
            shares = 0.0
        market_value = row.get("hold_amount", row.get("market_value"))
        try:
            market_value = None if market_value is None or market_value == "" else float(market_value)
        except (TypeError, ValueError):
            market_value = None
        cost = row.get("cost")
        try:
            cost = None if cost is None or cost == "" else float(cost)
        except (TypeError, ValueError):
            cost = None
        remark = "" if row.get("remark") is None else str(row.get("remark")).strip()
        cumulative = row.get("cumulative_profit")
        try:
            cumulative = None if cumulative is None or cumulative == "" else float(cumulative)
        except (TypeError, ValueError):
            cumulative = None
        yesterday = row.get("yesterday_profit")
        try:
            yesterday = None if yesterday is None or yesterday == "" else float(yesterday)
        except (TypeError, ValueError):
            yesterday = None

        mask = pool["基金代码"] == code
        if mask.any():
            pool.loc[mask, "持有份额"] = shares
            if market_value is not None:
                pool.loc[mask, "持仓市值"] = market_value
            if cost is not None:
                pool.loc[mask, COST_COLUMN] = cost
            if remark:
                pool.loc[mask, NOTE_COLUMN] = remark
            if cumulative is not None:
                pool.loc[mask, CUM_PROFIT_COLUMN] = cumulative
            updated += 1
        else:
            new_row = {column: "" for column in pool.columns}
            new_row["基金代码"] = code
            new_row["基金名称"] = name
            new_row["赛道归类"] = _tag_sector(code, name)
            new_row["持有份额"] = shares
            new_row["买入日期"] = ""
            new_row[COST_COLUMN] = 0.0 if cost is None else cost
            new_row["持仓市值"] = 0.0 if market_value is None else market_value
            new_row[NOTE_COLUMN] = remark
            new_row[CUM_PROFIT_COLUMN] = "" if cumulative is None else cumulative
            pool = pd.concat([pool, pd.DataFrame([new_row])], ignore_index=True)
            added += 1
        if yesterday is not None:
            yesterday_map[code] = yesterday
        if cumulative is not None:
            cumulative_map[code] = cumulative
    _write_pool(pool)
    save_profile_meta(profile_meta)
    return True, {"added": added, "updated": updated, "skipped": skipped}


def apply_transaction_import(rows):
    """按买入/卖出调整份额，并写入 source=ocr_import 的交易日志。"""
    create_undo_snapshot()
    pool = _ensure_row_columns(_read_pool())
    payload = _read_trade_log()
    applied = 0
    skipped = 0
    warnings = []
    for row in rows or []:
        if not row.get("include", True):
            skipped += 1
            continue
        code = str(row.get("fund_code") or "").strip()
        name = str(row.get("fund_name") or "").strip()
        action = str(row.get("action") or "").strip()
        if not code or action not in {"买入", "卖出"}:
            skipped += 1
            continue
        try:
            shares = abs(float(row.get("shares") or 0.0))
        except (TypeError, ValueError):
            skipped += 1
            continue
        if shares <= 0:
            skipped += 1
            continue
        date_text = str(row.get("date") or "").strip() or datetime.now().strftime("%Y-%m-%d")
        mask = pool["基金代码"] == code
        if mask.any():
            current = float(pd.to_numeric(pool.loc[mask, "持有份额"], errors="coerce").fillna(0).iloc[0])
        else:
            current = 0.0
            new_row = {column: "" for column in pool.columns}
            new_row["基金代码"] = code
            new_row["基金名称"] = name
            new_row["赛道归类"] = _tag_sector(code, name)
            new_row["持有份额"] = 0.0
            new_row["买入日期"] = date_text if action == "买入" else ""
            new_row[COST_COLUMN] = 0.0
            new_row["持仓市值"] = 0.0
            pool = pd.concat([pool, pd.DataFrame([new_row])], ignore_index=True)
            mask = pool["基金代码"] == code
        if action == "买入":
            pool.loc[mask, "持有份额"] = current + shares
            if not str(pool.loc[mask, "买入日期"].iloc[0] or "").strip():
                pool.loc[mask, "买入日期"] = date_text
        else:
            next_shares = current - shares
            if next_shares < 0:
                warnings.append(f"{code} 卖出后份额将为负，已置 0")
                next_shares = 0.0
            pool.loc[mask, "持有份额"] = next_shares
        payload["records"].append(
            {
                "fund_code": code,
                "action": action,
                "shares": shares,
                "date": date_text,
                "source": "ocr_import",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        applied += 1
    _write_pool(pool)
    _write_trade_log(payload)
    return True, {"applied": applied, "skipped": skipped, "warnings": warnings}
