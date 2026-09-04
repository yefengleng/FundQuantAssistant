import json
import re
import sys
import time

import akshare as ak
import pandas as pd
import requests


EMPTY_NAV_COLUMNS = ["date", "nav", "change_pct"]

COLUMN_RENAME_MAP = {
    "净值日期": "date",
    "单位净值": "nav",
    "日增长率": "change_pct",
    "date": "date",
    "nav": "nav",
    "change_pct": "change_pct",
}


def _safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "gbk"
        print(str(message).encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _empty_nav_df():
    return pd.DataFrame(columns=EMPTY_NAV_COLUMNS)


def _fetch_raw_history(fund_code):
    """
    必须按约定调用 ak.fund_open_fund_daily_em(symbol=fund_code)。
    当前 akshare 该接口已改为全市场快照、不再接收 symbol，
    遇到 TypeError 时回退到单基历史净值接口 fund_open_fund_info_em。
    """
    try:
        return ak.fund_open_fund_daily_em(symbol=fund_code)
    except TypeError:
        return ak.fund_open_fund_info_em(
            symbol=fund_code,
            indicator="单位净值走势",
            period="成立来",
        )


def _clean_fund_history(raw_df):
    """清洗 akshare 返回的净值表，统一为 date / nav / change_pct。"""
    if raw_df is None or raw_df.empty:
        raise ValueError("返回空数据")

    df = raw_df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(columns=COLUMN_RENAME_MAP)

    missing_required = [col for col in ("date", "nav") if col not in df.columns]
    if missing_required:
        raise KeyError(f"缺少必要字段: {missing_required}，实际列名: {list(raw_df.columns)}")

    if "change_pct" not in df.columns:
        df["change_pct"] = pd.NA

    df = df[EMPTY_NAV_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")

    change_raw = df["change_pct"].astype(str).str.replace("%", "", regex=False).str.strip()
    df["change_pct"] = pd.to_numeric(change_raw, errors="coerce")

    df = df.dropna(subset=["date", "nav"])
    if df.empty:
        raise ValueError("清洗后数据为空")

    df = df.sort_values("date", ascending=True).drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)


def fetch_fund_history(fund_code, retry_times=3):
    """
    拉取单只基金的历史净值。

    参数:
        fund_code: 基金代码，如 '001186'
        retry_times: 网络失败或空数据时的最大重试次数，默认 3

    返回:
        包含 date、nav、change_pct 的 DataFrame；失败时返回空表。
    """
    last_error = None
    fund_code = str(fund_code).strip()

    for attempt in range(1, retry_times + 1):
        try:
            raw_df = _fetch_raw_history(fund_code)
            return _clean_fund_history(raw_df)
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            requests.exceptions.RequestException,
        ) as exc:
            last_error = exc
            _safe_print(f"⚠️ 网络异常 [{fund_code}] 第 {attempt}/{retry_times} 次: {exc}")
        except (ValueError, KeyError) as exc:
            last_error = exc
            _safe_print(f"⚠️ 数据异常 [{fund_code}] 第 {attempt}/{retry_times} 次: {exc}")
        except Exception as exc:
            last_error = exc
            _safe_print(f"⚠️ 未知异常 [{fund_code}] 第 {attempt}/{retry_times} 次: {exc}")

        if attempt < retry_times:
            try:
                time.sleep(2)
            except Exception as sleep_exc:
                _safe_print(f"⚠️ 重试等待异常 [{fund_code}]: {sleep_exc}")

    _safe_print(f"❌ 拉取失败 [{fund_code}]，已重试 {retry_times} 次，最后错误: {last_error}")
    return _empty_nav_df()


def _empty_estimate():
    return {"nav_estimate": float("nan"), "change_pct": float("nan")}


def _to_estimate_float(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text or text.lower() in {"nan", "none", "--", "-"}:
        return float("nan")
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _parse_estimate_payload(payload):
    """从 akshare 返回值里抽出估算净值和涨跌幅；无法识别时返回 None。"""
    if payload is None:
        return None
    if isinstance(payload, dict):
        nav = None
        chg = None
        for key, value in payload.items():
            name = str(key)
            if nav is None and any(token in name for token in ("估算净值", "净值估算", "gsz", "nav_estimate", "估算值")):
                nav = _to_estimate_float(value)
            if chg is None and any(token in name for token in ("估算涨幅", "估算增长率", "gszzl", "change_pct", "涨跌幅", "日增长率")):
                chg = _to_estimate_float(value)
        if nav is None and "nav" in payload:
            nav = _to_estimate_float(payload.get("nav"))
        if chg is None and "change" in str(payload.keys()).lower():
            pass
        if nav is not None and pd.notna(nav):
            return {"nav_estimate": float(nav), "change_pct": float(chg) if chg is not None else float("nan")}
        return None

    if not isinstance(payload, pd.DataFrame) or payload.empty:
        return None

    df = payload.copy()
    df.columns = [str(col).strip() for col in df.columns]
    estimate_cols = [col for col in df.columns if any(token in col for token in ("估算净值", "净值估算", "估算值", "gsz"))]
    change_cols = [
        col
        for col in df.columns
        if any(token in col for token in ("估算涨幅", "估算增长率", "gszzl", "估算数据-估算增长率"))
    ]
    if not estimate_cols:
        return None
    last = df.iloc[-1]
    nav = _to_estimate_float(last[estimate_cols[0]])
    chg = _to_estimate_float(last[change_cols[0]]) if change_cols else float("nan")
    if pd.isna(nav):
        return None
    return {"nav_estimate": float(nav), "change_pct": float(chg) if pd.notna(chg) else float("nan")}


def _quote(nav, chg, gztime=None, source=None):
    nav_f = _to_estimate_float(nav)
    chg_f = _to_estimate_float(chg)
    if pd.isna(nav_f):
        return None
    result = {
        "nav_estimate": float(nav_f),
        "change_pct": float(chg_f) if pd.notna(chg_f) else float("nan"),
    }
    if gztime not in (None, ""):
        result["gztime"] = str(gztime).strip()
    if source:
        result["source"] = str(source)
    return result


def _quote_from_gsz_fields(payload):
    """从 dict 中精确读取 gsz / gszzl（避免 gszzl 被当成 gsz）。"""
    if not isinstance(payload, dict):
        return None
    fields = {str(key).strip().lower(): value for key, value in payload.items()}
    gztime = fields.get("gztime") or fields.get("gz_time")
    return _quote(fields.get("gsz"), fields.get("gszzl"), gztime=gztime, source="pingzhongdata")


def _extract_js_json(text, var_name):
    """提取 ``var name = {...}`` 或 ``var name = [...]`` 后的 JSON。"""
    match = re.search(rf"var\s+{re.escape(var_name)}\s*=\s*", text, flags=re.I)
    if not match:
        return None
    i = match.end()
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] not in "{[":
        return None
    opener = text[i]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    quote = None
    escape = False
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in {'"', "'"}:
            in_str = True
            quote = ch
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                chunk = text[i : j + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    return None
    return None


def _extract_js_number(text, var_name):
    """提取 ``var gsz = "3.23"`` / ``"gsz": 3.23`` 这类数值。"""
    pattern = re.compile(
        rf"(?:var\s+{re.escape(var_name)}\s*=|(?:['\"]{re.escape(var_name)}['\"])\s*:)\s*['\"]?"
        rf"([+-]?(?:\d+\.?\d*|\.\d+))",
        flags=re.I,
    )
    match = pattern.search(text or "")
    if not match:
        return None
    return match.group(1)


def _extract_js_string(text, var_name):
    pattern = re.compile(
        rf"(?:var\s+{re.escape(var_name)}\s*=|(?:['\"]{re.escape(var_name)}['\"])\s*:)\s*['\"]([^'\"]+)['\"]",
        flags=re.I,
    )
    match = pattern.search(text or "")
    if not match:
        return None
    return match.group(1)


PINGZHONG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://fund.eastmoney.com/",
}
VALUATION_LAST_URLS = (
    "https://fundcomapi.tiantianfunds.com/mm/newCore/FundValuationLast",
    "https://fundcomapi.eastmoney.com/mm/newCore/FundValuationLast",
)


def _fetch_fundgz_js(fund_code):
    """天天基金 jsonpgz 估值（与 pingzhongdata 同源盘中估算）。"""
    url = f"https://fundgz.1234567.com.cn/js/{fund_code}.js"
    try:
        response = requests.get(url, headers=PINGZHONG_HEADERS, timeout=5)
        response.raise_for_status()
        text = (response.text or "").strip()
        match = re.search(r"jsonpgz\s*\(\s*(\{.*\})\s*\)\s*;?\s*$", text, flags=re.S)
        if not match:
            match = re.search(r"(\{.*\})", text, flags=re.S)
        if not match:
            return None
        payload = json.loads(match.group(1))
        parsed = _quote_from_gsz_fields(payload)
        if parsed:
            parsed["source"] = "fundgz"
            return parsed
        return _quote(payload.get("gsz"), payload.get("gszzl"), gztime=payload.get("gztime"), source="fundgz")
    except Exception as exc:
        _safe_print(f"⚠️ fundgz 估值失败 [{fund_code}]: {exc}")
        return None


def _fetch_pingzhong_estimate(fund_code):
    """
    天天基金 pingzhongdata 接口。

    GET https://fund.eastmoney.com/pingzhongdata/{fund_code}.js
    优先解析 ``var pingzhongdata = {...}`` 中的 gsz / gszzl；
    收盘后 gsz 常等于当日净值、gszzl 为 0.00，视为有效。
    盘中若 JS 无 gsz，回退到同源 fundgz jsonpgz。
    """
    url = f"https://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
    try:
        response = requests.get(url, headers=PINGZHONG_HEADERS, timeout=5)
        response.raise_for_status()
        text = response.text or ""
        payload = _extract_js_json(text, "pingzhongdata")
        parsed = _quote_from_gsz_fields(payload) if payload else None
        if parsed:
            parsed["source"] = "pingzhongdata"
            return parsed
        if isinstance(payload, dict):
            parsed = _parse_estimate_payload(payload)
            if parsed:
                parsed["source"] = "pingzhongdata"
                return parsed
        gsz = _extract_js_number(text, "gsz")
        gszzl = _extract_js_number(text, "gszzl")
        gztime = _extract_js_string(text, "gztime") or _extract_js_string(text, "gz_time")
        parsed = _quote(gsz, gszzl, gztime=gztime, source="pingzhongdata")
        if parsed:
            return parsed
    except Exception as exc:
        _safe_print(f"⚠️ pingzhongdata 估值失败 [{fund_code}]: {exc}")
    return _fetch_fundgz_js(fund_code)


def _rows_from_valuation_payload(payload):
    if not isinstance(payload, dict):
        return []
    for key in ("data", "Data", "Datas"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    nested = payload.get("Data")
    if isinstance(nested, dict):
        for key in ("list", "List", "datas", "Datas"):
            rows = nested.get(key)
            if isinstance(rows, list):
                return rows
    return []


def _fetch_valuation_last(fund_codes):
    """天天基金 FundValuationLast：盘中返回 GSZ/GSZZL，替代已失效的 fundgz。"""
    codes = [str(code).strip() for code in (fund_codes or []) if str(code).strip()]
    if not codes:
        return {}
    params = {
        "FCODES": ",".join(codes),
        "FIELDS": "FCODE,SHORTNAME,GSZZL,GZTIME,GSZ,NAV,PDATE",
    }
    last_error = None
    for url in VALUATION_LAST_URLS:
        try:
            response = requests.get(url, params=params, headers=PINGZHONG_HEADERS, timeout=5)
            response.raise_for_status()
            payload = response.json()
            results = {}
            for row in _rows_from_valuation_payload(payload):
                if not isinstance(row, dict):
                    continue
                code = str(row.get("FCODE") or row.get("fcode") or "").strip()
                if not code:
                    continue
                gztime = row.get("GZTIME") if "GZTIME" in row else row.get("gztime")
                parsed = _quote(
                    row.get("GSZ") if "GSZ" in row else row.get("gsz"),
                    row.get("GSZZL") if "GSZZL" in row else row.get("gszzl"),
                    gztime=gztime,
                    source="FundValuationLast",
                )
                if not parsed:
                    # 收盘后估值字段为空时，当日净值仍可用；涨跌幅按 0.00 处理。
                    nav = row.get("NAV") if "NAV" in row else row.get("nav")
                    parsed = _quote(nav, 0.0, gztime=gztime, source="FundValuationLast")
                if parsed:
                    results[code] = parsed
            return results
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        _safe_print(f"⚠️ FundValuationLast 估值失败: {last_error}")
    return {}


def _fetch_eastmoney_mnf(fund_codes):
    """东方财富移动端估值接口，可一次查多只。盘中返回 GSZ/GSZZL。"""
    import uuid

    codes = [str(code).strip() for code in (fund_codes or []) if str(code).strip()]
    if not codes:
        return {}
    url = "https://fundmobapi.eastmoney.com/FundMNewApi/FundMNFInfo"
    params = {
        "pageIndex": 1,
        "pageSize": max(len(codes), 20),
        "plat": "Android",
        "appType": "ttjj",
        "product": "EFund",
        "Version": "6.5.5",
        "deviceid": str(uuid.uuid4()),
        "Fcodes": ",".join(codes),
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 12) AppleWebKit/537.36",
        "Referer": "https://fund.eastmoney.com/",
    }
    response = requests.get(url, headers=headers, params=params, timeout=5)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("Datas") or []
    results = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = str(row.get("FCODE") or "").strip()
        if not code:
            continue
        parsed = _quote(
            row.get("GSZ"),
            row.get("GSZZL"),
            gztime=row.get("GZTIME"),
            source="FundMNFInfo",
        )
        if parsed:
            results[code] = parsed
    return results


def fetch_realtime_estimate(fund_code):
    """
    拉取单只基金的实时估值和估算涨跌幅。

    1. 天天基金 pingzhongdata（gsz / gszzl）
    2. 天天基金 FundValuationLast
    3. 东方财富 FundMNFInfo 兜底

    返回:
        {'nav_estimate': float, 'change_pct': float}
        change_pct 为百分比点数，如 0.39 代表 +0.39%。失败时为 nan。
        收盘后 gszzl 为 0.00 属于正常值。
    """
    code = "" if fund_code is None else str(fund_code).strip()
    empty = _empty_estimate()
    if not code:
        return empty

    try:
        parsed = _fetch_pingzhong_estimate(code)
        if parsed:
            return parsed

        batch = _fetch_valuation_last([code])
        if code in batch:
            return batch[code]

        batch = _fetch_eastmoney_mnf([code])
        if code in batch:
            return batch[code]
    except Exception as exc:
        _safe_print(f"⚠️ 实时估值失败 [{code}]: {exc}")
    return empty


def fetch_realtime_estimates(fund_codes, progress_callback=None):
    """批量拉取实时估值，返回 {fund_code: {nav_estimate, change_pct}}。"""
    codes = [str(code).strip() for code in (fund_codes or []) if str(code).strip()]
    results = {code: _empty_estimate() for code in codes}
    total = len(codes)
    if not codes:
        return results

    if callable(progress_callback):
        try:
            progress_callback(0, total, "批量估值")
        except Exception:
            pass
    try:
        results.update(_fetch_valuation_last(codes))
    except Exception as exc:
        _safe_print(f"⚠️ 批量实时估值失败: {exc}")

    missing = [code for code in codes if pd.isna(results.get(code, {}).get("nav_estimate"))]
    for index, code in enumerate(missing, start=1):
        if callable(progress_callback):
            try:
                progress_callback(index, max(len(missing), 1), code)
            except Exception:
                pass
        results[code] = fetch_realtime_estimate(code)
        if index < len(missing):
            try:
                time.sleep(0.15)
            except Exception:
                pass
    return results
