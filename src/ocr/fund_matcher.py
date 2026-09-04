import json
import os
import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher, get_close_matches
from functools import lru_cache

from ..data_layer.loader import PROJECT_ROOT, _safe_print


CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "fund_name_cache.json")
LOG_PATH = os.path.join(PROJECT_ROOT, "data", "fund_match.log")
MATCH_THRESHOLD = 0.75
HIGH_THRESHOLD = 0.90
RESOLVE_HIGH_THRESHOLD = 0.90
RESOLVE_PARTIAL_THRESHOLD = 0.70
CACHE_MAX_DAYS = 30
CANDIDATE_LIMIT = 8
_CODE_RE = re.compile(r"^\d{6}$")
_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

_FULLWIDTH_MAP = str.maketrans(
    {
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "［": "[",
        "］": "]",
        "，": ",",
        "。": ".",
        "；": ";",
        "：": ":",
        "　": " ",
    }
)
_TYPE_WORDS = (
    "ETF联接",
    "指数增强",
    "灵活配置",
    "股票型",
    "债券型",
    "混合型",
    "(QDII)",
    "QDII",
    "LOF",
    "FOF",
    "ETF",
    "联接",
    "指数",
    "混合",
    "股票",
    "债券",
    "基金",
    "型",
)
_NOISE_TAGS = (
    "金选指数基金",
    "金选优选",
    "金选",
    "定投",
    "跟投",
    "自选",
)
_CLASS_RE = re.compile(r"[ABCEabce]$")
_NON_WORD_RE = re.compile(r"[^\w\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+")

_INDEX = None
_INDEX_STAMP = None


def _today():
    return datetime.now().strftime("%Y-%m-%d")


def _now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def preprocess_name(name):
    """去首尾空格、统一全半角括号、去掉金选等噪声。"""
    text = "" if name is None else str(name).strip()
    text = text.translate(_FULLWIDTH_MAP)
    for tag in _NOISE_TAGS:
        text = text.replace(tag, "")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _split_class_suffix(text):
    value = str(text or "")
    if len(value) > 2 and _CLASS_RE.search(value) and re.search(r"[\u4e00-\u9fff]", value[:-1]):
        return value[:-1], value[-1].upper()
    return value, ""


def strip_type_words(name):
    text, cls = _split_class_suffix(preprocess_name(name))
    changed = True
    while changed and text:
        changed = False
        for word in _TYPE_WORDS:
            if text.endswith(word) and len(text) > len(word) + 1:
                text = text[: -len(word)]
                changed = True
                break
            wrapped = f"({word})" if not word.startswith("(") else word
            if wrapped in text:
                text = text.replace(wrapped, "")
                changed = True
    return (text + cls).strip()


def exact_key(name):
    text = preprocess_name(name)
    text = _NON_WORD_RE.sub("", text)
    return text.lower()


def _name_variants(name):
    raw = preprocess_name(name)
    stripped = strip_type_words(name)
    key = exact_key(name)
    variants = {raw.lower(), stripped.lower(), key}
    body, cls = _split_class_suffix(raw)
    if cls:
        variants.add(body.lower())
        variants.add(exact_key(body))
    return {item for item in variants if item}


def _token_sort_key(name):
    tokens = _TOKEN_RE.findall(strip_type_words(name).lower())
    return "".join(sorted(tokens))


def _load_cache_file():
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, dict) and (payload.get("items") or payload.get("user_mapping")):
            payload.setdefault("items", [])
            payload.setdefault("user_mapping", {})
            return payload
    except Exception:
        pass
    return None


def _cache_is_fresh(payload):
    if not payload:
        return False
    stamp = str(payload.get("update_time") or "")[:10]
    try:
        updated = datetime.strptime(stamp, "%Y-%m-%d")
    except ValueError:
        return False
    return datetime.now() - updated < timedelta(days=CACHE_MAX_DAYS)


def _fetch_remote_funds():
    import akshare as ak

    frame = ak.fund_name_em()
    if frame is None or frame.empty:
        return []
    code_col = "基金代码" if "基金代码" in frame.columns else frame.columns[0]
    name_col = "基金简称" if "基金简称" in frame.columns else frame.columns[2]
    items = []
    for _, row in frame.iterrows():
        code = str(row.get(code_col) or "").strip()
        name = str(row.get(name_col) or "").strip()
        if code and name:
            items.append({"code": code, "name": name})
    return items


def _write_cache(items=None, user_mapping=None, update_time=None):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cached = _load_cache_file() or {}
    payload = {
        "update_time": update_time or cached.get("update_time") or _today(),
        "items": list(items) if items is not None else list(cached.get("items") or []),
        "user_mapping": dict(user_mapping) if user_mapping is not None else dict(cached.get("user_mapping") or {}),
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
    global _INDEX, _INDEX_STAMP
    _INDEX = None
    _INDEX_STAMP = None
    return payload


def load_fund_name_cache(force=False):
    """读取全量基金名称缓存；超过 30 天（约每月）或 force 时重新拉取。"""
    cached = _load_cache_file()
    if not force and _cache_is_fresh(cached):
        return cached.get("items") or []
    try:
        items = _fetch_remote_funds()
        if items:
            mapping = (cached or {}).get("user_mapping") or {}
            _write_cache(items=items, user_mapping=mapping, update_time=_today())
            _safe_print(f"✅ 基金名称缓存已更新，共 {len(items)} 只")
            return items
    except Exception as exc:
        _safe_print(f"⚠️ 拉取基金名称列表失败，改用本地缓存: {exc}")
    if cached and cached.get("items"):
        return cached["items"]
    return []


def get_code_name_map(force=False):
    """基金代码 -> 名称。force=True 时重新调用 ak.fund_name_em() 并写入缓存。"""
    items = load_fund_name_cache(force=bool(force))
    mapping = {}
    for item in items or []:
        code = str(item.get("code") or "").strip()
        name = str(item.get("name") or "").strip()
        if code and name:
            mapping[code] = name
    return mapping


def lookup_fund_name_by_code(fund_code):
    """按基金代码取名称：先读本地缓存，没有再强制刷新全量列表。"""
    code = str(fund_code or "").strip()
    if not code:
        return ""
    name = str(get_code_name_map(force=False).get(code) or "").strip()
    if name:
        return name
    return str(get_code_name_map(force=True).get(code) or "").strip()


def load_user_mapping():
    cached = _load_cache_file() or {}
    mapping = cached.get("user_mapping") or {}
    return mapping if isinstance(mapping, dict) else {}


def remember_user_mapping(fund_name, fund_code, matched_name=""):
    return remember_user_mappings([(fund_name, fund_code, matched_name)])


def remember_user_mappings(entries):
    """批量写入名称-代码映射，避免反复整文件保存。"""
    mapping = load_user_mapping()
    changed = False
    for fund_name, fund_code, matched_name in entries or []:
        name = preprocess_name(fund_name)
        code = str(fund_code or "").strip()
        if not name or not code:
            continue
        record = {
            "fund_code": code,
            "fund_name": str(matched_name or fund_name or "").strip(),
            "updated_at": _now_text(),
        }
        mapping[name] = record
        raw = str(fund_name or "").strip()
        if raw:
            mapping[raw] = record
        changed = True
        _log_match("user_fix", query=str(fund_name or ""), fund_code=code, reason="用户手动修正", matched_name=record["fund_name"])
    if not changed:
        return False
    _write_cache(user_mapping=mapping)
    return True


def _log_match(event, query="", fund_code="", reason="", score=None, matched_name=""):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        payload = {
            "time": _now_text(),
            "event": event,
            "query": str(query or ""),
            "fund_code": str(fund_code or ""),
            "reason": str(reason or ""),
            "score": score,
            "matched_name": str(matched_name or ""),
        }
        with open(LOG_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _lookup_user_mapping(fund_name):
    mapping = load_user_mapping()
    if not mapping:
        return None
    for key in (str(fund_name or "").strip(), preprocess_name(fund_name), exact_key(fund_name)):
        item = mapping.get(key)
        if isinstance(item, dict) and item.get("fund_code"):
            return item
        if isinstance(item, str) and item.strip():
            return {"fund_code": item.strip(), "fund_name": str(fund_name or "")}
    return None


def _build_index(items):
    exact = {}
    cores = {}
    names = []
    for item in items or []:
        code = str(item.get("code") or "").strip()
        name = str(item.get("name") or "").strip()
        if not code or not name:
            continue
        record = {"code": code, "name": name}
        for key in _name_variants(name):
            exact.setdefault(key, []).append(record)
        core = re.sub(r"[^\u4e00-\u9fff]", "", strip_type_words(name))[:4]
        if len(core) >= 2:
            cores.setdefault(core[:2], []).append(record)
        names.append(name)
    return {"exact": exact, "cores": cores, "names": names, "items": items or []}


def _get_index():
    global _INDEX, _INDEX_STAMP
    items = load_fund_name_cache(force=False)
    stamp = (items[0].get("code") if items else "") + str(len(items))
    if _INDEX is not None and _INDEX_STAMP == stamp:
        return _INDEX
    _INDEX = _build_index(items)
    _INDEX_STAMP = stamp
    return _INDEX


def _score(query, candidate_name):
    query_raw = preprocess_name(query)
    cand_raw = preprocess_name(candidate_name)
    if query_raw and query_raw == cand_raw:
        return 1.0
    if exact_key(query) and exact_key(query) == exact_key(candidate_name):
        return 1.0
    lefts = _name_variants(query)
    rights = _name_variants(candidate_name)
    if lefts & rights:
        q_cls = _split_class_suffix(query_raw)[1]
        c_cls = _split_class_suffix(cand_raw)[1]
        return 1.0 if q_cls == c_cls or not q_cls or not c_cls else 0.97

    best = 0.0
    q_strip = strip_type_words(query).lower()
    c_strip = strip_type_words(candidate_name).lower()
    if q_strip and c_strip:
        if q_strip == c_strip:
            best = 0.97
        elif q_strip in c_strip or c_strip in q_strip:
            shorter, longer = (q_strip, c_strip) if len(q_strip) <= len(c_strip) else (c_strip, q_strip)
            best = max(best, 0.84 + 0.13 * (len(shorter) / max(len(longer), 1)))
        best = max(best, SequenceMatcher(None, q_strip, c_strip).ratio())
        best = max(best, SequenceMatcher(None, _token_sort_key(query), _token_sort_key(candidate_name)).ratio())
    return best


def _pool_for_query(query, index):
    pool = []
    seen = set()

    def _add(record):
        code = record.get("code")
        if code and code not in seen:
            seen.add(code)
            pool.append(record)

    for key in _name_variants(query):
        for record in index["exact"].get(key, []):
            _add(record)
    core = re.sub(r"[^\u4e00-\u9fff]", "", strip_type_words(query))
    if len(core) >= 2:
        for record in index["cores"].get(core[:2], []):
            _add(record)
    if len(pool) < 12:
        close_names = get_close_matches(preprocess_name(query), index["names"], n=20, cutoff=0.55)
        name_map = {item.get("name"): item for item in index["items"]}
        for name in close_names:
            record = name_map.get(name)
            if record:
                _add({"code": record.get("code"), "name": record.get("name")})
    return pool


def _rank_matches(fund_name, limit=CANDIDATE_LIMIT, min_score=MATCH_THRESHOLD):
    query = "" if fund_name is None else str(fund_name).strip()
    if not query:
        return []
    index = _get_index()
    ranked = []
    for item in _pool_for_query(query, index):
        score = _score(query, item.get("name"))
        if score < float(min_score):
            continue
        ranked.append(
            {
                "fund_code": str(item.get("code") or "").strip(),
                "fund_name": str(item.get("name") or "").strip(),
                "score": round(float(score), 4),
                "confidence": round(float(score), 4),
            }
        )
    ranked.sort(key=lambda row: (row["score"], row["fund_name"] == query), reverse=True)
    return ranked[: max(int(limit), 1)]


def list_fund_candidates(fund_name, limit=CANDIDATE_LIMIT):
    return _rank_matches(fund_name, limit=limit, min_score=MATCH_THRESHOLD)


def search_fund_names(query, limit=12):
    """按名称或代码实时过滤本地缓存，供导入页搜索框使用。"""
    text = preprocess_name(query).lower()
    if not text:
        return []
    items = load_fund_name_cache(force=False)
    hits = []
    for item in items or []:
        code = str(item.get("code") or "").strip()
        name = str(item.get("name") or "").strip()
        blob = f"{code}{name}{preprocess_name(name)}".lower()
        if text in blob or text in exact_key(name):
            hits.append({"fund_code": code, "fund_name": name, "score": 1.0 if text == code else 0.8})
        if len(hits) >= max(int(limit), 1):
            break
    if len(hits) < max(int(limit), 1):
        extra = _rank_matches(query, limit=limit, min_score=0.55)
        seen = {row["fund_code"] for row in hits}
        for row in extra:
            if row["fund_code"] in seen:
                continue
            hits.append(row)
            if len(hits) >= max(int(limit), 1):
                break
    return hits[: max(int(limit), 1)]


def match_fund_code(fund_name, threshold=MATCH_THRESHOLD, limit=CANDIDATE_LIMIT):
    """
    匹配基金代码。始终返回字典：
    high(>90%) / partial(75%~90%) / fail(<75%)，candidates 为相似度 > 75% 的列表。
    """
    query = "" if fund_name is None else str(fund_name).strip()
    empty = {
        "fund_name": query,
        "fund_code": "",
        "matched_name": "",
        "confidence": 0.0,
        "score": 0.0,
        "confident": False,
        "match_tier": "fail",
        "fail_reason": "名称未找到",
        "candidates": [],
    }
    if not query:
        empty["fail_reason"] = "名称为空"
        _log_match("fail", query=query, reason="名称为空", score=0.0)
        return empty

    mapped = _lookup_user_mapping(query)
    if mapped:
        return {
            "fund_name": query,
            "fund_code": str(mapped.get("fund_code") or "").strip(),
            "matched_name": str(mapped.get("fund_name") or query).strip(),
            "confidence": 1.0,
            "score": 1.0,
            "confident": True,
            "match_tier": "high",
            "fail_reason": "",
            "candidates": [
                {
                    "fund_code": str(mapped.get("fund_code") or "").strip(),
                    "fund_name": str(mapped.get("fund_name") or query).strip(),
                    "score": 1.0,
                    "confidence": 1.0,
                }
            ],
        }

    if re.fullmatch(r"\d{6}", query):
        items = load_fund_name_cache(force=False)
        hit = next((item for item in items if str(item.get("code") or "").strip() == query), None)
        if hit:
            return {
                "fund_name": query,
                "fund_code": query,
                "matched_name": str(hit.get("name") or ""),
                "confidence": 1.0,
                "score": 1.0,
                "confident": True,
                "match_tier": "high",
                "fail_reason": "",
                "candidates": [{"fund_code": query, "fund_name": str(hit.get("name") or ""), "score": 1.0, "confidence": 1.0}],
            }

    top = _rank_matches(query, limit=max(int(limit), CANDIDATE_LIMIT), min_score=float(threshold))
    best = top[0] if top else None
    if not best:
        near = _rank_matches(query, limit=3, min_score=0.45)
        reason = "名称未找到" if not near else "相似度低"
        _log_match("fail", query=query, reason=reason, score=(near[0]["score"] if near else 0.0))
        empty["fail_reason"] = reason
        empty["candidates"] = near
        empty["confidence"] = float(near[0]["score"]) if near else 0.0
        empty["score"] = empty["confidence"]
        return empty

    score = float(best["score"])
    second = float(top[1]["score"]) if len(top) > 1 else 0.0
    if preprocess_name(best["fund_name"]) == preprocess_name(query) or exact_key(best["fund_name"]) == exact_key(query):
        q_cls = _split_class_suffix(preprocess_name(query))[1]
        c_cls = _split_class_suffix(preprocess_name(best["fund_name"]))[1]
        if not q_cls or not c_cls or q_cls == c_cls:
            tier = "high"
    elif score >= HIGH_THRESHOLD and (score - second >= 0.03 or second < HIGH_THRESHOLD):
        tier = "high"
    elif score >= float(threshold):
        tier = "partial"
    else:
        tier = "fail"
    if tier == "fail":
        _log_match("fail", query=query, reason="相似度低", score=score, matched_name=best["fund_name"])
    return {
        "fund_name": query,
        "fund_code": best["fund_code"] if tier != "fail" else "",
        "matched_name": best["fund_name"],
        "confidence": score,
        "score": score,
        "confident": tier == "high",
        "match_tier": tier,
        "fail_reason": "" if tier != "fail" else "相似度低",
        "candidates": top,
    }


def normalize_fund_identifier(identifier):
    text = "" if identifier is None else str(identifier).strip()
    text = text.translate(_FULLWIDTH_MAP).translate(_FULLWIDTH_DIGITS)
    return text.strip()


def _empty_resolve(query, kind="", reason="标识为空"):
    return {
        "query": query,
        "kind": kind,
        "status": "fail",
        "value": None,
        "fund_code": "",
        "matched_name": "",
        "score": 0.0,
        "fail_reason": reason,
        "candidates": [],
    }


def _lookup_fund_by_code(code, allow_network=True):
    """先查本地名称缓存，未命中再按需用 akshare 验证代码是否存在。"""
    code = str(code or "").strip()
    if not _CODE_RE.fullmatch(code):
        return None
    items = load_fund_name_cache(force=False)
    for item in items or []:
        if str(item.get("code") or "").strip() == code:
            name = str(item.get("name") or "").strip()
            return {"code": code, "name": name}
    if not allow_network:
        return None
    try:
        import akshare as ak

        frame = ak.fund_open_fund_daily_em(symbol=code)
        if frame is None or getattr(frame, "empty", True):
            return None
        name = ""
        for key in ("基金简称", "基金名称", "name"):
            if hasattr(frame, "columns") and key in frame.columns:
                name = str(frame[key].iloc[0] or "").strip()
                if name:
                    break
        return {"code": code, "name": name}
    except Exception as exc:
        _safe_print(f"⚠️ 校验基金代码 {code} 失败: {exc}")
        return None


def _resolve_from_name_match(query, matched):
    score = float(matched.get("score") or matched.get("confidence") or 0.0)
    candidates = []
    seen = set()
    for item in matched.get("candidates") or []:
        code = str(item.get("fund_code") or "").strip()
        item_score = float(item.get("score") or item.get("confidence") or 0.0)
        if not code or code in seen or item_score < RESOLVE_PARTIAL_THRESHOLD:
            continue
        seen.add(code)
        candidates.append(
            {
                "fund_code": code,
                "fund_name": str(item.get("fund_name") or "").strip(),
                "score": item_score,
            }
        )
    best_code = str(matched.get("fund_code") or "").strip()
    best_name = str(matched.get("matched_name") or "").strip()
    if score > RESOLVE_HIGH_THRESHOLD and best_code:
        return {
            "query": query,
            "kind": "name",
            "status": "ok",
            "value": best_code,
            "fund_code": best_code,
            "matched_name": best_name,
            "score": score,
            "fail_reason": "",
            "candidates": candidates or [
                {"fund_code": best_code, "fund_name": best_name, "score": score}
            ],
        }
    if score >= RESOLVE_PARTIAL_THRESHOLD and candidates:
        return {
            "query": query,
            "kind": "name",
            "status": "candidates",
            "value": [item["fund_code"] for item in candidates],
            "fund_code": "",
            "matched_name": best_name,
            "score": score,
            "fail_reason": "",
            "candidates": candidates,
        }
    reason = str(matched.get("fail_reason") or "相似度低于 70%，请手动输入基金代码")
    return {
        "query": query,
        "kind": "name",
        "status": "fail",
        "value": None,
        "fund_code": "",
        "matched_name": best_name,
        "score": score,
        "fail_reason": reason,
        "candidates": candidates,
    }


@lru_cache(maxsize=1024)
def resolve_fund_identifier(identifier: str, allow_network=True):
    """
    识别基金代码或基金名称。

    - 6 位数字视为代码：存在则返回该代码，否则失败。
    - 否则按名称调用 match_fund_code：
      相似度 > 90% 返回代码；70%~90% 返回候选代码列表；< 70% 失败。

    返回字典，其中 value 为 str（唯一代码）、list[str]（候选）或 None（失败）。
    """
    raw = "" if identifier is None else str(identifier)
    query = normalize_fund_identifier(raw)
    if query != raw:
        return resolve_fund_identifier(query, allow_network)

    if not query:
        result = _empty_resolve("", reason="标识为空")
        _log_match("resolve_fail", query=query, reason=result["fail_reason"], score=0.0)
        return result

    if _CODE_RE.fullmatch(query):
        hit = _lookup_fund_by_code(query, allow_network=bool(allow_network))
        if hit:
            result = {
                "query": query,
                "kind": "code",
                "status": "ok",
                "value": query,
                "fund_code": query,
                "matched_name": hit.get("name") or "",
                "score": 1.0,
                "fail_reason": "",
                "candidates": [
                    {"fund_code": query, "fund_name": hit.get("name") or "", "score": 1.0}
                ],
            }
            _log_match(
                "resolve_code",
                query=query,
                fund_code=query,
                reason="按基金代码识别",
                score=1.0,
                matched_name=result["matched_name"],
            )
            return result
        result = _empty_resolve(query, kind="code", reason="未找到该基金代码，请确认后手动输入")
        _log_match("resolve_fail", query=query, reason=result["fail_reason"], score=0.0)
        _safe_print(f"⚠️ 基金代码不存在：{query}")
        return result

    matched = match_fund_code(query, threshold=RESOLVE_PARTIAL_THRESHOLD) or {}
    result = _resolve_from_name_match(query, matched)
    if result["status"] == "ok":
        _log_match(
            "resolve_name",
            query=query,
            fund_code=result.get("fund_code") or "",
            reason="名称高置信匹配",
            score=result.get("score"),
            matched_name=result.get("matched_name") or "",
        )
    elif result["status"] == "candidates":
        _log_match(
            "resolve_candidates",
            query=query,
            fund_code="",
            reason="名称模糊匹配，待选择",
            score=result.get("score"),
            matched_name=result.get("matched_name") or "",
        )
    else:
        _log_match(
            "resolve_fail",
            query=query,
            reason=result.get("fail_reason") or "名称匹配失败",
            score=result.get("score") or 0.0,
            matched_name=result.get("matched_name") or "",
        )
    return result

