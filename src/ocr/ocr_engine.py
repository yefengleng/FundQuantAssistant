import os
import re
from datetime import datetime

from ..data_layer.loader import _safe_print


LOW_CONFIDENCE = 0.80
_OCR_ENGINE = None
_OCR_BACKEND = None

_NOISE_EXACT = {
    "持有金额", "持有金额(元)", "持有金额（元）", "持有市值", "当前市值",
    "持有份额", "可用份额", "日收益", "日收益率", "昨日收益", "累计收益", "持有收益",
    "持有收益率", "最新净值", "净值日期", "基金", "明细", "交易记录",
    "买入", "卖出", "申购", "赎回", "确认中", "交易成功", "支付宝",
    "更新于", "更多", "全部", "筛选", "金选", "金选指数基金", "定投", "跟投",
}

_REMARK_TAGS = (
    "金选指数基金",
    "金选优选",
    "金选",
    "定投",
    "跟投",
    "自选",
)

_FUND_HINTS = (
    "混合", "股票", "指数", "联接", "QDII", "LOF", "FOF", "债券",
    "货币", "ETF", "主题", "灵活配置", "增强",
)

_ACTION_MAP = {
    "买入": "买入",
    "申购": "买入",
    "定投": "买入",
    "卖出": "卖出",
    "赎回": "卖出",
}

_DATE_RE = re.compile(
    r"(20\d{2})\s*[-./年]\s*(\d{1,2})\s*[-./月]\s*(\d{1,2})"
)
_NUMBER_RE = re.compile(
    r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+\.\d+|[-+]?\d+"
)
_PERCENT_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*%")
_SIGNED_NUMBER_RE = re.compile(
    r"[-+]?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\s*\d+\.\d+|[-+]?\s*\d+"
)


def models_ready():
    home = os.path.expanduser("~")
    for name in (".paddleocr", ".paddlex", ".paddleocr2"):
        if os.path.isdir(os.path.join(home, name)):
            return True
    return False


def _init_paddle():
    from paddleocr import PaddleOCR

    try:
        return PaddleOCR(use_angle_cls=True, lang="ch", show_log=False), "paddleocr"
    except TypeError:
        pass
    try:
        return PaddleOCR(use_textline_orientation=True, lang="ch"), "paddleocr"
    except TypeError:
        return PaddleOCR(lang="ch"), "paddleocr"


def _init_rapidocr():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(), "rapidocr"


def get_ocr_engine(progress_callback=None):
    """懒加载 OCR。优先 PaddleOCR，失败时尝试 RapidOCR。"""
    global _OCR_ENGINE, _OCR_BACKEND
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE, _OCR_BACKEND
    if progress_callback:
        try:
            progress_callback("首次使用正在下载模型，请稍候...")
        except Exception:
            pass
    errors = []
    for factory in (_init_paddle, _init_rapidocr):
        try:
            engine, backend = factory()
            _OCR_ENGINE = engine
            _OCR_BACKEND = backend
            _safe_print(f"✅ OCR 引擎已就绪（{backend}）")
            return _OCR_ENGINE, _OCR_BACKEND
        except Exception as exc:
            errors.append(f"{factory.__name__}: {exc}")
    raise RuntimeError(
        "OCR 引擎初始化失败。请安装 paddlepaddle + paddleocr "
        "（或在 Python 3.14 下安装 rapidocr-onnxruntime）。 "
        + " | ".join(errors)
    )


def _normalize_box(box):
    if box is None:
        return None
    try:
        if len(box) == 4 and not isinstance(box[0], (list, tuple)):
            x1, y1, x2, y2 = [float(v) for v in box]
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        points = [[float(point[0]), float(point[1])] for point in box[:4]]
        if len(points) == 4:
            return points
    except Exception:
        return None
    return None


def _box_metrics(box):
    points = _normalize_box(box)
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "box": points,
        "x": (min(xs) + max(xs)) / 2.0,
        "y": (min(ys) + max(ys)) / 2.0,
        "x1": min(xs),
        "x2": max(xs),
        "y1": min(ys),
        "y2": max(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
    }


def _with_box(text, score, box=None):
    item = {"text": text, "confidence": float(score or 0.0)}
    metrics = _box_metrics(box)
    if metrics:
        item.update(metrics)
    return item


def _parse_paddle_v2(result):
    items = []
    if not result:
        return items
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if not page:
            continue
        for line in page:
            try:
                text = str(line[1][0]).strip()
                score = float(line[1][1])
                box = line[0] if line else None
            except Exception:
                continue
            if text:
                items.append(_with_box(text, score, box))
    return items


def _parse_paddle_v3(result):
    items = []
    rows = result if isinstance(result, list) else [result]
    for row in rows:
        texts = None
        scores = None
        if isinstance(row, dict):
            texts = row.get("rec_texts") or row.get("rec_text")
            scores = row.get("rec_scores") or row.get("rec_score")
        else:
            texts = getattr(row, "rec_texts", None) or getattr(row, "rec_text", None)
            scores = getattr(row, "rec_scores", None) or getattr(row, "rec_score", None)
            if texts is None and hasattr(row, "get"):
                texts = row.get("rec_texts")
                scores = row.get("rec_scores")
        if not texts:
            continue
        boxes = None
        if isinstance(row, dict):
            boxes = row.get("rec_boxes") or row.get("dt_polys") or row.get("rec_polys")
        else:
            boxes = getattr(row, "rec_boxes", None) or getattr(row, "dt_polys", None)
        if isinstance(texts, str):
            texts = [texts]
        if scores is None:
            scores = [1.0] * len(texts)
        elif not isinstance(scores, (list, tuple)):
            scores = [scores]
        if boxes is None:
            boxes = [None] * len(texts)
        for index, (text, score) in enumerate(zip(texts, scores)):
            value = str(text).strip()
            if value:
                box = boxes[index] if index < len(boxes) else None
                items.append(_with_box(value, score, box))
    return items


def _parse_rapidocr(result):
    items = []
    payload = result[0] if isinstance(result, tuple) else result
    if not payload:
        return items
    for line in payload:
        try:
            if isinstance(line, dict):
                text = str(line.get("text") or "").strip()
                score = float(line.get("score") or line.get("confidence") or 0.0)
                box = line.get("box") or line.get("dt_boxes")
            elif len(line) >= 3:
                box = line[0]
                text = str(line[1]).strip()
                score = float(line[2])
            else:
                continue
        except Exception:
            continue
        if text:
            items.append(_with_box(text, score, box))
    return items


def extract_text_from_image(image_path):
    """识别图片中的文字，返回 [{"text", "confidence"}, ...]。"""
    path = "" if image_path is None else str(image_path).strip()
    if not path or not os.path.isfile(path):
        return []
    engine, backend = get_ocr_engine()
    try:
        if backend == "rapidocr":
            return _parse_rapidocr(engine(path))
        if hasattr(engine, "ocr"):
            try:
                raw = engine.ocr(path, cls=True)
            except TypeError:
                raw = engine.ocr(path)
            parsed = _parse_paddle_v2(raw)
            if parsed:
                return parsed
        if hasattr(engine, "predict"):
            parsed = _parse_paddle_v3(engine.predict(path))
            if parsed:
                return parsed
    except Exception as exc:
        _safe_print(f"⚠️ OCR 识别失败: {exc}")
        return []
    return []


def _as_items(text_list):
    items = []
    for item in text_list or []:
        if isinstance(item, str):
            text = item.strip()
            if text:
                items.append({"text": text, "confidence": 1.0})
            continue
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            try:
                score = float(item.get("confidence") if item.get("confidence") is not None else 1.0)
            except (TypeError, ValueError):
                score = 1.0
            if not text:
                continue
            payload = dict(item)
            payload["text"] = text
            payload["confidence"] = score
            if payload.get("box") and "x" not in payload:
                payload.update(_box_metrics(payload.get("box")) or {})
            items.append(payload)
    return items


def parse_number(text):
    value = parse_signed_number(text)
    if value is None:
        return None
    return abs(value)


def parse_signed_number(text):
    raw = "" if text is None else str(text)
    cleaned = raw.replace(",", "").replace("，", "").replace(" ", "")
    if not cleaned:
        return None
    multiplier = 1.0
    if "万" in cleaned:
        multiplier = 10000.0
        cleaned = cleaned.replace("万", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group()) * multiplier
    except (TypeError, ValueError):
        return None


def parse_percent(text):
    match = _PERCENT_RE.search("" if text is None else str(text))
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _line_signed_numbers(text):
    values = []
    if parse_percent(text) is not None and "%" in str(text):
        leftover = _PERCENT_RE.sub(" ", str(text))
    else:
        leftover = str(text)
    for match in _SIGNED_NUMBER_RE.finditer(leftover):
        number = parse_signed_number(match.group())
        if number is not None:
            values.append(number)
    return values


def _collect_remarks(*texts):
    remarks = []
    blob = " ".join(str(text or "") for text in texts)
    for tag in _REMARK_TAGS:
        if tag not in blob or tag in remarks:
            continue
        if any(tag != existing and tag in existing for existing in remarks):
            continue
        remarks.append(tag)
    return " ".join(remarks)


def _line_numbers(text):
    values = []
    for match in _NUMBER_RE.finditer(str(text)):
        number = parse_number(match.group())
        if number is not None:
            values.append(number)
    return values


def _looks_like_fund_name(text):
    name = re.sub(r"\s+", "", str(text or ""))
    name = name.strip("·|-—_/:;,.。、")
    if len(name) < 4 or len(name) > 40:
        return ""
    if name in _NOISE_EXACT or name in _REMARK_TAGS:
        return ""
    if any(name == tag or name.startswith(tag) and len(name) <= len(tag) + 2 for tag in _REMARK_TAGS):
        return ""
    if _DATE_RE.search(name):
        return ""
    if parse_number(name) is not None and not any(hint in name for hint in _FUND_HINTS):
        return ""
    if any(hint in name for hint in _FUND_HINTS):
        return name
    if re.search(r"[A-Za-z]$", name) and len(name) >= 6 and re.search(r"[\u4e00-\u9fff]", name):
        return name
    return ""


def _looks_like_fund_identifier(text):
    """基金名称或 6 位基金代码。"""
    name = _looks_like_fund_name(text)
    if name:
        return name
    raw = re.sub(r"\s+", "", str(text or ""))
    raw = raw.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if re.fullmatch(r"\d{6}", raw):
        return raw
    return ""


def _detect_action(text):
    value = str(text or "")
    for word, action in _ACTION_MAP.items():
        if word in value:
            return action
    return ""


def _parse_date(text):
    match = _DATE_RE.search(str(text or ""))
    if not match:
        return ""
    year, month, day = match.groups()
    try:
        return datetime(int(year), int(month), int(day)).strftime("%Y-%m-%d")
    except ValueError:
        return ""


def _window_confidence(window):
    scores = [item.get("confidence") or 0.0 for item in window if item.get("confidence") is not None]
    if not scores:
        return 1.0
    return float(min(scores))


def extract_holdings(text_list):
    """从 OCR 文本提取持仓：基金名称、持有份额、持有市值。"""
    items = _as_items(text_list)
    results = []
    seen = set()
    for index, item in enumerate(items):
        name = _looks_like_fund_name(item["text"])
        if not name or name in seen:
            continue
        window = items[index:index + 10]
        shares = None
        market_value = None
        extra_numbers = []
        for line in window[1:]:
            text = line["text"]
            numbers = _line_numbers(text)
            if not numbers:
                continue
            if any(key in text for key in ("持有份额", "可用份额")) or (
                "份额" in text and "金额" not in text and "市值" not in text
            ):
                shares = numbers[0]
                continue
            if any(key in text for key in ("持有金额", "持有市值", "当前市值", "市值")):
                market_value = numbers[0]
                continue
            if _detect_action(text) or _parse_date(text):
                continue
            extra_numbers.extend(numbers)
        if market_value is None and extra_numbers:
            market_value = extra_numbers[0]
        if shares is None and len(extra_numbers) >= 2:
            shares = extra_numbers[1]
        if shares is None and market_value is None:
            continue
        seen.add(name)
        confidence = min(float(item.get("confidence") or 1.0), _window_confidence(window[:6]))
        if shares is None or market_value is None:
            confidence = min(confidence, 0.7)
        results.append(
            {
                "fund_name": name,
                "shares": shares,
                "market_value": market_value,
                "confidence": round(confidence, 4),
                "raw_text": " | ".join(line["text"] for line in window[:6]),
            }
        )
    return results


_HEADER_PATTERNS = (
    ("hold_return_rate", ("持有收益率",)),
    ("weight_pct", ("占比",)),
    ("cumulative_profit", ("累计收益",)),
    ("yesterday_profit", ("昨日收益", "日收益", "日盈亏")),
    ("hold_profit", ("持有收益",)),
    ("hold_amount", ("持有金额", "持有市值", "当前市值")),
    ("fund_name", ("基金名称",)),
)
_HEADER_MIN_HITS = 2
_ROW_Y_RATIO = 0.65


def _has_boxes(items):
    return bool(items) and sum(1 for item in items if item.get("x") is not None) >= max(3, len(items) // 3)


def _group_rows(items):
    located = [item for item in items if item.get("y") is not None]
    if not located:
        return [[item] for item in items]
    heights = [item.get("height") or 0.0 for item in located if (item.get("height") or 0) > 0]
    threshold = (sorted(heights)[len(heights) // 2] * _ROW_Y_RATIO) if heights else 18.0
    threshold = max(12.0, float(threshold))
    ordered = sorted(located, key=lambda item: (item.get("y") or 0.0, item.get("x") or 0.0))
    rows = []
    for item in ordered:
        if not rows:
            rows.append([item])
            continue
        current_y = sum(part.get("y") or 0.0 for part in rows[-1]) / len(rows[-1])
        if abs((item.get("y") or 0.0) - current_y) <= threshold:
            rows[-1].append(item)
        else:
            rows.append([item])
    for row in rows:
        row.sort(key=lambda item: item.get("x") or 0.0)
    return rows


def _classify_header_cell(text):
    value = str(text or "")
    for field, keys in _HEADER_PATTERNS:
        for key in keys:
            if key in value:
                if field == "hold_profit" and "率" in value:
                    continue
                if field == "fund_name" and len(value) > 8 and _looks_like_fund_name(value):
                    continue
                return field
    return ""


def _detect_header_map(rows):
    best = None
    best_hits = 0
    for row in rows:
        mapping = {}
        for cell in row:
            field = _classify_header_cell(cell.get("text"))
            if field and field not in mapping:
                mapping[field] = float(cell.get("x") or 0.0)
        hits = len(mapping)
        if hits > best_hits:
            best = mapping
            best_hits = hits
    if best is None:
        return None
    if best_hits >= 3:
        return best
    if best_hits >= 2 and "fund_name" in best and "hold_amount" in best:
        return best
    if best_hits >= 2 and "hold_amount" in best and "yesterday_profit" in best:
        return best
    return None


def _nearest_field(x_value, header_map, candidates=None):
    if not header_map:
        return ""
    if candidates is None:
        mapping = header_map
    else:
        mapping = {key: value for key, value in header_map.items() if key in candidates}
    if not mapping:
        return ""
    return min(mapping.items(), key=lambda item: abs(item[1] - float(x_value or 0.0)))[0]


def _parse_field_value(field, text):
    if field == "fund_name":
        return _looks_like_fund_name(text) or str(text or "").strip()
    if field in {"weight_pct", "hold_return_rate"}:
        return parse_percent(text)
    if field == "hold_amount":
        number = parse_signed_number(text)
        return None if number is None else abs(number)
    return parse_signed_number(text)


def _missing(value):
    if value is None or value == "":
        return True
    if isinstance(value, (list, tuple, dict, set)) and not value:
        return True
    return False


def _empty_holding():
    return {
        "fund_name": "",
        "hold_amount": None,
        "yesterday_profit": None,
        "hold_profit": None,
        "hold_return_rate": None,
        "cumulative_profit": None,
        "weight_pct": None,
        "remark": "",
        "confidence": 1.0,
        "raw_text": "",
        "parse_mode": "",
        "needs_review": False,
        "issues": [],
    }


def _token_stats(items):
    tokens = []
    for item in items:
        text = str(item.get("text") or "")
        percent = parse_percent(text)
        if percent is not None and "%" in text:
            tokens.append({"kind": "percent", "value": percent, "raw": text, "signed": text.strip().startswith(("+", "-"))})
            leftover = _PERCENT_RE.sub(" ", text)
        else:
            leftover = text
        for match in _SIGNED_NUMBER_RE.finditer(leftover):
            raw = match.group()
            number = parse_signed_number(raw)
            if number is None:
                continue
            tokens.append(
                {
                    "kind": "number",
                    "value": number,
                    "raw": raw,
                    "signed": str(raw).strip().startswith(("+", "-")),
                }
            )
    return tokens


def _flag_anomalous_holdings(record):
    """标记字段错位或数值异常，供预览高亮。"""
    issues = [str(item) for item in (record.get("issues") or []) if item]
    amount = None
    yesterday = None
    hold_profit = None
    try:
        amount = None if record.get("hold_amount") in {None, ""} else float(record.get("hold_amount"))
    except (TypeError, ValueError):
        issues.append("持有金额格式错误")
        record["needs_review"] = True
    try:
        yesterday = None if record.get("yesterday_profit") in {None, ""} else float(record.get("yesterday_profit"))
    except (TypeError, ValueError):
        issues.append("昨日收益格式错误")
        record["needs_review"] = True
    try:
        hold_profit = None if record.get("hold_profit") in {None, ""} else float(record.get("hold_profit"))
    except (TypeError, ValueError):
        issues.append("持有收益格式错误")
        record["needs_review"] = True

    if amount is None:
        if "持有金额格式错误" not in issues:
            issues.append("持有金额缺失")
        record["needs_review"] = True
    elif amount < 0:
        issues.append("持有金额为负数")
        record["needs_review"] = True
    elif amount == 0:
        issues.append("持有金额无效")
        record["needs_review"] = True
    if yesterday is not None and amount is not None and amount < abs(yesterday):
        issues.append("持有金额小于昨日收益，可能字段错位")
        record["needs_review"] = True
    if hold_profit is not None and amount is not None and abs(hold_profit) > amount * 5 and amount > 0:
        issues.append("持有收益相对持有金额过大，请核对")
        record["needs_review"] = True
    if not str(record.get("fund_name") or "").strip():
        issues.append("未识别到基金名称")
        record["needs_review"] = True
    # de-dup keep order
    seen = set()
    unique = []
    for item in issues:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    record["issues"] = unique
    return record


def assign_fields_by_heuristics(items):
    """按数值大小与正负号/百分号特征回填字段。"""
    record = _empty_holding()
    texts = [str(item.get("text") or "") for item in items]
    record["remark"] = _collect_remarks(*texts)
    record["raw_text"] = " | ".join(texts[:10])
    for item in items:
        name = _looks_like_fund_name(item.get("text"))
        if name:
            record["fund_name"] = name
            break
    tokens = _token_stats(items)
    percents = [token for token in tokens if token["kind"] == "percent"]
    numbers = [token for token in tokens if token["kind"] == "number"]
    if percents:
        record["weight_pct"] = percents[0]["value"]
    if len(percents) >= 2:
        record["hold_return_rate"] = percents[1]["value"]

    positives = [token for token in numbers if token["value"] > 0]
    used = set()
    if positives:
        largest = max(positives, key=lambda token: abs(token["value"]))
        record["hold_amount"] = abs(largest["value"])
        used.add(id(largest))

    signed = [token for token in numbers if id(token) not in used and (token["signed"] or token["value"] < 0)]
    signed.sort(key=lambda token: abs(token["value"]))
    if signed:
        record["yesterday_profit"] = signed[0]["value"]
        used.add(id(signed[0]))
    if len(signed) >= 2:
        record["hold_profit"] = signed[-1]["value"]
        used.add(id(signed[-1]))
    leftover_signed = [token for token in signed if id(token) not in used]
    if leftover_signed:
        record["cumulative_profit"] = leftover_signed[0]["value"]

    actual = len(tokens)
    record["needs_review"] = (
        actual < 3
        or record["hold_amount"] is None
        or record["fund_name"] == ""
        or (record["yesterday_profit"] is not None and record["hold_amount"] is not None
            and record["hold_amount"] < abs(record["yesterday_profit"]))
    )
    record["parse_mode"] = "heuristic"
    record["confidence"] = 0.68 if record["needs_review"] else 0.8
    _flag_anomalous_holdings(record)
    return record


def _fill_missing_from_heuristic(record, cells):
    if not cells:
        return record
    heuristic = assign_fields_by_heuristics(cells)
    filled = False
    for key, value in heuristic.items():
        if key.startswith("_") or key in {"parse_mode", "needs_review", "confidence", "issues", "raw_text"}:
            continue
        if _missing(record.get(key)) and not _missing(value):
            record[key] = value
            filled = True
    if filled and record.get("parse_mode") == "table" and record.get("hold_amount") is None:
        record["parse_mode"] = "heuristic"
    if heuristic.get("issues"):
        record.setdefault("issues", [])
        record["issues"] = list(record.get("issues") or []) + list(heuristic.get("issues") or [])
    return record


def _row_looks_like_header(row):
    if not row:
        return False
    if any(_looks_like_fund_name(cell.get("text")) for cell in row):
        return False
    hits = [cell for cell in row if _classify_header_cell(cell.get("text"))]
    return len(hits) >= 2


def _extract_by_table(items):
    rows = _group_rows(items)
    header_map = _detect_header_map(rows)
    if not header_map:
        return []
    results = []
    current = None
    for row in rows:
        if _row_looks_like_header(row):
            if current and current.get("fund_name"):
                results.append(current)
                current = None
            continue
        name_cell = next((cell for cell in row if _looks_like_fund_name(cell.get("text"))), None)
        if name_cell is not None:
            if current and current.get("fund_name"):
                results.append(current)
            current = _empty_holding()
            current["fund_name"] = _looks_like_fund_name(name_cell.get("text"))
            current["parse_mode"] = "table"
            current["confidence"] = float(name_cell.get("confidence") or 1.0)
            current["_cells"] = []
        elif current is None:
            continue
        current.setdefault("_cells", [])
        current["_cells"].extend(row)
        texts = [str(cell.get("text") or "") for cell in row]
        current["remark"] = " ".join(part for part in (current.get("remark"), _collect_remarks(*texts)) if part).strip()
        current["raw_text"] = " | ".join(part for part in (current.get("raw_text"), *texts) if part)
        for cell in row:
            text = cell.get("text")
            name = _looks_like_fund_name(text)
            if name:
                if not current.get("fund_name"):
                    current["fund_name"] = name
                continue
            if parse_percent(text) is not None and "%" in str(text):
                field = _nearest_field(cell.get("x"), header_map, {"weight_pct", "hold_return_rate"})
            elif parse_signed_number(text) is not None:
                field = _nearest_field(
                    cell.get("x"),
                    header_map,
                    {"hold_amount", "yesterday_profit", "hold_profit", "cumulative_profit"},
                )
            else:
                field = _nearest_field(cell.get("x"), header_map)
            if not field or field == "fund_name":
                continue
            value = _parse_field_value(field, text)
            if value in {None, ""}:
                continue
            current[field] = value
            current["confidence"] = min(float(current.get("confidence") or 1.0), float(cell.get("confidence") or 1.0))
    if current and current.get("fund_name"):
        results.append(current)
    finalized = []
    for record in results:
        cells = record.pop("_cells", []) or []
        filled = sum(
            1
            for key in ("hold_amount", "yesterday_profit", "hold_profit", "hold_return_rate", "weight_pct")
            if record.get(key) is not None
        )
        if filled < 3 or record.get("hold_amount") is None:
            record = _fill_missing_from_heuristic(record, cells)
            record["needs_review"] = True
        _flag_anomalous_holdings(record)
        if record.get("needs_review"):
            record["confidence"] = min(float(record.get("confidence") or 1.0), 0.7)
        if record.get("fund_name"):
            finalized.append(record)
    return finalized


def extract_holdings_full(text_list=None, image_path=None):
    """
    从支付宝持仓截图提取完整字段。优先按坐标+表头分列，失败则按数值特征回填。
    """
    items = []
    if image_path:
        items = extract_text_from_image(image_path)
    if not items and text_list is not None:
        items = _as_items(text_list)
    if not items:
        return []

    if _has_boxes(items):
        table_rows = _extract_by_table(items)
        if table_rows:
            return table_rows

    if _has_boxes(items):
        visual_rows = _group_rows(items)
        blocks = []
        current = []
        for row in visual_rows:
            joined = " ".join(cell.get("text") or "" for cell in row)
            if _looks_like_fund_name(joined) or any(_looks_like_fund_name(cell.get("text")) for cell in row):
                if current:
                    blocks.append(current)
                current = list(row)
            elif current:
                current.extend(row)
        if current:
            blocks.append(current)
        results = []
        for block in blocks:
            labeled = _extract_labeled_block(block)
            if labeled.get("hold_amount") is None or labeled.get("needs_review"):
                heuristic = assign_fields_by_heuristics(block)
                for key, value in heuristic.items():
                    if key.startswith("_"):
                        continue
                    if _missing(labeled.get(key)) and not _missing(value):
                        labeled[key] = value
                labeled["parse_mode"] = "heuristic"
                labeled["needs_review"] = True
            results.append(labeled)
        if results:
            for record in results:
                _flag_anomalous_holdings(record)
            return results

    results = []
    seen = set()
    items = _as_items(items)
    for index, item in enumerate(items):
        name = _looks_like_fund_name(item["text"])
        if not name or name in seen:
            continue
        window = items[index:index + 14]
        labeled = _extract_labeled_block(window)
        if labeled.get("hold_amount") is None:
            heuristic = assign_fields_by_heuristics(window)
            for key, value in heuristic.items():
                if key.startswith("_"):
                    continue
                if _missing(labeled.get(key)) and not _missing(value):
                    labeled[key] = value
            labeled["parse_mode"] = "heuristic"
            labeled["needs_review"] = True
        if labeled.get("fund_name"):
            seen.add(labeled["fund_name"])
            _flag_anomalous_holdings(labeled)
            results.append(labeled)
    return results


def _extract_labeled_block(window):
    record = _empty_holding()
    texts = [str(line.get("text") or "") for line in window]
    record["raw_text"] = " | ".join(texts[:8])
    record["remark"] = _collect_remarks(*texts)
    record["parse_mode"] = "labeled"
    pending = ""
    for line in window:
        text = str(line.get("text") or "")
        name = _looks_like_fund_name(text)
        if name and record["fund_name"] and name != record["fund_name"]:
            break
        if name and not record["fund_name"]:
            record["fund_name"] = name
            record["confidence"] = min(float(record.get("confidence") or 1.0), float(line.get("confidence") or 1.0))
        rate = parse_percent(text)
        signed = _line_signed_numbers(text)

        def _fill(field, value, as_amount=False):
            if value is None:
                return False
            record[field] = abs(value) if as_amount else value
            return True

        labeled = True
        if "持有收益率" in text or ("收益率" in text and rate is not None):
            pending = "" if _fill("hold_return_rate", rate) else "hold_return_rate"
        elif "占比" in text:
            pending = "" if _fill("weight_pct", rate) else "weight_pct"
        elif "累计收益" in text:
            pending = "" if _fill("cumulative_profit", signed[0] if signed else None) else "cumulative_profit"
        elif "持有收益" in text and "率" not in text:
            pending = "" if _fill("hold_profit", signed[0] if signed else None) else "hold_profit"
        elif any(key in text for key in ("日收益", "昨日收益", "日盈亏")):
            pending = "" if _fill("yesterday_profit", signed[0] if signed else None) else "yesterday_profit"
        elif any(key in text for key in ("持有金额", "持有市值", "当前市值")):
            pending = "" if _fill("hold_amount", signed[0] if signed else None, as_amount=True) else "hold_amount"
        else:
            labeled = False

        if labeled:
            continue
        if pending == "hold_return_rate" and rate is not None:
            record["hold_return_rate"] = rate
            pending = ""
            continue
        if pending == "weight_pct" and rate is not None:
            record["weight_pct"] = rate
            pending = ""
            continue
        if pending == "hold_amount" and signed:
            record["hold_amount"] = abs(signed[0])
            pending = ""
            continue
        if pending and signed:
            record[pending] = signed[0]
            pending = ""
            continue
        if rate is not None and record["weight_pct"] is None:
            record["weight_pct"] = rate
        elif rate is not None and record["hold_return_rate"] is None:
            record["hold_return_rate"] = rate
    record["needs_review"] = record["hold_amount"] is None or not record["fund_name"]
    if record["needs_review"]:
        record["confidence"] = min(float(record.get("confidence") or 1.0), 0.72)
    return record


def extract_transactions(text_list):
    """从 OCR 文本提取交易：买入/卖出、基金名称、份额、日期。"""
    items = _as_items(text_list)
    results = []
    used_names = set()
    action_indices = [idx for idx, item in enumerate(items) if _detect_action(item["text"])]
    for pos, index in enumerate(action_indices):
        item = items[index]
        action = _detect_action(item["text"])
        next_index = action_indices[pos + 1] if pos + 1 < len(action_indices) else min(len(items), index + 12)
        name_start = max(0, index - 2)
        if pos > 0:
            name_start = max(name_start, action_indices[pos - 1] + 1)
        window = items[name_start:next_index]
        forward = items[index:next_index]
        name = ""
        shares = None
        fallback_shares = None
        date_text = ""
        for line in window:
            if not name:
                found = _looks_like_fund_name(line["text"])
                if found:
                    name = found
        for line in forward:
            text = line["text"]
            numbers = _line_numbers(text)
            if numbers and shares is None:
                if any(key in text for key in ("金额", "市值", "净值", "费率", "%")):
                    pass
                elif any(key in text for key in ("份额", "数量", "份")):
                    shares = numbers[0]
                elif not _looks_like_fund_name(text) and not _parse_date(text) and not _detect_action(text):
                    if fallback_shares is None:
                        fallback_shares = numbers[0]
            if not date_text:
                date_text = _parse_date(text)
        if shares is None:
            shares = fallback_shares
        if not name or shares is None:
            continue
        key = (action, name, shares, date_text)
        if key in used_names:
            continue
        used_names.add(key)
        confidence = min(float(item.get("confidence") or 1.0), _window_confidence(window))
        if not date_text:
            confidence = min(confidence, 0.72)
        results.append(
            {
                "action": action,
                "fund_name": name,
                "shares": shares,
                "date": date_text,
                "confidence": round(confidence, 4),
                "raw_text": " | ".join(line["text"] for line in forward[:8]),
            }
        )
    return results


def _resolve_ocr_sector(text):
    try:
        from ..factor_layer.sector_classifier import resolve_sector_label
    except Exception:
        return ""
    return resolve_sector_label(text)


def _extract_sector_pairs_table(items):
    rows = _group_rows(items)
    header_idx = None
    name_x = None
    sector_x = None
    for index, row in enumerate(rows):
        texts = [str(cell.get("text") or "") for cell in row]
        has_name = any(
            "基金名称" in text or "基金代码" in text or "基金标识" in text
            or text.strip() in {"名称", "基金", "代码"}
            for text in texts
        )
        has_sector = any("赛道" in text or (text.strip() in {"行业", "主题"} ) for text in texts)
        if not (has_name and has_sector):
            continue
        header_idx = index
        for cell in row:
            text = str(cell.get("text") or "")
            if "基金名称" in text or "基金代码" in text or "基金标识" in text or text.strip() in {"名称", "基金", "代码"}:
                name_x = cell.get("x")
            if "赛道" in text or text.strip() in {"行业", "主题"}:
                sector_x = cell.get("x")
        break
    if header_idx is None or name_x is None or sector_x is None:
        return []

    results = []
    seen = set()
    for row in rows[header_idx + 1:]:
        name = ""
        sector = ""
        name_dist = 1e9
        sector_dist = 1e9
        conf = 1.0
        for cell in row:
            text = str(cell.get("text") or "")
            x = float(cell.get("x") or 0.0)
            score = float(cell.get("confidence") or 1.0)
            fund = _looks_like_fund_identifier(text)
            if fund and abs(x - name_x) < name_dist:
                name = fund
                name_dist = abs(x - name_x)
                conf = min(conf, score)
            resolved = _resolve_ocr_sector(text)
            if resolved and not fund and abs(x - sector_x) < sector_dist:
                sector = resolved
                sector_dist = abs(x - sector_x)
                conf = min(conf, score)
        if not name or not sector or name in seen:
            continue
        seen.add(name)
        results.append(
            {
                "fund_name": name,
                "identifier": name,
                "sector": sector,
                "confidence": round(conf, 4),
                "raw_text": " | ".join(str(cell.get("text") or "") for cell in row),
            }
        )
    return results


def _extract_sector_pairs_sequential(items):
    results = []
    seen = set()
    ordered = _as_items(items)
    for index, item in enumerate(ordered):
        name = _looks_like_fund_identifier(item.get("text"))
        if not name or name in seen:
            continue
        sector = ""
        conf = float(item.get("confidence") or 1.0)
        window = ordered[index + 1:index + 8]
        for line in window:
            text = str(line.get("text") or "")
            if _looks_like_fund_identifier(text):
                break
            if "基金名称" in text or "基金代码" in text or "基金标识" in text or "赛道" in text:
                continue
            resolved = _resolve_ocr_sector(text)
            if resolved:
                sector = resolved
                conf = min(conf, float(line.get("confidence") or 1.0))
                break
        if not sector:
            continue
        seen.add(name)
        results.append(
            {
                "fund_name": name,
                "identifier": name,
                "sector": sector,
                "confidence": round(conf, 4),
                "raw_text": " | ".join(
                    [str(item.get("text") or "")] + [str(line.get("text") or "") for line in window[:4]]
                ),
            }
        )
    return results


def extract_sector_pairs(text_list=None, image_path=None):
    """
    从截图提取「基金名称 + 赛道」对。

    优先按表头「基金名称」「赛道」分列；否则按基金名称后紧跟的赛道关键词匹配。
    """
    items = []
    if image_path:
        items = extract_text_from_image(image_path)
    if not items and text_list is not None:
        items = _as_items(text_list)
    if not items:
        return []
    results = []
    if _has_boxes(items):
        results = _extract_sector_pairs_table(items)
    if not results:
        results = _extract_sector_pairs_sequential(items)
    return results
