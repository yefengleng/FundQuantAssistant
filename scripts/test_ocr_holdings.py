"""持仓 OCR 分列、启发式回填与 CSV 粘贴解析的回归测试。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _cell(text, x, y, w=40, h=16):
    return {
        "text": text,
        "confidence": 0.99,
        "x": x,
        "y": y,
        "x1": x - w / 2,
        "x2": x + w / 2,
        "y1": y - h / 2,
        "y2": y + h / 2,
        "width": w,
        "height": h,
        "box": [[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2], [x - w / 2, y + h / 2]],
    }


def test_table_by_boxes():
    from src.ocr.ocr_engine import extract_holdings_full

    items = [
        _cell("基金名称", 80, 20),
        _cell("持有金额", 240, 20),
        _cell("昨日收益", 360, 20),
        _cell("持有收益", 480, 20),
        _cell("占比", 580, 20),
        _cell("持有收益率", 700, 20),
        _cell("东方人工智能主题混合C", 80, 60),
        _cell("201579.85", 240, 60),
        _cell("+106.02", 360, 60),
        _cell("-30123.45", 480, 60),
        _cell("12.34%", 580, 60),
        _cell("-13.22%", 700, 60),
    ]
    rows = extract_holdings_full(text_list=items)
    assert rows, "应按坐标还原出行"
    row = rows[0]
    assert "人工智能" in row["fund_name"]
    assert abs(float(row["hold_amount"]) - 201579.85) < 0.01
    assert abs(float(row["yesterday_profit"]) - 106.02) < 0.01
    assert abs(float(row["hold_profit"]) + 30123.45) < 0.01
    assert abs(float(row["weight_pct"]) - 12.34) < 0.01
    assert abs(float(row["hold_return_rate"]) + 13.22) < 0.01
    assert row.get("parse_mode") == "table"
    print("ok table-by-box", row["fund_name"], row["hold_amount"])


def test_heuristic_assignment():
    from src.ocr.ocr_engine import assign_fields_by_heuristics

    items = [
        {"text": "华夏国证半导体芯片ETF联接C"},
        {"text": "12.34%"},
        {"text": "-8.50%"},
        {"text": "+88.21"},
        {"text": "-15200.33"},
        {"text": "88012.00"},
        {"text": "+320.10"},
    ]
    record = assign_fields_by_heuristics(items)
    assert abs(record["hold_amount"] - 88012.00) < 0.01
    assert abs(record["weight_pct"] - 12.34) < 0.01
    assert abs(record["hold_return_rate"] + 8.50) < 0.01
    assert abs(record["yesterday_profit"] - 88.21) < 0.01
    assert abs(record["hold_profit"] + 15200.33) < 0.01
    print("ok heuristic", record["hold_amount"], record["yesterday_profit"], record["hold_profit"])


def test_anomaly_flag():
    from src.ocr.ocr_engine import _flag_anomalous_holdings

    record = {
        "fund_name": "测试基金混合C",
        "hold_amount": -12,
        "yesterday_profit": 100,
        "hold_profit": 1,
        "issues": [],
    }
    _flag_anomalous_holdings(record)
    assert record["needs_review"]
    assert any("负数" in item for item in record["issues"])
    print("ok anomaly", record["issues"])


def test_csv_parse():
    from src.ocr.importer import parse_holdings_csv, validate_holdings_preview

    text = (
        "基金名称,持有金额(元),占比(%),昨日收益(元),持有收益(元),持有收益率(%),累计收益(元),备注\n"
        "东方人工智能主题混合C,201579.85,12.34,+106.02,-30123.45,-13.22,1357.80,定投\n"
    )
    rows, error, meta = parse_holdings_csv(text)
    assert not error, error
    assert len(rows) == 1
    assert abs(rows[0]["hold_amount"] - 201579.85) < 0.01
    assert rows[0]["parse_mode"] == "csv"

    bad = "foo,bar\n1,2\n"
    rows, error, meta = parse_holdings_csv(bad)
    assert error and meta and meta.get("need_column_map")
    print("ok csv parse / need map")

    warnings = validate_holdings_preview(
        [{"include": True, "fund_name": "A", "hold_amount": -1, "yesterday_profit": 2}]
    )
    assert any("负数" in item for item in warnings)
    print("ok validate negative amount")


def main():
    test_table_by_boxes()
    test_heuristic_assignment()
    test_anomaly_flag()
    test_csv_parse()
    print("全部验证通过。")


if __name__ == "__main__":
    main()
