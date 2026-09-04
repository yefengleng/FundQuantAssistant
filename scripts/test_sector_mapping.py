import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

from src.factor_layer import sector_classifier as sc


def _use_temp_mapping():
    handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    handle.write(b"{}\n")
    handle.close()
    sc.GLOBAL_MAPPING_PATH = handle.name
    sc.invalidate_global_mapping_cache()
    sc._MEMORY_CACHE.clear()
    return handle.name


def test_mapping_cache_and_save():
    path = _use_temp_mapping()
    first = sc.load_global_sector_mapping()
    second = sc.load_global_sector_mapping()
    assert first == {} and second == {}
    sc.set_global_sector("000001", "半导体")
    loaded = sc.load_global_sector_mapping()
    assert loaded["000001"] == "半导体"
    sc.invalidate_global_mapping_cache()
    again = sc.load_global_sector_mapping()
    assert again["000001"] == "半导体"
    mtime = os.path.getmtime(path)
    cached = sc.load_global_sector_mapping()
    assert cached["000001"] == "半导体"
    assert os.path.getmtime(path) == mtime
    print("ok mapping cache")


def test_auto_tag_prefers_global():
    _use_temp_mapping()
    name = "华夏国证半导体芯片ETF联接C"
    auto = sc.auto_tag_fund("999999", name, allow_network=False)
    assert auto["source"] in {"name", "holdings", "fallback"}
    assert auto["sector"] == "半导体"
    sc.set_global_sector("999999", "消费")
    mapped = sc.auto_tag_fund("999999", name, allow_network=False)
    assert mapped == {"sector": "消费", "source": "global_manual"}
    sc.remove_global_sector("999999")
    restored = sc.auto_tag_fund("999999", name, allow_network=False)
    assert restored["source"] != "global_manual"
    assert restored["sector"] == "半导体"
    sc.set_global_sector("999999", "消费")
    ignored = sc.auto_tag_fund("999999", name, allow_network=False, ignore_global=True)
    assert ignored["source"] != "global_manual"
    assert ignored["sector"] == "半导体"
    print("ok auto_tag precedence")


def test_apply_overlay():
    _use_temp_mapping()
    sc.set_global_sector("000001", "CPO")
    frame = pd.DataFrame({"基金代码": ["000001", "000002"], "赛道归类": ["消费", "新能源"]})
    out = sc.apply_global_sector_map(frame)
    assert out.loc[out["基金代码"] == "000001", "赛道归类"].iloc[0] == "CPO"
    assert out.loc[out["基金代码"] == "000002", "赛道归类"].iloc[0] == "新能源"
    print("ok overlay")


def test_parse_csv():
    rows, error = sc.parse_sector_csv_text(
        "基金标识,赛道\n华夏国证半导体芯片ETF联接C,半导体\n012349,新能源\n"
    )
    assert not error
    assert len(rows) == 2
    assert rows[0]["identifier"] == "华夏国证半导体芯片ETF联接C"
    assert rows[0]["sector"] == "半导体"
    assert rows[1]["identifier"] == "012349"
    empty, msg = sc.parse_sector_csv_text("   ")
    assert empty == [] and msg
    print("ok parse csv")


def test_resolve_identifier():
    from src.ocr.fund_matcher import resolve_fund_identifier

    resolve_fund_identifier.cache_clear()
    empty = resolve_fund_identifier("  ", allow_network=False)
    assert empty["value"] is None and empty["status"] == "fail"

    missing = resolve_fund_identifier("999998", allow_network=False)
    assert missing["kind"] == "code"
    assert missing["value"] is None
    assert missing["status"] == "fail"

    named = resolve_fund_identifier("华夏国证半导体芯片ETF联接C", allow_network=False)
    assert named["status"] in {"ok", "candidates", "fail"}
    if named["status"] == "ok":
        assert named["value"] and len(str(named["value"])) == 6
        assert named["kind"] == "name"

    again = resolve_fund_identifier("999998", allow_network=False)
    assert again["status"] == "fail"
    info = resolve_fund_identifier.cache_info()
    assert info.hits >= 1
    print("ok resolve identifier")


def test_extract_sector_pairs_code():
    from src.ocr.ocr_engine import extract_sector_pairs

    pairs = extract_sector_pairs(
        text_list=[
            "012349",
            "半导体",
            "易方达蓝筹精选混合",
            "消费",
        ]
    )
    idents = {item.get("identifier") or item.get("fund_name"): item["sector"] for item in pairs}
    assert idents.get("012349") == "半导体"
    assert idents.get("易方达蓝筹精选混合") == "消费"
    print("ok ocr identifier pairs")


def test_extract_sector_pairs_text():
    from src.ocr.ocr_engine import extract_sector_pairs

    pairs = extract_sector_pairs(
        text_list=[
            "基金名称",
            "赛道",
            "华夏国证半导体芯片ETF联接C",
            "半导体",
            "易方达蓝筹精选混合",
            "消费",
        ]
    )
    names = {item["fund_name"]: item["sector"] for item in pairs}
    assert "华夏国证半导体芯片ETF联接C" in names
    assert names["华夏国证半导体芯片ETF联接C"] == "半导体"
    assert names.get("易方达蓝筹精选混合") == "消费"
    print("ok ocr sequential pairs")


def test_sector_limits_dynamic():
    handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    handle.close()
    os.remove(handle.name)
    old_path = sc.GLOBAL_SECTOR_LIMITS_PATH
    sc.GLOBAL_SECTOR_LIMITS_PATH = handle.name
    try:
        limits = sc.get_sector_limits()
        assert os.path.isfile(handle.name)
        assert "半导体" in limits
        assert abs(float(limits["半导体"]) - 0.25) < 1e-9
        _, added = sc.ensure_sector_limit("机器人")
        assert added is True
        loaded = sc.get_sector_limits()
        assert abs(float(loaded["机器人"]) - 0.10) < 1e-9
        _, added_again = sc.ensure_sector_limit("机器人")
        assert added_again is False
        sc.set_sector_limit("机器人", 0.18)
        assert abs(float(sc.get_sector_limits()["机器人"]) - 0.18) < 1e-9
        from config.settings import get_strategy_profile

        profile = get_strategy_profile("aggressive")
        assert abs(float(profile["SECTOR_LIMITS"]["机器人"]) - 0.18) < 1e-9
        result = sc.delete_sector_limit("机器人")
        assert result["ok"] is True
        assert "机器人" not in sc.get_sector_limits()
        print("ok sector limits")
    finally:
        sc.GLOBAL_SECTOR_LIMITS_PATH = old_path
        try:
            os.remove(handle.name)
        except OSError:
            pass


def test_batch_import_last_wins_and_diff():
    _use_temp_mapping()
    handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    handle.close()
    Path(handle.name).write_text(
        '{"半导体": 0.25, "CPO": 0.12, "新能源": 0.10, "美股": 0.13, "其他": 0.08}\n',
        encoding="utf-8",
    )
    old_limits = sc.GLOBAL_SECTOR_LIMITS_PATH
    sc.GLOBAL_SECTOR_LIMITS_PATH = handle.name
    try:
        def resolver(identifier, allow_network=True):
            ident = str(identifier)
            names = {
                "000001": "华夏国证半导体芯片ETF联接C",
                "华夏国证半导体芯片ETF联接C": "华夏国证半导体芯片ETF联接C",
            }
            codes = {
                "000001": "000001",
                "华夏国证半导体芯片ETF联接C": "000001",
            }
            if ident not in codes:
                return {"status": "fail", "fail_reason": "未找到", "fund_code": ""}
            return {
                "status": "ok",
                "value": codes[ident],
                "fund_code": codes[ident],
                "matched_name": names[ident],
            }

        pairs = [
            {"identifier": "000001", "sector": "消费", "fund_name": "华夏国证半导体芯片ETF联接C"},
            {"identifier": "华夏国证半导体芯片ETF联接C", "sector": "机器人"},
            {"identifier": "未知XX", "sector": "新能源"},
        ]
        report = sc.apply_batch_sector_import(pairs, resolver=resolver, allow_network=False)
        mapping = sc.load_global_sector_mapping()
        assert mapping["000001"] == "机器人"
        assert report["imported"] == 1
        assert report["duplicates"] and report["duplicates"][0]["出现次数"] == 2
        assert "机器人" in report["new_sectors"]
        assert any(item.get("identifier") == "未知XX" for item in report["skipped"])
        diffs = {row["您输入的赛道"]: row["系统自动识别的赛道"] for row in report["diff_rows"]}
        assert diffs.get("机器人") == "半导体"
        limits = sc.get_sector_limits()
        assert abs(float(limits["机器人"]) - 0.10) < 1e-9

        keywords_before = sc.load_sector_mapping()
        sc.clear_global_sector_mapping()
        assert sc.load_global_sector_mapping() == {}
        assert sc.load_sector_mapping() == keywords_before
        print("ok batch import")
    finally:
        sc.GLOBAL_SECTOR_LIMITS_PATH = old_limits
        try:
            os.remove(handle.name)
        except OSError:
            pass


def test_apply_global_sector_edits():
    _use_temp_mapping()
    sc.set_global_sector("000001", "半导体")
    result = sc.apply_global_sector_edits({"000001": sc.AUTO_MATCH_LABEL, "000002": "CPO"})
    assert result["changed"] is True
    mapping = sc.load_global_sector_mapping()
    assert "000001" not in mapping
    assert mapping["000002"] == "CPO"
    print("ok overview edits")


def main():
    test_mapping_cache_and_save()
    test_auto_tag_prefers_global()
    test_apply_overlay()
    test_parse_csv()
    test_sector_limits_dynamic()
    test_batch_import_last_wins_and_diff()
    test_apply_global_sector_edits()
    test_extract_sector_pairs_text()
    test_extract_sector_pairs_code()
    test_resolve_identifier()
    print("all sector mapping tests passed")


if __name__ == "__main__":
    main()
