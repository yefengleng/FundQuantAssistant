import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

COMPILE_TARGETS = [
    ROOT / "app" / "streamlit_app.py",
    ROOT / "config" / "settings.py",
]
for folder in ("src",):
    COMPILE_TARGETS.extend(sorted((ROOT / folder).rglob("*.py")))

IMPORT_MODULES = [
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
    "akshare",
    "config.settings",
    "src.data_layer.loader",
    "src.data_layer.fetcher",
    "src.data_layer.market_clock",
    "src.data_layer.realtime_cache",
    "src.factor_layer.indicators",
    "src.factor_layer.scorer",
    "src.factor_layer.portfolio_utils",
    "src.factor_layer.sector_classifier",
    "src.factor_layer.sector_analysis",
    "src.factor_layer.comparator",
    "src.strategy_layer.constraints",
    "src.strategy_layer.filters",
    "src.strategy_layer.signal_generator",
    "src.strategy_layer.weekly_scanner",
    "src.strategy_layer.intraday_monitor",
    "src.ocr.ocr_engine",
    "src.ocr.fund_matcher",
    "src.ocr.importer",
    "src.backtest.metrics",
    "src.backtest.strategies",
    "src.backtest.engine",
]

OPTIONAL_MODULES = [
    "backtrader",
    "rapidocr_onnxruntime",
    "paddleocr",
    "paddle",
    "cv2",
    "streamlit_autorefresh",
]


def compile_files():
    print("=== 语法检查 ===", flush=True)
    failed = 0
    for path in COMPILE_TARGETS:
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            print(f"OK  {path.relative_to(ROOT)}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"FAIL  {path.relative_to(ROOT)}  {exc}", flush=True)
    return failed


def import_modules(names, optional=False):
    title = "可选依赖" if optional else "关键模块导入"
    print(f"\n=== {title} ===", flush=True)
    failed = 0
    for name in names:
        print(f"... {name}", flush=True)
        try:
            importlib.import_module(name)
            print(f"OK  {name}", flush=True)
        except Exception as exc:
            failed += 1
            tag = "SKIP" if optional else "FAIL"
            print(f"{tag}  {name}  {type(exc).__name__}: {exc}", flush=True)
    return 0 if optional else failed


def check_symbol():
    print("\n=== 关键符号 ===", flush=True)
    from src.factor_layer.scorer import get_unheld_funds_score, batch_score_funds
    from src.factor_layer.sector_classifier import get_sector_limits
    from src.ocr.fund_matcher import resolve_fund_identifier
    from src.strategy_layer.signal_generator import generate_trading_signal, generate_buy_candidates

    _ = (
        get_unheld_funds_score,
        batch_score_funds,
        get_sector_limits,
        resolve_fund_identifier,
        generate_trading_signal,
        generate_buy_candidates,
    )
    print("OK  scorer.get_unheld_funds_score", flush=True)
    print("OK  sector_classifier.get_sector_limits", flush=True)
    print("OK  fund_matcher.resolve_fund_identifier", flush=True)
    print("OK  signal_generator.generate_trading_signal / generate_buy_candidates", flush=True)


def print_cache_hints():
    home = Path.home()
    print("\n=== Streamlit 缓存路径（若模块错误仍在，可删除后重启） ===", flush=True)
    for path in (
        home / ".streamlit",
        home / ".cache" / "streamlit",
        home / "AppData" / "Local" / "streamlit",
        ROOT / ".streamlit",
    ):
        mark = "存在" if path.exists() else "不存在"
        print(f"{mark}  {path}", flush=True)
    print("请先停掉占用 8502 的旧进程，再启动：", flush=True)
    print("  taskkill /F /IM streamlit.exe", flush=True)
    print("  netstat -ano | findstr :8502", flush=True)
    print("  .\\.venv\\Scripts\\python.exe -m streamlit run app/streamlit_app.py --server.port 8502", flush=True)
    print("浏览器请用 Ctrl+F5 强制刷新。Windows 缓存目录通常是 %USERPROFILE%\\.streamlit", flush=True)


def main():
    failed = compile_files()
    failed += import_modules(IMPORT_MODULES, optional=False)
    import_modules(OPTIONAL_MODULES, optional=True)
    try:
        check_symbol()
    except Exception as exc:
        failed += 1
        print(f"FAIL  关键符号  {exc}")
    print_cache_hints()
    print("\n=== 结论 ===")
    if failed:
        print(f"发现 {failed} 个必须修复的问题。")
        return 1
    print("语法与关键导入正常。若看板仍报 Failed to fetch dynamically imported module：")
    print("1) 停掉旧的 streamlit 进程后重新启动")
    print("2) 浏览器强制刷新（Ctrl+F5），或换无痕窗口")
    print("3) 打开开发者工具 Console，把具体 404/JS 报错发出来")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
