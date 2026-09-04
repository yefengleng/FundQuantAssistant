# 仓位与风控阈值配置
#
# 所有进攻/防御切换必须走 STRATEGY_MODE，禁止在业务代码里另开一套判断。
# 可选值：'defensive'（防御型）、'aggressive'（进攻型）、'auto'（按沪深300与均线自动切换）

STRATEGY_MODE = "aggressive"

AUTO_MODE_RULE = {
    "ma_period": 120,
    "index_code": "000300",
}

STRATEGY_MODE_LABELS = {
    "defensive": "防御型",
    "aggressive": "进攻型",
    "auto": "自动识别",
}

# 进攻型沿用原阈值；防御型收紧总仓位、赛道与单基上限。熔断线两边共用。
STRATEGY_PROFILES = {
    "aggressive": {
        "TOTAL_EQUITY_LIMIT": 0.85,
        "RETREAT_LIMIT": -0.18,
        "SINGLE_FUND_LIMIT": 0.12,
        "SECTOR_LIMITS": {
            "半导体": 0.25,
            "CPO": 0.12,
            "新能源": 0.10,
            "美股": 0.13,
            "其他": 0.08,
        },
    },
    "defensive": {
        "TOTAL_EQUITY_LIMIT": 0.50,
        "RETREAT_LIMIT": -0.18,
        "SINGLE_FUND_LIMIT": 0.08,
        "SECTOR_LIMITS": {
            "半导体": 0.15,
            "CPO": 0.08,
            "新能源": 0.06,
            "美股": 0.08,
            "其他": 0.05,
        },
    },
}

# 兼容旧 import：未显式传模式时视为进攻型阈值
TOTAL_EQUITY_LIMIT = STRATEGY_PROFILES["aggressive"]["TOTAL_EQUITY_LIMIT"]
RETREAT_LIMIT = STRATEGY_PROFILES["aggressive"]["RETREAT_LIMIT"]
SECTOR_LIMITS = STRATEGY_PROFILES["aggressive"]["SECTOR_LIMITS"]
SINGLE_FUND_LIMIT = STRATEGY_PROFILES["aggressive"]["SINGLE_FUND_LIMIT"]

SECTOR_TOP_N = 3
ENABLE_SECTOR_TOP_N = True
SECTOR_TOP_N_ACTIVE_MODES = ["aggressive"]
# 赛道精简赎回的资金默认视为转入货币基金，下月再分配。

# 赛道识别模式：'name_only'（仅名称）或 'hybrid'（混合模式，默认）
SECTOR_CLASSIFY_MODE = "hybrid"

OPERATION_FREQUENCY = "weekly"
OPERATION_FREQUENCY_LABELS = {
    "monthly": "月度调仓",
    "weekly": "周调仓",
}

# 展示层默认时间窗口（仅用于看板图表/表格，不影响调仓逻辑）
DISPLAY_WINDOW_DAYS = 60
DISPLAY_WINDOW_CHOICES = (20, 60, 120)

# 调仓与风控使用的固定窗口；未定义时沿用 60。
REBALANCE_WINDOW_DAYS = 60

# 周调仓：每周五为正式调仓日；盘中紧急信号可当日触发。
# 涨跌幅阈值为百分比点数（-7.0 表示 -7%），回撤为小数。
WEEKLY_SCAN = {
    "run_weekday": 4,
    "retreat_limit": -0.18,
    "fund_intraday_drop_pct": -7.0,
    "sector_intraday_drop_pct": -5.0,
    "crash_warn_pct": -5.0,
    "clear_fund_pct": -10.0,
    "sector_week_drop": -0.10,
    "score_drop_points": 15.0,
    "score_drop_ratio": 0.30,
    "lookback_days": 5,
    "reduce_ratio": 0.30,
    "emergency_reduce_ratio": 0.50,
    "meltdown_equity_cap": 0.50,
}


def normalize_strategy_mode(mode):
    """把输入规整为 defensive / aggressive / auto，非法值按防御处理。"""
    text = "" if mode is None else str(mode).strip().lower()
    if text in {"defensive", "aggressive", "auto"}:
        return text
    return "defensive"


def get_strategy_profile(mode):
    """
    返回生效模式对应的风控参数副本。

    mode 只能是 defensive 或 aggressive；auto 必须先解析成二者之一再调用。
    """
    key = "aggressive" if str(mode).strip().lower() == "aggressive" else "defensive"
    profile = STRATEGY_PROFILES[key]
    try:
        from src.factor_layer.sector_classifier import get_sector_limits

        sector_limits = get_sector_limits()
    except Exception:
        sector_limits = dict(profile["SECTOR_LIMITS"])
    return {
        "TOTAL_EQUITY_LIMIT": profile["TOTAL_EQUITY_LIMIT"],
        "RETREAT_LIMIT": profile["RETREAT_LIMIT"],
        "SINGLE_FUND_LIMIT": profile["SINGLE_FUND_LIMIT"],
        "SECTOR_LIMITS": sector_limits,
    }


def is_sector_top_n_active(effective_mode):
    """赛道精简是否对当前生效模式打开。列表含 both 时进攻/防御都生效。"""
    if not ENABLE_SECTOR_TOP_N:
        return False
    modes = [str(item).strip().lower() for item in (SECTOR_TOP_N_ACTIVE_MODES or [])]
    if "both" in modes:
        return True
    key = str(effective_mode or "").strip().lower()
    return key in {"defensive", "aggressive"} and key in modes


def normalize_operation_frequency(value):
    text = "" if value is None else str(value).strip().lower()
    if text in {"weekly", "monthly"}:
        return text
    return "monthly"


def normalize_display_window(value):
    """把展示窗口规整为 20 / 60 / 120，非法值回落到 DISPLAY_WINDOW_DAYS。"""
    try:
        days = int(value)
    except (TypeError, ValueError):
        days = int(DISPLAY_WINDOW_DAYS)
    if days in DISPLAY_WINDOW_CHOICES:
        return days
    return int(DISPLAY_WINDOW_DAYS)
