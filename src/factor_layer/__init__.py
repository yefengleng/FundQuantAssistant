__all__ = [
    "calculate_score",
    "batch_score_funds",
    "get_unheld_funds_score",
    "scan_market_funds",
    "load_current_holdings",
]


def __getattr__(name):
    if name == "load_current_holdings":
        from .portfolio_utils import load_current_holdings

        return load_current_holdings
    if name in {"calculate_score", "batch_score_funds", "get_unheld_funds_score", "scan_market_funds"}:
        import importlib

        from . import scorer

        if not hasattr(scorer, name):
            scorer = importlib.reload(scorer)
        return getattr(scorer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
