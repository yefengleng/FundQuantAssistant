__all__ = ["generate_trading_signal", "generate_buy_candidates"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        import src.factor_layer.scorer as scorer

        if not hasattr(scorer, "get_unheld_funds_score"):
            importlib.reload(scorer)
    except Exception:
        pass

    from . import signal_generator

    if not hasattr(signal_generator, name):
        signal_generator = importlib.reload(signal_generator)
    return getattr(signal_generator, name)
