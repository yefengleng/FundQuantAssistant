import pandas as pd
import plotly.graph_objects as go


def _normalize(series):
    values = pd.to_numeric(series, errors="coerce")
    valid = values[values > 0]
    if valid.empty:
        return values
    return values / float(valid.iloc[0]) * 100.0


def plot_equity_curve(equity_df):
    fig = go.Figure()
    if equity_df is None or equity_df.empty:
        fig.update_layout(title="策略净值曲线", height=420)
        return fig
    frame = equity_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=_normalize(frame["strategy"]),
            mode="lines",
            name="策略净值",
            line=dict(width=2.6, color="#2563eb"),
        )
    )
    has_benchmark = "benchmark" in frame.columns and frame["benchmark"].notna().any()
    if has_benchmark:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=_normalize(frame["benchmark"]),
                mode="lines",
                name="沪深300",
                line=dict(width=2.0, color="#94a3b8", dash="dash"),
            )
        )
    fig.update_layout(
        title="策略净值 vs 沪深300（起点=100）" if has_benchmark else "策略净值曲线（起点=100）",
        xaxis_title="日期",
        yaxis_title="归一化净值",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_drawdown_curve(equity_df):
    fig = go.Figure()
    if equity_df is None or equity_df.empty:
        fig.update_layout(title="回撤曲线", height=320)
        return fig
    frame = equity_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if "drawdown" in frame.columns:
        drawdown = pd.to_numeric(frame["drawdown"], errors="coerce")
    else:
        series = pd.to_numeric(frame["strategy"], errors="coerce")
        peak = series.cummax()
        drawdown = series / peak.replace(0, pd.NA) - 1.0
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=drawdown * 100.0,
            mode="lines",
            name="回撤",
            line=dict(width=2.0, color="#dc2626"),
            fill="tozeroy",
            fillcolor="rgba(220, 38, 38, 0.12)",
        )
    )
    fig.update_layout(
        title="回撤曲线",
        xaxis_title="日期",
        yaxis_title="回撤（%）",
        hovermode="x unified",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig
