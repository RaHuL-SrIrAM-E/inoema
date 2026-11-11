"""Plotly charts for the Payments workspace (INOEMA palette via the shared PLOTLY_LAYOUT). Pure functions: DataFrame in, Figure out."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dashboard.components import BORDER, CORAL, MUTED, PRIMARY_TEXT, TEAL
from dashboard.payments.ui import plotly_layout


def _colors(df: pd.DataFrame) -> list[str]:
    return [TEAL if f else CORAL for f in df["Feasible"]]


def recovery_ci_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=df["Policy"], y=df["Recovery rate"], marker_color=_colors(df),
                           error_y=dict(type="data", symmetric=False, array=(df["RR CI high"] - df["Recovery rate"]).tolist(),
                                        arrayminus=(df["Recovery rate"] - df["RR CI low"]).tolist(), color=PRIMARY_TEXT, thickness=1.2),
                           customdata=df[["Feasible"]].values, hovertemplate="%{x}<br>recovery rate %{y:.1%}<br>feasible: %{customdata[0]}<extra></extra>"))
    fig.update_layout(**plotly_layout(height=340, yaxis=dict(title="Recovery rate (95% Wilson CI)", tickformat=".0%", gridcolor=BORDER), xaxis=dict(tickangle=-30),
                                      title=dict(text="Recovery rate · teal = feasible, coral = infeasible (SIMULATED)", font=dict(size=13))))
    return fig


def score_chart(df: pd.DataFrame) -> go.Figure:
    d = df.sort_values("Score")
    fig = go.Figure(go.Bar(y=d["Policy"], x=d["Score"], orientation="h", marker_color=_colors(d),
                           error_x=dict(type="data", symmetric=False, array=(d["Score CI high"] - d["Score"]).fillna(0).tolist(),
                                        arrayminus=(d["Score"] - d["Score CI low"]).fillna(0).tolist(), color=PRIMARY_TEXT, thickness=1.2)))
    fig.update_layout(**plotly_layout(height=max(260, 34 * len(d) + 120), xaxis=dict(title="Objective score (deterministic; 95% CI over trials)", gridcolor=BORDER),
                                      title=dict(text="Objective score · infeasible policies shown in coral are never selectable", font=dict(size=13))))
    return fig


def tradeoff_scatter(df: pd.DataFrame) -> go.Figure:
    """One trace per policy (legend lists names, so near-identical policies stay legible); hover for the numbers."""
    fig = go.Figure()
    mx = max(df["Processing cost / trial"].max(), 1e-9)
    teal_shades = ["#2CB6A4", "#7BC7BB", "#1E8A7A", "#A5DDD5", "#146B5E", "#CDEDE9"]
    n_f = 0
    for _, r in df.iterrows():
        if r["Feasible"]:
            color, n_f = teal_shades[n_f % len(teal_shades)], n_f + 1
        else:
            color = CORAL
        fig.add_trace(go.Scatter(x=[r["Avg recovery (min)"]], y=[r["Recovered revenue / trial"]], mode="markers", name=f"{r['Policy']}{'' if r['Feasible'] else ' (infeasible)'}",
                                 marker=dict(color=color, size=12 + 34 * r["Processing cost / trial"] / mx, opacity=0.8, symbol="circle" if r["Feasible"] else "x", line=dict(color="#fff", width=1)),
                                 hovertemplate=f"{r['Policy']}<br>avg recovery %{{x:.0f}} min<br>recovered revenue %{{y:,.0f}}<extra></extra>"))
    fig.update_layout(**plotly_layout(height=400, xaxis=dict(title="Average recovery time (minutes)", gridcolor=BORDER), yaxis=dict(title="Recovered revenue per trial", gridcolor=BORDER),
                                      title=dict(text="Revenue vs latency trade-off · marker size = processing cost · × = infeasible", font=dict(size=13)), legend=dict(orientation="h", y=-0.25)))
    return fig


def sensitivity_line(frames: dict, parameter: str) -> go.Figure:
    pts = pd.DataFrame(frames["points"])
    pts = pts[pts["parameter"] == parameter]
    base = pd.DataFrame(frames["baseline"])
    fig = go.Figure()
    palette = [TEAL, CORAL, PRIMARY_TEXT, MUTED, "#7BC7BB", "#F5A79B"]
    for i, pol in enumerate(sorted(pts["policy"].unique())):
        d = pts[pts["policy"] == pol].sort_values("value")
        b = float(base[base["policy"] == pol]["score"].iloc[0]) if (base["policy"] == pol).any() else None
        fig.add_trace(go.Scatter(x=d["value"], y=d["score"], mode="lines+markers", name=pol, line=dict(color=palette[i % len(palette)]),
                                 marker=dict(size=[13 if (w and w != pol) else 8 for w in d["winner"]], symbol=["diamond" if w == pol else "circle" for w in d["winner"]])))
        if b is not None:
            fig.add_hline(y=b, line=dict(color=palette[i % len(palette)], dash="dot", width=1), opacity=0.35)
    fig.update_layout(**plotly_layout(height=380, xaxis=dict(title=f"{parameter} (swept; ASSUMED range)", gridcolor=BORDER), yaxis=dict(title="Objective score", gridcolor=BORDER),
                                      title=dict(text="Objective score under assumption variations · diamond = winner at that point · dotted = baseline score", font=dict(size=13)),
                                      legend=dict(orientation="h", y=-0.25)))
    return fig


def sensitivity_heatmap(frames: dict) -> go.Figure:
    """Which policy wins at each swept point (rows = parameters)."""
    pts = pd.DataFrame(frames["points"])
    w = pts.drop_duplicates(["parameter", "value"])[["parameter", "value", "winner"]]
    policies = sorted(w["winner"].dropna().unique())
    code = {p: i for i, p in enumerate(policies)}
    params = list(dict.fromkeys(w["parameter"]))
    maxn = max(len(w[w["parameter"] == p]) for p in params)
    z, text = [], []
    for p in params:
        d = w[w["parameter"] == p].sort_values("value")
        z.append([code.get(x) for x in d["winner"]] + [None] * (maxn - len(d)))
        text.append([f"{v:g}: {x}" for v, x in zip(d["value"], d["winner"])] + [""] * (maxn - len(d)))
    colorscale = [[i / max(len(policies) - 1, 1), c] for i, c in enumerate([TEAL, CORAL, PRIMARY_TEXT, MUTED, "#7BC7BB", "#F5A79B"][:len(policies)])]
    fig = go.Figure(go.Heatmap(z=z, text=text, texttemplate="%{text}", hoverinfo="text", colorscale=colorscale, showscale=False, xgap=2, ygap=2))
    fig.update_layout(**plotly_layout(height=60 + 46 * len(params), yaxis=dict(autorange="reversed", labelalias={}), xaxis=dict(showticklabels=False),
                                      title=dict(text="Winner at every sweep point (colour = winning policy)", font=dict(size=13))))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(params))), ticktext=params)
    return fig
