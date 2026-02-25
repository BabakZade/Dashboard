from __future__ import annotations

import re
import sys
import os
from pathlib import Path

import joblib  # kept (you’ll enable later)
import pandas as pd
import numpy as np

from dash import dcc, html, ctx, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# --- import metrics (your current sys.path trick) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.metrics import compute_all, metric_labels
from pages.cost_function import nonlinear_cost


# -----------------------------
# Initial settings
# -----------------------------
settings = {
    "slice_window": 7,
    "early_penalty": 1,
    "late_penalty": 10,
    "cost_reactive": 200,
    "cost_predictive": 20,
}

ASSETS_DIR = "assets"
MODELS_DIR = "assets/costleap"
VALIDATION_CSV = str(Path(ASSETS_DIR) / "validation.csv")


# -----------------------------
# Shared styles
# -----------------------------
pill_row_style = {
    "display": "flex",
    "alignItems": "center",
    "gap": "10px",
    "marginBottom": "10px",
}

pill_label_style = {
    "width": "170px",
    "whiteSpace": "nowrap",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
    "fontWeight": 600,
}

pill_wrap_style = {
    "display": "flex",
    "alignItems": "stretch",
    "height": "34px",
    "borderRadius": "999px",
    "overflow": "hidden",
    "border": "1px solid #ccc",
    "backgroundColor": "#f3f4f6",
}

pill_btn_style = {
    "width": "38px",
    "border": "0px",
    "backgroundColor": "transparent",
    "cursor": "pointer",
    "fontSize": "18px",
    "lineHeight": "34px",
    "padding": "0px",
}

pill_value_style = {
    "minWidth": "56px",
    "padding": "0 10px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "fontFamily": "monospace",
    "borderLeft": "1px solid #d1d5db",
    "borderRight": "1px solid #d1d5db",
}

def param_three_zone_pill(label: str, value_id: str, dec_id: str, inc_id: str, initial_value: int):
    return html.Div(
        [
            html.Div(label, style=pill_label_style),
            html.Div(
                [
                    html.Button("−", id=dec_id, n_clicks=0, style=pill_btn_style),
                    html.Div(id=value_id, children=initial_value, style=pill_value_style),
                    html.Button("+", id=inc_id, n_clicks=0, style=pill_btn_style),
                ],
                style=pill_wrap_style,
            ),
        ],
        style=pill_row_style,
    )


dropdown_style_hidden = {
    "display": "none",
    "position": "absolute",
    "top": "45px",
    "right": "0px",
    "zIndex": "9999",
    "border": "1px solid #ddd",
    "backgroundColor": "white",
    "width": "380px",          # ✅ fixed width
    "boxSizing": "border-box", # ✅ include padding in width
    "padding": "15px",
    "borderRadius": "8px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.15)",
    "maxHeight": "560px",
    "overflowY": "auto",
}

CARD_STYLE = {"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginBottom": "12px"}
HIDDEN_STYLE = {"display": "none"}
VISIBLE_CARD_STYLE = {**CARD_STYLE, "display": "block"}


# -----------------------------
# Settings -> model filename
# -----------------------------
def settings_slug(s: dict) -> str:
    def fmt(x):
        if isinstance(x, float):
            return str(x).replace(".", "p")
        return str(x)

    return (
        f"Model_"
        f"W{fmt(s['slice_window'])}_"
        f"EP{fmt(s['early_penalty'])}_"
        f"LP{fmt(s['late_penalty'])}_"
        f"CR{fmt(s['cost_reactive'])}_"
        f"CP{fmt(s['cost_predictive'])}.pkl"
    )


# -----------------------------
# Model filename parsing
# -----------------------------
MODEL_RE = re.compile(
    r"^Model_"
    r"W(?P<slice_window>\d+)_"
    r"EP(?P<early_penalty>\d+)_"
    r"LP(?P<late_penalty>\d+)_"
    r"CR(?P<cost_reactive>\d+)_"
    r"CP(?P<cost_predictive>\d+)\.pkl$"
)

def _parse_model_name(filename: str) -> dict | None:
    m = MODEL_RE.match(filename)
    if not m:
        return None
    d = m.groupdict()
    return {
        "slice_window": int(d["slice_window"]),
        "early_penalty": int(d["early_penalty"]),
        "late_penalty": int(d["late_penalty"]),
        "cost_reactive": int(d["cost_reactive"]),
        "cost_predictive": int(d["cost_predictive"]),
    }


# -----------------------------
# Pretty settings display
# -----------------------------
SETTING_LABELS = {
    "slice_window": "Slice length (days)",
    "early_penalty": "Early penalty",
    "late_penalty": "Late penalty",
    "cost_reactive": "Reactive cost",
    "cost_predictive": "Predictive cost",
}

def format_settings_for_display(s: dict) -> str:
    lines = []
    for k in ["slice_window", "early_penalty", "late_penalty", "cost_reactive", "cost_predictive"]:
        lines.append(f"- {SETTING_LABELS.get(k, k)}: {s.get(k)}")
    return "\n".join(lines)


# -----------------------------
# Similarity by % difference
# -----------------------------
def _pct_close(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-9)
    pct_diff = abs(a - b) / denom
    return max(0.0, 1.0 - pct_diff)

def similarity_percent(target: dict, candidate: dict) -> float:
    weights = {
        "slice_window": 3.0,
        "early_penalty": 2.0,
        "late_penalty": 2.0,
        "cost_reactive": 0.5,
        "cost_predictive": 0.5,
    }
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        num += w * _pct_close(float(target[k]), float(candidate[k]))
        den += w
    return num / den if den else 0.0

def top_k_models(target: dict, model_dir: str = MODELS_DIR, k: int = 3) -> list[dict]:
    assets_path = Path(model_dir)
    rows = []
    for p in assets_path.glob("*.pkl"):
        parsed = _parse_model_name(p.name)
        if parsed is None:
            continue
        score = similarity_percent(target, parsed)
        rows.append({"name": p.name, "score": score, "parsed": parsed})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:k]


# -----------------------------
# Hooks
# -----------------------------
_MODEL_CACHE: dict[str, object] = {}

def load_model(model_path: str):
    """
    TEMP: model loading disabled. Returns None always.
    (kept for the pipeline UI: select -> load -> metrics -> graphs)
    """
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    model = None
    return model

def train_new_model(target_settings: dict, save_to: str):
    """
    TODO: replace with your real training.
    MUST write a .pkl at save_to for real usage.
    """
    return {"trained_for_pretty": format_settings_for_display(target_settings), "saved_to": save_to}

def predict_rul_from_validation_csv(model, validation_csv_path: str):
    """
    DEMO:
    - rul_true comes from column 'rul'
    - rul_pred is randomly generated with the same length
    """
    p = Path(validation_csv_path)
    if not p.exists():
        raise FileNotFoundError(f"validation.csv not found at: {validation_csv_path}")

    df = pd.read_csv(p, sep = ";", decimal= ",")

    if "rul" not in df.columns:
        raise ValueError(
            "validation.csv must contain a column named 'rul' for this demo. "
            f"Found columns: {list(df.columns)[:40]}..."
        )

    y_true = df["rul"].to_numpy(dtype=float).reshape(-1)
    y_pred = df["y_pred"].to_numpy(dtype=float).reshape(-1)


    return y_true, y_pred


def make_cost_fn_from_settings(s: dict, *, leadtime: float):
    """
    Returns a function cost_fn(yt, yp) -> float for compute_all.
    Uses the page settings 's' to fill nonlinear_cost params.
    """
    C_PR = float(s["cost_predictive"])   # predictive cost (early)
    C_RE = float(s["cost_reactive"])     # reactive cost (late)
    ALPHA = float(s["early_penalty"])
    BETA  = float(s["late_penalty"])

    def cost_fn(yt: np.ndarray, yp: np.ndarray) -> float:
        yt = np.asarray(yt, dtype=float).reshape(-1)  # true
        yp = np.asarray(yp, dtype=float).reshape(-1)  # pred
        if yt.size != yp.size:
            raise ValueError("y_true and y_pred must have the same length")

        # per-sample costs, then average
        costs = []
        for t, p in zip(yt, yp):
            costs.append(nonlinear_cost(p, t, leadtime, C_PR, ALPHA, C_RE, BETA))
        return float(np.mean(costs))

    return cost_fn

# -----------------------------
# UI rendering helpers
# -----------------------------

def render_metrics_section(metrics: dict | None):
    if not metrics:
        return html.Div("No metrics available (no y_true/y_pred yet).", style={"opacity": 0.7})

    # pull thresholds that compute_all already returns
    min_err = float(metrics.get("min_err", 2.0))
    max_err = float(metrics.get("max_err", 10.0))
    crit_lt = float(metrics.get("critical_true_lt", 10.0))

    labels = metric_labels(min_err, max_err, crit_lt)

    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    # ---------- styles ----------
    legend_style = {
        "display": "flex",
        "gap": "14px",
        "alignItems": "center",
        "fontSize": "12px",
        "opacity": 0.85,
        "marginBottom": "10px",
    }
    legend_item_style = {"display": "flex", "alignItems": "center", "gap": "6px"}
    swatch_dark = {"width": "12px", "height": "12px", "borderRadius": "2px", "backgroundColor": "#111827"}  # dark
    swatch_light = {"width": "12px", "height": "12px", "borderRadius": "2px", "backgroundColor": "#E5E7EB", "border": "1px solid #D1D5DB"}  # light

    row_style = {"marginBottom": "10px"}
    title_row_style = {"display": "flex", "justifyContent": "space-between", "gap": "10px", "fontSize": "13px"}
    label_style = {"fontWeight": 600, "flex": "1 1 auto", "minWidth": "0px"}
    value_style = {"fontFamily": "monospace", "whiteSpace": "nowrap", "flex": "0 0 auto"}

    bar_outer = {
        "marginTop": "6px",
        "height": "12px",
        "borderRadius": "999px",
        "backgroundColor": "#F3F4F6",
        "border": "1px solid #E5E7EB",
        "overflow": "hidden",
        "display": "flex",
    }
    bar_dark = {"height": "100%", "backgroundColor": "#111827"}   # conservative
    bar_light = {"height": "100%", "backgroundColor": "#E5E7EB"}  # other

    section_title_style = {"marginTop": "8px", "marginBottom": "6px", "fontSize": "12px", "fontWeight": 700, "opacity": 0.9}

    # ---------- helper to render one stacked KPI row ----------
    def stacked_kpi_row(label_key: str, total_key: str, cons_key: str):
        total = metrics.get(total_key)
        cons = metrics.get(cons_key)

        # handle Nones
        if total is None or cons is None:
            return html.Div(
                [
                    html.Div(
                        [
                            html.Span(labels.get(label_key, label_key), style=label_style),
                            html.Span(": —", style=value_style),
                        ],
                        style=title_row_style,
                    )
                ],
                style=row_style,
            )

        # clamp for safety
        total_f = float(total)
        cons_f = float(cons)
        total_f = max(0.0, min(100.0, total_f))
        cons_f = max(0.0, min(total_f, cons_f))
        other_f = max(0.0, total_f - cons_f)

        # widths as % of 100 (not of "total") so bar shows absolute prevalence
        cons_w = f"{cons_f:.4f}%"
        other_w = f"{other_f:.4f}%"

        return html.Div(
            [
                html.Div(
                    [
                        html.Span(labels.get(label_key, label_key), style=label_style),
                        html.Span(f": {total_f:.2f}%", style=value_style),
                    ],
                    style=title_row_style,
                ),
                html.Div(
                    [
                        html.Div(style={**bar_dark, "width": cons_w}),
                        html.Div(style={**bar_light, "width": other_w}),
                    ],
                    style=bar_outer,
                    title=f"Conservative: {cons_f:.2f}% | Other: {other_f:.2f}% (Total: {total_f:.2f}%)",
                ),
                html.Div(
                    f"Conservative (err ≤ 0): {cons_f:.2f}%   |   Other (err > 0): {other_f:.2f}%",
                    style={"fontSize": "12px", "opacity": 0.8, "marginTop": "4px"},
                ),
            ],
            style=row_style,
        )

    # ---------- render ----------
    return html.Div(
        [
            html.Div(
                f"Thresholds: ε1={min_err:g}, ε2={max_err:g} | Critical: true RUL < {crit_lt:g}",
                style={"opacity": 0.75, "marginBottom": "8px", "fontSize": "12px"},
            ),

            # Legend (once)
            html.Div(
                [
                    html.Div([html.Div(style=swatch_dark), html.Span("Conservative (err ≤ 0)")], style=legend_item_style),
                    html.Div([html.Div(style=swatch_light), html.Span("Other (err > 0)")], style=legend_item_style),
                ],
                style=legend_style,
            ),

            # MSE + Cost (text)
            html.Div(
                [
                    html.Div(
                        [html.Span(labels.get("mse", "mse"), style={"fontWeight": 600}), html.Span(f": {fmt(metrics.get('mse'))}")],
                        style={"marginBottom": "6px", "fontSize": "13px"},
                    ),
                    html.Div(
                        [html.Span(labels.get("cost", "cost"), style={"fontWeight": 600}), html.Span(f": {fmt(metrics.get('cost'))}")],
                        style={"marginBottom": "10px", "fontSize": "13px"},
                    ),
                ]
            ),

            # Overall section
            html.Div("Overall", style=section_title_style),
            stacked_kpi_row("error_within_0_min_pct", "error_within_0_min_pct", "error_within_0_min_pct_neg"),
            stacked_kpi_row("error_within_min_max_pct", "error_within_min_max_pct", "error_within_min_max_pct_neg"),
            stacked_kpi_row("error_largerthan_max_pct", "error_largerthan_max_pct", "error_largerthan_max_pct_neg"),

            # Critical section
            html.Div("Critical subset (true < threshold)", style=section_title_style),
            stacked_kpi_row("error_within_0_min_pct_critical", "error_within_0_min_pct_critical", "error_within_0_min_pct_critical_neg"),
            stacked_kpi_row("error_within_min_max_pct_critical", "error_within_min_max_pct_critical", "error_within_min_max_pct_critical_neg"),
            stacked_kpi_row("error_largerthan_max_pct_critical", "error_largerthan_max_pct_critical", "error_largerthan_max_pct_critical_neg"),
        ]
    )

def render_graphs_section(y_true, y_pred, *, s: dict, leadtime: float, thresh: float):
    if y_true is None or y_pred is None:
        return html.Div("No graphs available (no y_true/y_pred yet).", style={"opacity": 0.7})

    import plotly.graph_objects as go

    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    err = yp - yt

    # -------------------------
    # 1) Pred vs True (binned mean, LINE ONLY)
    # -------------------------
    bins = 25  # tweak as needed
    yt_min = float(np.nanmin(yt))
    yt_max = float(np.nanmax(yt))
    edges = np.linspace(yt_min, yt_max, bins + 1)
    bin_ids = np.digitize(yt, edges) - 1  # 0..bins-1

    x_centers = (edges[:-1] + edges[1:]) / 2.0
    mean_pred = np.full(bins, np.nan)

    for b in range(bins):
        m = bin_ids == b
        if np.any(m):
            mean_pred[b] = float(np.mean(yp[m]))

    valid = np.isfinite(mean_pred)
    x_plot = x_centers[valid]
    y_plot = mean_pred[valid]

    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="lines",           # ✅ line only (no markers)
            name="Mean(pred) per true-bin",
        )
    )

    # y=x reference line
    mn = float(np.nanmin(x_plot)) if x_plot.size else yt_min
    mx = float(np.nanmax(x_plot)) if x_plot.size else yt_max
    fig_scatter.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="y=x"))

    fig_scatter.update_layout(
        title="Predicted RUL vs True RUL (binned mean)",
        xaxis_title="True RUL (bin center)",
        yaxis_title="Mean predicted RUL",
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    # -------------------------
    # 2) Error distribution
    # -------------------------
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=err, name="Error (pred-true)"))
    fig_hist.update_layout(
        title="Prediction Error Distribution",
        xaxis_title="Error (pred - true)",
        yaxis_title="Count",
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    # -------------------------
    # 3) Error vs True RUL (binned mean, LINE ONLY — like graph 1)
    # -------------------------
    bins = 25  # keep same as graph 1 for consistency
    yt_min = float(np.nanmin(yt))
    yt_max = float(np.nanmax(yt))
    edges = np.linspace(yt_min, yt_max, bins + 1)
    bin_ids = np.digitize(yt, edges) - 1  # 0..bins-1

    x_centers = (edges[:-1] + edges[1:]) / 2.0
    mean_err = np.full(bins, np.nan)

    for b in range(bins):
        m = bin_ids == b
        if np.any(m):
            mean_err[b] = float(np.mean(err[m]))

    valid = np.isfinite(mean_err)
    x_plot = x_centers[valid]
    y_plot = mean_err[valid]

    fig_err_vs_true = go.Figure()
    fig_err_vs_true.add_trace(
        go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="lines",  # ✅ no markers
            name="Mean(err) per true-bin",
        )
    )
    fig_err_vs_true.add_hline(y=0, line_dash="dash")
    fig_err_vs_true.update_layout(
        title="Signed Error vs True RUL (binned mean)",
        xaxis_title="True RUL (bin center)",
        yaxis_title="Mean error (pred - true)",
        height=340,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    # -------------------------
    # -------------------------
    # 4) Cost curve for ONE record (random pick) — COST ONLY
    #    + colored vertical lines with legend on the right
    # -------------------------
    i = int(np.random.randint(0, yt.size))
    t_i = float(yt[i])
    p_i = float(yp[i])

    # cost params from settings
    C_PR = float(s["cost_predictive"])
    C_RE = float(s["cost_reactive"])
    ALPHA = float(s["early_penalty"])
    BETA = float(s["late_penalty"])

    # x-grid for predicted RUL in the cost curve
    x_max = float(max(200.0, np.nanmax(yt), np.nanmax(yp), thresh * 2))
    x_grid = np.linspace(0.0, x_max, 600)

    # per-record cost curve
    costs = np.array([nonlinear_cost(p, t_i, leadtime, C_PR, ALPHA, C_RE, BETA) for p in x_grid], dtype=float)

    # a "random prediction" baseline point (uniform over range)
    rand_p = float(np.random.uniform(0.0, x_max))
    rand_cost = float(nonlinear_cost(rand_p, t_i, leadtime, C_PR, ALPHA, C_RE, BETA))
    pred_cost = float(nonlinear_cost(p_i, t_i, leadtime, C_PR, ALPHA, C_RE, BETA))

    y_min = float(np.nanmin(costs))
    y_max = float(np.nanmax(costs))

    fig_cost = go.Figure()

    # cost curve
    fig_cost.add_trace(go.Scatter(x=x_grid, y=costs, mode="lines", name="Cost"))

    # markers for selected pred + random pred
    fig_cost.add_trace(go.Scatter(x=[p_i], y=[pred_cost], mode="markers", name=f"Pred cost = {pred_cost:.1f}"))
    fig_cost.add_trace(go.Scatter(x=[rand_p], y=[rand_cost], mode="markers", name=f"Random cost = {rand_cost:.1f}"))

    # --- vertical lines as traces so they appear in legend ---
    fig_cost.add_trace(
        go.Scatter(
            x=[p_i, p_i],
            y=[y_min, y_max],
            mode="lines",
            name=f"Pred = {p_i:.1f}",
            line=dict(width=2, dash="dash", color="#1f77b4"),
            hoverinfo="skip",
        )
    )
    fig_cost.add_trace(
        go.Scatter(
            x=[t_i, t_i],
            y=[y_min, y_max],
            mode="lines",
            name=f"True = {t_i:.1f}",
            line=dict(width=2, dash="dash", color="#2ca02c"),
            hoverinfo="skip",
        )
    )
    fig_cost.add_trace(
        go.Scatter(
            x=[float(thresh), float(thresh)],
            y=[y_min, y_max],
            mode="lines",
            name=f"Thresh = {thresh:g}",
            line=dict(width=2, dash="dash", color="#d62728"),
            hoverinfo="skip",
        )
    )

    fig_cost.update_layout(
        title=f"Cost curve (Tr={t_i:.1f}, Pr={p_i:.1f}, LT={leadtime:g})",
        xaxis_title="Predicted RUL",
        yaxis_title="Cost",
        height=340,
        margin=dict(l=40, r=160, t=50, b=40),  # extra right margin for legend
        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
    )

    # -------------------------
    # Responsive 2x2 grid
    # - wide: 2 columns
    # - tiny: 1 column
    # -------------------------
    cell_style = {
        "flex": "1 1 480px",      # grows; wraps if narrow
        "minWidth": "320px",      # prevents too tiny
    }
    grid_style = {
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "12px",
    }

    return html.Div(
        [
            html.Div(dcc.Graph(figure=fig_scatter, config={"displayModeBar": False}), style=cell_style),
            html.Div(dcc.Graph(figure=fig_err_vs_true, config={"displayModeBar": False}), style=cell_style),
            html.Div(dcc.Graph(figure=fig_hist, config={"displayModeBar": False}), style=cell_style),
            html.Div(dcc.Graph(figure=fig_cost, config={"displayModeBar": False}), style=cell_style),
        ],
        style=grid_style,
    )


def build_main_page_layout():
    return html.Div(
        id="main-area",
        children=[
            html.Div(
                id="main-placeholder",
                children=html.Div(
                    [
                        html.H3("Cost sensitive model"),
                        html.Div("Select a model from Settings to run: Select → Load → Metrics → Plots.", style={"opacity": 0.8}),
                    ],
                    style={"border": "1px dashed #bbb", "borderRadius": "10px", "padding": "16px"},
                ),
            ),

            html.Div(
                id="card-loading-training",
                style=HIDDEN_STYLE,
                children=[
                    html.H3("Model Overview"),
                    dcc.Loading(type="circle", children=html.Div(id="section-loading-training")),
                ],
            ),

            html.Div(
                id="card-metrics",
                style=HIDDEN_STYLE,
                children=[
                    html.H3("Performance KPIs"),
                    dcc.Loading(type="circle", children=html.Div(id="section-metrics")),
                ],
            ),

            html.Div(
                id="card-graphs",
                style=HIDDEN_STYLE,
                children=[
                    html.H3("Diagnostics & Plots"),
                    dcc.Loading(type="circle", children=html.Div(id="section-graphs")),
                ],
            ),
        ],
        style={"minHeight": "400px", "marginTop": "20px"},
    )


# -----------------------------
# +/- helpers
# -----------------------------
def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


# -------------------------------------------------
# Layout
# -------------------------------------------------
def layout():
    desired = settings_slug(settings)

    return html.Div(
        [
            dcc.Store(id="settings-store", data=settings),
            dcc.Store(id="selected-model-store", data={"model_name": None, "source": "none", "match_percent": None}),

            html.Div(
                [
                    html.I(className="fa fa-cogs", id="gear-icon", style={"fontSize": "30px", "cursor": "pointer", "color": "black"}),
                    html.Div(
                        id="settings-panel",
                        children=[
                            html.Div(
                                [
                                    html.H3("Dataset Settings", style={"textAlign": "center"}),

                                    html.Div("Desired model:", style={"fontWeight": "bold"}),
                                    html.Div(id="desired-model-text", children=desired, style={"fontFamily": "monospace", "fontSize": "12px"}),
                                    html.Div(id="model-status-text", style={"marginTop": "8px", "fontSize": "13px"}),

                                    html.Label("Available / closest models:", style={"marginTop": "10px"}),
                                    dcc.Dropdown(id="model-dropdown", options=[], value=None, clearable=False, style={"fontSize": "12px"}),

                                    html.Div(
                                        [
                                            html.Button("Select model", id="btn-select-model", n_clicks=0),
                                            html.Button("Train new model", id="btn-train-new", n_clicks=0, style={"marginLeft": "8px"}),
                                        ],
                                        style={"marginTop": "10px"},
                                    ),

                                    html.Div("Selected model:", style={"fontWeight": "bold", "marginTop": "10px"}),
                                    html.Div(id="selected-model-text", style={"fontSize": "12px", "fontFamily": "monospace"}),

                                    html.Hr(style={"margin": "14px 0"}),

                                    param_three_zone_pill("Slice length (days)", "slice_window_value", "slice_window_decrease", "slice_window_increase", settings["slice_window"]),
                                    param_three_zone_pill("Early penalty", "early_penalty_value", "early_penalty_decrease", "early_penalty_increase", settings["early_penalty"]),
                                    param_three_zone_pill("Late penalty", "late_penalty_value", "late_penalty_decrease", "late_penalty_increase", settings["late_penalty"]),
                                    param_three_zone_pill("Reactive cost", "cost_reactive_value", "cost_reactive_decrease", "cost_reactive_increase", settings["cost_reactive"]),
                                    param_three_zone_pill("Predictive cost", "cost_predictive_value", "cost_predictive_decrease", "cost_predictive_increase", settings["cost_predictive"]),
                                ],
                                style={"width": "100%", "padding": "10px"},
                            )
                        ],
                        style=dropdown_style_hidden,
                    ),
                ],
                style={"position": "relative", "display": "flex", "justifyContent": "flex-end", "padding": "10px"},
            ),

            html.Div(id="main-click-catcher", n_clicks=0, children=[build_main_page_layout()]),
        ]
    )


# -------------------------------------------------
# Callbacks
# -------------------------------------------------
def register_callbacks(app):
    # toggle / auto-close settings panel
    @app.callback(
        Output("settings-panel", "style"),
        Input("gear-icon", "n_clicks"),
        Input("main-click-catcher", "n_clicks"),
        Input("btn-select-model", "n_clicks"),
        Input("btn-train-new", "n_clicks"),
        Input("slice_window_decrease", "n_clicks"),
        Input("slice_window_increase", "n_clicks"),
        Input("early_penalty_decrease", "n_clicks"),
        Input("early_penalty_increase", "n_clicks"),
        Input("late_penalty_decrease", "n_clicks"),
        Input("late_penalty_increase", "n_clicks"),
        Input("cost_reactive_decrease", "n_clicks"),
        Input("cost_reactive_increase", "n_clicks"),
        Input("cost_predictive_decrease", "n_clicks"),
        Input("cost_predictive_increase", "n_clicks"),
        State("settings-panel", "style"),
        prevent_initial_call=True,
    )
    def toggle_or_close_settings_panel(
        gear_clicks,
        main_clicks,
        n_select,
        n_train,
        w_dec, w_inc,
        ep_dec, ep_inc,
        lp_dec, lp_inc,
        cr_dec, cr_inc,
        cp_dec, cp_inc,
        current_style,
    ):
        def _open_style():
            s = dropdown_style_hidden.copy()
            s["display"] = "block"
            return s

        def _is_open(style):
            return bool(style) and style.get("display") == "block"

        trig = ctx.triggered_id
        is_open_now = _is_open(current_style)

        # Gear toggles open/close (same as old)
        if trig == "gear-icon":
            return dropdown_style_hidden if is_open_now else _open_style()

        # If open, ONLY close on outside click or after select/train
        if is_open_now and trig in ("main-click-catcher", "btn-select-model", "btn-train-new"):
            return dropdown_style_hidden

        # Otherwise keep it as-is (so +/- won't close it)
        return no_update


    # update settings-store + visible values
    @app.callback(
        Output("settings-store", "data"),
        Output("slice_window_value", "children"),
        Output("early_penalty_value", "children"),
        Output("late_penalty_value", "children"),
        Output("cost_reactive_value", "children"),
        Output("cost_predictive_value", "children"),
        Input("slice_window_decrease", "n_clicks"),
        Input("slice_window_increase", "n_clicks"),
        Input("early_penalty_decrease", "n_clicks"),
        Input("early_penalty_increase", "n_clicks"),
        Input("late_penalty_decrease", "n_clicks"),
        Input("late_penalty_increase", "n_clicks"),
        Input("cost_reactive_decrease", "n_clicks"),
        Input("cost_reactive_increase", "n_clicks"),
        Input("cost_predictive_decrease", "n_clicks"),
        Input("cost_predictive_increase", "n_clicks"),
        State("settings-store", "data"),
        prevent_initial_call=True,
    )
    def update_settings(
        w_dec, w_inc,
        ep_dec, ep_inc,
        lp_dec, lp_inc,
        cr_dec, cr_inc,
        cp_dec, cp_inc,
        data,
    ):
        if not ctx.triggered_id:
            raise PreventUpdate

        s = dict(data) if data else dict(settings)
        trig = ctx.triggered_id

        if trig == "slice_window_decrease":
            s["slice_window"] = _clamp_int(s["slice_window"] - 1, 1, 10_000)
        elif trig == "slice_window_increase":
            s["slice_window"] = _clamp_int(s["slice_window"] + 1, 1, 10_000)

        elif trig == "early_penalty_decrease":
            s["early_penalty"] = _clamp_int(s["early_penalty"] - 1, 0, 10_000)
        elif trig == "early_penalty_increase":
            s["early_penalty"] = _clamp_int(s["early_penalty"] + 1, 0, 10_000)

        elif trig == "late_penalty_decrease":
            s["late_penalty"] = _clamp_int(s["late_penalty"] - 1, 0, 10_000)
        elif trig == "late_penalty_increase":
            s["late_penalty"] = _clamp_int(s["late_penalty"] + 1, 0, 10_000)

        elif trig == "cost_reactive_decrease":
            s["cost_reactive"] = _clamp_int(s["cost_reactive"] - 10, 0, 1_000_000)
        elif trig == "cost_reactive_increase":
            s["cost_reactive"] = _clamp_int(s["cost_reactive"] + 10, 0, 1_000_000)

        elif trig == "cost_predictive_decrease":
            s["cost_predictive"] = _clamp_int(s["cost_predictive"] - 1, 0, 1_000_000)
        elif trig == "cost_predictive_increase":
            s["cost_predictive"] = _clamp_int(s["cost_predictive"] + 1, 0, 1_000_000)
        else:
            raise PreventUpdate

        return (
            s,
            s["slice_window"],
            s["early_penalty"],
            s["late_penalty"],
            s["cost_reactive"],
            s["cost_predictive"],
        )


    # Update desired model text + dropdown options + status message
    @app.callback(
        Output("desired-model-text", "children"),
        Output("model-status-text", "children"),
        Output("model-dropdown", "options"),
        Output("model-dropdown", "value"),
        Input("settings-store", "data"),
        prevent_initial_call=False,
    )
    def update_model_choices(s):
        if not s:
            raise PreventUpdate

        desired = settings_slug(s)
        assets_path = Path(MODELS_DIR)
        desired_exists = (assets_path / desired).exists()

        options = []
        default_value = None

        if desired_exists:
            options.append({"label": f"✅ {desired} (100.0%)", "value": desired})
            default_value = desired
            status = html.Div("Exact model exists. Click 'Select model' to run it.", style={"color": "#0a7a0a"})
        else:
            top3 = top_k_models(s, MODELS_DIR, k=3)
            if not top3:
                status = html.Div(f"Model not found and no matching models exist in {MODELS_DIR}/.", style={"color": "#b00020"})
            else:
                status = html.Div("Exact model not found. Pick one of the closest models or train a new one.", style={"color": "#b00020"})
                for row in top3:
                    options.append({"label": f"{row['name']} ({row['score']*100:.1f}%)", "value": row["name"]})
                default_value = top3[0]["name"]

        return desired, status, options, default_value


    # =========================================================
    # ONE callback for the whole pipeline:
    # Select/Train -> Load (skipped) -> Metrics -> Graphs
    #
    # - Load step is shown but returns None for now.
    # - Metrics/Graphs use validation.csv 'rul' and demo rul_pred.
    # =========================================================
    @app.callback(
        Output("selected-model-store", "data"),
        Output("selected-model-text", "children"),
        Output("section-loading-training", "children"),
        Output("section-metrics", "children"),
        Output("section-graphs", "children"),
        Output("card-loading-training", "style"),
        Output("card-metrics", "style"),
        Output("card-graphs", "style"),
        Output("main-placeholder", "style"),
        Input("btn-select-model", "n_clicks"),
        Input("btn-train-new", "n_clicks"),
        State("settings-store", "data"),
        State("model-dropdown", "value"),
        prevent_initial_call=True,
    )
    def run_full_pipeline(n_select, n_train, s, dropdown_value):
        if not s:
            raise PreventUpdate

        assets_path = Path(MODELS_DIR)
        desired = settings_slug(s)
        desired_path = assets_path / desired
        trig = ctx.triggered_id

        if trig == "btn-select-model":
            if not dropdown_value:
                raise PreventUpdate

            chosen_name = dropdown_value
            chosen_path = assets_path / chosen_name

            if not chosen_path.exists():
                err = html.Div("Model file not found.", style={"color": "#b00020"})
                return (
                    {"model_name": None, "source": "none", "match_percent": None},
                    "None selected",
                    err,
                    html.Div("No metrics.", style={"opacity": 0.7}),
                    html.Div("No graphs.", style={"opacity": 0.7}),
                    VISIBLE_CARD_STYLE,
                    VISIBLE_CARD_STYLE,
                    {**VISIBLE_CARD_STYLE, "marginBottom": "0px"},
                    {"display": "none"},
                )

            if chosen_name == desired and desired_path.exists():
                data = {"model_name": chosen_name, "source": "exact", "match_percent": 100.0}
            else:
                top3 = top_k_models(s, MODELS_DIR, k=3)
                mp = None
                for row in top3:
                    if row["name"] == chosen_name:
                        mp = row["score"] * 100
                        break
                data = {"model_name": chosen_name, "source": "closest", "match_percent": mp}

            mp_text = "" if data["match_percent"] is None else f", {data['match_percent']:.1f}%"
            selected_text = f"{chosen_name} ({data['source']}{mp_text})"

            model = load_model(str(chosen_path))  # None

            loading_ui = html.Pre(
                f"Step 1: SELECT MODEL ✅\n"
                f"Model file: {chosen_name}\n"
                f"Path: {chosen_path}\n"
                f"Source: {data['source']}{mp_text}\n\n"
                f"Step 2: LOAD MODEL ⏭️ (skipped — loading disabled for demo)\n\n"
                f"Settings:\n{format_settings_for_display(s)}",
                style={"whiteSpace": "pre-wrap"},
            )

        elif trig == "btn-train-new":
            train_new_model(s, save_to=str(desired_path))

            data = {"model_name": desired, "source": "trained", "match_percent": 100.0}
            selected_text = f"{desired} (trained, 100.0%)"

            model = load_model(str(desired_path))  # None

            loading_ui = html.Pre(
                f"Step 1: TRAIN NEW MODEL ✅ (placeholder)\n"
                f"Would save to: {desired_path}\n\n"
                f"Step 2: LOAD MODEL ⏭️ (skipped — loading disabled for demo)\n\n"
                f"Settings:\n{format_settings_for_display(s)}",
                style={"whiteSpace": "pre-wrap"},
            )
        else:
            raise PreventUpdate

        # Step 3: predictions -> metrics (demo)
        y_true, y_pred = predict_rul_from_validation_csv(model, VALIDATION_CSV)

        leadtime = float(s["slice_window"])  #
        cost_fn = make_cost_fn_from_settings(s, leadtime=leadtime)

        metrics_dict = compute_all(
            y_true=y_true,
            y_pred=y_pred,
            min_err=leadtime,
            max_err=3 * leadtime,
            critical_true_lt=10.0,
            cost_fn=cost_fn,
        )
        metrics_ui = render_metrics_section(metrics_dict)

        # Step 4: graphs
        graphs_ui = render_graphs_section(
            y_true,
            y_pred,
            s=s,
            leadtime=leadtime,
            thresh= 10 * leadtime,  # same as max_err you used in compute_all
        )
        return (
            data,
            selected_text,
            loading_ui,
            metrics_ui,
            graphs_ui,
            VISIBLE_CARD_STYLE,
            VISIBLE_CARD_STYLE,
            {**VISIBLE_CARD_STYLE, "marginBottom": "0px"},
            {"display": "none"},
        )