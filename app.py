# app.py

import importlib.util
import sys

REQUIRED = ["dash", "plotly", "numpy"]
missing = [p for p in REQUIRED if importlib.util.find_spec(p) is None]
if missing:
    print("❌ Missing required packages:", ", ".join(missing))
    print("Install with:")
    print(f"  {sys.executable} -m pip install " + " ".join(missing))
    raise SystemExit(1)

from dash import Dash, dcc, html, Input, Output, State
from dash import dash_table
import plotly.graph_objects as go
import numpy as np


RUL_TRUE = 14
LEADTIME = 7

C_PR = 2000
ALPHA = 100
C_RE = 10000
BETA = 500

X_MIN, X_MAX = 0, 30

COL_GREEN = "#3fc918"   # pred <= true
COL_BLUE  = "#2c7be5"   # pred > true and diff <= leadtime
COL_RED   = "#d62728"   # diff > leadtime

app = Dash(__name__)
app.title = "Cost-sensitive predictive maintenace"


def clamp(x, lo=X_MIN, hi=X_MAX):
    return max(lo, min(hi, x))


def regime_color(pred: float, true: float) -> str:
    if pred <= true:
        return COL_GREEN
    diff = pred - true
    return COL_RED if diff > LEADTIME else COL_BLUE


def split_x_by_regime(xs: np.ndarray):
    g = xs <= RUL_TRUE
    b = (xs > RUL_TRUE) & (xs <= RUL_TRUE + LEADTIME)
    r = xs > RUL_TRUE + LEADTIME
    return g, b, r


def linear_cost_global(rul_pred: float):
    if rul_pred <= RUL_TRUE:
        return C_PR + ALPHA * (RUL_TRUE - rul_pred)
    eff = float(np.minimum(rul_pred - RUL_TRUE, LEADTIME))
    return C_RE + BETA * eff


def nonlinear_cost_global(rul_pred: float):
    if rul_pred <= RUL_TRUE:
        return C_PR + ALPHA * (RUL_TRUE - rul_pred) ** 2
    eff = float(np.minimum(rul_pred - RUL_TRUE, LEADTIME))
    return C_RE + BETA * (eff ** 2)


def mse_loss_global(rul_pred: float):
    return (rul_pred - RUL_TRUE) ** 2


def timeline_figure(rul_pred: float):
    col = regime_color(rul_pred, RUL_TRUE)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[X_MIN, X_MAX],
            y=[0, 0],
            mode="lines",
            line={"width": 5},
            hoverinfo="skip",
        )
    )

    fig.add_vline(
        x=RUL_TRUE,
        line_width=3,
        line_dash="dash",
        annotation_text=f"Failure = {RUL_TRUE}",
        annotation_position="top",
        annotation_yref="paper",
        annotation_y=1,
    )

    fig.add_vline(
        x=rul_pred,
        line_width=4,
        line_color=col,
        annotation_text=f"Prediction = {rul_pred:.1f}",
        annotation_position="top",
        annotation_yref="paper",
        annotation_y=0.2,
        annotation_x=rul_pred + 0.3,
        annotation_textangle=90,
        annotation_font=dict(color=col),
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=50),
        xaxis=dict(range=[X_MIN, X_MAX], title=None),
        yaxis=dict(visible=False),
        showlegend=False,
        clickmode="event+select",
    )

    return fig


def cost_text_and_value(rul_pred: float):
    col = regime_color(rul_pred, RUL_TRUE)

    if rul_pred <= RUL_TRUE:
        cost = linear_cost_global(rul_pred)
        note = html.Span(
            f"cost = predictive maintenance + α·(Remaining Useful life) = "
            f"{C_PR} + {ALPHA}·({RUL_TRUE} - {rul_pred:.1f})",
            style={"color": col, "fontWeight": "600"},
        )
        return note, cost

    downtime = rul_pred - RUL_TRUE
    eff = float(np.minimum(downtime, LEADTIME))
    cost = C_RE + BETA * eff
    shown_eff = LEADTIME if downtime > LEADTIME else downtime

    note = html.Span(
        f"cost = reactive maintenance + β·MIN(Downtime, Leadtime) = "
        f"{C_RE} + {BETA}·({shown_eff:.1f})",
        style={"color": col, "fontWeight": "600"},
    )
    return note, cost


def part2_cost_breakdown_fig(rul_pred: float, kind: str):
    xs = np.linspace(X_MIN, X_MAX, 240)
    g, b, r = split_x_by_regime(xs)

    if kind == "linear":
        ys = np.where(
            xs <= RUL_TRUE,
            C_PR + ALPHA * (RUL_TRUE - xs),
            C_RE + BETA * np.minimum(xs - RUL_TRUE, LEADTIME),
        )
        title = "Linear cost"
        y_sel = linear_cost_global(rul_pred)
    else:
        ys = np.where(
            xs <= RUL_TRUE,
            C_PR + ALPHA * (RUL_TRUE - xs) ** 2,
            C_RE + BETA * (np.minimum(xs - RUL_TRUE, LEADTIME) ** 2),
        )
        title = "Non-linear cost"
        y_sel = nonlinear_cost_global(rul_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs[g], y=ys[g], mode="lines", name="Predictive", line=dict(color=COL_GREEN)))
    fig.add_trace(go.Scatter(x=xs[b], y=ys[b], mode="lines", name="Reactive ≤ leadtime", line=dict(color=COL_BLUE)))
    fig.add_trace(go.Scatter(x=xs[r], y=ys[r], mode="lines", name="Reactive > leadtime", line=dict(color=COL_RED)))

    fig.add_vline(x=RUL_TRUE, line_dash="dash")
    fig.add_vline(x=RUL_TRUE + LEADTIME, line_dash="dot")

    fig.add_trace(
        go.Scatter(
            x=[rul_pred],
            y=[y_sel],
            mode="markers",
            name="Selected",
            marker=dict(size=10, color=regime_color(rul_pred, RUL_TRUE)),
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=40),
        title=title,
        xaxis_title="rul_pred",
        yaxis_title="cost",
        showlegend=False,   # 👈 THIS
    )
    return fig


def part2_both_costs_fig(rul_pred: float):
    xs = np.linspace(X_MIN, X_MAX, 240)
    g, b, r = split_x_by_regime(xs)

    y_lin = np.where(
        xs <= RUL_TRUE,
        C_PR + ALPHA * (RUL_TRUE - xs),
        C_RE + BETA * np.minimum(xs - RUL_TRUE, LEADTIME),
    )

    y_nonlin = np.where(
        xs <= RUL_TRUE,
        C_PR + ALPHA * (RUL_TRUE - xs) ** 2,
        C_RE + BETA * (np.minimum(xs - RUL_TRUE, LEADTIME) ** 2),
    )

    fig = go.Figure()

    for mask, col, lbl in [
        (g, COL_GREEN, "Predictive"),
        (b, COL_BLUE, "Reactive ≤ leadtime"),
        (r, COL_RED, "Reactive > leadtime"),
    ]:
        fig.add_trace(go.Scatter(x=xs[mask], y=y_lin[mask], mode="lines", line=dict(color=col), name=f"Linear – {lbl}"))
        fig.add_trace(go.Scatter(x=xs[mask], y=y_nonlin[mask], mode="lines", line=dict(color=col, dash="dash"), name=f"Non-linear – {lbl}"))

    fig.add_vline(x=RUL_TRUE, line_dash="dash")
    fig.add_vline(x=RUL_TRUE + LEADTIME, line_dash="dot")

    col_sel = regime_color(rul_pred, RUL_TRUE)
    fig.add_trace(go.Scatter(x=[rul_pred], y=[linear_cost_global(rul_pred)], mode="markers", name="Selected (linear)", marker=dict(size=10, color=col_sel)))
    fig.add_trace(go.Scatter(x=[rul_pred], y=[nonlinear_cost_global(rul_pred)], mode="markers", name="Selected (non-linear)", marker=dict(size=10, color=col_sel)))

    fig.update_layout(
        height=480, 
        margin=dict(l=20, r=20, t=60, b=40),
        title="Linear vs Non-linear cost (colored by regime)",
        xaxis_title="rul_pred",
        yaxis_title="cost",
        showlegend=False,
    )

    return fig


def part3_mse_fig(rul_pred: float):
    xs = np.linspace(X_MIN, X_MAX, 240)
    ys = (xs - RUL_TRUE) ** 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="MSE loss"))
    fig.add_vline(x=RUL_TRUE, line_dash="dash", annotation_text="True RUL", annotation_position="top")
    fig.add_trace(
        go.Scatter(
            x=[rul_pred],
            y=[mse_loss_global(rul_pred)],
            mode="markers",
            name="Selected",
            marker=dict(size=10, color=regime_color(rul_pred, RUL_TRUE)),
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=80, b=40),
        title="Loss = (Prdicted - True)²<br>" \
        " ",
        xaxis_title="rul_pred",
        yaxis_title="loss",
        showlegend=False,
    )
    return fig


def part3_costloss_fig(rul_pred: float):
    xs = np.linspace(X_MIN, X_MAX, 240)
    g, b, r = split_x_by_regime(xs)

    ys = np.where(
        xs <= RUL_TRUE,
        C_PR + ALPHA * (RUL_TRUE - xs) ** 2,
        C_RE + BETA * (np.minimum(xs - RUL_TRUE, LEADTIME) ** 2),
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs[g], y=ys[g], mode="lines", line=dict(color=COL_GREEN)))
    fig.add_trace(go.Scatter(x=xs[b], y=ys[b], mode="lines", line=dict(color=COL_BLUE)))
    fig.add_trace(go.Scatter(x=xs[r], y=ys[r], mode="lines", line=dict(color=COL_RED)))

    fig.add_vline(x=RUL_TRUE, line_dash="dash", annotation_text="Failure", annotation_position="top")
    fig.add_vline(x=RUL_TRUE + LEADTIME, line_dash="dot", annotation_text="Failure + Leadtime", annotation_position="top")

    fig.add_trace(
        go.Scatter(
            x=[rul_pred],
            y=[nonlinear_cost_global(rul_pred)],
            mode="markers",
            name="Selected",
            marker=dict(size=10, color=regime_color(rul_pred, RUL_TRUE)),
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=80, b=40),
        title=(
            "Loss = if (Predicted < True) "
            "(Cₚ + α·(Predicted − True)²)<br>"
            "else (Cᵣ + β·min(Leadtime², (Predicted − True)²))"
        ),
        xaxis_title="rul_pred",
        yaxis_title="loss",
        showlegend=False,
    )

    return fig


def part3_both_losses_fig(rul_pred: float):
    xs = np.linspace(X_MIN, X_MAX, 240)
    g, b, r = split_x_by_regime(xs)

    y_mse = (xs - RUL_TRUE) ** 2
    y_cost = np.where(
        xs <= RUL_TRUE,
        C_PR + ALPHA * (RUL_TRUE - xs) ** 2,
        C_RE + BETA * (np.minimum(xs - RUL_TRUE, LEADTIME) ** 2),
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=xs, y=y_mse, mode="lines", name="MSE loss"))
    fig.add_trace(go.Scatter(x=xs[g], y=y_cost[g], mode="lines", name="Cost loss — Predictive", line=dict(color=COL_GREEN)))
    fig.add_trace(go.Scatter(x=xs[b], y=y_cost[b], mode="lines", name="Cost loss — Reactive ≤ leadtime", line=dict(color=COL_BLUE)))
    fig.add_trace(go.Scatter(x=xs[r], y=y_cost[r], mode="lines", name="Cost loss — Reactive > leadtime", line=dict(color=COL_RED)))

    fig.add_vline(x=RUL_TRUE, line_dash="dash", annotation_text="Failure", annotation_position="top")
    fig.add_vline(x=RUL_TRUE + LEADTIME, line_dash="dot", annotation_text="Failure + Leadtime", annotation_position="top")

    col_sel = regime_color(rul_pred, RUL_TRUE)
    fig.add_trace(go.Scatter(x=[rul_pred], y=[mse_loss_global(rul_pred)], mode="markers", name="Selected (MSE)", marker=dict(size=10, color=col_sel)))
    fig.add_trace(go.Scatter(x=[rul_pred], y=[nonlinear_cost_global(rul_pred)], mode="markers", name="Selected (Cost)", marker=dict(size=10, color=col_sel)))

    fig.update_layout(
        height=480,
        margin=dict(l=20, r=20, t=60, b=40),
        xaxis_title="rul_pred",
        yaxis_title="loss",
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)"
        ),
    )
    return fig


# -----------------------------
# Part 4 (synthetic examples)
# -----------------------------
def record_cost_linear(pred: float, true: float) -> float:
    if pred <= true:
        return C_PR + ALPHA * (true - pred)
    diff = pred - true
    eff = float(np.minimum(diff, LEADTIME))
    return C_RE + BETA * eff


def record_cost_nonlinear(pred: float, true: float) -> float:
    if pred <= true:
        return C_PR + ALPHA * (true - pred) ** 2
    diff = pred - true
    eff = float(np.minimum(diff, LEADTIME))
    return C_RE + BETA * (eff ** 2)


def make_examples(n: int = 18, seed: int = 7):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    true = rng.uniform(4, 24, n)

    # Model 1 (MSE-trained): symmetric noise
    pred_mse = true + rng.normal(0, 4.5, n)

    # Model 2 (cost-trained): penalize late predictions more -> shrink positive errors
    noise2 = rng.normal(0, 4.5, n)
    noise2 = np.where(noise2 > 0, noise2 * 0.35, noise2 * 1.0)
    pred_cost = true + noise2

    pred_mse = np.clip(pred_mse, X_MIN, X_MAX)
    pred_cost = np.clip(pred_cost, X_MIN, X_MAX)

    rows = []
    for i in range(n):
        mse1 = float((pred_mse[i] - true[i]) ** 2)
        mse2 = float((pred_cost[i] - true[i]) ** 2)

        cost1 = float(record_cost_linear(pred_mse[i], true[i]))
        cost2 = float(record_cost_linear(pred_cost[i], true[i]))

        rows.append(
            dict(
                id=i + 1,
                f1=float(f1[i]),
                f2=float(f2[i]),
                rul_true=float(true[i]),
                pred_model1=float(pred_mse[i]),
                pred_model2=float(pred_cost[i]),
                mse_model1=mse1,
                mse_model2=mse2,
                cost_model1=cost1,
                cost_model2=cost2,
            )
        )
    return rows


def part4_summary_fig(rows):
    mse1 = np.mean([r["mse_model1"] for r in rows])
    mse2 = np.mean([r["mse_model2"] for r in rows])
    c1 = np.mean([r["cost_model1"] for r in rows])
    c2 = np.mean([r["cost_model2"] for r in rows])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Model 1 (MSE)", "Model 2 (Cost)"], y=[mse1, mse2], name="Avg MSE"))
    fig.add_trace(go.Bar(x=["Model 1 (MSE)", "Model 2 (Cost)"], y=[c1, c2], name="Avg Cost"))

    fig.update_layout(
        height=320,
        barmode="group",
        title="Part 4 — Average metrics across examples",
        xaxis_title="Model",
        yaxis_title="Value",
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig


def part4_scatter_fig(rows, selected_id=None):
    x = [r["pred_model1"] for r in rows]
    y = [r["pred_model2"] for r in rows]
    colors = [regime_color(r["pred_model2"], r["rul_true"]) for r in rows]  # color by model2 regime vs true
    text = [f"id={r['id']} | true={r['rul_true']:.1f}" for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            text=text,
            marker=dict(size=10, color=colors),
            name="Examples",
        )
    )
    fig.add_trace(go.Scatter(x=[X_MIN, X_MAX], y=[X_MIN, X_MAX], mode="lines", name="y=x"))

    if selected_id is not None:
        r = next((rr for rr in rows if rr["id"] == selected_id), None)
        if r is not None:
            fig.add_trace(
                go.Scatter(
                    x=[r["pred_model1"]],
                    y=[r["pred_model2"]],
                    mode="markers",
                    marker=dict(size=16, symbol="circle-open", line=dict(width=3)),
                    name=f"Selected id={selected_id}",
                )
            )

    fig.update_layout(
        height=320,
        title="Part 4 — Predictions: Model 1 vs Model 2 (colored by regime)",
        xaxis_title="Model 1 prediction",
        yaxis_title="Model 2 prediction",
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig


def part4_true_vs_pred_scatter_fig(rows, selected_id=None):
    y_true = np.array([r["rul_true"] for r in rows])
    y_m1 = np.array([r["pred_model1"] for r in rows])
    y_m2 = np.array([r["pred_model2"] for r in rows])

    lo = min(y_true.min(), y_m1.min(), y_m2.min())
    hi = max(y_true.max(), y_m1.max(), y_m2.max())

    fig = go.Figure()

    # Perfect prediction line
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Perfect prediction (y = x)",
        )
    )

    # Model 1 — MSE (outline only)
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_m1,
            mode="markers",
            name="Model 1 (MSE)",
            marker=dict(
                symbol="circle",
                size=10,
                color="rgba(0,0,0,0)",
                line=dict(color="black", width=2),
            ),
        )
    )

    # Model 2 — Cost-sensitive (outline only)
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_m2,
            mode="markers",
            name="Model 2 (Cost)",
            marker=dict(
                symbol="triangle-up",
                size=12,
                color="rgba(0,0,0,0)",
                line=dict(color="black", width=2),
            ),
        )
    )

    # 🔥 Highlight BOTH points if a row is selected
    if selected_id is not None:
        r = next((rr for rr in rows if rr["id"] == selected_id), None)
        if r is not None:
            # Model 1 highlight
            fig.add_trace(
                go.Scatter(
                    x=[r["rul_true"]],
                    y=[r["pred_model1"]],
                    mode="markers",
                    marker=dict(
                        symbol="circle-open",
                        size=20,
                        line=dict(color="black", width=3),
                    ),
                    showlegend=False,
                )
            )

            # Model 2 highlight
            fig.add_trace(
                go.Scatter(
                    x=[r["rul_true"]],
                    y=[r["pred_model2"]],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up-open",
                        size=22,
                        line=dict(color="black", width=3),
                    ),
                    showlegend=False,
                )
            )

    fig.update_layout(
        height=340,
        title="Part 4 — True RUL vs Predicted RUL",
        xaxis_title="True RUL",
        yaxis_title="Predicted RUL",
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)"
        ),
        margin=dict(l=20, r=20, t=60, b=40),
    )

    return fig

examples_data = make_examples(n=18, seed=7)


app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "16px"},
    children=[
        dcc.Store(id="examples_store", data=examples_data),

        html.H2("Cost-sensitive Predictive maintenance"),

        # ✅ Global controls (always visible)
        html.Div(
            style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginBottom": "12px"},
            children=[
                html.Div("Select predicted RUL:", style={"fontWeight": 700}),
                dcc.Slider(
                    id="rul_pred",
                    min=X_MIN, max=X_MAX, step=0.5, value=RUL_TRUE,
                    tooltip={"placement": "bottom", "always_visible": False},
                    marks=None,
                )
            ],
        ),

        dcc.Tabs(
            id="main_tabs",
            value="tab1",
            children=[
                dcc.Tab(
                    label="Diffirent costs",
                    value="tab1",
                    children=[
                        html.Div(
                            style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginTop": "12px"},
                            children=[
                                html.H4("Too early vs. too late prediction"),
                                html.Div(id="cost_note", style={"marginTop": "10px", "fontSize": "16px"}),
                                html.Div(
                                    style={"position": "relative"},
                                    children=[
                                        dcc.Graph(
                                            id="timeline",
                                            figure=timeline_figure(RUL_TRUE),
                                            style={"marginBottom": "0px"},
                                            config={"displayModeBar": False},
                                        ),
                                        
                                    ],
                                ),
                                html.Div(id="cost_value", style={"marginTop": "6px", "fontSize": "20px", "fontWeight": 700}),
                                html.Div(
                                    f"Predictive cost={C_PR}, α={ALPHA}, Reactive cost={C_RE}, β={BETA}, Failure={RUL_TRUE}, Leadtime={LEADTIME}",
                                    style={"marginTop": "8px", "opacity": 0.7},
                                ),
                            ],
                        )
                    ],
                ),

                dcc.Tab(
                    label="Cost function",
                    value="tab2",
                    children=[
                        html.Div(
                            style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginTop": "12px"},
                            children=[
                                html.H4("Linear vs. NonLinear cost function"),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(style={"fontWeight": 1200}),
                                                dcc.Graph(id="cost_lin_parts_fig", config={"displayModeBar": False}),
                                            ],
                                            style={"flex": 1},
                                        ),
                                        html.Div(
                                            [
                                                dcc.Graph(id="cost_nonlin_parts_fig", config={"displayModeBar": False}),
                                            ],
                                            style={"flex": 1},
                                        ),
                                    ],
                                    style={"display": "flex", "gap": "10px", "marginBottom": "10px"},
                                ),
                                dcc.Graph(id="cost_both_fig", config={"displayModeBar": False}),
                            ],
                        )
                    ],
                ),

                dcc.Tab(
                    label="Loss function",
                    value="tab3",
                    children=[
                        html.Div(
                            style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginTop": "12px"},
                            children=[

                                # Row: two side-by-side graphs
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div("MSE:", style={"fontWeight": 600}),
                                                dcc.Graph(id="mse_fig", config={"displayModeBar": False}),
                                            ],
                                            style={"flex": 1},
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Cost:", style={"fontWeight": 600}),
                                                dcc.Graph(id="costloss_fig", config={"displayModeBar": False}),
                                            ],
                                            style={"flex": 1},
                                        ),
                                    ],
                                    style={"display": "flex", "gap": "10px"},
                                ),

                                # Below: combined graph
                                html.Div("MSE vs. Cost", style={"fontWeight": 600, "marginTop": "10px"}),
                                dcc.Graph(id="bothloss_fig", config={"displayModeBar": False}),
                            ],
                        )
                    ],
                ),

                dcc.Tab(
                    label="Synthetic examples",
                    value="tab4",
                    children=[
                        html.Div(
                            style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginTop": "12px"},
                            children=[
                                html.H4("4) Synthetic examples — Model 1 (MSE) vs Model 2 (Cost)"),
                                html.Div(
                                    "Each example has 2 features, a true RUL, and two predictions (one per model). "
                                    "We show per-example MSE and expected maintenance cost (linear cost metric).",
                                    style={"opacity": 0.85, "marginBottom": "8px"},
                                ),
                                dash_table.DataTable(
                                    id="examples_table",
                                    columns=[
                                        {"name": "id", "id": "id"},
                                        {"name": "f1", "id": "f1", "type": "numeric", "format": {"specifier": ".2f"}},
                                        {"name": "f2", "id": "f2", "type": "numeric", "format": {"specifier": ".2f"}},
                                        {"name": "true_rul", "id": "rul_true", "type": "numeric", "format": {"specifier": ".2f"}},
                                        {"name": "pred_model1 (MSE)", "id": "pred_model1", "type": "numeric", "format": {"specifier": ".2f"}},
                                        {"name": "pred_model2 (Cost)", "id": "pred_model2", "type": "numeric", "format": {"specifier": ".2f"}},
                                        {"name": "mse_model1", "id": "mse_model1", "type": "numeric", "format": {"specifier": ".2f"}},
                                        {"name": "mse_model2", "id": "mse_model2", "type": "numeric", "format": {"specifier": ".2f"}},
                                        {"name": "cost_model1", "id": "cost_model1", "type": "numeric", "format": {"specifier": ".0f"}},
                                        {"name": "cost_model2", "id": "cost_model2", "type": "numeric", "format": {"specifier": ".0f"}},
                                    ],
                                    data=examples_data,
                                    page_size=8,
                                    sort_action="native",
                                    row_selectable="single",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontFamily": "sans-serif", "fontSize": "13px", "padding": "6px"},
                                    style_header={"fontWeight": "700"},
                                ),
                                html.Div(
                                    [
                                        html.Div([dcc.Graph(id="examples_summary_fig", config={"displayModeBar": False})], style={"flex": 1}),
                                        html.Div([dcc.Graph(id="examples_scatter_fig", config={"displayModeBar": False})], style={"flex": 1}),
                                    ],
                                    style={"display": "flex", "gap": "10px", "marginTop": "10px"},
                                ),
                                html.Div(id="example_details", style={"marginTop": "8px", "fontSize": "14px"}),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ],
)




@app.callback(
    Output("rul_pred", "value"),
    Input("timeline", "clickData"),
    State("rul_pred", "value"),
)
def move_slider_by_click(clickData, current):
    if not clickData:
        return current
    x = clickData["points"][0]["x"]
    return float(clamp(x))


@app.callback(
    Output("timeline", "figure"),
    Output("cost_note", "children"),
    Output("cost_value", "children"),
    Output("cost_lin_parts_fig", "figure"),
    Output("cost_nonlin_parts_fig", "figure"),
    Output("cost_both_fig", "figure"),
    Output("mse_fig", "figure"),
    Output("costloss_fig", "figure"),
    Output("bothloss_fig", "figure"),
    Input("rul_pred", "value"),
)
def update_parts_1_to_3(rul_pred):
    rul_pred = float(rul_pred)

    fig1 = timeline_figure(rul_pred)
    note, cost = cost_text_and_value(rul_pred)

    fig_lin_parts = part2_cost_breakdown_fig(rul_pred, "linear")
    fig_nonlin_parts = part2_cost_breakdown_fig(rul_pred, "nonlinear")
    fig_both = part2_both_costs_fig(rul_pred)

    fig_mse = part3_mse_fig(rul_pred)
    fig_costloss = part3_costloss_fig(rul_pred)
    fig_bothloss = part3_both_losses_fig(rul_pred)

    return (
        fig1,
        note,
        f"Total cost: €{cost:,.0f}",
        fig_lin_parts,
        fig_nonlin_parts,
        fig_both,
        fig_mse,
        fig_costloss,
        fig_bothloss,
    )



@app.callback(
    Output("examples_summary_fig", "figure"),
    Output("examples_scatter_fig", "figure"),
    Output("example_details", "children"),
    Input("examples_store", "data"),
    Input("examples_table", "selected_rows"),
)
def update_part4(rows, selected_rows):
    selected_id = None
    if selected_rows:
        selected_id = rows[selected_rows[0]]["id"]

    fig_summary = part4_summary_fig(rows)
    fig_scatter = part4_true_vs_pred_scatter_fig(rows, selected_id=selected_id)

    if selected_id is None:
        details = "Select a row to see the per-example comparison."
        return fig_summary, fig_scatter, details

    r = next(rr for rr in rows if rr["id"] == selected_id)
    better_mse = "Model 1" if r["mse_model1"] < r["mse_model2"] else "Model 2"
    better_cost = "Model 1" if r["cost_model1"] < r["cost_model2"] else "Model 2"

    details = html.Div(
        [
            html.Div(
                f"Example id={r['id']} | true={r['rul_true']:.2f} | "
                f"pred1={r['pred_model1']:.2f} | pred2={r['pred_model2']:.2f}",
                style={"fontWeight": "700"},
            ),
            html.Div(f"MSE: model1={r['mse_model1']:.2f}, model2={r['mse_model2']:.2f} → better: {better_mse}"),
            html.Div(f"Cost: model1=€{r['cost_model1']:.0f}, model2=€{r['cost_model2']:.0f} → better: {better_cost}"),
        ]
    )

    return fig_summary, fig_scatter, details



if __name__ == "__main__":

    print("Starting Dash on http://127.0.0.1:8050/")
    app.run(debug=True, host="127.0.0.1", port=8050)
