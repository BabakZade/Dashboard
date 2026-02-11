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

from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash import dash_table
import plotly.graph_objects as go
import numpy as np


# =======================
# Global parameters
# =======================
RUL_TRUE = 14
LEADTIME = 7

C_PR = 2000
ALPHA = 100
C_RE = 10000
BETA = 500

X_MIN, X_MAX = 0, 30

COL_GREEN = "#3fc918"
COL_BLUE = "#2c7be5"
COL_RED = "#d62728"


# =======================
# App
# =======================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Cost-sensitive predictive maintenance"


# =======================
# Helpers
# =======================
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
    return C_RE + BETA * (eff**2)


def record_cost_linear(pred: float, true: float) -> float:
    if pred <= true:
        return C_PR + ALPHA * (true - pred)
    diff = pred - true
    eff = float(np.minimum(diff, LEADTIME))
    return C_RE + BETA * eff


# =======================
# Figures (Page 1: Cost function)
# =======================
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
    fig.add_trace(go.Scatter(x=xs[g], y=ys[g], mode="lines", line=dict(color=COL_GREEN)))
    fig.add_trace(go.Scatter(x=xs[b], y=ys[b], mode="lines", line=dict(color=COL_BLUE)))
    fig.add_trace(go.Scatter(x=xs[r], y=ys[r], mode="lines", line=dict(color=COL_RED)))

    fig.add_vline(x=RUL_TRUE, line_dash="dash")
    fig.add_vline(x=RUL_TRUE + LEADTIME, line_dash="dot")

    fig.add_trace(
        go.Scatter(
            x=[rul_pred],
            y=[y_sel],
            mode="markers",
            marker=dict(size=10, color=regime_color(rul_pred, RUL_TRUE)),
            showlegend=False,
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=40),
        title=title,
        xaxis_title="rul_pred",
        yaxis_title="cost",
        showlegend=False,
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
    for mask, col in [(g, COL_GREEN), (b, COL_BLUE), (r, COL_RED)]:
        fig.add_trace(go.Scatter(x=xs[mask], y=y_lin[mask], mode="lines", line=dict(color=col)))
        fig.add_trace(go.Scatter(x=xs[mask], y=y_nonlin[mask], mode="lines", line=dict(color=col, dash="dash")))

    fig.add_vline(x=RUL_TRUE, line_dash="dash")
    fig.add_vline(x=RUL_TRUE + LEADTIME, line_dash="dot")

    col_sel = regime_color(rul_pred, RUL_TRUE)
    fig.add_trace(go.Scatter(x=[rul_pred], y=[linear_cost_global(rul_pred)], mode="markers", marker=dict(size=10, color=col_sel), showlegend=False))
    fig.add_trace(go.Scatter(x=[rul_pred], y=[nonlinear_cost_global(rul_pred)], mode="markers", marker=dict(size=10, color=col_sel), showlegend=False))

    fig.update_layout(
        height=480,
        margin=dict(l=20, r=20, t=60, b=40),
        title="Linear vs Non-linear cost (colored by regime)",
        xaxis_title="rul_pred",
        yaxis_title="cost",
        showlegend=False,
    )
    return fig


# =======================
# Benchmark: data + output generation
# =======================
def make_benchmark_data(n: int = 18, seed: int = 7):
    rng = np.random.default_rng(seed)

    product = rng.choice(["tire", "brake"], size=n, p=[0.55, 0.45])
    distance = rng.uniform(20, 450, n)   # f1
    avg_speed = rng.uniform(30, 120, n)  # f2
    rul_true = rng.uniform(4, 24, n)

    rows = []
    for i in range(n):
        rows.append(
            dict(
                id=i + 1,
                product=str(product[i]),
                f1=float(distance[i]),
                f2=float(avg_speed[i]),
                rul_true=float(rul_true[i]),
            )
        )
    return rows


def build_benchmark_output_rows(data_rows, seed: int = 7):
    rng = np.random.default_rng(seed)

    out = []
    for r in data_rows:
        true = float(r["rul_true"])

        pred_model1 = true + float(rng.normal(0, 4.5))  # classic model (MSE-like)
        noise = float(rng.normal(0, 4.5))
        noise = noise * 0.35 if noise > 0 else noise
        pred_model2 = true + noise  # cost-sensitive

        pred_model1 = float(np.clip(pred_model1, X_MIN, X_MAX))
        pred_model2 = float(np.clip(pred_model2, X_MIN, X_MAX))

        mse_model1 = float((pred_model1 - true) ** 2)
        mse_model2 = float((pred_model2 - true) ** 2)

        cost_model1 = float(record_cost_linear(pred_model1, true))
        cost_model2 = float(record_cost_linear(pred_model2, true))

        out.append(
            dict(
                id=r["id"],
                rul_true=true,
                pred_model1=pred_model1,
                pred_model2=pred_model2,
                mse_model1=mse_model1,
                mse_model2=mse_model2,
                cost_model1=cost_model1,
                cost_model2=cost_model2,
            )
        )
    return out


def part4_summary_fig(rows):
    if not rows:
        fig = go.Figure()
        fig.update_layout(
            height=320,
            title="Average metrics across synthetic examples",
            margin=dict(l=20, r=20, t=60, b=40),
        )
        return fig

    mse1 = np.mean([r["mse_model1"] for r in rows])
    mse2 = np.mean([r["mse_model2"] for r in rows])
    c1 = np.mean([r["cost_model1"] for r in rows])
    c2 = np.mean([r["cost_model2"] for r in rows])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Classic", "Cost-sensitive"], y=[mse1, mse2], name="Avg MSE"))
    fig.add_trace(go.Bar(x=["Classic", "Cost-sensitive"], y=[c1, c2], name="Avg Cost"))

    fig.update_layout(
        height=320,
        barmode="group",
        title="Average metrics across examples",
        xaxis_title="Model",
        yaxis_title="Value",
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig


def part4_true_vs_pred_scatter_fig(rows, selected_id=None):
    if not rows:
        fig = go.Figure()
        fig.update_layout(
            height=340,
            title="True RUL vs Predicted RUL",
            xaxis_title="True RUL",
            yaxis_title="Predicted RUL",
            margin=dict(l=20, r=20, t=60, b=40),
        )
        return fig

    y_true = np.array([r["rul_true"] for r in rows])
    y_m1 = np.array([r["pred_model1"] for r in rows])
    y_m2 = np.array([r["pred_model2"] for r in rows])

    lo = float(min(y_true.min(), y_m1.min(), y_m2.min()))
    hi = float(max(y_true.max(), y_m1.max(), y_m2.max()))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Perfect prediction (y = x)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_m1,
            mode="markers",
            name="Classic model",
            marker=dict(symbol="circle", size=10, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_m2,
            mode="markers",
            name="Cost-sensitive model",
            marker=dict(symbol="triangle-up", size=12, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
        )
    )

    if selected_id is not None:
        r = next((rr for rr in rows if rr["id"] == selected_id), None)
        if r is not None:
            fig.add_trace(
                go.Scatter(
                    x=[r["rul_true"]],
                    y=[r["pred_model1"]],
                    mode="markers",
                    marker=dict(symbol="circle-open", size=20, line=dict(color="black", width=3)),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[r["rul_true"]],
                    y=[r["pred_model2"]],
                    mode="markers",
                    marker=dict(symbol="triangle-up-open", size=22, line=dict(color="black", width=3)),
                    showlegend=False,
                )
            )

    fig.update_layout(
        height=340,
        title="True RUL vs Predicted RUL",
        xaxis_title="True RUL",
        yaxis_title="Predicted RUL",
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig


benchmark_data = make_benchmark_data(n=18, seed=7)


# =======================
# UI blocks
# =======================
def slider_block(slider_id: str):
    return html.Div(
        style={
            "position": "sticky",
            "top": "0px",
            "zIndex": 300,  # below overlay/drawer
            "backgroundColor": "white",
            "border": "1px solid #ddd",
            "borderRadius": "10px",
            "padding": "12px",
            "marginBottom": "12px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
        },
        children=[
            html.Div("Select predicted RUL:", style={"fontWeight": 700}),
            dcc.Slider(
                id=slider_id,
                min=X_MIN,
                max=X_MAX,
                step=0.5,
                value=RUL_TRUE,
                tooltip={"placement": "bottom", "always_visible": False},
                marks=None,
            ),
            html.Div(
                style={
                    "marginTop": "12px",
                    "fontSize": "13px",
                    "opacity": 0.85,
                    "lineHeight": "1.6",
                    "backgroundColor": "#f8f9fa",
                    "padding": "10px",
                    "borderRadius": "6px",
                },
                children=[
                    html.Div("Assumptions used in this example:", style={"fontWeight": 600}),
                    html.Div(f"• Predictive cost: €{C_PR:,}"),
                    html.Div(f"• Reactive cost: €{C_RE:,}"),
                    html.Div(f"• α (early penalty): €{ALPHA} per day before failure"),
                    html.Div(f"• β (downtime penalty): €{BETA} per day after failure"),
                    html.Div(f"• Failure time: {RUL_TRUE} days"),
                    html.Div(f"• Leadtime: {LEADTIME} days"),
                ],
            ),
        ],
    )


def drawer_link(text, href):
    return dcc.Link(
        text,
        href=href,
        style={
            "display": "block",
            "padding": "10px 12px",
            "textDecoration": "none",
            "color": "#111",
            "borderRadius": "8px",
        },
    )


def empty_page(title):
    return html.Div(
        style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "16px"},
        children=[html.H3(title, style={"marginTop": 0}), html.Div("Coming soon…", style={"opacity": 0.75})],
    )


# =======================
# Layout
# =======================
app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px"},
    children=[
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="menu_open", data=False),
        dcc.Store(id="benchmark_data_store", data=benchmark_data),

        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "12px"},
            children=[
                html.Button(
                    "☰",
                    id="menu_btn",
                    n_clicks=0,
                    style={
                        "fontSize": "20px",
                        "padding": "8px 12px",
                        "borderRadius": "10px",
                        "border": "1px solid #ddd",
                        "background": "white",
                        "cursor": "pointer",
                    },
                ),
                html.H2("Cost-sensitive Predictive Maintenance", style={"margin": 0}),
            ],
        ),

        html.Div(
            id="overlay",
            n_clicks=0,
            style={
                "display": "none",
                "position": "fixed",
                "top": 0,
                "left": 0,
                "right": 0,
                "bottom": 0,
                "backgroundColor": "rgba(0,0,0,0.25)",
                "zIndex": 2500,
            },
        ),

        html.Div(
            id="drawer",
            style={
                "display": "none",
                "position": "fixed",
                "top": 0,
                "left": 0,
                "height": "100%",
                "width": "260px",
                "backgroundColor": "white",
                "borderRight": "1px solid #ddd",
                "padding": "14px",
                "zIndex": 3000,
                "boxShadow": "2px 0 12px rgba(0,0,0,0.10)",
            },
            children=[
                html.Div("Menu", style={"fontWeight": 800, "marginBottom": "10px"}),
                drawer_link("Page 1 — Cost function", "/cost-function"),
                drawer_link("Page 2 — Data Simulator", "/data-simulator"),
                drawer_link("Page 3 — RUL distribution", "/rul-distribution"),
                drawer_link("Page 4 — Cost sensitive model", "/cost-sensitive-model"),
                drawer_link("Page 5 — Benchmark", "/benchmark"),
                html.Hr(),
                html.Div("Tip: click outside the drawer to close.", style={"fontSize": "12px", "opacity": 0.7}),
            ],
        ),

        html.Div(id="page_content"),
    ],
)


# =======================
# Menu open/close
# =======================
@app.callback(
    Output("menu_open", "data"),
    Input("menu_btn", "n_clicks"),
    Input("overlay", "n_clicks"),
    Input("url", "pathname"),
    State("menu_open", "data"),
)
def toggle_menu(n_btn, n_overlay, pathname, is_open):
    if not callback_context.triggered:
        return is_open

    trig = callback_context.triggered[0]["prop_id"].split(".")[0]
    if trig == "menu_btn":
        return not is_open
    return False


@app.callback(
    Output("drawer", "style"),
    Output("overlay", "style"),
    Input("menu_open", "data"),
)
def set_drawer_styles(is_open):
    drawer_style = {
        "display": "block" if is_open else "none",
        "position": "fixed",
        "top": 0,
        "left": 0,
        "height": "100%",
        "width": "260px",
        "backgroundColor": "white",
        "borderRight": "1px solid #ddd",
        "padding": "14px",
        "zIndex": 3000,
        "boxShadow": "2px 0 12px rgba(0,0,0,0.10)",
    }
    overlay_style = {
        "display": "block" if is_open else "none",
        "position": "fixed",
        "top": 0,
        "left": 0,
        "right": 0,
        "bottom": 0,
        "backgroundColor": "rgba(0,0,0,0.25)",
        "zIndex": 2500,
    }
    return drawer_style, overlay_style


# =======================
# Page renderer
# =======================
@app.callback(
    Output("page_content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    if not pathname or pathname == "/":
        pathname = "/cost-function"

    if pathname == "/cost-function":
        return html.Div(
            children=[
                slider_block("rul_pred_cost"),
                html.Div(
                    style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginTop": "12px"},
                    children=[
                        html.H4("Too early vs. too late prediction"),
                        html.Div(id="cost_note", style={"marginTop": "10px", "fontSize": "16px"}),
                        dcc.Graph(id="timeline", figure=timeline_figure(RUL_TRUE), config={"displayModeBar": False}),
                        html.Div(id="cost_value", style={"marginTop": "6px", "fontSize": "20px", "fontWeight": 700}),
                    ],
                ),
                html.Div(
                    style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginTop": "12px"},
                    children=[
                        html.H4("Linear vs. NonLinear cost function"),
                        html.Div(
                            [
                                html.Div([dcc.Graph(id="cost_lin_parts_fig", config={"displayModeBar": False})], style={"flex": 1}),
                                html.Div([dcc.Graph(id="cost_nonlin_parts_fig", config={"displayModeBar": False})], style={"flex": 1}),
                            ],
                            style={"display": "flex", "gap": "10px", "marginBottom": "10px"},
                        ),
                        dcc.Graph(id="cost_both_fig", config={"displayModeBar": False}),
                    ],
                ),
            ]
        )

    if pathname == "/data-simulator":
        return empty_page("Page 2 — Data Simulator")

    if pathname == "/rul-distribution":
        return empty_page("Page 3 — RUL distribution")

    if pathname == "/cost-sensitive-model":
        return empty_page("Page 4 — Cost sensitive model")

    if pathname == "/benchmark":
        return html.Div(
            children=[
                dcc.Tabs(
                    id="p5_tabs",
                    value="p5_data",
                    children=[
                        dcc.Tab(
                            label="Data",
                            value="p5_data",
                            children=[
                                dash_table.DataTable(
                                    id="benchmark_data_table",
                                    columns=[
                                        {"name": "id", "id": "id"},
                                        {"name": "product", "id": "product"},
                                        {"name": "f1 (distance traveled)", "id": "f1", "type": "numeric", "format": {"specifier": ".1f"}},
                                        {"name": "f2 (avg speed)", "id": "f2", "type": "numeric", "format": {"specifier": ".1f"}},
                                        {"name": "rul_true", "id": "rul_true", "type": "numeric", "format": {"specifier": ".2f"}},
                                    ],
                                    data=benchmark_data,
                                    page_size=8,
                                    sort_action="native",
                                    row_selectable="single",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontFamily": "sans-serif", "fontSize": "13px", "padding": "6px"},
                                    style_header={"fontWeight": "700"},
                                ),
                            ],
                        ),
                        dcc.Tab(
                            label="Output",
                            value="p5_output",
                            children=[
                                dash_table.DataTable(
                                    id="benchmark_output_table",
                                    columns=[
                                        {"name": "id", "id": "id"},
                                        {"name": "classic model", "id": "pred_model1", "type": "numeric", "format": {"specifier": ".2f"}},
                                        {"name": "cost sensitive model", "id": "pred_model2", "type": "numeric", "format": {"specifier": ".2f"}},
                                        {"name": "mse (classic)", "id": "mse_model1", "type": "numeric", "format": {"specifier": ".2f"}},
                                        {"name": "mse (cost)", "id": "mse_model2", "type": "numeric", "format": {"specifier": ".2f"}},
                                        {"name": "cost (classic)", "id": "cost_model1", "type": "numeric", "format": {"specifier": ".0f"}},
                                        {"name": "cost (cost)", "id": "cost_model2", "type": "numeric", "format": {"specifier": ".0f"}},
                                    ],
                                    data=[],
                                    page_size=8,
                                    sort_action="native",
                                    row_selectable="single",
                                    style_table={"overflowX": "auto"},
                                    style_cell={"fontFamily": "sans-serif", "fontSize": "13px", "padding": "6px"},
                                    style_header={"fontWeight": "700"},
                                ),
                                html.Div(
                                    [
                                        html.Div([dcc.Graph(id="benchmark_summary_fig", config={"displayModeBar": False})], style={"flex": 1}),
                                        html.Div([dcc.Graph(id="benchmark_scatter_fig", config={"displayModeBar": False})], style={"flex": 1}),
                                    ],
                                    style={"display": "flex", "gap": "10px", "marginTop": "10px"},
                                ),
                                html.Div(id="benchmark_details", style={"marginTop": "8px", "fontSize": "14px"}),
                            ],
                        ),
                    ],
                )
            ]
        )

    return empty_page("404 — Page not found")


# =======================
# Page 1 interactions
# =======================
@app.callback(
    Output("rul_pred_cost", "value"),
    Input("timeline", "clickData"),
    State("rul_pred_cost", "value"),
    prevent_initial_call=True,
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
    Input("rul_pred_cost", "value"),
)
def update_cost_page(rul_pred):
    rul_pred = float(rul_pred)

    fig1 = timeline_figure(rul_pred)
    note, cost = cost_text_and_value(rul_pred)

    fig_lin_parts = part2_cost_breakdown_fig(rul_pred, "linear")
    fig_nonlin_parts = part2_cost_breakdown_fig(rul_pred, "nonlinear")
    fig_both = part2_both_costs_fig(rul_pred)

    return (
        fig1,
        note,
        f"Total cost: €{cost:,.0f}",
        fig_lin_parts,
        fig_nonlin_parts,
        fig_both,
    )


# =======================
# Benchmark: fill output table
# =======================
@app.callback(
    Output("benchmark_output_table", "data"),
    Input("benchmark_data_store", "data"),
)
def fill_benchmark_output_table(data_rows):
    if not data_rows:
        return []
    return build_benchmark_output_rows(data_rows, seed=7)


# =======================
# Benchmark: graphs + details (your exact logic, adapted)
# =======================
@app.callback(
    Output("benchmark_summary_fig", "figure"),
    Output("benchmark_scatter_fig", "figure"),
    Output("benchmark_details", "children"),
    Input("benchmark_output_table", "data"),
    Input("benchmark_output_table", "selected_rows"),
)
def update_benchmark_output(rows, selected_rows):
    selected_id = None
    if selected_rows:
        selected_id = rows[selected_rows[0]]["id"]

    fig_summary = part4_summary_fig(rows)
    fig_scatter = part4_true_vs_pred_scatter_fig(rows, selected_id=selected_id)

    if selected_id is None:
        return fig_summary, fig_scatter, "Select a row to see the per-example comparison."

    r = next(rr for rr in rows if rr["id"] == selected_id)
    better_mse = "Classic" if r["mse_model1"] < r["mse_model2"] else "Cost-sensitive"
    better_cost = "Classic" if r["cost_model1"] < r["cost_model2"] else "Cost-sensitive"

    details = html.Div(
        [
            html.Div(
                f"Example id={r['id']} | true={r['rul_true']:.2f} | "
                f"classic={r['pred_model1']:.2f} | cost-sensitive={r['pred_model2']:.2f}",
                style={"fontWeight": "700"},
            ),
            html.Div(f"MSE: classic={r['mse_model1']:.2f}, cost={r['mse_model2']:.2f} → better: {better_mse}"),
            html.Div(f"Cost: classic=€{r['cost_model1']:.0f}, cost=€{r['cost_model2']:.0f} → better: {better_cost}"),
        ]
    )

    return fig_summary, fig_scatter, details


if __name__ == "__main__":
    print("Starting Dash on http://127.0.0.1:8050/")
    app.run(debug=True, host="127.0.0.1", port=8050)
