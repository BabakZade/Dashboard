# pages/cost_function.py
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np

X_MIN, X_MAX = 0, 30

COL_GREEN = "#3fc918"
COL_BLUE = "#2c7be5"
COL_RED = "#d62728"

DEFAULTS = {
    "RUL_TRUE": 14,
    "LEADTIME": 7,
    "C_PR": 2000,
    "ALPHA": 100,
    "C_RE": 10000,
    "BETA": 500,
}

import plotly.io as pio

pio.templates.default = "plotly_white"

def clamp(x, lo=X_MIN, hi=X_MAX):
    return max(lo, min(hi, x))


def regime_color(rul_pred: float, rul_true: float, leadtime: float) -> str:
    if rul_pred <= rul_true:
        return COL_GREEN
    diff = rul_pred - rul_true
    return COL_RED if diff > leadtime else COL_BLUE


def split_x_by_regime(xs: np.ndarray, rul_true: float, leadtime: float):
    g = xs <= rul_true
    b = (xs > rul_true) & (xs <= rul_true + leadtime)
    r = xs > rul_true + leadtime
    return g, b, r


def linear_cost(rul_pred: float, rul_true: float, leadtime: float, C_PR: float, ALPHA: float, C_RE: float, BETA: float):
    if rul_pred <= rul_true:
        return C_PR + ALPHA * (rul_true - rul_pred)
    eff = float(np.minimum(rul_pred - rul_true, leadtime))
    return C_RE + BETA * eff


def nonlinear_cost(rul_pred: float, rul_true: float, leadtime: float, C_PR: float, ALPHA: float, C_RE: float, BETA: float):
    if rul_pred <= rul_true:
        return C_PR + ALPHA * (rul_true - rul_pred) ** 2
    eff = float(np.minimum(rul_pred - rul_true, leadtime))
    return C_RE + BETA * (eff**2)


def timeline_figure(rul_pred: float, rul_true: float, leadtime: float, C_PR: float, ALPHA: float, C_RE: float, BETA: float):
    col = regime_color(rul_pred, rul_true, leadtime)
    fig = go.Figure()

    fig.add_vline(
        x=rul_true,
        line_width=3,
        line_dash="dash",
        annotation_text=f"Failure = {rul_true:g}",
        annotation_position="top",
        annotation_yref="paper",
        annotation_y=1,
    )

    fig.add_vline(
        x=rul_pred,
        line_width=4,
        line_color=col,
        annotation_text=f"Cost = {linear_cost(rul_pred, rul_true, leadtime, C_PR, ALPHA, C_RE, BETA):.1f}",
        annotation_position="top",
        annotation_yref="paper",
        annotation_y=0.2,
        annotation_x=rul_pred + 0.3,
        annotation_textangle=90,
        annotation_font=dict(color=col),
    )

    fig.update_layout(
        autosize=True,
        height=230,
        margin=dict(l=20, r=20, t=60, b=50),
        xaxis=dict(range=[X_MIN, X_MAX], title=None),
        yaxis=dict(visible=False),
        showlegend=False,
        clickmode="event+select",
    )
    return fig


def cost_text_and_value(rul_pred: float, rul_true: float, leadtime: float, C_PR: float, ALPHA: float, C_RE: float, BETA: float):
    col = regime_color(rul_pred, rul_true, leadtime)

    if rul_pred <= rul_true:
        cost = linear_cost(rul_pred, rul_true, leadtime, C_PR, ALPHA, C_RE, BETA)
        note = html.Span(
            f"cost = Cₚ + α·(True − Predicted) = {C_PR:g} + {ALPHA:g}·({rul_true:g} − {rul_pred:.1f})",
            style={"color": col, "fontWeight": "700"},
        )
        return note, cost

    downtime = rul_pred - rul_true
    eff = float(np.minimum(downtime, leadtime))
    cost = C_RE + BETA * eff
    shown_eff = leadtime if downtime > leadtime else downtime

    note = html.Span(
        f"cost = Cᵣ + β·min(Downtime, Leadtime) = {C_RE:g} + {BETA:g}·({shown_eff:.1f})",
        style={"color": col, "fontWeight": "700"},
    )
    return note, cost


def part2_cost_breakdown_fig(rul_pred: float, kind: str, rul_true: float, leadtime: float, C_PR: float, ALPHA: float, C_RE: float, BETA: float):
    xs = np.linspace(X_MIN, X_MAX, 240)
    g, b, r = split_x_by_regime(xs, rul_true, leadtime)

    if kind == "linear":
        ys = np.where(
            xs <= rul_true,
            C_PR + ALPHA * (rul_true - xs),
            C_RE + BETA * np.minimum(xs - rul_true, leadtime),
        )
        title = "Linear cost"
        y_sel = linear_cost(rul_pred, rul_true, leadtime, C_PR, ALPHA, C_RE, BETA)
    else:
        ys = np.where(
            xs <= rul_true,
            C_PR + ALPHA * (rul_true - xs) ** 2,
            C_RE + BETA * (np.minimum(xs - rul_true, leadtime) ** 2),
        )
        title = "Non-linear cost"
        y_sel = nonlinear_cost(rul_pred, rul_true, leadtime, C_PR, ALPHA, C_RE, BETA)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs[g], y=ys[g], mode="lines", line=dict(color=COL_GREEN)))
    fig.add_trace(go.Scatter(x=xs[b], y=ys[b], mode="lines", line=dict(color=COL_BLUE)))
    fig.add_trace(go.Scatter(x=xs[r], y=ys[r], mode="lines", line=dict(color=COL_RED)))

    fig.add_vline(x=rul_true, line_dash="dash")
    fig.add_vline(x=rul_true + leadtime, line_dash="dot")

    fig.add_trace(
        go.Scatter(
            x=[rul_pred],
            y=[y_sel],
            mode="markers",
            marker=dict(size=10, color=regime_color(rul_pred, rul_true, leadtime)),
            showlegend=False,
        )
    )

    fig.update_layout(
        autosize=True,
        height=300,
        margin=dict(l=20, r=20, t=60, b=40),
        title=title,
        xaxis_title="Maintenance time",
        yaxis_title="cost",
        showlegend=False,
    )
    return fig


def part2_both_costs_fig(rul_pred: float, rul_true: float, leadtime: float, C_PR: float, ALPHA: float, C_RE: float, BETA: float):
    xs = np.linspace(X_MIN, X_MAX, 240)
    g, b, r = split_x_by_regime(xs, rul_true, leadtime)

    y_lin = np.where(
        xs <= rul_true,
        C_PR + ALPHA * (rul_true - xs),
        C_RE + BETA * np.minimum(xs - rul_true, leadtime),
    )

    y_nonlin = np.where(
        xs <= rul_true,
        C_PR + ALPHA * (rul_true - xs) ** 2,
        C_RE + BETA * (np.minimum(xs - rul_true, leadtime) ** 2),
    )

    fig = go.Figure()
    for mask, col in [(g, COL_GREEN), (b, COL_BLUE), (r, COL_RED)]:
        fig.add_trace(go.Scatter(x=xs[mask], y=y_lin[mask], mode="lines", line=dict(color=col)))
        fig.add_trace(go.Scatter(x=xs[mask], y=y_nonlin[mask], mode="lines", line=dict(color=col, dash="dash")))

    fig.add_vline(x=rul_true, line_dash="dash")
    fig.add_vline(x=rul_true + leadtime, line_dash="dot")

    col_sel = regime_color(rul_pred, rul_true, leadtime)
    fig.add_trace(go.Scatter(x=[rul_pred], y=[linear_cost(rul_pred, rul_true, leadtime, C_PR, ALPHA, C_RE, BETA)], mode="markers", marker=dict(size=10, color=col_sel), showlegend=False))
    fig.add_trace(go.Scatter(x=[rul_pred], y=[nonlinear_cost(rul_pred, rul_true, leadtime, C_PR, ALPHA, C_RE, BETA)], mode="markers", marker=dict(size=10, color=col_sel), showlegend=False))

    fig.update_layout(
        autosize=True,
        height=420,
        margin=dict(l=20, r=20, t=60, b=40),
        title="Linear vs Non-linear cost",
        xaxis_title="Maintenance time",
        yaxis_title="cost",
        showlegend=False,
    )
    return fig


def slider_block_inline():
    return html.Div(
        style={
            "border": "1px solid #ddd",
            "borderRadius": "14px",
            "padding": "12px",
            "background": "white",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
        },
        children=[
            html.Div("Maintenance time", style={"fontWeight": 900, "marginBottom": "6px"}),
            dcc.Slider(
                id="rul_pred_cost",
                className="slider-green",
                min=X_MIN,
                max=X_MAX,
                step=0.5,
                value=DEFAULTS["RUL_TRUE"],
                tooltip={"placement": "bottom", "always_visible": False},
                marks=None,
            ),
        ],
    )


def _param_slider(label, slider_id, min_v, max_v, step, value):
    return html.Div(
        style={
            "padding": "10px 0",
            "borderBottom": "1px solid #f1f1f1",
        },
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "6px",
                },
                children=[
                    html.Div(label, style={"fontSize": "13px", "fontWeight": 900}),
                    html.Div(id=f"{slider_id}_value", style={"fontSize": "13px", "fontWeight": 950}),
                ],
            ),
            dcc.Slider(
                id=slider_id,
                className= "slider-green",
                min=min_v,
                max=max_v,
                step=step,
                value=value,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
    )


def _summary_row(left, right):
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr auto",
            "gap": "10px",
            "padding": "8px 0",
            "borderTop": "1px solid #f3f3f3",
        },
        children=[
            html.Div(left, style={"fontSize": "12px", "fontWeight": 800, "color": "#333"}),
            html.Div(right, style={"fontSize": "12px", "fontWeight": 950, "whiteSpace": "nowrap"}),
        ],
    )


def assumptions_side_panel():
    return html.Div(
        style={
            "background": "white",
            "border": "1px solid #eee",
            "borderRadius": "14px",
            "padding": "14px",
            "boxShadow": "0 10px 26px rgba(0,0,0,0.06)",
        },
        children=[

            # ===== HEADER =====
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "marginBottom": "8px",
                },
                children=[
                    html.Div("Assumptions", style={"fontSize": "15px", "fontWeight": 950}),
                    html.Div(
                        "Context",
                        style={
                            "fontSize": "12px",
                            "fontWeight": 900,
                            "padding": "4px 10px",
                            "borderRadius": "999px",
                            "background": "#f5f7ff",
                            "border": "1px solid #e6e9ff",
                        },
                    ),
                ],
            ),

            # ===== SHORT EXPLANATION =====
            html.Div(
                "These parameters define the maintenance cost structure. "
                "Adjust them to see how early vs late decisions affect total cost.",
                style={
                    "fontSize": "12px",
                    "color": "#555",
                    "lineHeight": "1.6",
                    "marginBottom": "12px",
                },
            ),

            # ===== SUMMARY SECTION (TOP) =====
            html.Div(
                "Current configuration",
                style={"fontSize": "13px", "fontWeight": 900, "marginBottom": "6px"},
            ),

            html.Div(id="p1_assumptions_rows"),

            html.Hr(style={"margin": "14px 0"}),

            # ===== SLIDERS SECTION =====
            html.Div(
                "Adjust parameters",
                style={"fontSize": "13px", "fontWeight": 900, "marginBottom": "6px"},
            ),

            html.Div(
                "Move sliders to explore different economic scenarios.",
                style={
                    "fontSize": "12px",
                    "color": "#666",
                    "marginBottom": "10px",
                },
            ),

            _param_slider("Preventive cost (Cₚ)", "p1_C_PR", 0, 20000, 100, DEFAULTS["C_PR"]),
            _param_slider("Reactive cost (Cᵣ)", "p1_C_RE", 0, 50000, 500, DEFAULTS["C_RE"]),
            _param_slider("Early penalty (α) €/day", "p1_ALPHA", 0, 1000, 10, DEFAULTS["ALPHA"]),
            _param_slider("Downtime penalty (β) €/day", "p1_BETA", 0, 5000, 50, DEFAULTS["BETA"]),
            _param_slider("Failure time", "p1_RUL_TRUE", X_MIN, X_MAX, 1, DEFAULTS["RUL_TRUE"]),
            _param_slider("Leadtime", "p1_LEADTIME", 0, X_MAX, 1, DEFAULTS["LEADTIME"]),
        ],
    )




def decision_block():
    return html.Div(
        style={"border": "1px solid #ddd", "borderRadius": "14px", "padding": "12px"},
        children=[
            html.H4("Decision & cost", style={"marginTop": 0}),
            html.Div(id="cost_note", style={"marginTop": "10px", "fontSize": "16px"}),
            dcc.Graph(id="timeline", config={"displayModeBar": False, "responsive": True}),
            html.Div(id="cost_value", style={"marginTop": "6px", "fontSize": "20px", "fontWeight": 950}),
        ],
    )


def curves_block():
    return html.Div(
        style={"border": "1px solid #ddd", "borderRadius": "14px", "padding": "12px", "marginTop": "12px"},
        children=[
            html.H4("Cost curves", style={"marginTop": 0}),
            html.Div(
                [
                    html.Div([dcc.Graph(id="cost_lin_parts_fig", config={"displayModeBar": False, "responsive": True})], style={"flex": 1, "minWidth": 0}),
                    html.Div([dcc.Graph(id="cost_nonlin_parts_fig", config={"displayModeBar": False, "responsive": True})], style={"flex": 1, "minWidth": 0}),
                ],
                style={"display": "flex", "gap": "10px", "marginBottom": "10px"},
            ),
            dcc.Graph(id="cost_both_fig", config={"displayModeBar": False, "responsive": True}),
        ],
    )


def layout():
    return html.Div(
        className="p1-grid",
        children=[
            html.Div(
                className="p1-main",
                children=[
                    slider_block_inline(),
                    html.Div(style={"height": "12px"}),
                    decision_block(),
                    curves_block(),
                ],
            ),
            html.Div(
                className="p1-side",
                children=[assumptions_side_panel()],
            ),
        ],
    )


def register_callbacks(app):
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
        Output("p1_assumptions_rows", "children"),
        Output("p1_C_PR_value", "children"),
        Output("p1_C_RE_value", "children"),
        Output("p1_ALPHA_value", "children"),
        Output("p1_BETA_value", "children"),
        Output("p1_RUL_TRUE_value", "children"),
        Output("p1_LEADTIME_value", "children"),
        Input("rul_pred_cost", "value"),
        Input("p1_C_PR", "value"),
        Input("p1_C_RE", "value"),
        Input("p1_ALPHA", "value"),
        Input("p1_BETA", "value"),
        Input("p1_RUL_TRUE", "value"),
        Input("p1_LEADTIME", "value"),
    )
    def update_cost_page(rul_pred, C_PR, C_RE, ALPHA, BETA, RUL_TRUE, LEADTIME):
        rul_pred = float(rul_pred)
        C_PR = float(C_PR)
        C_RE = float(C_RE)
        ALPHA = float(ALPHA)
        BETA = float(BETA)
        RUL_TRUE = float(RUL_TRUE)
        LEADTIME = float(LEADTIME)

        fig1 = timeline_figure(rul_pred, RUL_TRUE, LEADTIME, C_PR, ALPHA, C_RE, BETA)
        note, cost = cost_text_and_value(rul_pred, RUL_TRUE, LEADTIME, C_PR, ALPHA, C_RE, BETA)
        fig_lin_parts = part2_cost_breakdown_fig(rul_pred, "linear", RUL_TRUE, LEADTIME, C_PR, ALPHA, C_RE, BETA)
        fig_nonlin_parts = part2_cost_breakdown_fig(rul_pred, "nonlinear", RUL_TRUE, LEADTIME, C_PR, ALPHA, C_RE, BETA)
        fig_both = part2_both_costs_fig(rul_pred, RUL_TRUE, LEADTIME, C_PR, ALPHA, C_RE, BETA)

        rows = [
            _summary_row("Preventive (Cₚ)", f"€{C_PR:,.0f}"),
            _summary_row("Reactive (Cᵣ)", f"€{C_RE:,.0f}"),
            _summary_row("Early penalty (α)", f"€{ALPHA:,.0f}/day"),
            _summary_row("Downtime pen. (β)", f"€{BETA:,.0f}/day"),
            _summary_row("Failure time", f"{RUL_TRUE:g}"),
            _summary_row("Leadtime", f"{LEADTIME:g}"),
        ]

        return (
            fig1,
            note,
            f"Total cost: €{cost:,.0f}",
            fig_lin_parts,
            fig_nonlin_parts,
            fig_both,
            rows,
            f"€{C_PR:,.0f}",
            f"€{C_RE:,.0f}",
            f"{ALPHA:,.0f} €/day",
            f"{BETA:,.0f} €/day",
            f"{RUL_TRUE:g}",
            f"{LEADTIME:g}",
        )
