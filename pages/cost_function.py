# pages/cost_function.py
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np

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
        height=380,
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
            f"cost = Cₚ + α·(True − Predicted) = {C_PR} + {ALPHA}·({RUL_TRUE} − {rul_pred:.1f})",
            style={"color": col, "fontWeight": "600"},
        )
        return note, cost

    downtime = rul_pred - RUL_TRUE
    eff = float(np.minimum(downtime, LEADTIME))
    cost = C_RE + BETA * eff
    shown_eff = LEADTIME if downtime > LEADTIME else downtime

    note = html.Span(
        f"cost = Cᵣ + β·min(Downtime, Leadtime) = {C_RE} + {BETA}·({shown_eff:.1f})",
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
        height=420,
        margin=dict(l=20, r=20, t=60, b=40),
        title="Linear vs Non-linear cost",
        xaxis_title="rul_pred",
        yaxis_title="cost",
        showlegend=False,
    )
    return fig


def slider_block():
    return html.Div(
        id="p1_slider_block",
        style={
            "position": "sticky",
            "top": "64px",
            "zIndex": 10,
            "backgroundColor": "white",
            "border": "1px solid #ddd",
            "borderRadius": "12px",
            "padding": "12px",
            "marginBottom": "12px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
        },
        children=[
            html.Div("Select predicted RUL:", style={"fontWeight": 700}),
            dcc.Slider(
                id="rul_pred_cost",
                min=X_MIN, max=X_MAX, step=0.5, value=RUL_TRUE,
                tooltip={"placement": "bottom", "always_visible": False},
                marks=None,
            ),
            html.Div(
                style={
                    "marginTop": "10px",
                    "fontSize": "13px",
                    "opacity": 0.9,
                    "lineHeight": "1.6",
                    "backgroundColor": "#f8f9fa",
                    "padding": "10px",
                    "borderRadius": "10px",
                },
                children=[
                    html.Div("Assumptions:", style={"fontWeight": 700, "marginBottom": "4px"}),
                    html.Div(f"• Predictive cost (Cₚ): €{C_PR:,}"),
                    html.Div(f"• Reactive cost (Cᵣ): €{C_RE:,}"),
                    html.Div(f"• α (early penalty): €{ALPHA} per day"),
                    html.Div(f"• β (downtime penalty): €{BETA} per day"),
                    html.Div(f"• Failure: {RUL_TRUE}"),
                    html.Div(f"• Leadtime: {LEADTIME}"),
                ],
            ),
        ],
    )


def layout():
    return html.Div(
        children=[
            slider_block(),
            html.Div(
                style={"border": "1px solid #ddd", "borderRadius": "12px", "padding": "12px"},
                children=[
                    html.H4("Decision & cost", style={"marginTop": 0}),
                    html.Div(id="cost_note", style={"marginTop": "10px", "fontSize": "16px"}),
                    dcc.Graph(id="timeline", figure=timeline_figure(RUL_TRUE), config={"displayModeBar": False}),
                    html.Div(id="cost_value", style={"marginTop": "6px", "fontSize": "20px", "fontWeight": 800}),
                ],
            ),
            html.Div(
                style={"border": "1px solid #ddd", "borderRadius": "12px", "padding": "12px", "marginTop": "12px"},
                children=[
                    html.H4("Cost curves", style={"marginTop": 0}),
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
