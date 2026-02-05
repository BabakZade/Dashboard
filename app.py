# app.py
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

RUL_TRUE = 14

C_PR = 2000
ALPHA = 100
C_RE = 10000
BETA = 500

X_MIN, X_MAX = 0, 30

app = Dash(__name__)
app.title = "RR1 Demo — Merged Slider + Timeline"

def clamp(x, lo=X_MIN, hi=X_MAX):
    return max(lo, min(hi, x))

def timeline_figure(rul_pred: float):
    fig = go.Figure()

    # timeline line (thicker helps visual merge)
    fig.add_trace(
        go.Scatter(
            x=[X_MIN, X_MAX],
            y=[0, 0],
            mode="lines",
            line={"width": 5},
            hoverinfo="skip",
        )
    )

    # failure marker
    fig.add_vline(
        x=RUL_TRUE,
        line_width=3,
        line_dash="dash",
        annotation_text=f"Failure = {RUL_TRUE}",
        annotation_position="top",
        annotation_yref="paper",
        annotation_y=1,
    )

    # prediction marker
    fig.add_vline(
        x=rul_pred,
        line_width=4,
        annotation_text=f"Prediction = {rul_pred:.1f}",
        annotation_position="top",
        annotation_yref="paper",
        annotation_y=0.2,
        annotation_x=rul_pred + 0.3,
        annotation_textangle=90,
    )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=50),
        xaxis=dict(range=[X_MIN, X_MAX], title=None),
        yaxis=dict(visible=False),
        showlegend=False,
        title="Part 1 — Timeline + merged slider",
        clickmode="event+select",
    )

    return fig

def cost_text_and_value(rul_pred: float):
    if rul_pred <= RUL_TRUE:
        RUL = RUL_TRUE - rul_pred
        cost = C_PR + ALPHA * RUL

        note = html.Span(
            f"cost = predictive maintenance + α·(Remaining Useful life) = "
            f"{C_PR} + {ALPHA}·({RUL_TRUE} - {rul_pred:.1f})",
            style={"color": "#2c7be5"}  # calm blue
        )
    else:
        downtime = rul_pred - RUL_TRUE
        cost = C_RE + BETA * downtime

        note = html.Span(
            f"cost = reactive maintenance + β·(Downtime) = "
            f"{C_RE} + {BETA}·({rul_pred:.1f} - {RUL_TRUE})",
            style={"color": "#d62728", "fontWeight": "600"}  # RED
        )

    return note, cost

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "16px"},
    children=[
        html.H2("RR1 Demo — Simple Page (4 Parts)"),

        html.Div(
            style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginBottom": "12px"},
            children=[
                html.H4("1) Timeline with merged slider"),

                html.Div(
                    style={"position": "relative"},
                    children=[
                        dcc.Graph(
                            id="timeline",
                            figure=timeline_figure(RUL_TRUE),
                            style={"marginBottom": "0px"},
                            config={"displayModeBar": False},
                        ),

                        # Slider overlay: spans full graph width + matches plot margins via padding
                        html.Div(
                            style={
                                "position": "absolute",
                                "left": "0px",
                                "right": "0px",
                                "bottom": "120px",
                                "padding": "0 20px",  # matches fig margin l/r
                                "background": "rgba(255,255,255,0.0)",
                            },
                            children=[
                                dcc.Slider(
                                    id="rul_pred",
                                    min=X_MIN,
                                    max=X_MAX,
                                    step=0.5,
                                    value=RUL_TRUE,
                                    tooltip={"placement": "bottom", "always_visible": False},
                                    marks=None,
                                )
                            ],
                        ),
                    ],
                ),

                html.Div(id="cost_note", style={"marginTop": "10px", "fontSize": "16px"}),
                html.Div(id="cost_value", style={"marginTop": "6px", "fontSize": "20px", "fontWeight": 700}),
                html.Div(
                    f"Predictive cost ={C_PR}, α={ALPHA}, Reactive cost ={C_RE}, β={BETA}, Failure point={RUL_TRUE}",
                    style={"marginTop": "8px", "opacity": 0.7},
                ),
            ],
        ),

        html.Div(
            style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginBottom": "12px"},
            children=[html.H4("2) Lead time impact (empty for now)"), html.Div("—")],
        ),

        html.Div(
            style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginBottom": "12px"},
            children=[html.H4("3) MSE vs Cost function (empty for now)"), html.Div("—")],
        ),

        html.Div(
            style={"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px"},
            children=[html.H4("4) Extra / Scenario comparison (empty for now)"), html.Div("—")],
        ),
    ],
)

# Click on the timeline to move the slider (optional, feels “merged”)
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

# Update visuals + cost when slider changes
@app.callback(
    Output("timeline", "figure"),
    Output("cost_note", "children"),
    Output("cost_value", "children"),
    Input("rul_pred", "value"),
)
def update_part1(rul_pred):
    fig = timeline_figure(float(rul_pred))
    note, cost = cost_text_and_value(float(rul_pred))
    return fig, note, f"Total cost: €{cost:,.0f}"

if __name__ == "__main__":
    print("Starting Dash on http://127.0.0.1:8050/")
    app.run(debug=True, host="127.0.0.1", port=8050)
