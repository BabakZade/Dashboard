# pages/benchmark.py
from dash import dcc, html
from dash import dash_table
from dash import Input, Output
import plotly.graph_objects as go
import numpy as np

X_MIN, X_MAX = 0, 30

C_PR = 2000
ALPHA = 100
C_RE = 10000
BETA = 500


def record_cost_linear(pred: float, true: float, leadtime: int) -> float:
    if pred <= true:
        return C_PR + ALPHA * (true - pred)
    diff = pred - true
    eff = float(np.minimum(diff, leadtime))
    return C_RE + BETA * eff


def make_examples(n: int = 18, seed: int = 7):
    rng = np.random.default_rng(seed)

    product = rng.choice(["tire", "brake"], size=n, p=[0.55, 0.45])
    distance = rng.uniform(20, 450, n)
    avg_speed = rng.uniform(30, 120, n)
    rul_true = rng.uniform(4, 24, n)
    leadtime = rng.integers(3, 11, size=n).astype(int)

    rows = []
    for i in range(n):
        rows.append(
            dict(
                id=i + 1,
                product=str(product[i]),
                f1=float(distance[i]),
                f2=float(avg_speed[i]),
                rul_true=float(rul_true[i]),
                leadtime=int(leadtime[i]),
            )
        )
    return rows


def build_output_rows(data_rows, seed: int = 7):
    rng = np.random.default_rng(seed)

    out = []
    for r in data_rows:
        true = float(r["rul_true"])
        lt = int(r["leadtime"])

        pred_classic = true + float(rng.normal(0, 4.5))

        noise = float(rng.normal(0, 4.5))
        noise = noise * 0.35 if noise > 0 else noise
        pred_cost = true + noise

        pred_classic = float(np.clip(pred_classic, X_MIN, X_MAX))
        pred_cost = float(np.clip(pred_cost, X_MIN, X_MAX))

        mse_classic = float((pred_classic - true) ** 2)
        mse_cost = float((pred_cost - true) ** 2)

        cost_classic = float(record_cost_linear(pred_classic, true, lt))
        cost_cost = float(record_cost_linear(pred_cost, true, lt))

        out.append(
            dict(
                id=int(r["id"]),
                rul_true=true,
                product=str(r["product"]),
                leadtime=lt,
                pred_classic=pred_classic,
                pred_cost_sensitive=pred_cost,
                mse_classic=mse_classic,
                mse_cost_sensitive=mse_cost,
                cost_classic=cost_classic,
                cost_cost_sensitive=cost_cost,
            )
        )
    return out


DATA_ROWS = make_examples(n=18, seed=7)
OUT_ROWS = build_output_rows(DATA_ROWS, seed=7)


def _summary_fig(output_rows):
    mse1 = np.mean([r["mse_classic"] for r in output_rows])
    mse2 = np.mean([r["mse_cost_sensitive"] for r in output_rows])
    c1 = np.mean([r["cost_classic"] for r in output_rows])
    c2 = np.mean([r["cost_cost_sensitive"] for r in output_rows])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Classic", "Cost-sensitive"], y=[mse1, mse2], name="Avg MSE"))
    fig.add_trace(go.Bar(x=["Classic", "Cost-sensitive"], y=[c1, c2], name="Avg Cost"))
    fig.update_layout(height=300, barmode="group", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def layout():
    return html.Div(
        children=[
            dcc.Tabs(
                id="bench_tabs",
                value="bench_data",
                children=[
                    dcc.Tab(
                        label="Data",
                        value="bench_data",
                        children=[
                            html.Div(
                                style={"border": "1px solid #eee", "borderRadius": "14px", "padding": "12px", "marginTop": "12px"},
                                children=[
                                    dash_table.DataTable(
                                        id="benchmark_data_table",
                                        columns=[
                                            {"name": "id", "id": "id"},
                                            {"name": "product", "id": "product"},
                                            {"name": "distance (f1)", "id": "f1", "type": "numeric", "format": {"specifier": ".1f"}},
                                            {"name": "avg speed (f2)", "id": "f2", "type": "numeric", "format": {"specifier": ".1f"}},
                                            {"name": "rul_true", "id": "rul_true", "type": "numeric", "format": {"specifier": ".1f"}},
                                            {"name": "leadtime", "id": "leadtime", "type": "numeric", "format": {"specifier": ".0f"}},
                                        ],
                                        data=DATA_ROWS,
                                        page_size=8,
                                        sort_action="native",
                                        row_selectable="single",
                                        style_table={"overflowX": "auto"},
                                        style_cell={"fontFamily": "sans-serif", "fontSize": "13px", "padding": "6px"},
                                        style_header={"fontWeight": "800"},
                                    ),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Output",
                        value="bench_output",
                        children=[
                            html.Div(
                                style={"border": "1px solid #eee", "borderRadius": "14px", "padding": "12px", "marginTop": "12px"},
                                children=[
                                    dash_table.DataTable(
                                        id="benchmark_output_table",
                                        columns=[
                                            {"name": "id", "id": "id"},
                                            {"name": "leadtime", "id": "leadtime", "type": "numeric", "format": {"specifier": ".0f"}},
                                            {"name": "classic", "id": "pred_classic", "type": "numeric", "format": {"specifier": ".1f"}},
                                            {"name": "cost-sensitive", "id": "pred_cost_sensitive", "type": "numeric", "format": {"specifier": ".1f"}},
                                            {"name": "mse classic", "id": "mse_classic", "type": "numeric", "format": {"specifier": ".1f"}},
                                            {"name": "mse cost", "id": "mse_cost_sensitive", "type": "numeric", "format": {"specifier": ".1f"}},
                                            {"name": "cost classic", "id": "cost_classic", "type": "numeric", "format": {"specifier": ".0f"}},
                                            {"name": "cost cost", "id": "cost_cost_sensitive", "type": "numeric", "format": {"specifier": ".0f"}},
                                        ],
                                        data=OUT_ROWS,
                                        page_size=8,
                                        sort_action="native",
                                        row_selectable="single",
                                        style_table={"overflowX": "auto"},
                                        style_cell={"fontFamily": "sans-serif", "fontSize": "13px", "padding": "6px"},
                                        style_header={"fontWeight": "800"},
                                    ),
                                ],
                            ),

                            html.Div(
                                style={"marginTop": "12px", "border": "1px solid #eee", "borderRadius": "14px", "padding": "12px"},
                                children=[
                                    html.Div("Quick benchmark summary", style={"fontWeight": 800, "marginBottom": "8px"}),
                                    dcc.Graph(id="benchmark_summary_fig", config={"displayModeBar": False}),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ]
    )


def register_callbacks(app):
    @app.callback(
        Output("benchmark_summary_fig", "figure"),
        Input("bench_out_store", "data"),
    )
    def _update_summary(out_rows):
        out_rows = out_rows or []
        if not out_rows:
            return go.Figure()
        return _summary_fig(out_rows)
