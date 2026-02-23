# pages/benchmark.py
from dash import dcc, html
from dash import dash_table
from dash import Input, Output
import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd

# =========================
# Constants
# =========================
X_MIN, X_MAX = 0, 30
C_PR = 2000
ALPHA = 100
C_RE = 10000
BETA = 500


# =========================
# Cost (linear)
# =========================
def record_cost_linear(pred: float, true: float, leadtime: int) -> float:
    if pred <= true:
        return C_PR + ALPHA * (true - pred)
    diff = pred - true
    eff = float(np.minimum(diff, leadtime))
    return C_RE + BETA * eff


# =========================
# Load data from CSV
# =========================
def make_examples():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, "assets", "test.csv")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    df = pd.read_csv(file_path, sep=";", decimal=",")
    df.columns = df.columns.str.strip()

    # best-effort numeric conversions
    for col in ["id", "car_type", "failure_type", "rul", "leadtime", "route_ratio", "speed", "load"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df) + 1)

    if "leadtime" not in df.columns:
        df["leadtime"] = 0

    df["leadtime"] = df["leadtime"].fillna(0).astype(int)

    return df.to_dict("records")


def process_input_rows(data_raw):
    out = []
    for r in data_raw:
        true = r.get("rul", None)
        if true is None or (isinstance(true, float) and np.isnan(true)):
            continue

        true = float(true)
        lt = int(r.get("leadtime", 0) or 0)

        car = "van" if int(r.get("car_type", 0) or 0) == 0 else "truck"
        failure = "tire" if int(r.get("failure_type", 0) or 0) == 0 else "brake"

        out.append(
            dict(
                id=int(r.get("id", 0) or 0),
                rul_true=true,
                car=car,
                failure=failure,
                leadtime=lt,
                route=r.get("route", 0),
                route_ratio=r.get("route_ratio", 0),
                speed=r.get("speed", 0),
                load=r.get("load", 0),
            )
        )
    return out


# =========================
# Build output rows (two models)
# =========================
def generate_random_predictions(data_raw, rul_column="rul", seed: int = 7):
    rng = np.random.default_rng(seed)
    actual = np.array([float(r.get(rul_column, np.nan)) for r in data_raw], dtype=float)
    actual = np.where(np.isnan(actual), 0.0, actual)

    m1 = actual + rng.normal(0, 5, size=len(actual))
    m2 = actual + rng.normal(0, 5, size=len(actual))
    m1 = np.clip(m1, X_MIN, X_MAX)
    m2 = np.clip(m2, X_MIN, X_MAX)
    return m1, m2


def build_output_rows_page1(data_rows, preds):
    out = []
    for r, pred in zip(data_rows, preds):
        true = float(r["rul_true"])
        pred = float(pred)
        out.append(
            dict(
                id=r["id"],
                rul_true=true,
                pred_classic=pred,
                mse_classic=float((pred - true) ** 2),
                cost_classic=float(record_cost_linear(pred, true, int(r["leadtime"]))),
            )
        )
    return out


def build_output_rows_page2(data_rows, preds):
    out = []
    for r, pred in zip(data_rows, preds):
        true = float(r["rul_true"])
        pred = float(pred)
        out.append(
            dict(
                id=r["id"],
                rul_true=true,
                pred_cost_sensitive=pred,
                mse_cost_sensitive=float((pred - true) ** 2),
                cost_cost_sensitive=float(record_cost_linear(pred, true, int(r["leadtime"]))),
            )
        )
    return out


def combine_output_rows(out1, out2):
    combined = []
    for r1, r2 in zip(out1, out2):
        combined.append({**r1, **r2})
    return combined


# =========================
# Figures
# =========================
def _summary_fig(rows):
    mse1 = float(np.mean([r["mse_classic"] for r in rows]))
    mse2 = float(np.mean([r["mse_cost_sensitive"] for r in rows]))
    c1 = float(np.mean([r["cost_classic"] for r in rows]))
    c2 = float(np.mean([r["cost_cost_sensitive"] for r in rows]))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Classic", "Cost-sensitive"], y=[mse1, mse2], name="Avg MSE"))
    fig.add_trace(go.Bar(x=["Classic", "Cost-sensitive"], y=[c1, c2], name="Avg Cost"))
    fig.update_layout(height=300, barmode="group", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def benchmark_true_vs_pred_scatter_fig(rows, selected_id=None):
    # always show scatter; highlight selected if any
    fig = go.Figure()
    if not rows:
        fig.update_layout(
            height=340,
            title="True RUL vs Predicted RUL",
            xaxis_title="True RUL",
            yaxis_title="Predicted RUL",
            margin=dict(l=20, r=20, t=60, b=40),
        )
        return fig

    y_true = np.array([r["rul_true"] for r in rows], dtype=float)
    y_m1 = np.array([r["pred_classic"] for r in rows], dtype=float)
    y_m2 = np.array([r["pred_cost_sensitive"] for r in rows], dtype=float)

    lo = float(min(y_true.min(), y_m1.min(), y_m2.min()))
    hi = float(max(y_true.max(), y_m1.max(), y_m2.max()))

    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x"))

    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_m1,
            mode="markers",
            name="Classic",
            marker=dict(symbol="circle", size=10, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_m2,
            mode="markers",
            name="Cost-sensitive",
            marker=dict(symbol="triangle-up", size=12, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
        )
    )

    if selected_id is not None:
        r = next((rr for rr in rows if rr.get("id") == selected_id), None)
        if r is not None:
            fig.add_trace(
                go.Scatter(
                    x=[r["rul_true"]],
                    y=[r["pred_classic"]],
                    mode="markers",
                    marker=dict(symbol="circle-open", size=20, line=dict(color="black", width=3)),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[r["rul_true"]],
                    y=[r["pred_cost_sensitive"]],
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


# =========================
# Pre-load data once (module-level)
# =========================
data_raw = make_examples()
DATA_ROWS = process_input_rows(data_raw)

m1_preds, m2_preds = generate_random_predictions(data_raw, rul_column="rul", seed=7)
OUT1 = build_output_rows_page1(DATA_ROWS, m1_preds[: len(DATA_ROWS)])
OUT2 = build_output_rows_page2(DATA_ROWS, m2_preds[: len(DATA_ROWS)])
OUT_ROWS = combine_output_rows(OUT1, OUT2)


# =========================
# Layout
# =========================
def layout():
    if not DATA_ROWS:
        return html.Div(
            style={"border": "1px solid #eee", "borderRadius": "14px", "padding": "12px"},
            children=[
                html.H3("Benchmark"),
                html.Div("No data loaded (assets/test.csv missing or empty).", style={"opacity": 0.8}),
            ],
        )

    data_columns = [{"name": col, "id": col} for col in DATA_ROWS[0].keys()]

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
                                        columns=data_columns,
                                        data=DATA_ROWS,
                                        page_size=8,
                                        sort_action="native",
                                        filter_action="native",
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
                                            {"name": "true", "id": "rul_true", "type": "numeric", "format": {"specifier": ".1f"}},
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
                                        filter_action="native",
                                        row_selectable="single",
                                        style_table={"overflowX": "auto"},
                                        style_cell={"fontFamily": "sans-serif", "fontSize": "13px", "padding": "6px"},
                                        style_header={"fontWeight": "800"},
                                    ),
                                ],
                            ),
                            # ✅ Summary + Scatter always visible side-by-side (old behavior)
                            html.Div(
                                style={"display": "flex", "gap": "10px", "marginTop": "12px"},
                                children=[
                                    html.Div(
                                        style={"flex": 1, "border": "1px solid #eee", "borderRadius": "14px", "padding": "12px"},
                                        children=[
                                            html.Div("Quick benchmark summary", style={"fontWeight": 800, "marginBottom": "8px"}),
                                            dcc.Graph(id="benchmark_summary_fig", config={"displayModeBar": False}),
                                        ],
                                    ),
                                    html.Div(
                                        style={"flex": 1, "border": "1px solid #eee", "borderRadius": "14px", "padding": "12px"},
                                        children=[
                                            html.Div("Scatter (select a row to highlight)", style={"fontWeight": 800, "marginBottom": "8px"}),
                                            dcc.Graph(id="benchmark_scatter_fig", config={"displayModeBar": False}),
                                            html.Div(id="benchmark_details", style={"marginTop": "8px", "fontSize": "14px"}),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ]
    )


# =========================
# Callbacks
# =========================
def register_callbacks(app):
    @app.callback(
        Output("benchmark_summary_fig", "figure"),
        Input("benchmark_output_table", "data"),
    )
    def _update_summary(out_rows):
        out_rows = out_rows or []
        if not out_rows:
            return go.Figure()
        return _summary_fig(out_rows)

    @app.callback(
        Output("benchmark_scatter_fig", "figure"),
        Output("benchmark_details", "children"),
        Input("benchmark_output_table", "data"),
        Input("benchmark_output_table", "derived_virtual_data"),
        Input("benchmark_output_table", "selected_rows"),
    )
    def _update_scatter(rows, derived_rows, selected_rows):
        rows = rows or []
        view = derived_rows if derived_rows is not None else rows

        selected_id = None
        if selected_rows and view and 0 <= selected_rows[0] < len(view):
            selected_id = view[selected_rows[0]].get("id")

        fig = benchmark_true_vs_pred_scatter_fig(rows, selected_id=selected_id)

        if selected_id is None:
            return fig, "Select a row in the table to highlight it."

        r = next((rr for rr in rows if rr.get("id") == selected_id), None)
        if not r:
            return fig, "Selected row not found."

        better_mse = "Classic" if r["mse_classic"] < r["mse_cost_sensitive"] else "Cost-sensitive"
        better_cost = "Classic" if r["cost_classic"] < r["cost_cost_sensitive"] else "Cost-sensitive"

        details = html.Div(
            [
                html.Div(
                    f"id={r['id']} | true={r['rul_true']:.2f} | classic={r['pred_classic']:.2f} | cost-sensitive={r['pred_cost_sensitive']:.2f}",
                    style={"fontWeight": 800},
                ),
                html.Div(f"MSE: classic={r['mse_classic']:.2f}, cost={r['mse_cost_sensitive']:.2f} → better: {better_mse}"),
                html.Div(f"Cost: classic=€{r['cost_classic']:.0f}, cost=€{r['cost_cost_sensitive']:.0f} → better: {better_cost}"),
            ]
        )
        return fig, details