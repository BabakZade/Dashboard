import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

SCENARIOS = {
    "A - Failure is catastrophic": dict(reactive=10000, predictive=1000, downtime=500, early=200, late=2000, leadtime=3),
    "B - Preventive is expensive": dict(reactive=5000, predictive=2500, downtime=300, early=1000, late=1000, leadtime=3),
    "C - Downtime dominates": dict(reactive=6000, predictive=1200, downtime=2000, early=400, late=1200, leadtime=5),
}

def cost_eval(y_true, y_pred, T, c):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    do_maint = y_pred <= T
    late = (y_true <= 0) & (~do_maint)
    early = do_maint & (y_true > T)
    ontime = do_maint & (y_true <= T) & (y_true > 0)

    preventive_cost = do_maint.sum() * c["predictive"]
    reactive_cost = late.sum() * c["reactive"]

    downtime_days = np.maximum(0.0, c["leadtime"] - y_pred)
    downtime_cost = downtime_days.sum() * c["downtime"]

    early_pen = early.sum() * c["early"]
    late_pen = late.sum() * c["late"]

    total = preventive_cost + reactive_cost + downtime_cost + early_pen + late_pen

    breakdown = {
        "preventive": preventive_cost,
        "reactive": reactive_cost,
        "downtime": downtime_cost,
        "early_penalty": early_pen,
        "late_penalty": late_pen,
        "total": total,
        "counts": dict(early=int(early.sum()), late=int(late.sum()), ontime=int(ontime.sum()))
    }
    return breakdown

def make_df(y_true, y_pred_base, y_pred_cost):
    return pd.DataFrame({
        "true_rul": y_true,
        "pred_baseline": y_pred_base,
        "pred_cost": y_pred_cost,
    })

rng = np.random.default_rng(7)
y_true = rng.normal(loc=15, scale=8, size=600).clip(min=-5, max=60)
y_pred_baseline = (y_true + rng.normal(0, 6, size=len(y_true))).clip(min=-5, max=60)
y_pred_cost = (y_true + rng.normal(-1.5, 5, size=len(y_true))).clip(min=-5, max=60)

df = make_df(y_true, y_pred_baseline, y_pred_cost)

app = Dash(__name__)

app.layout = html.Div([
    html.H2("RR1: Cost-Sensitive Metrics Demo"),

    html.Div([
        html.Div([
            html.Div("Scenario"),
            dcc.Dropdown(list(SCENARIOS.keys()), list(SCENARIOS.keys())[0], id="scenario")
        ], style={"flex": 2, "padding": "8px"}),

        html.Div([
            html.Div("Decision Threshold T"),
            dcc.Slider(0, 40, 1, value=10, id="T", marks=None, tooltip={"placement": "bottom", "always_visible": True})
        ], style={"flex": 3, "padding": "8px"}),
    ], style={"display": "flex"}),

    html.Div([
        dcc.Graph(id="total_cost"),
        dcc.Graph(id="timing_plot"),
    ], style={"display": "flex"}),

    html.Div([
        dcc.Graph(id="breakdown"),
        html.Div(id="counts_box", style={"padding": "12px", "fontSize": "16px"})
    ], style={"display": "flex"})
])

@app.callback(
    Output("total_cost", "figure"),
    Output("timing_plot", "figure"),
    Output("breakdown", "figure"),
    Output("counts_box", "children"),
    Input("scenario", "value"),
    Input("T", "value"),
)
def update(scenario_name, T):
    c = SCENARIOS[scenario_name]

    b0 = cost_eval(df["true_rul"], df["pred_baseline"], T, c)
    b1 = cost_eval(df["true_rul"], df["pred_cost"], T, c)

    cost_df = pd.DataFrame([
        {"model": "Baseline", "total_cost": b0["total"]},
        {"model": "Cost-sensitive", "total_cost": b1["total"]},
    ])
    fig_total = px.bar(cost_df, x="model", y="total_cost", title="Total Cost (€)")

    long = df.melt(id_vars=["true_rul"], value_vars=["pred_baseline", "pred_cost"], var_name="model", value_name="pred_rul")
    long["model"] = long["model"].replace({"pred_baseline": "Baseline", "pred_cost": "Cost-sensitive"})
    fig_timing = px.scatter(long, x="true_rul", y="pred_rul", color="model", title="True RUL vs Predicted RUL")
    fig_timing.add_hline(y=T)

    breakdown_df = pd.DataFrame([
        {"model": "Baseline", **{k: b0[k] for k in ["preventive", "reactive", "downtime", "early_penalty", "late_penalty"]}},
        {"model": "Cost-sensitive", **{k: b1[k] for k in ["preventive", "reactive", "downtime", "early_penalty", "late_penalty"]}},
    ]).melt(id_vars="model", var_name="component", value_name="cost")
    fig_breakdown = px.bar(breakdown_df, x="model", y="cost", color="component", title="Cost Breakdown (€)")

    counts_text = html.Div([
        html.Div(f"Baseline counts: early={b0['counts']['early']} | on-time={b0['counts']['ontime']} | late={b0['counts']['late']}"),
        html.Div(f"Cost-sensitive counts: early={b1['counts']['early']} | on-time={b1['counts']['ontime']} | late={b1['counts']['late']}"),
        html.Div(f"Lead time: {c['leadtime']} days")
    ])

    return fig_total, fig_timing, fig_breakdown, counts_text

if __name__ == "__main__":
    app.run(debug=True)
