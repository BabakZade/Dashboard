# pages/rul_distribution.py
import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output

from core.pymc_model import model_is_fitted, predict_machine

# =============================================================================
# Data helpers
# =============================================================================

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_DATASET_DIR = os.path.join(_ASSETS_DIR, "simulated_dataset")

_COVARIATES = [
    ("route_ratio", "Route Ratio"),
    ("speed",       "Speed"),
    ("load",        "Load"),
    ("car_type",    "Car Type"),
    ("region",      "Region"),
    ("route",       "Route"),
]

VEHICLE_TYPES = {"Vans": 0, "Trucks": 1}

FAILURE_TYPES = {np.nan: 0,"Tires": 1, "Brakes": 2}


def _load_split(split: str) -> pd.DataFrame:
    """Load and concatenate all time-series CSVs for a given split (train/test)."""
    pattern = os.path.join(_DATASET_DIR, split, "tbm_ts_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        one_file_raw_data = pd.read_csv(f, index_col=None, header=0)

        one_file_raw_data["car_type"] = one_file_raw_data["car_type"].map(VEHICLE_TYPES)
        one_file_raw_data["failure_type"] = one_file_raw_data["failure_type"].map(FAILURE_TYPES)

        original_machine_id = one_file_raw_data["machine_id"].astype(int)
        one_file_raw_data["machine_id"] = (
            one_file_raw_data["car_type"].astype(int) * 1_000 + original_machine_id
        )

        dfs.append(one_file_raw_data)
    df = pd.concat(dfs, ignore_index=True)
    return df


def _machine_ids(df: pd.DataFrame) -> list[int]:
    return sorted(df["machine_id"].unique().astype(int).tolist())


# Pre-load both splits once at import time so callbacks are fast
_CACHE: dict[str, pd.DataFrame] = {}


def _get_df(split: str) -> pd.DataFrame:
    if split not in _CACHE:
        _CACHE[split] = _load_split(split)
    return _CACHE[split]


# =============================================================================
# Figure builder
# =============================================================================

def _covariate_figure(df: pd.DataFrame, machine_id: int) -> go.Figure:
    machine_data = df[df["machine_id"] == machine_id].sort_values("time")

    fault_times = machine_data.loc[machine_data["failure_type"] > 1, "time"].tolist()

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[label for _, label in _COVARIATES],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    for idx, (col, label) in enumerate(_COVARIATES):
        row, col_pos = divmod(idx, 3)
        row += 1
        col_pos += 1

        fig.add_trace(
            go.Scatter(
                x=machine_data["time"],
                y=machine_data[col],
                mode="lines",
                name=label,
                showlegend=False,
                line=dict(width=1.5),
            ),
            row=row, col=col_pos,
        )

        for ft in fault_times:
            failure_type = machine_data.loc[machine_data["time"] == ft, "failure_type"].iloc[0]
            line_color = "red" if failure_type == 2 else "orange"
            fig.add_vline(
                x=ft,
                line_color=line_color,
                line_dash="dash",
                line_width=1,
                opacity=0.5,
                row=row, col=col_pos,
            )

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=60, b=20),
        title_text=f"Covariates — machine {machine_id}",
        title_x=0.5,
    )
    fig.update_xaxes(title_text="Time")

    return fig


# =============================================================================
# Prediction figure builders
# =============================================================================

def _prediction_figure(result: dict, failure_type: str) -> go.Figure:
    """Two-row figure: TTF quantiles (top) and failure probability (bottom)."""
    t = result["time"]
    fault_times = t[result["fault_indicator"].astype(bool)]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"{failure_type} — Time to Failure (predicted vs actual)",
            f"{failure_type} — Instantaneous failure probability",
        ],
        vertical_spacing=0.15,
        shared_xaxes=True,
    )

    # ── Row 1: TTF ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([result["ttf_95"], result["ttf_05"][::-1]]),
        fill="toself", fillcolor="rgba(99,110,250,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90 % CI", showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=result["ttf_50"],
        mode="lines", name="Pred TTF (median)",
        line=dict(color="#636EFA", width=2),
    ), row=1, col=1)

    ttf_true = result["ttf_true"]
    valid = np.isfinite(ttf_true)
    if valid.any():
        fig.add_trace(go.Scatter(
            x=t[valid], y=ttf_true[valid],
            mode="lines", name="Actual TTF",
            line=dict(color="black", dash="dashdot", width=1.5),
        ), row=1, col=1)

    # ── Row 2: failure probability ───────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([result["prob_hi"], result["prob_lo"][::-1]]),
        fill="toself", fillcolor="rgba(239,85,59,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90 % CI ", showlegend=True,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=result["prob_mean"],
        mode="lines", name="Failure prob (mean)",
        line=dict(color="#EF553B", width=2),
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=t[result["fault_indicator"].astype(bool)],
        y=np.ones(result["fault_indicator"].sum()),
        name="Observed failure",
        marker_color="rgba(0,0,0,0.25)",
        width=0.8,
        showlegend=True,
    ), row=2, col=1)

    # ── Fault-time vertical lines (both rows) ───────────────────────────────
    for ft in fault_times:
        for r in (1, 2):
            fig.add_shape(
                type="line",
                xref=f"x{'' if r == 1 else r}",
                yref=f"y{'' if r == 1 else r} domain",
                x0=ft, x1=ft, y0=0, y1=1,
                line=dict(color="red", dash="dot", width=1),
                opacity=0.6,
            )

    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=60, b=30),
        legend=dict(orientation="h", y=-0.08),
    )
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="TTF", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1)
    return fig


# =============================================================================
# Layout & callbacks
# =============================================================================

def layout():
    # Bootstrap machine-id options from train split; callback will update them
    df_train = _get_df("train")
    ids = _machine_ids(df_train) if not df_train.empty else []
    id_options = [{"label": str(i), "value": i} for i in ids]
    default_id = ids[0] if ids else None

    # ── Model status banner ──────────────────────────────────────────────────
    if model_is_fitted():
        model_banner = html.Div(
            "Model fitted ✓",
            style={"color": "#2e7d32", "fontWeight": "600", "marginBottom": "8px"},
        )
    else:
        model_banner = html.Div(
            [
                html.Span(
                    "No fitted model found. Run the fitting script first:",
                    style={"marginRight": "8px"},
                ),
                html.Code(
                    "/home/lavinius.ioangliga/Projects/Dashboard/venv/bin/python "
                    "scripts/fit_model.py",
                    style={"fontSize": "0.85em", "background": "#f5f5f5", "padding": "2px 6px"},
                ),
            ],
            style={"color": "#b71c1c", "marginBottom": "8px"},
        )

    return html.Div(
        style={"border": "1px solid #ddd", "borderRadius": "12px", "padding": "16px"},
        children=[
            # ── Section 1: Covariate explorer ────────────────────────────────
            html.H3("Covariate explorer", style={"marginTop": 0}),
            html.Div(
                "Time-series of each sensor covariate for a single machine. "
                "Red dashed lines mark failure events.",
                style={"opacity": 0.7, "marginBottom": "14px"},
            ),
            html.Div(
                style={"display": "flex", "gap": "24px", "alignItems": "center", "marginBottom": "12px"},
                children=[
                    html.Div([
                        html.Label("Split", style={"fontWeight": "600", "marginRight": "8px"}),
                        dcc.RadioItems(
                            id="rul_split",
                            options=[
                                {"label": "Train", "value": "train"},
                                {"label": "Test",  "value": "test"},
                            ],
                            value="train",
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "16px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("Machine ID", style={"fontWeight": "600", "marginRight": "8px"}),
                        dcc.Dropdown(
                            id="rul_machine_id",
                            options=id_options,
                            value=default_id,
                            clearable=False,
                            style={"width": "120px", "display": "inline-block", "verticalAlign": "middle"},
                        ),
                    ]),
                ],
            ),
            dcc.Graph(id="rul_covariate_plot", config={"displayModeBar": False}),

            html.Hr(style={"margin": "28px 0"}),

            # ── Section 2: RUL & failure probability ─────────────────────────
            html.H3("RUL & failure probability (Weibull PH model)",
                    style={"marginTop": 0}),
            model_banner,
            html.Div(
                "Predicted time-to-failure and instantaneous failure probability "
                "from the Bayesian Weibull proportional-hazards model. "
                "Shaded bands = 90 % posterior credible interval. "
                "Red dotted lines = observed failures.",
                style={"opacity": 0.7, "marginBottom": "14px"},
            ),
            html.Div(
                style={"display": "flex", "gap": "24px", "alignItems": "center", "marginBottom": "12px"},
                children=[
                    html.Div([
                        html.Label("Failure type", style={"fontWeight": "600", "marginRight": "8px"}),
                        dcc.RadioItems(
                            id="pred_failure_type",
                            options=[
                                {"label": "Tires",  "value": "Tires"},
                                {"label": "Brakes", "value": "Brakes"},
                            ],
                            value="Tires",
                            inline=True,
                            inputStyle={"marginRight": "4px"},
                            labelStyle={"marginRight": "16px"},
                        ),
                    ]),
                ],
            ),
            dcc.Loading(
                id="pred_loading",
                type="circle",
                children=dcc.Graph(id="rul_pred_plot", config={"displayModeBar": False}),
            ),

            html.Hr(style={"margin": "28px 0"}),

            # ── Section 3: Model graph ────────────────────────────────────────
            html.H3("Bayesian network model structure", style={"marginTop": 0}),
            html.Div(
                "Graphical representation of the Weibull proportional-hazards model "
                "(generated by PyMC model_to_graphviz).",
                style={"opacity": 0.7, "marginBottom": "14px"},
            ),
            html.Img(
                src="/assets/model_graph.png",
                style={
                    "maxWidth": "100%",
                    "display": "block",
                    "margin": "0 auto",
                    "borderRadius": "8px",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.12)",
                },
            ),
        ],
    )


def register_callbacks(app):

    @app.callback(
        Output("rul_machine_id", "options"),
        Output("rul_machine_id", "value"),
        Input("rul_split", "value"),
    )
    def update_machine_options(split):
        df = _get_df(split)
        ids = _machine_ids(df) if not df.empty else []
        options = [{"label": str(i), "value": i} for i in ids]
        value = ids[0] if ids else None
        return options, value

    @app.callback(
        Output("rul_covariate_plot", "figure"),
        Input("rul_split", "value"),
        Input("rul_machine_id", "value"),
    )
    def update_covariate_plot(split, machine_id):
        df = _get_df(split)
        if df.empty or machine_id is None:
            return go.Figure()
        return _covariate_figure(df, int(machine_id))

    @app.callback(
        Output("rul_pred_plot", "figure"),
        Input("rul_split", "value"),
        Input("rul_machine_id", "value"),
        Input("pred_failure_type", "value"),
    )
    def update_prediction_plot(split, machine_id, failure_type):
        if not model_is_fitted() or machine_id is None:
            fig = go.Figure()
            fig.update_layout(
                height=200,
                annotations=[dict(
                    text="Model not fitted — run scripts/fit_model.py first",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=14, color="#999"),
                )],
            )
            return fig

        df = _get_df(split)
        if df.empty:
            return go.Figure()

        machine_df = df[df["machine_id"] == int(machine_id)]
        if machine_df.empty:
            return go.Figure()

        result = predict_machine(machine_df, failure_type=failure_type,
                                 cache_key=(split, int(machine_id), failure_type))
        if result is None:
            return go.Figure()

        return _prediction_figure(result, failure_type)

