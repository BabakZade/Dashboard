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
# Lifetime maintenance cost helpers
# =============================================================================

def _find_trigger_indices_threshold(
    result: dict, lead_time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold rule: trigger in window when ttf_50 first drops ≤ lead_time."""
    ttf_50 = result["ttf_50"]
    fault_indicator = result["fault_indicator"].astype(bool)
    failure_indices = np.where(fault_indicator)[0]

    trigger_idxs, reactive_idxs = [], []
    prev_idx = 0
    for fi in failure_indices:
        window_ttf = ttf_50[prev_idx : fi + 1]
        hits = np.where(window_ttf <= lead_time)[0]
        if len(hits) > 0:
            trigger_idxs.append(prev_idx + int(hits[0]))
        else:
            reactive_idxs.append(int(fi))
        prev_idx = fi + 1

    return np.array(trigger_idxs, dtype=int), np.array(reactive_idxs, dtype=int)


def _find_trigger_indices_cost_optimal(
    result: dict, c_predictive: float, c_reactive: float, lead_time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cost-optimal rule: trigger when P(fail within lead_time steps) ≥ C_pr / C_re.

    The probability threshold equals the cost ratio C_pr / C_re.  A large
    reactive cost relative to predictive cost lowers the threshold, causing
    earlier triggers.  Failure probability is computed from the product of
    per-step survival probabilities (1 − prob_mean) over a look-ahead window
    of length lead_time.
    """
    prob_mean = result["prob_mean"]
    fault_indicator = result["fault_indicator"].astype(bool)
    failure_indices = np.where(fault_indicator)[0]

    prob_threshold = min(c_predictive / max(c_reactive, 1e-9), 1.0)

    trigger_idxs, reactive_idxs = [], []
    prev_idx = 0
    for fi in failure_indices:
        triggered = False
        for i in range(prev_idx, fi + 1):
            horizon_end = min(i + int(lead_time), fi + 1)
            hazards = np.clip(prob_mean[i:horizon_end], 0.0, 1.0)
            p_fail = 1.0 - float(np.prod(1.0 - hazards))
            if p_fail >= prob_threshold:
                trigger_idxs.append(i)
                triggered = True
                break
        if not triggered:
            reactive_idxs.append(int(fi))
        prev_idx = fi + 1

    return np.array(trigger_idxs, dtype=int), np.array(reactive_idxs, dtype=int)


def _get_trigger_indices(
    result: dict, lead_time: float, c_pred: float, c_react: float, rule: str
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the chosen trigger rule; returns (trigger_indices, reactive_indices)."""
    if rule == "cost_optimal":
        return _find_trigger_indices_cost_optimal(result, c_pred, c_react, lead_time)
    return _find_trigger_indices_threshold(result, lead_time)


def _optimal_maintenance_figure(
    result: dict, lead_time: float, c_pred: float, c_react: float,
    rule: str, failure_type: str,
) -> go.Figure:
    """TTF (actual + predicted + CI) with optimal-maintenance trigger lines."""
    t = result["time"]
    ttf_true = result["ttf_true"]
    fault_times = t[result["fault_indicator"].astype(bool)]
    trigger_idxs, reactive_idxs = _get_trigger_indices(result, lead_time, c_pred, c_react, rule)
    trigger_times = t[trigger_idxs] if len(trigger_idxs) > 0 else np.array([])
    reactive_times = t[reactive_idxs] if len(reactive_idxs) > 0 else np.array([])

    fig = go.Figure()

    # ── 90 % CI band ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([result["ttf_95"], result["ttf_05"][::-1]]),
        fill="toself", fillcolor="rgba(99,110,250,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90 % CI",
    ))

    # ── Predicted TTF (median) ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t, y=result["ttf_50"],
        mode="lines", name="Pred TTF (median)",
        line=dict(color="#636EFA", width=2),
    ))

    # ── Actual TTF ───────────────────────────────────────────────────────────
    valid = np.isfinite(ttf_true)
    if valid.any():
        fig.add_trace(go.Scatter(
            x=t[valid], y=ttf_true[valid],
            mode="lines", name="Actual TTF",
            line=dict(color="black", dash="dashdot", width=1.5),
        ))

    # ── Threshold / look-ahead annotation (horizontal) ────────────────────────
    if rule == "cost_optimal":
        p_thr = min(c_pred / max(c_react, 1e-9), 1.0)
        hline_label = f"Look-ahead H={lead_time}  │  p_thr={p_thr:.2f} (C_pr/C_re)"
    else:
        hline_label = f"Lead time = {lead_time}"
    fig.add_hline(
        y=lead_time,
        line_color="orange", line_dash="dash", line_width=1.5, opacity=0.75,
        annotation_text=hline_label,
        annotation_position="top right",
        annotation_font=dict(color="orange", size=11),
    )

    # ── Observed failures (red dotted) ────────────────────────────────────────
    for ft in fault_times:
        fig.add_vline(
            x=ft, line_color="red", line_dash="dot",
            line_width=1.2, opacity=0.5,
        )

    # ── Predictive maintenance triggers (green dashed) ────────────────────────
    for tt in trigger_times:
        fig.add_vline(
            x=tt, line_color="#00CC96", line_dash="dash",
            line_width=2, opacity=0.85,
        )

    # ── Unforeseeable reactive events (orange dotted) ─────────────────────────
    for rt in reactive_times:
        fig.add_vline(
            x=rt, line_color="darkorange", line_dash="dot",
            line_width=2, opacity=0.85,
        )

    # ── Legend proxies (add_vline has no legend support) ─────────────────────
    if len(trigger_times) > 0:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"Predictive trigger  ×{len(trigger_times)}",
            line=dict(color="#00CC96", dash="dash", width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        name=f"Observed failure  ×{len(fault_times)}",
        line=dict(color="red", dash="dot", width=2),
    ))
    if len(reactive_times) > 0:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"Reactive (unforeseeable)  ×{len(reactive_times)}",
            line=dict(color="darkorange", dash="dot", width=2),
        ))

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=60, b=40),
        title_text=(
            f"{failure_type} — Optimal maintenance schedule  "
            f"({len(trigger_times)} predictive  |  {len(reactive_times)} reactive)"
        ),
        title_x=0.5,
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="TTF")
    return fig


def _compute_lifetime_costs(
    result: dict,
    c_predictive: float,
    c_reactive: float,
    lead_time: float,
    rule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time, cumulative_model_guided_cost, cumulative_always_reactive_cost).

    Uses the selected trigger rule to determine when model-guided maintenance
    fires; the always-reactive baseline pays c_reactive at every failure.
    """
    t = result["time"]
    fault_indicator = result["fault_indicator"].astype(bool)
    n = len(t)

    failure_indices = np.where(fault_indicator)[0]
    trigger_idxs, _ = _get_trigger_indices(result, lead_time, c_predictive, c_reactive, rule)
    trigger_idx_set = set(trigger_idxs.tolist())

    mg_costs = np.zeros(n)
    ar_costs = np.zeros(n)

    prev_idx = 0
    for fi in failure_indices:
        # Find whether a trigger fired in this window [prev_idx, fi]
        window_trigger = next(
            (i for i in range(prev_idx, fi + 1) if i in trigger_idx_set), None
        )
        
        prev_idx = fi + 1

        if window_trigger is not None:
            mg_costs[window_trigger] += c_predictive
            
        else:
            mg_costs[fi] += c_reactive

        ar_costs[fi] += c_reactive

    return t, np.cumsum(mg_costs), np.cumsum(ar_costs)


def _lifetime_cost_figure(
    result: dict,
    c_predictive: float,
    c_reactive: float,
    lead_time: float,
    rule: str,
    failure_type: str,
) -> go.Figure:
    t, cumcost_mg, cumcost_ar = _compute_lifetime_costs(
        result, c_predictive, c_reactive, lead_time, rule
    )
    fault_times = t[result["fault_indicator"].astype(bool)]
    total_savings = float(cumcost_ar[-1] - cumcost_mg[-1]) if len(t) > 0 else 0.0
    savings_pct = (total_savings / cumcost_ar[-1] * 100) if cumcost_ar[-1] > 0 else 0.0

    fig = go.Figure()

    # ── savings shading ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([cumcost_ar, cumcost_mg[::-1]]),
        fill="toself",
        fillcolor="rgba(0,204,150,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"Savings  ({savings_pct:.1f} %)",
        showlegend=True,
    ))

    # ── always-reactive baseline ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t, y=cumcost_ar,
        mode="lines",
        name="Always Reactive",
        line=dict(color="#EF553B", width=2.5),
    ))

    # ── model-guided predictive ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t, y=cumcost_mg,
        mode="lines",
        name="Model-Guided Predictive",
        line=dict(color="#00CC96", width=2.5),
    ))

    # ── observed-failure markers ─────────────────────────────────────────────
    for ft in fault_times:
        fig.add_vline(
            x=ft, line_color="red", line_dash="dot",
            line_width=1, opacity=0.45,
        )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=30),
        title_text=(
            f"{failure_type} — Cumulative maintenance cost  "
            f"| Model-guided saves {total_savings:,.0f} units  ({savings_pct:.1f} %)"
        ),
        title_x=0.5,
        legend=dict(orientation="h", y=-0.14),
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Cumulative cost")
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

            # ── Section 3: Lifetime maintenance cost estimator ────────────────
            html.H3("Lifetime maintenance cost estimator", style={"marginTop": 0}),
            html.Div(
                "Simulates cumulative maintenance cost over the machine's lifetime. "
                "The model-guided strategy schedules predictive maintenance whenever "
                "the predicted TTF drops to or below the lead time threshold. "
                "The always-reactive baseline pays the full emergency cost at every "
                "observed failure.",
                style={"opacity": 0.7, "marginBottom": "14px"},
            ),
            # ── Trigger rule selector ────────────────────────────────────────
            html.Div(
                style={"marginBottom": "14px", "display": "flex", "alignItems": "center", "gap": "12px"},
                children=[
                    html.Label("Trigger rule", style={"fontWeight": "600", "whiteSpace": "nowrap"}),
                    dcc.RadioItems(
                        id="lcost_rule",
                        options=[
                            {
                                "label": "Lead-time threshold  (trigger when ttf₅₀ ≤ lead time)",
                                "value": "threshold",
                            },
                            {
                                "label": "Cost-optimal  (trigger when P(fail in next H steps) ≥ C_pr / C_re)",
                                "value": "cost_optimal",
                            },
                        ],
                        value="threshold",
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "24px"},
                    ),
                ],
            ),

            # ── Parameter controls ──────────────────────────────────────────
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3, 1fr)",
                    "gap": "20px",
                    "marginBottom": "16px",
                    "background": "#f9f9f9",
                    "borderRadius": "8px",
                    "padding": "16px",
                },
                children=[
                    html.Div([
                        html.Label(
                            "Predictive maintenance cost (C_pr)",
                            style={"fontWeight": "600", "display": "block", "marginBottom": "6px"},
                        ),
                        dcc.Slider(
                            id="lcost_c_pred",
                            min=5, max=500, step=5, value=20,
                            marks={5: "5", 100: "100", 250: "250", 500: "500"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),
                    html.Div([
                        html.Label(
                            "Reactive maintenance cost (C_re)",
                            style={"fontWeight": "600", "display": "block", "marginBottom": "6px"},
                        ),
                        dcc.Slider(
                            id="lcost_c_react",
                            min=50, max=2000, step=50, value=200,
                            marks={50: "50", 500: "500", 1000: "1000", 2000: "2000"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),
                    html.Div([
                        html.Label(
                            "Lead time threshold",
                            style={"fontWeight": "600", "display": "block", "marginBottom": "6px"},
                        ),
                        dcc.Slider(
                            id="lcost_lead_time",
                            min=1, max=90, step=1, value=25,
                            marks={1: "1", 15: "15", 30: "30", 45: "45", 60: "60", 75: "75", 90: "90"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),
                ],
            ),
            # ── Optimal schedule graph ──────────────────────────────────
            html.Div(
                "Green dashed lines = scheduled predictive maintenance. "
                "Orange dotted = unforeseeable reactive events. "
                "Red dotted = observed failures. "
                "Orange dashed horizontal = lead-time / look-ahead threshold.",
                style={"opacity": 0.65, "fontSize": "0.88em", "marginBottom": "10px"},
            ),
            html.Div(id="lcost_rule_hint", style={"marginBottom": "8px"}),

            dcc.Loading(
                id="lcost_schedule_loading",
                type="circle",
                children=dcc.Graph(id="lcost_trigger_plot", config={"displayModeBar": False}),
            ),

            # ── Summary chips ───────────────────────────────────────────
            html.Div(id="lcost_summary", style={"marginBottom": "12px", "marginTop": "18px"}),
            dcc.Loading(
                id="lcost_loading",
                type="circle",
                children=dcc.Graph(id="lcost_plot", config={"displayModeBar": False}),
            ),

            html.Hr(style={"margin": "28px 0"}),

            # ── Section 4: Model graph ────────────────────────────────────────
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

    @app.callback(
        Output("lcost_trigger_plot", "figure"),
        Output("lcost_plot", "figure"),
        Output("lcost_summary", "children"),
        Output("lcost_rule_hint", "children"),
        Input("rul_split", "value"),
        Input("rul_machine_id", "value"),
        Input("pred_failure_type", "value"),
        Input("lcost_c_pred", "value"),
        Input("lcost_c_react", "value"),
        Input("lcost_lead_time", "value"),
        Input("lcost_rule", "value"),
    )
    def update_lifetime_cost_plot(split, machine_id, failure_type,
                                  c_pred, c_react, lead_time, rule):
        def _empty(msg="Model not fitted — run scripts/fit_model.py first"):
            f = go.Figure()
            f.update_layout(
                height=200,
                annotations=[dict(
                    text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=14, color="#999"),
                )],
            )
            return f

        if not model_is_fitted() or machine_id is None:
            return _empty(), _empty(), "", ""

        df = _get_df(split)
        if df.empty:
            return go.Figure(), go.Figure(), "", ""

        machine_df = df[df["machine_id"] == int(machine_id)]
        if machine_df.empty:
            return go.Figure(), go.Figure(), "", ""

        result = predict_machine(machine_df, failure_type=failure_type,
                                 cache_key=(split, int(machine_id), failure_type))
        if result is None:
            return go.Figure(), go.Figure(), "", ""

        schedule_fig = _optimal_maintenance_figure(
            result, lead_time, c_pred, c_react, rule, failure_type
        )

        _t, cumcost_mg, cumcost_ar = _compute_lifetime_costs(
            result, c_pred, c_react, lead_time, rule
        )
        total_ar  = float(cumcost_ar[-1]) if len(_t) > 0 else 0.0
        total_mg  = float(cumcost_mg[-1]) if len(_t) > 0 else 0.0
        savings   = total_ar - total_mg
        savings_pct = (savings / total_ar * 100) if total_ar > 0 else 0.0

        chip_style_base = {
            "display": "inline-block",
            "borderRadius": "6px",
            "padding": "6px 14px",
            "marginRight": "10px",
            "fontWeight": "600",
            "fontSize": "0.9em",
        }
        summary = html.Div([
            html.Span(f"Reactive total: {total_ar:,.0f}",
                      style={**chip_style_base, "background": "#fdecea", "color": "#b71c1c"}),
            html.Span(f"Model-guided total: {total_mg:,.0f}",
                      style={**chip_style_base, "background": "#e8f5e9", "color": "#1b5e20"}),
            html.Span(f"Savings: {savings:,.0f}  ({savings_pct:.1f} %)",
                      style={**chip_style_base, "background": "#e3f2fd", "color": "#0d47a1"}),
        ])

        if rule == "cost_optimal":
            p_thr = min(c_pred / max(c_react, 1e-9), 1.0)
            hint_text = (
                f"ℹ️  Cost-optimal rule active — "
                f"triggering when P(fail within next {lead_time} steps) ≥ "
                f"{p_thr:.2f}  (= C_pr / C_re = {c_pred} / {c_react})"
            )
        else:
            hint_text = (
                f"ℹ️  Threshold rule active — "
                f"triggering when predicted median TTF ≤ {lead_time}"
            )
        rule_hint = html.Div(
            hint_text,
            style={"fontSize": "0.85em", "color": "#555", "fontStyle": "italic"},
        )

        cost_fig = _lifetime_cost_figure(result, c_pred, c_react, lead_time, rule, failure_type)
        return schedule_fig, cost_fig, summary, rule_hint

