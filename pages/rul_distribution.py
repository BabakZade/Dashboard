# pages/rul_distribution.py
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output

from core.cbm_model import predict_machine_rolling

# =============================================================================
# Data helpers
# =============================================================================

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_DATASET_DIR = os.path.join(_ASSETS_DIR, "CBM_dataset")

_COVARIATES = [
    ("degra_level_observed", "Degradation Level"),
    ("brake_temperature",    "Brake Temperature"),
    ("vibration_level",      "Vibration Level"),
    ("speed",                "Speed"),
    ("load",                 "Load"),
    ("region",               "Region"),
]


def _load_split(split: str) -> pd.DataFrame:
    """Load the single CSV for a given split (train/test)."""
    path = os.path.join(_DATASET_DIR, split, f"complete_df_{split}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=None, header=0)


def _machine_ids(df: pd.DataFrame) -> list[int]:
    return sorted(df["machine_id"].unique().astype(int).tolist())


# Pre-load both splits once at import time so callbacks are fast
_CACHE: dict[str, pd.DataFrame] = {}


def _get_df(split: str) -> pd.DataFrame:
    if split not in _CACHE:
        _CACHE[split] = _load_split(split)
    return _CACHE[split]


# Cache for rolling predictions (expensive)
_ROLLING_CACHE: dict[tuple, dict] = {}


# =============================================================================
# Figure builder — Covariate explorer
# =============================================================================

def _covariate_figure(df: pd.DataFrame, machine_id: int) -> go.Figure:
    machine_data = df[df["machine_id"] == machine_id].sort_values("time")
    segments = machine_data["segment_id"].unique()

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[label for _, label in _COVARIATES],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    colours = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
               "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]

    for idx, (col, label) in enumerate(_COVARIATES):
        row, col_pos = divmod(idx, 3)
        row += 1
        col_pos += 1

        for i, seg_id in enumerate(segments):
            seg_data = machine_data[machine_data["segment_id"] == seg_id].sort_values("time")
            fig.add_trace(
                go.Scatter(
                    x=seg_data["time"],
                    y=seg_data[col],
                    mode="lines",
                    name=f"Seg {seg_id}" if idx == 0 else None,
                    showlegend=(idx == 0),
                    legendgroup=f"seg_{seg_id}",
                    line=dict(width=1.5, color=colours[i % len(colours)]),
                ),
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

def _rolling_rul_figure(rolling: dict, machine_id: int) -> go.Figure:
    """Rolling RUL prediction plot."""
    t = rolling["time"]
    rul_true = rolling["rul_true"]

    fig = go.Figure()

    # RUL 5-95 % band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([rolling["rul_q95"], rolling["rul_q05"][::-1]]),
        fill="toself", fillcolor="rgba(99,110,250,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Predicted RUL 5-95 %",
    ))

    # RUL median
    fig.add_trace(go.Scatter(
        x=t, y=rolling["rul_q50"],
        mode="lines", name="Predicted RUL (median)",
        line=dict(color="#636EFA", width=2),
    ))

    # Ground truth RUL
    fig.add_trace(go.Scatter(
        x=t, y=rul_true,
        mode="lines", name="Ground truth RUL",
        line=dict(color="black", dash="dashdot", width=1.5),
    ))

    # Fault markers
    fault_times = t[rolling["fault_indicator"].astype(bool)]
    for ft in fault_times:
        fig.add_vline(
            x=ft, line_color="red", line_dash="dot", line_width=1, opacity=0.6,
        )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=60, b=30),
        title_text=f"Machine {machine_id} — Online RUL prediction (rolling)",
        title_x=0.5,
        legend=dict(orientation="h", y=-0.12),
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="RUL")
    return fig


# =============================================================================
# Lifetime maintenance cost helpers (RUL-based triggers)
# =============================================================================

def _find_trigger_indices_threshold(
    rolling: dict, lead_time: float,
    c_pred: float = 0.0, c_react: float = 0.0,
    c_early: float = 0.0, c_downtime: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold rule: trigger when rul_q50 first drops <= effective threshold."""
    rul_50 = rolling["rul_q50"]
    fault_indicator = rolling["fault_indicator"].astype(bool)
    failure_indices = np.where(fault_indicator)[0]

    if c_early > 0:
        c_react_eff = c_react + c_downtime * lead_time
        thr = lead_time + (c_react_eff - c_pred) / c_early
    else:
        thr = lead_time

    trigger_idxs, reactive_idxs = [], []
    prev_idx = 0
    for fi in failure_indices:
        window_rul = rul_50[prev_idx: fi + 1]
        valid_rul = np.isfinite(window_rul)
        hits = np.where(valid_rul & (window_rul <= thr))[0]
        if len(hits) > 0:
            trigger_idxs.append(prev_idx + int(hits[0]))
        else:
            reactive_idxs.append(int(fi))
        prev_idx = fi + 1

    return np.array(trigger_idxs, dtype=int), np.array(reactive_idxs, dtype=int)


def _find_trigger_indices_cost_optimal(
    rolling: dict, c_predictive: float, c_reactive: float, lead_time: float,
    c_early: float = 0.0, c_downtime: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cost-optimal rule using RUL posterior samples for failure probability.

    At each step i, failure probability is computed as:
        P(fail within H) = P(RUL <= H)
    estimated from the posterior RUL samples at that step.
    """
    rul_50 = rolling["rul_q50"]
    rul_samples_matrix = rolling.get("rul_samples_matrix")
    fault_indicator = rolling["fault_indicator"].astype(bool)
    failure_indices = np.where(fault_indicator)[0]

    c_react_eff = c_reactive + c_downtime * lead_time

    trigger_idxs, reactive_idxs = [], []
    prev_idx = 0
    for fi in failure_indices:
        triggered = False
        for i in range(prev_idx, fi + 1):
            if not np.isfinite(rul_50[i]):
                continue
            c_pred_eff = c_predictive + c_early * max(0.0, float(rul_50[i]) - lead_time)
            prob_threshold = min(c_pred_eff / max(c_react_eff, 1e-9), 1.0)

            if rul_samples_matrix is not None:
                rul_samples_i = rul_samples_matrix[i]
                valid_samples = np.isfinite(rul_samples_i) & (rul_samples_i >= 0.0)
                if np.any(valid_samples):
                    p_fail = float(np.mean(rul_samples_i[valid_samples] <= lead_time))
                else:
                    p_fail = 0.0
            else:
                # Fallback for older cached payloads that do not include samples.
                p_fail = 0.0

            if p_fail >= prob_threshold:
                trigger_idxs.append(i)
                triggered = True
                break
        if not triggered:
            reactive_idxs.append(int(fi))
        prev_idx = fi + 1

    return np.array(trigger_idxs, dtype=int), np.array(reactive_idxs, dtype=int)


def _get_trigger_indices(
    rolling: dict, lead_time: float, c_pred: float, c_react: float, rule: str,
    c_early: float = 0.0, c_downtime: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if rule == "cost_optimal":
        return _find_trigger_indices_cost_optimal(
            rolling, c_pred, c_react, lead_time, c_early, c_downtime
        )
    return _find_trigger_indices_threshold(
        rolling, lead_time, c_pred, c_react, c_early, c_downtime
    )


def _optimal_maintenance_figure(
    rolling: dict, lead_time: float, c_pred: float, c_react: float,
    rule: str, c_early: float = 0.0, c_downtime: float = 0.0,
) -> go.Figure:
    """RUL (actual + predicted + CI) with optimal-maintenance trigger lines."""
    t = rolling["time"]
    rul_true = rolling["rul_true"]
    fault_times = t[rolling["fault_indicator"].astype(bool)]
    trigger_idxs, reactive_idxs = _get_trigger_indices(
        rolling, lead_time, c_pred, c_react, rule, c_early, c_downtime
    )
    trigger_times = t[trigger_idxs] if len(trigger_idxs) > 0 else np.array([])
    reactive_times = t[reactive_idxs] if len(reactive_idxs) > 0 else np.array([])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([rolling["rul_q95"], rolling["rul_q05"][::-1]]),
        fill="toself", fillcolor="rgba(99,110,250,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90 % CI",
    ))

    fig.add_trace(go.Scatter(
        x=t, y=rolling["rul_q50"],
        mode="lines", name="Pred RUL (median)",
        line=dict(color="#636EFA", width=2),
    ))

    valid = np.isfinite(rul_true)
    if valid.any():
        fig.add_trace(go.Scatter(
            x=t[valid], y=rul_true[valid],
            mode="lines", name="Actual RUL",
            line=dict(color="black", dash="dashdot", width=1.5),
        ))

    c_react_eff = c_react + c_downtime * lead_time
    if rule == "cost_optimal":
        p_thr_base = min(c_pred / max(c_react_eff, 1e-9), 1.0)
        hline_label = f"Look-ahead H={lead_time}  |  base p_thr={p_thr_base:.2f}"
        hline_y = lead_time
    else:
        if c_early > 0:
            thr = lead_time + (c_react_eff - c_pred) / c_early
            hline_label = f"Eff. threshold = {thr:.1f}"
            hline_y = thr
        else:
            hline_label = f"Lead time = {lead_time}"
            hline_y = lead_time

    fig.add_hline(
        y=hline_y,
        line_color="orange", line_dash="dash", line_width=1.5, opacity=0.75,
        annotation_text=hline_label,
        annotation_position="top right",
        annotation_font=dict(color="orange", size=11),
    )

    for ft in fault_times:
        fig.add_vline(x=ft, line_color="red", line_dash="dot", line_width=1.2, opacity=0.5)
    for tt in trigger_times:
        fig.add_vline(x=tt, line_color="#00CC96", line_dash="dash", line_width=2, opacity=0.85)
    for rt in reactive_times:
        fig.add_vline(x=rt, line_color="darkorange", line_dash="dot", line_width=2, opacity=0.85)

    if len(trigger_times) > 0:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"Predictive trigger  x{len(trigger_times)}",
            line=dict(color="#00CC96", dash="dash", width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        name=f"Observed failure  x{len(fault_times)}",
        line=dict(color="red", dash="dot", width=2),
    ))
    if len(reactive_times) > 0:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"Reactive (unforeseeable)  x{len(reactive_times)}",
            line=dict(color="darkorange", dash="dot", width=2),
        ))

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=60, b=40),
        title_text=(
            f"Optimal maintenance schedule  "
            f"({len(trigger_times)} predictive  |  {len(reactive_times)} reactive)"
        ),
        title_x=0.5,
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="RUL")
    return fig


def _compute_lifetime_costs(
    rolling: dict,
    c_predictive: float,
    c_reactive: float,
    lead_time: float,
    rule: str,
    c_early: float = 0.0,
    c_downtime: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = rolling["time"]
    fault_indicator = rolling["fault_indicator"].astype(bool)
    n = len(t)

    failure_indices = np.where(fault_indicator)[0]
    trigger_idxs, _ = _get_trigger_indices(
        rolling, lead_time, c_predictive, c_reactive, rule, c_early, c_downtime
    )
    trigger_idx_set = set(trigger_idxs.tolist())

    c_react_eff = c_reactive + c_downtime * lead_time

    mg_costs = np.zeros(n)
    ar_costs = np.zeros(n)

    prev_idx = 0
    for fi in failure_indices:
        window_trigger = next(
            (i for i in range(prev_idx, fi + 1) if i in trigger_idx_set), None
        )
        prev_idx = fi + 1
        ar_costs[fi] += c_react_eff

        if window_trigger is not None:
            actual_gap = float(t[fi] - t[window_trigger])
            excess_early = max(0.0, actual_gap - lead_time)
            mg_costs[window_trigger] += c_predictive + c_early * excess_early
        else:
            mg_costs[fi] += c_react_eff

    return t, np.cumsum(mg_costs), np.cumsum(ar_costs)


def _lifetime_cost_figure(
    rolling: dict,
    c_predictive: float,
    c_reactive: float,
    lead_time: float,
    rule: str,
    c_early: float = 0.0,
    c_downtime: float = 0.0,
) -> go.Figure:
    t, cumcost_mg, cumcost_ar = _compute_lifetime_costs(
        rolling, c_predictive, c_reactive, lead_time, rule, c_early, c_downtime
    )
    fault_times = t[rolling["fault_indicator"].astype(bool)]
    total_savings = float(cumcost_ar[-1] - cumcost_mg[-1]) if len(t) > 0 else 0.0
    savings_pct = (total_savings / cumcost_ar[-1] * 100) if cumcost_ar[-1] > 0 else 0.0

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([cumcost_ar, cumcost_mg[::-1]]),
        fill="toself", fillcolor="rgba(0,204,150,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name=f"Savings  ({savings_pct:.1f} %)",
    ))

    fig.add_trace(go.Scatter(
        x=t, y=cumcost_ar,
        mode="lines", name="Always Reactive",
        line=dict(color="#EF553B", width=2.5),
    ))

    fig.add_trace(go.Scatter(
        x=t, y=cumcost_mg,
        mode="lines", name="Model-Guided Predictive",
        line=dict(color="#00CC96", width=2.5),
    ))

    for ft in fault_times:
        fig.add_vline(x=ft, line_color="red", line_dash="dot", line_width=1, opacity=0.45)

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=30),
        title_text=(
            f"Cumulative maintenance cost  "
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
    df_train = _get_df("train")
    ids = _machine_ids(df_train) if not df_train.empty else []
    id_options = [{"label": str(i), "value": i} for i in ids]
    default_id = ids[0] if ids else None

    return html.Div(
        style={"border": "1px solid #ddd", "borderRadius": "12px", "padding": "16px"},
        children=[
            # ── Section 1: Covariate explorer ────────────────────────────────
            html.H3("Covariate explorer", style={"marginTop": 0}),
            html.Div(
                "Time-series of each sensor covariate for a single machine. "
                "Each segment is shown in a different colour.",
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

            # ── Section 2: Rolling RUL + maintenance cost ─────────────────────
            html.H3("Rolling RUL prediction & maintenance cost", style={"marginTop": 0}),
            html.Div(
                "Online RUL predictions using fast importance resampling (grows "
                "observation window one step at a time). "
                "The maintenance cost estimator uses these rolling predictions "
                "to schedule predictive vs. reactive maintenance.",
                style={"opacity": 0.7, "marginBottom": "14px"},
            ),

            dcc.Loading(
                id="rolling_loading",
                type="circle",
                children=dcc.Graph(id="rolling_rul_plot", config={"displayModeBar": False}),
            ),

            html.Hr(style={"margin": "18px 0"}),

            # ── Trigger rule selector ────────────────────────────────────────
            html.Div(
                style={"marginBottom": "14px", "display": "flex", "alignItems": "center", "gap": "12px"},
                children=[
                    html.Label("Trigger rule", style={"fontWeight": "600", "whiteSpace": "nowrap"}),
                    dcc.RadioItems(
                        id="lcost_rule",
                        options=[
                            {
                                "label": "Lead-time threshold  (trigger when RUL median <= lead time)",
                                "value": "threshold",
                            },
                            {
                                "label": "Cost-optimal  (trigger when P(fail) >= C_pr / C_re)",
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
                            "Lead time threshold (H)",
                            style={"fontWeight": "600", "display": "block", "marginBottom": "6px"},
                        ),
                        dcc.Slider(
                            id="lcost_lead_time",
                            min=1, max=90, step=1, value=25,
                            marks={1: "1", 15: "15", 30: "30", 45: "45", 60: "60", 75: "75", 90: "90"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),
                    html.Div([
                        html.Label(
                            "Early replacement penalty (C_early)",
                            style={"fontWeight": "600", "display": "block", "marginBottom": "6px"},
                        ),
                        dcc.Slider(
                            id="lcost_c_early",
                            min=0, max=50, step=1, value=0,
                            marks={0: "0", 10: "10", 25: "25", 50: "50"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),
                    html.Div([
                        html.Label(
                            "Downtime cost (C_dt)",
                            style={"fontWeight": "600", "display": "block", "marginBottom": "6px"},
                        ),
                        dcc.Slider(
                            id="lcost_c_downtime",
                            min=0, max=50, step=1, value=0,
                            marks={0: "0", 10: "10", 25: "25", 50: "50"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ]),
                ],
            ),

            html.Div(
                "Green dashed = scheduled predictive maintenance. "
                "Orange dotted = reactive. Red dotted = observed failures.",
                style={"opacity": 0.65, "fontSize": "0.88em", "marginBottom": "10px"},
            ),
            html.Div(id="lcost_rule_hint", style={"marginBottom": "8px"}),

            dcc.Loading(
                id="lcost_schedule_loading",
                type="circle",
                children=dcc.Graph(id="lcost_trigger_plot", config={"displayModeBar": False}),
            ),

            html.Div(id="lcost_summary", style={"marginBottom": "12px", "marginTop": "18px"}),
            dcc.Loading(
                id="lcost_loading",
                type="circle",
                children=dcc.Graph(id="lcost_plot", config={"displayModeBar": False}),
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
        Output("rolling_rul_plot", "figure"),
        Input("rul_split", "value"),
        Input("rul_machine_id", "value"),
    )
    def update_rolling_rul_plot(split, machine_id):
        if machine_id is None:
            return go.Figure()

        cache_key = (split, int(machine_id))
        if cache_key in _ROLLING_CACHE:
            rolling = _ROLLING_CACHE[cache_key]
        else:
            df = _get_df(split)
            if df.empty:
                return go.Figure()
            machine_df = df[df["machine_id"] == int(machine_id)]
            if machine_df.empty:
                return go.Figure()
            rolling = predict_machine_rolling(machine_df)
            _ROLLING_CACHE[cache_key] = rolling

        return _rolling_rul_figure(rolling, int(machine_id))

    @app.callback(
        Output("lcost_trigger_plot", "figure"),
        Output("lcost_plot", "figure"),
        Output("lcost_summary", "children"),
        Output("lcost_rule_hint", "children"),
        Input("rul_split", "value"),
        Input("rul_machine_id", "value"),
        Input("lcost_c_pred", "value"),
        Input("lcost_c_react", "value"),
        Input("lcost_lead_time", "value"),
        Input("lcost_rule", "value"),
        Input("lcost_c_early", "value"),
        Input("lcost_c_downtime", "value"),
    )
    def update_lifetime_cost_plot(split, machine_id,
                                  c_pred, c_react, lead_time, rule,
                                  c_early, c_downtime):
        c_early = c_early or 0.0
        c_downtime = c_downtime or 0.0

        def _empty(msg="Select a machine"):
            f = go.Figure()
            f.update_layout(
                height=200,
                annotations=[dict(
                    text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=14, color="#999"),
                )],
            )
            return f

        if machine_id is None:
            return _empty(), _empty(), "", ""

        cache_key = (split, int(machine_id))
        if cache_key in _ROLLING_CACHE:
            rolling = _ROLLING_CACHE[cache_key]
        else:
            df = _get_df(split)
            if df.empty:
                return go.Figure(), go.Figure(), "", ""
            machine_df = df[df["machine_id"] == int(machine_id)]
            if machine_df.empty:
                return go.Figure(), go.Figure(), "", ""
            rolling = predict_machine_rolling(machine_df)
            _ROLLING_CACHE[cache_key] = rolling

        schedule_fig = _optimal_maintenance_figure(
            rolling, lead_time, c_pred, c_react, rule, c_early, c_downtime,
        )

        _t, cumcost_mg, cumcost_ar = _compute_lifetime_costs(
            rolling, c_pred, c_react, lead_time, rule, c_early, c_downtime
        )
        total_ar = float(cumcost_ar[-1]) if len(_t) > 0 else 0.0
        total_mg = float(cumcost_mg[-1]) if len(_t) > 0 else 0.0
        savings = total_ar - total_mg
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

        c_react_eff = c_react + c_downtime * lead_time
        if rule == "cost_optimal":
            p_thr_base = min(c_pred / max(c_react_eff, 1e-9), 1.0)
            hint_text = (
                f"Cost-optimal rule — using posterior P(RUL <= H); "
                f"base threshold: {p_thr_base:.2f}"
            )
        else:
            if c_early > 0:
                thr = lead_time + (c_react_eff - c_pred) / c_early
                hint_text = f"Threshold rule — effective threshold = {thr:.1f}"
            else:
                hint_text = f"Threshold rule — trigger when predicted median RUL <= {lead_time}"

        rule_hint = html.Div(
            hint_text,
            style={"fontSize": "0.85em", "color": "#555", "fontStyle": "italic"},
        )

        cost_fig = _lifetime_cost_figure(
            rolling, c_pred, c_react, lead_time, rule, c_early, c_downtime,
        )
        return schedule_fig, cost_fig, summary, rule_hint
