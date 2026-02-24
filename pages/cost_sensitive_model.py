from __future__ import annotations

import re
from pathlib import Path

from dash import dcc, html, ctx, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# --- import metrics (your current sys.path trick) ---
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.metrics import compute_all


# -----------------------------
# Initial settings
# -----------------------------
settings = {
    "slice_window": 7,
    "early_penalty": 1,
    "late_penalty": 10,
    "cost_reactive": 200,
    "cost_predictive": 20,
}

ASSETS_DIR = "assets"

# -----------------------------
# Shared styles
# -----------------------------
# --- Three-zone pill ( - | value | + ) ---
pill_row_style = {
    "display": "flex",
    "alignItems": "center",
    "gap": "10px",
    "marginBottom": "10px",
}

pill_label_style = {
    "width": "170px",
    "whiteSpace": "nowrap",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
    "fontWeight": 600,
}

pill_wrap_style = {
    "display": "flex",
    "alignItems": "stretch",
    "height": "34px",
    "borderRadius": "999px",
    "overflow": "hidden",
    "border": "1px solid #ccc",
    "backgroundColor": "#f3f4f6",
}

pill_btn_style = {
    "width": "38px",
    "border": "0px",
    "backgroundColor": "transparent",
    "cursor": "pointer",
    "fontSize": "18px",
    "lineHeight": "34px",
    "padding": "0px",
}

pill_value_style = {
    "minWidth": "56px",
    "padding": "0 10px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "fontFamily": "monospace",
    "borderLeft": "1px solid #d1d5db",
    "borderRight": "1px solid #d1d5db",
}

def param_three_zone_pill(label: str, value_id: str, dec_id: str, inc_id: str, initial_value: int):
    """
    Label   ( - | value | + )   in ONE connected pill
    """
    return html.Div(
        [
            html.Div(label, style=pill_label_style),
            html.Div(
                [
                    html.Button("−", id=dec_id, n_clicks=0, style=pill_btn_style),
                    html.Div(id=value_id, children=initial_value, style=pill_value_style),
                    html.Button("+", id=inc_id, n_clicks=0, style=pill_btn_style),
                ],
                style=pill_wrap_style,
            ),
        ],
        style=pill_row_style,
    )


dropdown_style_hidden = {
    "display": "none",
    "position": "absolute",
    "top": "45px",
    "right": "0px",
    "zIndex": "9999",
    "border": "1px solid #ddd",
    "backgroundColor": "white",
    "minWidth": "380px",
    "padding": "15px",
    "borderRadius": "8px",
    "boxShadow": "0 4px 12px rgba(0,0,0,0.15)",
    "maxHeight": "560px",
    "overflowY": "auto",
}

# main page card styles
CARD_STYLE = {"border": "1px solid #ddd", "borderRadius": "10px", "padding": "12px", "marginBottom": "12px"}
HIDDEN_STYLE = {"display": "none"}
VISIBLE_CARD_STYLE = {**CARD_STYLE, "display": "block"}


# -----------------------------
# Settings -> model filename
# -----------------------------
def settings_slug(s: dict) -> str:
    def fmt(x):
        if isinstance(x, float):
            return str(x).replace(".", "p")
        return str(x)

    return (
        f"Model_"
        f"W{fmt(s['slice_window'])}_"
        f"EP{fmt(s['early_penalty'])}_"
        f"LP{fmt(s['late_penalty'])}_"
        f"CR{fmt(s['cost_reactive'])}_"
        f"CP{fmt(s['cost_predictive'])}.pkl"
    )


# -----------------------------
# Model filename parsing
# -----------------------------
MODEL_RE = re.compile(
    r"^Model_"
    r"W(?P<slice_window>\d+)_"
    r"EP(?P<early_penalty>\d+)_"
    r"LP(?P<late_penalty>\d+)_"
    r"CR(?P<cost_reactive>\d+)_"
    r"CP(?P<cost_predictive>\d+)\.pkl$"
)

def _parse_model_name(filename: str) -> dict | None:
    m = MODEL_RE.match(filename)
    if not m:
        return None
    d = m.groupdict()
    return {
        "slice_window": int(d["slice_window"]),
        "early_penalty": int(d["early_penalty"]),
        "late_penalty": int(d["late_penalty"]),
        "cost_reactive": int(d["cost_reactive"]),
        "cost_predictive": int(d["cost_predictive"]),
    }


# -----------------------------
# Pretty settings display
# -----------------------------
SETTING_LABELS = {
    "slice_window": "Slice length (days)",
    "early_penalty": "Early penalty",
    "late_penalty": "Late penalty",
    "cost_reactive": "Reactive cost",
    "cost_predictive": "Predictive cost",
}

def format_settings_for_display(s: dict) -> str:
    lines = []
    for k in ["slice_window", "early_penalty", "late_penalty", "cost_reactive", "cost_predictive"]:
        lines.append(f"- {SETTING_LABELS.get(k, k)}: {s.get(k)}")
    return "\n".join(lines)


# -----------------------------
# Similarity by % difference
# -----------------------------
def _pct_close(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-9)
    pct_diff = abs(a - b) / denom
    return max(0.0, 1.0 - pct_diff)

def similarity_percent(target: dict, candidate: dict) -> float:
    weights = {
        "slice_window": 3.0,
        "early_penalty": 2.0,
        "late_penalty": 2.0,
        "cost_reactive": 0.5,
        "cost_predictive": 0.5,
    }
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        num += w * _pct_close(float(target[k]), float(candidate[k]))
        den += w
    return num / den if den else 0.0

def top_k_models(target: dict, assets_dir: str = ASSETS_DIR, k: int = 3) -> list[dict]:
    assets_path = Path(assets_dir)
    rows = []
    for p in assets_path.glob("*.pkl"):
        parsed = _parse_model_name(p.name)
        if parsed is None:
            continue
        score = similarity_percent(target, parsed)
        rows.append({"name": p.name, "score": score, "parsed": parsed})
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:k]


# -----------------------------
# Your hooks (replace with real ones)
# -----------------------------
def load_model(model_path: str):
    # TODO: replace with joblib.load / pickle.load
    return {"loaded_from": model_path}

def train_new_model(target_settings: dict, save_to: str):
    # TODO: must actually save a .pkl at save_to for real usage
    return {"trained_for_pretty": format_settings_for_display(target_settings), "saved_to": save_to}

# TODO: Replace with your real inference code.
def get_predictions_and_truth(model, target_settings: dict):
    return None, None


# -----------------------------
# UI rendering helpers for main page
# -----------------------------
def render_metrics_section(metrics: dict | None):
    if not metrics:
        return html.Div("No metrics available (no y_true/y_pred yet).", style={"opacity": 0.7})

    keys = [
        "mse",
        "cost",
        "error_within_0_min_pct",
        "error_within_min_max_pct",
        "error_largerthan_max_pct",
        "error_within_0_min_pct_critical",
        "error_within_min_max_pct_critical",
        "error_largerthan_max_pct_critical",
    ]

    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    return html.Div(
        [html.Div([html.Span(k, style={"fontWeight": 600}), html.Span(f": {fmt(metrics.get(k))}")]) for k in keys],
        style={"fontFamily": "monospace", "fontSize": "12px", "lineHeight": "1.8"},
    )

def render_graphs_section(y_true, y_pred):
    if y_true is None or y_pred is None:
        return html.Div("No graphs available (no y_true/y_pred yet).", style={"opacity": 0.7})

    import numpy as np
    import plotly.graph_objects as go

    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    err = yp - yt

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=yt, y=yp, mode="markers", name="Pred vs True"))
    fig_scatter.add_trace(go.Scatter(x=[float(yt.min()), float(yt.max())], y=[float(yt.min()), float(yt.max())], mode="lines", name="y=x"))
    fig_scatter.update_layout(title="Predicted RUL vs True RUL", xaxis_title="True RUL", yaxis_title="Predicted RUL", height=360)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=err, name="Error (pred-true)"))
    fig_hist.update_layout(title="Prediction Error Distribution", xaxis_title="Error (pred-true)", yaxis_title="Count", height=360)

    return html.Div([dcc.Graph(figure=fig_scatter), dcc.Graph(figure=fig_hist)])


def build_main_page_layout():
    """
    IMPORTANT: All 3 sections start HIDDEN.
    They become visible ONLY after user selects/trains a model.
    """
    return html.Div(
        id="main-area",
        children=[
            # Placeholder shown before selecting model
            html.Div(
                id="main-placeholder",
                children=html.Div(
                    [
                        html.H3("Main Dashboard"),
                        html.Div("Select a model from Settings to show Loading/Training, Metrics, and Graphs.", style={"opacity": 0.8}),
                    ],
                    style={"border": "1px dashed #bbb", "borderRadius": "10px", "padding": "16px"},
                ),
            ),

            # Section 1 (hidden initially)
            html.Div(
                id="card-loading-training",
                style=HIDDEN_STYLE,
                children=[
                    html.H3("1) Loading & Training"),
                    html.Div(id="section-loading-training"),
                ],
            ),

            # Section 2 (hidden initially)
            html.Div(
                id="card-metrics",
                style=HIDDEN_STYLE,
                children=[
                    html.H3("2) Metrics"),
                    html.Div(id="section-metrics"),
                ],
            ),

            # Section 3 (hidden initially)
            html.Div(
                id="card-graphs",
                style=HIDDEN_STYLE,
                children=[
                    html.H3("3) Graphs"),
                    html.Div(id="section-graphs"),
                ],
            ),
        ],
        style={"minHeight": "400px", "marginTop": "20px"},
    )


# -----------------------------
# +/- helpers
# -----------------------------
def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


# -------------------------------------------------
# Layout
# -------------------------------------------------
def layout():
    desired = settings_slug(settings)

    return html.Div(
        [
            dcc.Store(id="settings-store", data=settings),
            dcc.Store(id="metrics-store", storage_type="session", data=None),
            dcc.Store(id="selected-model-store", data={"model_name": None, "source": "none", "match_percent": None}),

            # Top-right gear + settings panel
            html.Div(
                [
                    html.I(className="fa fa-cogs", id="gear-icon", style={"fontSize": "30px", "cursor": "pointer", "color": "black"}),
                    html.Div(
                        id="settings-panel",
                        children=[
                            html.Div(
                                [
                                    html.H3("Dataset Settings", style={"textAlign": "center"}),

                                    html.Div("Desired model:", style={"fontWeight": "bold"}),
                                    html.Div(id="desired-model-text", children=desired, style={"fontFamily": "monospace", "fontSize": "12px"}),
                                    html.Div(id="model-status-text", style={"marginTop": "8px", "fontSize": "13px"}),

                                    html.Label("Available / closest models:", style={"marginTop": "10px"}),
                                    dcc.Dropdown(id="model-dropdown", options=[], value=None, clearable=False, style={"fontSize": "12px"}),

                                    html.Div(
                                        [
                                            html.Button("Select model", id="btn-select-model", n_clicks=0),
                                            html.Button("Train new model", id="btn-train-new", n_clicks=0, style={"marginLeft": "8px"}),
                                        ],
                                        style={"marginTop": "10px"},
                                    ),

                                    html.Div("Selected model:", style={"fontWeight": "bold", "marginTop": "10px"}),
                                    html.Div(id="selected-model-text", style={"fontSize": "12px", "fontFamily": "monospace"}),

                                    html.Hr(style={"margin": "14px 0"}),

                                    # 
                                    param_three_zone_pill("Slice length (days)", "slice_window_value", "slice_window_decrease", "slice_window_increase", settings["slice_window"]),
                                    param_three_zone_pill("Early penalty", "early_penalty_value", "early_penalty_decrease", "early_penalty_increase", settings["early_penalty"]),
                                    param_three_zone_pill("Late penalty", "late_penalty_value", "late_penalty_decrease", "late_penalty_increase", settings["late_penalty"]),
                                    param_three_zone_pill("Reactive cost", "cost_reactive_value", "cost_reactive_decrease", "cost_reactive_increase", settings["cost_reactive"]),
                                    param_three_zone_pill("Predictive cost", "cost_predictive_value", "cost_predictive_decrease", "cost_predictive_increase", settings["cost_predictive"]),
                                    ],
                                style={"width": "100%", "padding": "10px"},
                            )
                        ],
                        style=dropdown_style_hidden,
                    ),
                ],
                style={"position": "relative", "display": "flex", "justifyContent": "flex-end", "padding": "10px"},
            ),

            # Click-catcher closes the panel
            html.Div(id="main-click-catcher", n_clicks=0, children=[build_main_page_layout()]),
        ]
    )


# -------------------------------------------------
# Callbacks
# -------------------------------------------------
def register_callbacks(app):
    # toggle / auto-close settings panel
    @app.callback(
        Output("settings-panel", "style"),
        Input("gear-icon", "n_clicks"),
        Input("main-click-catcher", "n_clicks"),
        Input("btn-select-model", "n_clicks"),
        Input("btn-train-new", "n_clicks"),
        Input("slice_window_decrease", "n_clicks"),
        Input("slice_window_increase", "n_clicks"),
        Input("early_penalty_decrease", "n_clicks"),
        Input("early_penalty_increase", "n_clicks"),
        Input("late_penalty_decrease", "n_clicks"),
        Input("late_penalty_increase", "n_clicks"),
        Input("cost_reactive_decrease", "n_clicks"),
        Input("cost_reactive_increase", "n_clicks"),
        Input("cost_predictive_decrease", "n_clicks"),
        Input("cost_predictive_increase", "n_clicks"),
        State("settings-panel", "style"),
        prevent_initial_call=True,
    )
    def toggle_or_close_settings_panel(
        gear_clicks,
        main_clicks,
        n_select,
        n_train,
        w_dec, w_inc,
        ep_dec, ep_inc,
        lp_dec, lp_inc,
        cr_dec, cr_inc,
        cp_dec, cp_inc,
        current_style,
    ):
        def _open_style():
            s = dropdown_style_hidden.copy()
            s["display"] = "block"
            return s

        def _is_open(style):
            return bool(style) and style.get("display") == "block"

        trig = ctx.triggered_id
        is_open_now = _is_open(current_style)

        if trig == "gear-icon":
            return dropdown_style_hidden if is_open_now else _open_style()

        if is_open_now:
            return dropdown_style_hidden

        return no_update


    # update settings-store + visible values
    @app.callback(
        Output("settings-store", "data"),
        Output("slice_window_value", "children"),
        Output("early_penalty_value", "children"),
        Output("late_penalty_value", "children"),
        Output("cost_reactive_value", "children"),
        Output("cost_predictive_value", "children"),
        Input("slice_window_decrease", "n_clicks"),
        Input("slice_window_increase", "n_clicks"),
        Input("early_penalty_decrease", "n_clicks"),
        Input("early_penalty_increase", "n_clicks"),
        Input("late_penalty_decrease", "n_clicks"),
        Input("late_penalty_increase", "n_clicks"),
        Input("cost_reactive_decrease", "n_clicks"),
        Input("cost_reactive_increase", "n_clicks"),
        Input("cost_predictive_decrease", "n_clicks"),
        Input("cost_predictive_increase", "n_clicks"),
        State("settings-store", "data"),
        prevent_initial_call=True,
    )
    def update_settings(
        w_dec, w_inc,
        ep_dec, ep_inc,
        lp_dec, lp_inc,
        cr_dec, cr_inc,
        cp_dec, cp_inc,
        data,
    ):
        if not ctx.triggered_id:
            raise PreventUpdate

        s = dict(data) if data else dict(settings)
        trig = ctx.triggered_id

        if trig == "slice_window_decrease":
            s["slice_window"] = _clamp_int(s["slice_window"] - 1, 1, 10_000)
        elif trig == "slice_window_increase":
            s["slice_window"] = _clamp_int(s["slice_window"] + 1, 1, 10_000)

        elif trig == "early_penalty_decrease":
            s["early_penalty"] = _clamp_int(s["early_penalty"] - 1, 0, 10_000)
        elif trig == "early_penalty_increase":
            s["early_penalty"] = _clamp_int(s["early_penalty"] + 1, 0, 10_000)

        elif trig == "late_penalty_decrease":
            s["late_penalty"] = _clamp_int(s["late_penalty"] - 1, 0, 10_000)
        elif trig == "late_penalty_increase":
            s["late_penalty"] = _clamp_int(s["late_penalty"] + 1, 0, 10_000)

        elif trig == "cost_reactive_decrease":
            s["cost_reactive"] = _clamp_int(s["cost_reactive"] - 10, 0, 1_000_000)
        elif trig == "cost_reactive_increase":
            s["cost_reactive"] = _clamp_int(s["cost_reactive"] + 10, 0, 1_000_000)

        elif trig == "cost_predictive_decrease":
            s["cost_predictive"] = _clamp_int(s["cost_predictive"] - 1, 0, 1_000_000)
        elif trig == "cost_predictive_increase":
            s["cost_predictive"] = _clamp_int(s["cost_predictive"] + 1, 0, 1_000_000)
        else:
            raise PreventUpdate

        return (
            s,
            s["slice_window"],
            s["early_penalty"],
            s["late_penalty"],
            s["cost_reactive"],
            s["cost_predictive"],
        )


    # Update desired model text + dropdown options + status message
    @app.callback(
        Output("desired-model-text", "children"),
        Output("model-status-text", "children"),
        Output("model-dropdown", "options"),
        Output("model-dropdown", "value"),
        Input("settings-store", "data"),
        prevent_initial_call=False,
    )
    def update_model_choices(s):
        if not s:
            raise PreventUpdate

        desired = settings_slug(s)
        assets_path = Path(ASSETS_DIR)
        desired_exists = (assets_path / desired).exists()

        options = []
        default_value = None

        if desired_exists:
            options.append({"label": f"✅ {desired} (100.0%)", "value": desired})
            default_value = desired
            status = html.Div("Exact model exists. Click 'Select model' to run it.", style={"color": "#0a7a0a"})
        else:
            top3 = top_k_models(s, ASSETS_DIR, k=3)
            if not top3:
                status = html.Div(f"Model not found and no matching models exist in {ASSETS_DIR}/.", style={"color": "#b00020"})
            else:
                status = html.Div("Exact model not found. Pick one of the closest models or train a new one.", style={"color": "#b00020"})
                for row in top3:
                    options.append({"label": f"{row['name']} ({row['score']*100:.1f}%)", "value": row["name"]})
                default_value = top3[0]["name"]

        return desired, status, options, default_value


    # Select model / Train new model -> SHOW sections + fill them
    @app.callback(
        Output("selected-model-store", "data"),
        Output("selected-model-text", "children"),
        Output("metrics-store", "data"),
        Output("section-loading-training", "children"),
        Output("section-metrics", "children"),
        Output("section-graphs", "children"),
        Output("card-loading-training", "style"),
        Output("card-metrics", "style"),
        Output("card-graphs", "style"),
        Output("main-placeholder", "style"),
        Input("btn-select-model", "n_clicks"),
        Input("btn-train-new", "n_clicks"),
        State("settings-store", "data"),
        State("model-dropdown", "value"),
        prevent_initial_call=True,
    )
    def select_or_train(n_select, n_train, s, dropdown_value):
        if not s:
            raise PreventUpdate

        assets_path = Path(ASSETS_DIR)
        desired = settings_slug(s)
        desired_path = assets_path / desired
        trig = ctx.triggered_id

        # ---------- Load/Train ----------
        if trig == "btn-select-model":
            if not dropdown_value:
                raise PreventUpdate

            chosen_name = dropdown_value
            chosen_path = assets_path / chosen_name

            if not chosen_path.exists():
                return (
                    {"model_name": None, "source": "none", "match_percent": None},
                    "None selected",
                    None,
                    no_update,
                    no_update,
                    no_update,
                    HIDDEN_STYLE,
                    HIDDEN_STYLE,
                    HIDDEN_STYLE,
                    {"display": "block"},
                )

            model = load_model(str(chosen_path))

            if chosen_name == desired and desired_path.exists():
                data = {"model_name": chosen_name, "source": "exact", "match_percent": 100.0}
            else:
                top3 = top_k_models(s, ASSETS_DIR, k=3)
                mp = None
                for row in top3:
                    if row["name"] == chosen_name:
                        mp = row["score"] * 100
                        break
                data = {"model_name": chosen_name, "source": "closest", "match_percent": mp}

            mp_text = "" if data["match_percent"] is None else f", {data['match_percent']:.1f}%"
            selected_text = f"{chosen_name} ({data['source']}{mp_text})"

            loading_ui = html.Pre(
                f"Action: LOAD MODEL\n"
                f"Model file: {chosen_name}\n"
                f"Path: {chosen_path}\n"
                f"Source: {data['source']}{mp_text}\n\n"
                f"Settings:\n{format_settings_for_display(s)}"
            )

        elif trig == "btn-train-new":
            train_new_model(s, save_to=str(desired_path))
            model = load_model(str(desired_path))

            data = {"model_name": desired, "source": "trained", "match_percent": 100.0}
            selected_text = f"{desired} (trained, 100.0%)"

            loading_ui = html.Pre(
                f"Action: TRAIN NEW MODEL\n"
                f"Saved to: {desired_path}\n\n"
                f"Settings:\n{format_settings_for_display(s)}"
            )
        else:
            raise PreventUpdate

        # ---------- Predictions -> Metrics -> Graphs ----------
        y_true, y_pred = get_predictions_and_truth(model, s)

        metrics_dict = None
        if y_true is not None and y_pred is not None:
            metrics_dict = compute_all(
                y_true=y_true,
                y_pred=y_pred,
                min_err=2.0,
                max_err=10.0,
                critical_true_lt=10.0,
                cost_fn=None,  # put your real cost function here
            )

        metrics_ui = render_metrics_section(metrics_dict)
        graphs_ui = render_graphs_section(y_true, y_pred)

        # SHOW all three cards; HIDE placeholder
        return (
            data,
            selected_text,
            metrics_dict,
            loading_ui,
            metrics_ui,
            graphs_ui,
            VISIBLE_CARD_STYLE,
            VISIBLE_CARD_STYLE,
            {**VISIBLE_CARD_STYLE, "marginBottom": "0px"},
            {"display": "none"},
        )