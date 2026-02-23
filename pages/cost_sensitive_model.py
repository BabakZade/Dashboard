from __future__ import annotations

import re
from pathlib import Path

import dash
from dash import dcc, html, ctx, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# -----------------------------
# Initial settings
# -----------------------------
settings = {
    "lead_time": 1,
    "rul_thresh": 60,
    "slice_window": 7,
    "slice_shift": 7,
    "cost_weight": 1.0,
    "early_penalty": 1,
    "late_penalty": 10,
    "cost_reactive": 200,
    "cost_predictive": 20,
}

ASSETS_DIR = "assets"

# -----------------------------
# Shared styles
# -----------------------------
row_style = {"display": "flex", "gap": "5px", "alignItems": "center", "marginBottom": "8px"}
box_style = {"padding": "5px", "border": "1px solid #ddd", "width": "60px", "textAlign": "center"}

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

# -----------------------------
# Settings -> model filename
# -----------------------------
def settings_slug(s: dict) -> str:
    def fmt(x):
        if isinstance(x, float):
            return str(x).replace(".", "p")
        return str(x)

    return (
        f"Model_LT{fmt(s['lead_time'])}_"
        f"TR{fmt(s['rul_thresh'])}_"
        f"W{fmt(s['slice_window'])}_"
        f"S{fmt(s['slice_shift'])}_"
        f"CW{fmt(s['cost_weight'])}_"
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
    r"LT(?P<lead_time>\d+)_"
    r"TR(?P<rul_thresh>\d+)_"
    r"W(?P<slice_window>\d+)_"
    r"S(?P<slice_shift>\d+)_"
    r"CW(?P<cost_weight>\d+(?:p\d+)?)_"
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
        "lead_time": int(d["lead_time"]),
        "rul_thresh": int(d["rul_thresh"]),
        "slice_window": int(d["slice_window"]),
        "slice_shift": int(d["slice_shift"]),
        "cost_weight": float(d["cost_weight"].replace("p", ".")),
        "early_penalty": int(d["early_penalty"]),
        "late_penalty": int(d["late_penalty"]),
        "cost_reactive": int(d["cost_reactive"]),
        "cost_predictive": int(d["cost_predictive"]),
    }

# -----------------------------
# Similarity by % difference
# -----------------------------
def _pct_close(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-9)
    pct_diff = abs(a - b) / denom
    return max(0.0, 1.0 - pct_diff)

def similarity_percent(target: dict, candidate: dict) -> float:
    weights = {
        "lead_time": 2.0,
        "rul_thresh": 4.0,
        "slice_window": 3.0,
        "slice_shift": 3.0,
        "cost_weight": 1.0,
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
    # TODO: your training code; must save a .pkl at save_to
    return {"trained_for": target_settings, "saved_to": save_to}

def run_model(model, target_settings: dict):
    # must return Dash components or string
    return html.Pre(f"Running model: {model}\n\nSettings:\n{target_settings}")

# -----------------------------
# +/- helpers
# -----------------------------
def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))

def _clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))

# -------------------------------------------------
# Layout
# -------------------------------------------------
def layout():
    desired = settings_slug(settings)

    return html.Div(
        [
            dcc.Store(id="settings-store", data=settings),

            # Which model is currently selected (not necessarily run)
            dcc.Store(
                id="selected-model-store",
                data={"model_name": None, "source": "none", "match_percent": None},
            ),

            html.Div(
                [
                    html.I(
                        className="fa fa-cogs",
                        id="gear-icon",
                        style={"fontSize": "30px", "cursor": "pointer", "color": "black"},
                    ),
                    html.Div(
                        id="settings-panel",
                        children=[
                            html.H3("Dataset Settings", style={"textAlign": "center"}),

                            html.Div(
                                [
                                    # Lead Time
                                    html.Label("Lead Time (days):"),
                                    html.Div(
                                        [
                                            html.Button("-", id="lead_time_decrease", n_clicks=0),
                                            html.Div(id="lead_time_value", children=settings["lead_time"], style=box_style),
                                            html.Button("+", id="lead_time_increase", n_clicks=0),
                                        ],
                                        style=row_style,
                                    ),

                                    # Threshold (TR)
                                    html.Label("Threshold (TR):"),
                                    html.Div(
                                        [
                                            html.Button("-", id="rul_thresh_decrease", n_clicks=0),
                                            html.Div(id="rul_thresh_value", children=settings["rul_thresh"], style=box_style),
                                            html.Button("+", id="rul_thresh_increase", n_clicks=0),
                                        ],
                                        style=row_style,
                                    ),

                                    # Slice Window
                                    html.Label("Slice Window (days):"),
                                    html.Div(
                                        [
                                            html.Button("-", id="slice_window_decrease", n_clicks=0),
                                            html.Div(id="slice_window_value", children=settings["slice_window"], style=box_style),
                                            html.Button("+", id="slice_window_increase", n_clicks=0),
                                        ],
                                        style=row_style,
                                    ),

                                    # Slice Shift
                                    html.Label("Slice Shift (days):"),
                                    html.Div(
                                        [
                                            html.Button("-", id="slice_shift_decrease", n_clicks=0),
                                            html.Div(id="slice_shift_value", children=settings["slice_shift"], style=box_style),
                                            html.Button("+", id="slice_shift_increase", n_clicks=0),
                                        ],
                                        style=row_style,
                                    ),

                                    # Cost Weight
                                    html.Label("Cost Weight:"),
                                    html.Div(
                                        [
                                            html.Button("-", id="cost_weight_decrease", n_clicks=0),
                                            html.Div(id="cost_weight_value", children=settings["cost_weight"], style=box_style),
                                            html.Button("+", id="cost_weight_increase", n_clicks=0),
                                        ],
                                        style=row_style,
                                    ),

                                    # Early Penalty
                                    html.Label("Early Penalty:"),
                                    html.Div(
                                        [
                                            html.Button("-", id="early_penalty_decrease", n_clicks=0),
                                            html.Div(id="early_penalty_value", children=settings["early_penalty"], style=box_style),
                                            html.Button("+", id="early_penalty_increase", n_clicks=0),
                                        ],
                                        style=row_style,
                                    ),

                                    # Late Penalty
                                    html.Label("Late Penalty:"),
                                    html.Div(
                                        [
                                            html.Button("-", id="late_penalty_decrease", n_clicks=0),
                                            html.Div(id="late_penalty_value", children=settings["late_penalty"], style=box_style),
                                            html.Button("+", id="late_penalty_increase", n_clicks=0),
                                        ],
                                        style=row_style,
                                    ),

                                    # Reactive cost
                                    html.Label("Reactive Cost:"),
                                    html.Div(
                                        [
                                            html.Button("-", id="cost_reactive_decrease", n_clicks=0),
                                            html.Div(id="cost_reactive_value", children=settings["cost_reactive"], style=box_style),
                                            html.Button("+", id="cost_reactive_increase", n_clicks=0),
                                        ],
                                        style=row_style,
                                    ),

                                    # Predictive cost
                                    html.Label("Predictive Cost:"),
                                    html.Div(
                                        [
                                            html.Button("-", id="cost_predictive_decrease", n_clicks=0),
                                            html.Div(id="cost_predictive_value", children=settings["cost_predictive"], style=box_style),
                                            html.Button("+", id="cost_predictive_increase", n_clicks=0),
                                        ],
                                        style=row_style,
                                    ),

                                    html.Hr(),

                                    # Model section
                                    html.Div("Desired model:", style={"fontWeight": "bold"}),
                                    html.Div(id="desired-model-text", children=desired, style={"fontFamily": "monospace", "fontSize": "12px"}),

                                    html.Div(id="model-status-text", style={"marginTop": "8px", "fontSize": "13px"}),

                                    html.Label("Available / closest models:", style={"marginTop": "10px"}),
                                    dcc.Dropdown(
                                        id="model-dropdown",
                                        options=[],
                                        value=None,
                                        clearable=False,
                                        style={"fontSize": "12px"},
                                    ),

                                    html.Div(
                                        [
                                            html.Button("Select model", id="btn-select-model", n_clicks=0),
                                            html.Button(
                                                "Train new model",
                                                id="btn-train-new",
                                                n_clicks=0,
                                                style={"marginLeft": "8px"},
                                            ),
                                        ],
                                        style={"marginTop": "10px"},
                                    ),

                                    html.Div("Selected model:", style={"fontWeight": "bold", "marginTop": "10px"}),
                                    html.Div(id="selected-model-text", style={"fontSize": "12px", "fontFamily": "monospace"}),
                                ],
                                style={"width": "100%", "padding": "10px"},
                            ),
                        ],
                        style=dropdown_style_hidden,
                    ),
                ],
                style={"position": "relative", "display": "flex", "justifyContent": "flex-end", "padding": "10px"},
            ),

            html.Div(id="main-area", style={"minHeight": "400px", "marginTop": "20px"}),
        ]
    )

# -------------------------------------------------
# Callbacks
# -------------------------------------------------
def register_callbacks(app):
    # show/hide settings panel
    @app.callback(
        Output("settings-panel", "style"),
        Input("gear-icon", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_settings_panel(gear_clicks):
        if gear_clicks and gear_clicks % 2 == 1:
            visible = dropdown_style_hidden.copy()
            visible["display"] = "block"
            return visible
        return dropdown_style_hidden

    # update settings-store + visible values
    @app.callback(
        Output("settings-store", "data"),
        Output("lead_time_value", "children"),
        Output("rul_thresh_value", "children"),
        Output("slice_window_value", "children"),
        Output("slice_shift_value", "children"),
        Output("cost_weight_value", "children"),
        Output("early_penalty_value", "children"),
        Output("late_penalty_value", "children"),
        Output("cost_reactive_value", "children"),
        Output("cost_predictive_value", "children"),
        Input("lead_time_decrease", "n_clicks"),
        Input("lead_time_increase", "n_clicks"),
        Input("rul_thresh_decrease", "n_clicks"),
        Input("rul_thresh_increase", "n_clicks"),
        Input("slice_window_decrease", "n_clicks"),
        Input("slice_window_increase", "n_clicks"),
        Input("slice_shift_decrease", "n_clicks"),
        Input("slice_shift_increase", "n_clicks"),
        Input("cost_weight_decrease", "n_clicks"),
        Input("cost_weight_increase", "n_clicks"),
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
        lt_dec, lt_inc,
        tr_dec, tr_inc,
        w_dec, w_inc,
        s_dec, s_inc,
        cw_dec, cw_inc,
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

        if trig == "lead_time_decrease":
            s["lead_time"] = _clamp_int(s["lead_time"] - 1, 0, 365)
        elif trig == "lead_time_increase":
            s["lead_time"] = _clamp_int(s["lead_time"] + 1, 0, 365)
        elif trig == "rul_thresh_decrease":
            s["rul_thresh"] = _clamp_int(s["rul_thresh"] - 1, 0, 10_000)
        elif trig == "rul_thresh_increase":
            s["rul_thresh"] = _clamp_int(s["rul_thresh"] + 1, 0, 10_000)
        elif trig == "slice_window_decrease":
            s["slice_window"] = _clamp_int(s["slice_window"] - 1, 1, 10_000)
        elif trig == "slice_window_increase":
            s["slice_window"] = _clamp_int(s["slice_window"] + 1, 1, 10_000)
        elif trig == "slice_shift_decrease":
            s["slice_shift"] = _clamp_int(s["slice_shift"] - 1, 1, 10_000)
        elif trig == "slice_shift_increase":
            s["slice_shift"] = _clamp_int(s["slice_shift"] + 1, 1, 10_000)
        elif trig == "cost_weight_decrease":
            s["cost_weight"] = round(_clamp_float(s["cost_weight"] - 0.1, 0.0, 1_000.0), 3)
        elif trig == "cost_weight_increase":
            s["cost_weight"] = round(_clamp_float(s["cost_weight"] + 0.1, 0.0, 1_000.0), 3)
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
            s["lead_time"],
            s["rul_thresh"],
            s["slice_window"],
            s["slice_shift"],
            s["cost_weight"],
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
                status = html.Div(f"Model not found and no models exist in {ASSETS_DIR}/.", style={"color": "#b00020"})
            else:
                status = html.Div("Exact model not found. Pick one of the closest models or train a new one.", style={"color": "#b00020"})
                for row in top3:
                    options.append(
                        {"label": f"{row['name']} ({row['score']*100:.1f}%)", "value": row["name"]}
                    )
                default_value = top3[0]["name"]

        # If nothing is available, keep dropdown empty
        return desired, status, options, default_value

    # Select model / Train new model (and run)
    @app.callback(
        Output("selected-model-store", "data"),
        Output("selected-model-text", "children"),
        Output("main-area", "children"),
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

        if trig == "btn-select-model":
            if not dropdown_value:
                raise PreventUpdate
            chosen_name = dropdown_value
            chosen_path = assets_path / chosen_name
            if not chosen_path.exists():
                # safety: dropdown might point to stale file
                return (
                    {"model_name": None, "source": "none", "match_percent": None},
                    "None selected",
                    html.Pre(f"Selected file does not exist:\n{chosen_name}"),
                )

            model = load_model(str(chosen_path))
            out = run_model(model, s)

            # set source
            if chosen_name == desired and desired_path.exists():
                data = {"model_name": chosen_name, "source": "exact", "match_percent": 100.0}
            else:
                # best-effort: compute match% if it was one of the closest
                top3 = top_k_models(s, ASSETS_DIR, k=3)
                mp = None
                for row in top3:
                    if row["name"] == chosen_name:
                        mp = row["score"] * 100
                        break
                data = {"model_name": chosen_name, "source": "closest", "match_percent": mp}

            mp_text = "" if data["match_percent"] is None else f", {data['match_percent']:.1f}%"
            return data, f"{chosen_name} ({data['source']}{mp_text})", out

        if trig == "btn-train-new":
            # train -> save exact filename -> load -> run
            train_new_model(s, save_to=str(desired_path))
            model = load_model(str(desired_path))
            out = run_model(model, s)
            data = {"model_name": desired, "source": "trained", "match_percent": 100.0}
            return data, f"{desired} (trained, 100.0%)", out

        raise PreventUpdate