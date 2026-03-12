from dash import html, dcc, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from pathlib import Path
from copy import deepcopy
import uuid
import yaml
import dash_bootstrap_components as dbc


MODAL_OVERLAY_HIDDEN = {
    "display": "none",
    "position": "fixed",
    "top": 0,
    "left": 0,
    "width": "100vw",
    "height": "100vh",
    "backgroundColor": "rgba(0,0,0,0.35)",
    "zIndex": 2000,
    "justifyContent": "center",
    "alignItems": "center",
}

MODAL_OVERLAY_VISIBLE = {
    **MODAL_OVERLAY_HIDDEN,
    "display": "flex",
}

MODAL_CONTENT_STYLE = {
    "width": "min(900px, 95vw)",
    "maxWidth": "95vw",
    "height": "min(760px, 90vh)",
    "maxHeight": "90vh",
    "backgroundColor": "white",
    "borderRadius": "16px",
    "boxShadow": "0 10px 30px rgba(0,0,0,0.2)",
    "padding": "24px",
    "display": "flex",
    "flexDirection": "column",
}

STEP_BADGE_STYLE = {
    "display": "inline-block",
    "padding": "6px 12px",
    "borderRadius": "999px",
    "backgroundColor": "#EEF2FF",
    "fontWeight": 600,
    "marginBottom": "16px",
}

NAV_BTN_ROW_STYLE = {
    "display": "flex",
    "justifyContent": "space-between",
    "alignItems": "center",
    "marginTop": "24px",
    "flexShrink": 0,
    "gap": "12px",
}

STEP_CONTENT_STYLE = {
    "border": "1px solid #E5E7EB",
    "borderRadius": "12px",
    "padding": "20px",
    "marginTop": "12px",
    "backgroundColor": "#FAFAFA",
}

WIZARD_BODY_SCROLL_STYLE = {
    "flex": "1",
    "overflowY": "auto",
    "paddingRight": "6px",
    "minHeight": 0,
}

FIELDS_2COL_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
    "gap": "20px 24px",
    "alignItems": "start",
}


DEFAULT_TRAINING_CONFIG = {
    "experiment": {
        "name": "cost_sensitivity",
        "readme": "Analyzing cost weighting",
        "top_n_settings": 3,
        "top_n_front": 3,
    },
    "general": {
        "precision": 4,
        "k_in_tanh": 1,
        "n_jobs": -1,
        "welch_bins_keep": 10,
    },
    "logging": {
        "console_steps": True,
        "console_info": True,
        "console_errors": True,
        "console_warnings": True,
        "console_debuge": True,
        "console_debuge_detail": True,
        "debug_save": True,
        "debug_dir": "logs",
        "debug_detail_custom_losses": False,
        "level": "DEBUG",
        "console_level": "INFO",
        "file_level": "DEBUG",
        "log_to_file": True,
        "log_filename": "run.log",
        "rotate_logs": True,
        "max_log_size_mb": 10,
        "backup_logs": 3,
        "verbosity": True,
        "save_model_diagnostic": True,
    },
    "savecsv": {
        "sep": ";",
        "decimal": ",",
        "index": False,
        "header": True,
        "encoding": "utf-8",
        "quoting": 0,
        "float_format": "%.5f",
    },
    "model": {
        "seed": 42,
        "train_frac": 0.7,
        "val_frac": 0.1,
        "test_frac": 0.2,
        "calibration_frac": 0.1,
        "outer_k_fold": 3,
        "inner_k_fold": 3,
        "test_count_perID": 1,
        "trials": 4,
        "tuning_time_limit": 1800,
        "primary_metric": "prmc",
        "secondary_metric": "rmse",
        "engine_id_overlap": False,
        "label_window_overlap": True,
        "top_n_features": 20,
        "min_req_features": 4,
        "max_req_features": 1000,
        "CI_alpha": 0.1,
        "f_engineer": True,
        "integer_regularization": 0.01,
        "upper_bound": {
            "quantile": 0.995,
            "regularization": 0.02,
        },
        "positive_prediction": True,
    },
    "figure": {
        "style": "whitegrid",
        "palette": "viridis",
        "dpi": 150,
        "fontsize": {
            "title": 14,
            "axis_title": 12,
            "tick": 9,
            "legend": 9,
            "annot": 7,
        },
        "save": True,
        "format": ["png", "svg"],
        "bin": 30,
        "focus": 2,
        "id_grid": 15,
    },
    "problem": {
        "lead_time": 1,
        "emrgency_response_time": 0,
        "rul_thresh": 60,
        "slice_window": 2,
        "slice_shift": 2,
        "cost_weight": 1.0,
        "early_penalty": 1,
        "late_penalty": 10,
        "emergency_penalty": 10,
        "cost_reactive": 200,
        "cost_predictive": 20,
        "normalize": True,
        "slicer_output": 1,
        "instance_dependent_cost_sensitive": False,
        "rul_distribution": {
            "error_dist": "normal",
            "sampling": 32,
            "shape_param": 1.5,
            "scale_param": 50,
        },
    },
    "cleaning": {
        "distinct_thresh": 0,
        "correlation_thresh": 0.98,
        "correlation_minimum": 0.08,
        "missing_col_thresh": 0.4,
        "missing_row_thresh": 0.5,
        "outlier_method": "iqr",
        "outlier_threshold": 3,
    },
    "paths": {
        "project_root": "",
        "models_src": "outputs/",
        "experiment_dir": "",
        "trained_models": "trained_models/",
        "model_analysis": "model_analysis/",
        "result_dir": "results/",
        "notebook_dir": "notebooks/",
        "scripts_dir": "scripts/",
    },
    "database": {
        "cmapss": {
            "raw": "data/cmapss/",
            "processed": "data/cmapss/processed/",
            "connection_string": "postgresql://user:pass@localhost/db",
            "query": "SELECT * FROM cmapss_table",
        },
        "ncmpss": {
            "raw": "data/ncmpss/",
            "processed": "data/ncmpss/processed/",
            "connection_string": "postgresql://user:pass@localhost/db",
            "query": "SELECT * FROM ncmpss_table",
        },
        "phm": {
            "raw": "data/phm/",
            "processed": "data/phm/processed/",
            "connection_string": "postgresql://user:pass@localhost/db",
            "query": "SELECT * FROM cmapss_table",
        },
        "phm2008": {
            "raw": "data/phm2008/",
            "processed": "data/phm2008/processed/",
            "connection_string": "postgresql://user:pass@localhost/db",
            "query": "SELECT * FROM cmapss_table",
        },
        "ev": {
            "raw": "data/ev/raw/",
            "processed": "data/ev/processed/",
            "connection_string": "mysql://user:pass@localhost/ev",
            "query": "SELECT * FROM ev_table",
        },
        "btry": {
            "raw": "data/btry/",
            "processed": "data/btry/processed/",
            "connection_string": "mysql://user:pass@localhost/btry",
            "query": "SELECT * FROM ev_table",
        },
        "simulated_bpost": {
            "raw": "data/simulated_bpost/",
            "processed": "data/simulated_bpost/processed/",
            "connection_string": "mysql://user:pass@localhost/btry",
            "query": "SELECT * FROM ev_table",
        },
    },
    "selected_dataset": "cmapss",
}


def _summary_row(label, value):
    return html.Div(
        [
            html.Div(label, className="wizard-summary-label"),
            html.Div(str(value if value is not None else "-"), className="wizard-summary-value"),
        ],
        className="wizard-summary-row",
    )


def _compact_number_row(
    unit_text,
    input_id,
    param_name,
    param_abr="abr",
    *,
    value=None,
    min_value=0,
    max_value=100,
    step=1,
    disabled=False,
    input_style=None,
    unit_width="60px",
    abr_width="70px",
    class_name="compact-number-input-progress-bar",
    persistence=True,
):
    return html.Div(
        [
            html.Div(
                [
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText(
                                unit_text,
                                className="wizard-input-addon wizard-input-addon-left",
                                style={
                                    "minWidth": unit_width,
                                    "justifyContent": "center",
                                    "borderBottom": "0",
                                    "borderBottomLeftRadius": "12px",
                                    "backgroundColor": "transparent",
                                },
                            ),
                            dbc.Input(
                                id=input_id,
                                type="number",
                                value=value if value is not None else None,
                                min=min_value,
                                max=max_value,
                                step=step,
                                disabled=disabled,
                                placeholder=param_name,
                                persistence=persistence,
                                persistence_type="session",
                                debounce=False,
                                className="wizard-number-input",
                                style={
                                    "flex": "1",
                                    "borderBottom": "0",
                                    "paddingBottom": "14px",
                                    "boxShadow": "none",
                                    "outline": "none",
                                    "backgroundColor": "transparent",
                                    "borderColor": "transparent",
                                    "backgroundClip": "padding-box",
                                    **(input_style or {}),
                                },
                            ),
                            dbc.InputGroupText(
                                param_abr,
                                className="wizard-input-addon wizard-input-addon-right",
                                style={
                                    "minWidth": abr_width,
                                    "justifyContent": "center",
                                    "borderBottom": "0",
                                    "backgroundColor": "transparent",
                                },
                            ),
                        ],
                        className="wizard-input-group",
                        style={"width": "100%"},
                    ),
                    dbc.Progress(
                        id=f"{input_id}-progress",
                        value=0,
                        striped=True,
                        animated=True,
                        style={
                            "position": "absolute",
                            "left": "0",
                            "right": "0",
                            "bottom": "0",
                            "height": "8px",
                            "borderTopLeftRadius": "0",
                            "borderTopRightRadius": "0",
                            "borderBottomLeftRadius": "12px",
                            "borderBottomRightRadius": "12px",
                            "overflow": "hidden",
                            "margin": "0",
                        },
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "100%",
                },
            ),
        ],
        className=class_name,
        style={"width": "100%"},
    )


def _compact_dropdown_row(
    dropdown_component,
    *,
    class_name="mb-0",
):
    return html.Div(
        dropdown_component,
        className=class_name,
        style={"width": "100%"},
    )


def build_training_config(form_data: dict, save_dir: Path, experiment_name: str) -> dict:
    config = deepcopy(DEFAULT_TRAINING_CONFIG)
    form_data = form_data or {}

    selected_dataset = form_data.get("dataset", config.get("selected_dataset", "cmapss"))
    instance_dependent = bool(form_data.get("instance_dependent_cost", False))

    config["experiment"]["name"] = experiment_name
    config["experiment"]["readme"] = f"Wizard-generated config for dataset {selected_dataset}"
    config["paths"]["experiment_dir"] = str(save_dir)
    config["selected_dataset"] = selected_dataset

    config["problem"]["slice_window"] = form_data.get("slice_window") or config["problem"]["slice_window"]
    config["problem"]["slice_shift"] = config["problem"]["slice_window"]
    config["problem"]["slicer_output"] = form_data.get("slicer_output") or config["problem"]["slicer_output"]
    config["problem"]["instance_dependent_cost_sensitive"] = instance_dependent
    config["problem"]["lead_time"] = form_data.get("lead_time") or config["problem"]["lead_time"]
    config["problem"]["cost_weight"] = form_data.get("cost_weight") or config["problem"]["cost_weight"]
    config["problem"]["early_penalty"] = form_data.get("early_penalty") or config["problem"]["early_penalty"]
    config["problem"]["late_penalty"] = form_data.get("late_penalty") or config["problem"]["late_penalty"]
    config["problem"]["cost_reactive"] = form_data.get("cost_reactive") or config["problem"]["cost_reactive"]
    config["problem"]["cost_predictive"] = form_data.get("cost_predictive") or config["problem"]["cost_predictive"]

    config["model"]["outer_k_fold"] = form_data.get("outer_k_fold") or config["model"]["outer_k_fold"]
    config["model"]["inner_k_fold"] = form_data.get("inner_k_fold") or config["model"]["inner_k_fold"]
    config["model"]["trials"] = form_data.get("trials") or config["model"]["trials"]
    config["model"]["tuning_time_limit"] = form_data.get("tuning_time_limit") or config["model"]["tuning_time_limit"]

    return config


def write_training_config(config: dict, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return config_path


def render_training_step(step: int, settings: dict, form_data: dict, validation_csv: str):
    titles = {
        1: "Data manipulation",
        2: "Cost function",
        3: "Training setup",
        4: "Dataset",
        5: "Preview",
    }

    settings = settings or {}
    form_data = form_data or {}

    if step == 1:
        body = html.Div(
            [
                html.H4("Data time-slice settings"),
                html.P("Define how the time window is sliced and represented for model input."),
                html.Div(
                    [
                        _compact_number_row(
                            "T",
                            "wizard-slice-window",
                            "Length of slice window",
                            param_abr="L_w",
                            value=form_data.get("slice_window"),
                            min_value=1,
                            max_value=100,
                            step=1,
                        ),
                        _compact_dropdown_row(
                            dcc.Dropdown(
                                id="wizard-slicer-output",
                                options=[
                                    {"label": "Choose slicer output", "value": 0},
                                    {"label": "Flatten all time-window features", "value": 1},
                                    {"label": "Average of each feature over the time window", "value": 2},
                                    {"label": "Standard deviation of each feature over the time window", "value": 3},
                                    {"label": "Average and standard deviation per feature", "value": 4},
                                    {"label": "Flattened features plus summary statistics", "value": 5},
                                ],
                                value=form_data.get("slicer_output"),
                                clearable=True,
                                placeholder="Choose slicer output",
                                persistence=True,
                                persistence_type="session",
                            )
                        ),
                    ],
                    className="wizard-grid-2",
                    style=FIELDS_2COL_STYLE,
                ),
            ],
            style=STEP_CONTENT_STYLE,
        )

    elif step == 2:
        instance_dependent = form_data.get(
            "instance_dependent_cost",
            settings.get("instance_dependent_cost", False),
        )
        manual_disabled = bool(instance_dependent)

        input_style = {}
        if manual_disabled:
            input_style = {
                "backgroundColor": "#F3F4F6",
                "color": "#9CA3AF",
                "cursor": "not-allowed",
            }

        section_style = {"marginTop": "0", "color": "#9CA3AF"} if manual_disabled else {"marginTop": "0"}

        body = html.Div(
            [
                html.H4("Cost function"),
                html.P("Define maintenance costs and fleet-related settings."),
                html.Label("Cost setting mode"),
                html.Div(
                    [
                        dbc.Switch(
                            id="wizard-instance-dependent-cost",
                            value=bool(instance_dependent),
                            label="Instance-dependent cost-sensitive",
                            className="mb-2",
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    "If selected, the data should include: weight, leadtime, reactive_cost, predictive_cost, downtime_cost, rul_cost.",
                    style={
                        "fontSize": "13px",
                        "color": "#6B7280",
                        "marginBottom": "18px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5(
                                    "Maintenance cost",
                                    id="title-maintenance-cost",
                                    style=section_style,
                                ),
                                _compact_number_row(
                                    "$",
                                    "wizard-cost-predictive",
                                    "Predictive maintenance cost",
                                    param_abr="C_pr",
                                    value=form_data.get("cost_predictive"),
                                    min_value=0,
                                    max_value=100,
                                    step=10,
                                    disabled=manual_disabled,
                                    input_style=input_style,
                                ),
                                _compact_number_row(
                                    "$",
                                    "wizard-cost-reactive",
                                    "Reactive maintenance cost",
                                    param_abr="C_re",
                                    value=form_data.get("cost_reactive"),
                                    min_value=0,
                                    max_value=10000,
                                    step=100,
                                    disabled=manual_disabled,
                                    input_style=input_style,
                                ),
                                _compact_number_row(
                                    "$/T",
                                    "wizard-early-penalty",
                                    "Early maintenance penalty",
                                    param_abr="α",
                                    value=form_data.get("early_penalty"),
                                    min_value=0,
                                    max_value=10,
                                    step=1,
                                    disabled=manual_disabled,
                                    input_style=input_style,
                                ),
                                _compact_number_row(
                                    "$/T",
                                    "wizard-late-penalty",
                                    "Late maintenance penalty",
                                    param_abr="β",
                                    value=form_data.get("late_penalty"),
                                    min_value=0,
                                    max_value=100,
                                    step=10,
                                    disabled=manual_disabled,
                                    input_style=input_style,
                                ),
                            ],
                            className="wizard-two-col-item",
                        ),
                        html.Div(
                            [
                                html.H5(
                                    "Fleet characteristics",
                                    id="title-fleet-characteristics",
                                    style=section_style,
                                ),
                                _compact_number_row(
                                    "day",
                                    "wizard-lead-time",
                                    "Lead time",
                                    param_abr="LT",
                                    value=form_data.get("lead_time"),
                                    min_value=0,
                                    max_value=365,
                                    step=7,
                                    disabled=manual_disabled,
                                    input_style=input_style,
                                ),
                                _compact_number_row(
                                    "w",
                                    "wizard-cost-weight",
                                    "Importance weight",
                                    param_abr="w",
                                    value=form_data.get("cost_weight"),
                                    min_value=0,
                                    max_value=10,
                                    step=1,
                                    disabled=manual_disabled,
                                    input_style=input_style,
                                ),
                            ],
                            className="wizard-two-col-item",
                        ),
                    ],
                    className="wizard-two-col",
                ),
            ],
            style=STEP_CONTENT_STYLE,
        )

    elif step == 3:
        body = html.Div(
            [
                html.H4("Training setup"),
                html.P("Define cross-validation and hyperparameter tuning settings."),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5("Cross-validation settings", style={"marginTop": "0", "marginBottom": "16px"}),
                                _compact_number_row(
                                    "K",
                                    "wizard-outer-k-fold",
                                    "#Outer K-fold CV",
                                    param_abr="K_out",
                                    value=form_data.get("outer_k_fold"),
                                    min_value=2,
                                    max_value=10,
                                    step=1,
                                ),
                                _compact_number_row(
                                    "K",
                                    "wizard-inner-k-fold",
                                    "#Inner K-fold",
                                    param_abr="K_in",
                                    value=form_data.get("inner_k_fold"),
                                    min_value=2,
                                    max_value=10,
                                    step=1,
                                ),
                            ],
                            className="wizard-two-col-item",
                        ),
                        html.Div(
                            [
                                html.H5("Hyperparameter tuning settings", style={"marginTop": "0", "marginBottom": "16px"}),
                                _compact_number_row(
                                    "n",
                                    "wizard-trials",
                                    "# Number of trials",
                                    param_abr="N_trials",
                                    value=form_data.get("trials", settings.get("trials", 4)),
                                    min_value=1,
                                    max_value=256,
                                    step=8,
                                ),
                                _compact_number_row(
                                    "Sec.",
                                    "wizard-tuning-time-limit",
                                    "Tuning time limit",
                                    param_abr="T_limit",
                                    value=form_data.get("tuning_time_limit"),
                                    min_value=1,
                                    max_value=7200,
                                    step=60,
                                ),
                            ],
                            className="wizard-two-col-item",
                        ),
                    ],
                    className="wizard-two-col",
                ),
            ],
            style=STEP_CONTENT_STYLE,
        )

    elif step == 4:
        body = html.Div(
            [
                html.H4("Dataset"),
                html.P("Select the dataset to use for training."),
                _compact_dropdown_row(
                    dcc.Dropdown(
                        id="wizard-dataset",
                        options=[
                            {"label": "Choose Dataset", "value": "choose_dataset"},
                            {"label": "ncmpss", "value": "ncmpss"},
                            {"label": "btry", "value": "btry"},
                            {"label": "phm2008", "value": "phm2008"},
                            {"label": "cmapss", "value": "cmapss"},
                            {"label": "phm", "value": "phm"},
                        ],
                        value=form_data.get("dataset"),
                        clearable=True,
                        placeholder="Choose dataset",
                        persistence=True,
                        persistence_type="session",
                    )
                ),
            ],
            style=STEP_CONTENT_STYLE,
        )

    elif step == 5:
        effective = dict(settings)
        effective.update(form_data or {})

        slicer_output_map = {
            0: "Choose slicer output",
            1: "Flatten all time-window features",
            2: "Average of each feature over the time window",
            3: "Standard deviation of each feature over the time window",
            4: "Average and standard deviation per feature",
            5: "Flattened features plus summary statistics",
        }

        dataset_map = {
            "choose_dataset": "Choose dataset",
            "ncmpss": "ncmpss",
            "btry": "btry",
            "phm2008": "phm2008",
            "cmapss": "cmapss",
            "phm": "phm",
        }

        preview_rows = [
            html.H4("Preview"),
            html.P("Review all settings before starting training."),
            html.H5("Data manipulation", style={"marginTop": "16px"}),
            _summary_row("Slice window", effective.get("slice_window")),
            _summary_row(
                "Time-window representation",
                slicer_output_map.get(effective.get("slicer_output"), effective.get("slicer_output")),
            ),
            html.H5("Cost function", style={"marginTop": "20px"}),
            _summary_row(
                "Cost setting mode",
                "Instance-dependent cost-sensitive" if effective.get("instance_dependent_cost") else "Standard",
            ),
        ]

        if not effective.get("instance_dependent_cost"):
            preview_rows.extend(
                [
                    _summary_row("Predictive maintenance cost", effective.get("cost_predictive")),
                    _summary_row("Reactive maintenance cost", effective.get("cost_reactive")),
                    _summary_row("Early maintenance penalty", effective.get("early_penalty")),
                    _summary_row("Late maintenance penalty", effective.get("late_penalty")),
                    _summary_row("Lead time", effective.get("lead_time")),
                    _summary_row("Importance weight", effective.get("cost_weight")),
                ]
            )
        else:
            preview_rows.append(
                html.Div(
                    "Manual cost and lead-time fields are ignored because instance-dependent cost-sensitive mode is selected.",
                    style={
                        "padding": "10px 0",
                        "color": "#6B7280",
                    },
                )
            )

        preview_rows.extend(
            [
                html.H5("Training setup", style={"marginTop": "20px"}),
                _summary_row("Outer K-fold", effective.get("outer_k_fold")),
                _summary_row("Inner K-fold", effective.get("inner_k_fold")),
                _summary_row("Number of trials", effective.get("trials")),
                _summary_row("Tuning time limit (seconds)", effective.get("tuning_time_limit")),
                html.H5("Dataset", style={"marginTop": "20px"}),
                _summary_row("Dataset", dataset_map.get(effective.get("dataset"), effective.get("dataset"))),
            ]
        )

        body = html.Div(preview_rows, style=STEP_CONTENT_STYLE)

    else:
        body = html.Div("Unknown step", style=STEP_CONTENT_STYLE)

    return html.Div(
        [
            html.Div(f"Step {step} of 5", style=STEP_BADGE_STYLE),
            html.H3(titles.get(step, "Wizard")),
            body,
        ]
    )


def build_training_wizard():
    return html.Div(
        id="train-wizard-modal",
        style=MODAL_OVERLAY_HIDDEN,
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("New model training", className="wizard-title"),
                            html.Button("✕", id="btn-close-wizard", n_clicks=0, className="wizard-close-btn"),
                        ],
                        className="wizard-header",
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "marginBottom": "12px",
                            "flexShrink": 0,
                            "gap": "12px",
                        },
                    ),
                    html.Div(
                        id="train-wizard-body",
                        style=WIZARD_BODY_SCROLL_STYLE,
                    ),
                    html.Div(
                        [
                            html.Button("Back", id="btn-wizard-back", n_clicks=0, className="wizard-nav-btn"),
                            html.Div(
                                [
                                    html.Button("Next", id="btn-wizard-next", n_clicks=0, className="wizard-nav-btn"),
                                    html.Button(
                                        "Start training",
                                        id="btn-wizard-finish",
                                        n_clicks=0,
                                        className="wizard-nav-btn wizard-nav-btn-primary",
                                        style={"marginLeft": "8px"},
                                    ),
                                ],
                                className="wizard-nav-actions",
                            ),
                        ],
                        className="wizard-nav-row",
                        style=NAV_BTN_ROW_STYLE,
                    ),
                ],
                className="wizard-modal-content",
                style=MODAL_CONTENT_STYLE,
            )
        ],
    )


def register_training_wizard_callbacks(
    app,
    *,
    settings_slug,
    train_new_model,
    MODELS_DIR,
    VISIBLE_CARD_STYLE,
    settings_default,
    VALIDATION_CSV,
):
    @app.callback(
        Output("train-wizard-open", "data"),
        Input("btn-train-new", "n_clicks"),
        Input("btn-close-wizard", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_close_training_wizard(n_open, n_close):
        trig = ctx.triggered_id
        if trig == "btn-train-new":
            return True
        if trig == "btn-close-wizard":
            return False
        raise PreventUpdate

    @app.callback(
        Output("train-wizard-step", "data"),
        Input("btn-train-new", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_training_wizard_step(n_open):
        return 1

    @app.callback(
        Output("train-wizard-step", "data", allow_duplicate=True),
        Input("btn-wizard-next", "n_clicks"),
        Input("btn-wizard-back", "n_clicks"),
        State("train-wizard-step", "data"),
        prevent_initial_call=True,
    )
    def move_training_wizard(n_next, n_back, step):
        step = step or 1
        trig = ctx.triggered_id

        if trig == "btn-wizard-next":
            return min(5, step + 1)
        if trig == "btn-wizard-back":
            return max(1, step - 1)

        raise PreventUpdate

    @app.callback(
        Output("train-wizard-modal", "style"),
        Input("train-wizard-open", "data"),
    )
    def display_training_wizard(is_open):
        return MODAL_OVERLAY_VISIBLE if is_open else MODAL_OVERLAY_HIDDEN

    @app.callback(
        Output("train-wizard-body", "children"),
        Input("train-wizard-step", "data"),
        State("settings-store", "data"),
        State("train-wizard-form-store", "data"),
    )
    def render_training_wizard_body(step, s, form_data):
        return render_training_step(
            step or 1,
            s or settings_default,
            form_data or {},
            VALIDATION_CSV,
        )

    @app.callback(
        Output("btn-wizard-back", "style"),
        Output("btn-wizard-next", "style"),
        Output("btn-wizard-finish", "style"),
        Input("train-wizard-step", "data"),
    )
    def update_wizard_nav_buttons(step):
        step = step or 1

        back_style = {}
        next_style = {}
        finish_style = {"marginLeft": "8px"}

        if step == 1:
            back_style["visibility"] = "hidden"
        if step == 5:
            next_style["display"] = "none"
        else:
            finish_style["display"] = "none"

        return back_style, next_style, finish_style

    @app.callback(
        Output("train-wizard-form-store", "data"),
        Input("wizard-slice-window", "value"),
        Input("wizard-slicer-output", "value"),
        Input("wizard-instance-dependent-cost", "value"),
        Input("wizard-cost-predictive", "value"),
        Input("wizard-cost-reactive", "value"),
        Input("wizard-early-penalty", "value"),
        Input("wizard-late-penalty", "value"),
        Input("wizard-lead-time", "value"),
        Input("wizard-cost-weight", "value"),
        Input("wizard-outer-k-fold", "value"),
        Input("wizard-inner-k-fold", "value"),
        Input("wizard-trials", "value"),
        Input("wizard-tuning-time-limit", "value"),
        Input("wizard-dataset", "value"),
        State("train-wizard-form-store", "data"),
        prevent_initial_call=False,
    )
    def persist_wizard_form_data(
        wizard_slice_window,
        wizard_slicer_output,
        wizard_instance_dependent_cost,
        wizard_cost_predictive,
        wizard_cost_reactive,
        wizard_early_penalty,
        wizard_late_penalty,
        wizard_lead_time,
        wizard_cost_weight,
        wizard_outer_k_fold,
        wizard_inner_k_fold,
        wizard_trials,
        wizard_tuning_time_limit,
        wizard_dataset,
        form_data,
    ):
        form_data = dict(form_data or {})

        if wizard_slice_window is not None:
            form_data["slice_window"] = wizard_slice_window

        if wizard_slicer_output not in (None, 0):
            form_data["slicer_output"] = wizard_slicer_output
        else:
            form_data.pop("slicer_output", None)

        if wizard_instance_dependent_cost is not None:
            form_data["instance_dependent_cost"] = bool(wizard_instance_dependent_cost)

        if wizard_cost_predictive is not None:
            form_data["cost_predictive"] = wizard_cost_predictive
        if wizard_cost_reactive is not None:
            form_data["cost_reactive"] = wizard_cost_reactive
        if wizard_early_penalty is not None:
            form_data["early_penalty"] = wizard_early_penalty
        if wizard_late_penalty is not None:
            form_data["late_penalty"] = wizard_late_penalty
        if wizard_lead_time is not None:
            form_data["lead_time"] = wizard_lead_time
        if wizard_cost_weight is not None:
            form_data["cost_weight"] = wizard_cost_weight

        if wizard_outer_k_fold is not None:
            form_data["outer_k_fold"] = wizard_outer_k_fold
        if wizard_inner_k_fold is not None:
            form_data["inner_k_fold"] = wizard_inner_k_fold
        if wizard_trials is not None:
            form_data["trials"] = wizard_trials
        if wizard_tuning_time_limit is not None:
            form_data["tuning_time_limit"] = wizard_tuning_time_limit

        if wizard_dataset not in (None, "choose_dataset"):
            form_data["dataset"] = wizard_dataset
        else:
            form_data.pop("dataset", None)

        return form_data

    @app.callback(
        Output("selected-model-store", "data", allow_duplicate=True),
        Output("selected-model-text", "children", allow_duplicate=True),
        Output("card-loading-training", "style", allow_duplicate=True),
        Output("card-metrics", "style", allow_duplicate=True),
        Output("card-graphs", "style", allow_duplicate=True),
        Output("main-placeholder", "style", allow_duplicate=True),
        Output("run-token", "data", allow_duplicate=True),
        Output("train-wizard-open", "data", allow_duplicate=True),
        Output("train-wizard-form-store", "data", allow_duplicate=True),
        Input("btn-wizard-finish", "n_clicks"),
        State("settings-store", "data"),
        State("train-wizard-form-store", "data"),
        prevent_initial_call=True,
    )
    def finish_training_from_wizard(n_finish, s, form_data):
        if not n_finish or not s:
            raise PreventUpdate

        form_data = dict(form_data or {})
        s = dict(s or {})

        form_data["dataset"] = form_data.get("dataset", s.get("dataset"))
        instance_dependent = form_data.get("instance_dependent_cost", False)

        for key in [
            "slice_window",
            "slicer_output",
            "instance_dependent_cost",
            "outer_k_fold",
            "inner_k_fold",
            "trials",
            "tuning_time_limit",
            "dataset",
        ]:
            if key in form_data and form_data[key] is not None:
                s[key] = form_data[key]

        if not instance_dependent:
            for key in [
                "cost_predictive",
                "cost_reactive",
                "early_penalty",
                "late_penalty",
                "lead_time",
                "cost_weight",
            ]:
                if key in form_data and form_data[key] is not None:
                    s[key] = form_data[key]

        assets_path = Path(MODELS_DIR)
        desired = settings_slug(s)
        desired_path = assets_path / desired
        desired_path.mkdir(parents=True, exist_ok=True)

        config = build_training_config(
            form_data=form_data,
            save_dir=desired_path,
            experiment_name=desired,
        )
        write_training_config(config=config, save_dir=desired_path)

        train_new_model(
            target_settings=s,
            save_to=str(desired_path),
        )

        data = {"model_name": desired, "source": "trained", "match_percent": 100.0}
        selected_text = f"{desired} (trained, 100.0%)"

        return (
            data,
            selected_text,
            VISIBLE_CARD_STYLE,
            VISIBLE_CARD_STYLE,
            {**VISIBLE_CARD_STYLE, "marginBottom": "0px"},
            {"display": "none"},
            str(uuid.uuid4()),
            False,
            form_data,
        )

    @app.callback(
        Output("wizard-cost-predictive", "disabled"),
        Output("wizard-cost-reactive", "disabled"),
        Output("wizard-early-penalty", "disabled"),
        Output("wizard-late-penalty", "disabled"),
        Output("wizard-lead-time", "disabled"),
        Output("wizard-cost-weight", "disabled"),
        Output("wizard-cost-predictive", "style"),
        Output("wizard-cost-reactive", "style"),
        Output("wizard-early-penalty", "style"),
        Output("wizard-late-penalty", "style"),
        Output("wizard-lead-time", "style"),
        Output("wizard-cost-weight", "style"),
        Output("title-maintenance-cost", "style"),
        Output("title-fleet-characteristics", "style"),
        Input("wizard-instance-dependent-cost", "value"),
        prevent_initial_call=False,
    )
    def toggle_instance_dependent_cost_fields(instance_value):
        instance_dependent = bool(instance_value)

        enabled_style = {}
        disabled_style = {
            "backgroundColor": "#F3F4F6",
            "color": "#9CA3AF",
            "cursor": "not-allowed",
        }

        section_enabled = {"marginTop": "0"}
        section_disabled = {"marginTop": "0", "color": "#9CA3AF"}

        field_style = disabled_style if instance_dependent else enabled_style
        heading_style = section_disabled if instance_dependent else section_enabled

        return (
            instance_dependent,
            instance_dependent,
            instance_dependent,
            instance_dependent,
            instance_dependent,
            instance_dependent,
            field_style,
            field_style,
            field_style,
            field_style,
            field_style,
            field_style,
            heading_style,
            heading_style,
        )

    @app.callback(
        Output("wizard-slice-window-progress", "value"),
        Output("wizard-slice-window-progress", "color"),
        Input("wizard-slice-window", "value"),
        prevent_initial_call=False,
    )
    def update_step1_progress(value):
        if value is None:
            p = 0
        else:
            value = max(1, min(value, 100))
            p = (value - 1) / (100 - 1) * 100

        if p < 25:
            color = "warning"
        elif p < 75:
            color = "success"
        else:
            color = "danger"

        return p, color

    @app.callback(
        Output("wizard-cost-predictive-progress", "value"),
        Output("wizard-cost-predictive-progress", "color"),
        Output("wizard-cost-reactive-progress", "value"),
        Output("wizard-cost-reactive-progress", "color"),
        Output("wizard-early-penalty-progress", "value"),
        Output("wizard-early-penalty-progress", "color"),
        Output("wizard-late-penalty-progress", "value"),
        Output("wizard-late-penalty-progress", "color"),
        Output("wizard-lead-time-progress", "value"),
        Output("wizard-lead-time-progress", "color"),
        Output("wizard-cost-weight-progress", "value"),
        Output("wizard-cost-weight-progress", "color"),
        Input("wizard-cost-predictive", "value"),
        Input("wizard-cost-reactive", "value"),
        Input("wizard-early-penalty", "value"),
        Input("wizard-late-penalty", "value"),
        Input("wizard-lead-time", "value"),
        Input("wizard-cost-weight", "value"),
        prevent_initial_call=False,
    )
    def update_step2_progress(
        cost_predictive,
        cost_reactive,
        early_penalty,
        late_penalty,
        lead_time,
        cost_weight,
    ):
        def pct(value, min_v, max_v):
            if value is None:
                return 0
            value = max(min_v, min(value, max_v))
            if max_v == min_v:
                return 0
            return (value - min_v) / (max_v - min_v) * 100

        def bar_color(p):
            if p < 25:
                return "warning"
            elif p < 75:
                return "success"
            return "danger"

        p1 = pct(cost_predictive, 0, 100)
        p2 = pct(cost_reactive, 0, 10000)
        p3 = pct(early_penalty, 0, 10)
        p4 = pct(late_penalty, 0, 100)
        p5 = pct(lead_time, 0, 365)
        p6 = pct(cost_weight, 0, 10)

        return (
            p1, bar_color(p1),
            p2, bar_color(p2),
            p3, bar_color(p3),
            p4, bar_color(p4),
            p5, bar_color(p5),
            p6, bar_color(p6),
        )

    @app.callback(
        Output("wizard-outer-k-fold-progress", "value"),
        Output("wizard-outer-k-fold-progress", "color"),
        Output("wizard-inner-k-fold-progress", "value"),
        Output("wizard-inner-k-fold-progress", "color"),
        Output("wizard-trials-progress", "value"),
        Output("wizard-trials-progress", "color"),
        Output("wizard-tuning-time-limit-progress", "value"),
        Output("wizard-tuning-time-limit-progress", "color"),
        Input("wizard-outer-k-fold", "value"),
        Input("wizard-inner-k-fold", "value"),
        Input("wizard-trials", "value"),
        Input("wizard-tuning-time-limit", "value"),
        prevent_initial_call=False,
    )
    def update_step3_progress(
        outer_k_fold,
        inner_k_fold,
        trials,
        tuning_time_limit,
    ):
        def pct(value, min_v, max_v):
            if value is None:
                return 0
            value = max(min_v, min(value, max_v))
            if max_v == min_v:
                return 0
            return (value - min_v) / (max_v - min_v) * 100

        def bar_color(p):
            if p < 25:
                return "warning"
            elif p < 75:
                return "success"
            return "danger"

        p1 = pct(outer_k_fold, 2, 10)
        p2 = pct(inner_k_fold, 2, 10)
        p3 = pct(trials, 1, 256)
        p4 = pct(tuning_time_limit, 1, 7200)

        return (
            p1, bar_color(p1),
            p2, bar_color(p2),
            p3, bar_color(p3),
            p4, bar_color(p4),
        )