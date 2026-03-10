from dash import html, dcc, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from pathlib import Path
import uuid

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
    "width": "900px",
    "maxWidth": "95vw",
    "height": "760px",
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


def _summary_row(label, value):
    return html.Div(
        [
            html.Div(label, style={"fontWeight": 600, "minWidth": "220px"}),
            html.Div(str(value if value is not None else "-")),
        ],
        style={
            "display": "flex",
            "gap": "12px",
            "padding": "8px 0",
            "borderBottom": "1px solid #E5E7EB",
        },
    )


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
                html.Label("Slice window"),
                dcc.Input(
                    id="wizard-slice-window",
                    type="number",
                    value=form_data.get("slice_window", settings.get("slice_window", 2)),
                    min=1,
                    step=1,
                    style={"width": "100%", "marginBottom": "12px"},
                ),
                html.Label("Time-window representation"),
                dcc.Dropdown(
                    id="wizard-slicer-output",
                    options=[
                        {"label": "Flatten all time-window features", "value": 1},
                        {"label": "Average of each feature over the time window", "value": 2},
                        {"label": "Standard deviation of each feature over the time window", "value": 3},
                        {"label": "Average and standard deviation per feature", "value": 4},
                        {"label": "Flattened features plus summary statistics", "value": 5},
                    ],
                    value=form_data.get("slicer_output", settings.get("slicer_output", 1)),
                    clearable=False,
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

        input_style = {
            "width": "100%",
            "marginBottom": "12px",
        }
        input_style_last = {
            "width": "100%",
            "marginBottom": "0px",
        }

        if manual_disabled:
            input_style = {
                **input_style,
                "backgroundColor": "#F3F4F6",
                "color": "#9CA3AF",
                "cursor": "not-allowed",
            }
            input_style_last = {
                **input_style_last,
                "backgroundColor": "#F3F4F6",
                "color": "#9CA3AF",
                "cursor": "not-allowed",
            }

        label_style = {"color": "#9CA3AF"} if manual_disabled else {}
        section_style = {"marginTop": "0", "color": "#9CA3AF"} if manual_disabled else {"marginTop": "0"}

        body = html.Div(
            [
                html.H4("Cost function"),
                html.P("Define maintenance costs and fleet-related settings."),
                html.Label("Cost setting mode"),
                dcc.Checklist(
                    id="wizard-instance-dependent-cost",
                    options=[
                        {
                            "label": " Instance-dependent cost-sensitive",
                            "value": True,
                        }
                    ],
                    value=[True] if instance_dependent else [],
                    inputStyle={"marginRight": "8px"},
                    style={"marginBottom": "6px"},
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
                                html.Label(
                                    "Predictive maintenance cost",
                                    id="label-cost-predictive",
                                    style=label_style,
                                ),
                                dcc.Input(
                                    id="wizard-cost-predictive",
                                    type="number",
                                    value=form_data.get("cost_predictive", settings.get("cost_predictive", 20)),
                                    min=0,
                                    step=1,
                                    disabled=manual_disabled,
                                    style=input_style,
                                ),
                                html.Label(
                                    "Reactive maintenance cost",
                                    id="label-cost-reactive",
                                    style=label_style,
                                ),
                                dcc.Input(
                                    id="wizard-cost-reactive",
                                    type="number",
                                    value=form_data.get("cost_reactive", settings.get("cost_reactive", 200)),
                                    min=0,
                                    step=1,
                                    disabled=manual_disabled,
                                    style=input_style,
                                ),
                                html.Label(
                                    "Early maintenance penalty",
                                    id="label-early-penalty",
                                    style=label_style,
                                ),
                                dcc.Input(
                                    id="wizard-early-penalty",
                                    type="number",
                                    value=form_data.get("early_penalty", settings.get("early_penalty", 1)),
                                    min=0,
                                    step=1,
                                    disabled=manual_disabled,
                                    style=input_style,
                                ),
                                html.Label(
                                    "Late maintenance penalty",
                                    id="label-late-penalty",
                                    style=label_style,
                                ),
                                dcc.Input(
                                    id="wizard-late-penalty",
                                    type="number",
                                    value=form_data.get("late_penalty", settings.get("late_penalty", 10)),
                                    min=0,
                                    step=1,
                                    disabled=manual_disabled,
                                    style=input_style,
                                ),
                                html.Label(
                                    "Emergency maintenance penalty",
                                    id="label-emergency-penalty",
                                    style=label_style,
                                ),
                                dcc.Input(
                                    id="wizard-emergency-penalty",
                                    type="number",
                                    value=form_data.get("emergency_penalty", settings.get("emergency_penalty", 10)),
                                    min=0,
                                    step=1,
                                    disabled=manual_disabled,
                                    style=input_style_last,
                                ),
                            ],
                            style={
                                "flex": "1",
                                "minWidth": "280px",
                                "opacity": 0.65 if manual_disabled else 1,
                            },
                        ),
                        html.Div(
                            [
                                html.H5(
                                    "Fleet characteristics",
                                    id="title-fleet-characteristics",
                                    style=section_style,
                                ),
                                html.Label(
                                    "Lead time",
                                    id="label-lead-time",
                                    style=label_style,
                                ),
                                dcc.Input(
                                    id="wizard-lead-time",
                                    type="number",
                                    value=form_data.get("lead_time", settings.get("lead_time", 1)),
                                    min=0,
                                    step=1,
                                    disabled=manual_disabled,
                                    style=input_style,
                                ),
                                html.Label(
                                    "Importance weight",
                                    id="label-cost-weight",
                                    style=label_style,
                                ),
                                dcc.Input(
                                    id="wizard-cost-weight",
                                    type="number",
                                    value=form_data.get("cost_weight", settings.get("cost_weight", 1.0)),
                                    min=0,
                                    step=0.1,
                                    disabled=manual_disabled,
                                    style=input_style_last,
                                ),
                            ],
                            style={
                                "flex": "1",
                                "minWidth": "280px",
                                "opacity": 0.65 if manual_disabled else 1,
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "24px",
                        "alignItems": "flex-start",
                        "flexWrap": "wrap",
                    },
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
                                html.H5("Cross-validation", style={"marginTop": "0"}),
                                html.Label("Outer K-fold"),
                                dcc.Input(
                                    id="wizard-outer-k-fold",
                                    type="number",
                                    value=form_data.get("outer_k_fold", settings.get("outer_k_fold", 5)),
                                    min=2,
                                    step=1,
                                    style={"width": "100%", "marginBottom": "12px"},
                                ),
                                html.Label("Inner K-fold"),
                                dcc.Input(
                                    id="wizard-inner-k-fold",
                                    type="number",
                                    value=form_data.get("inner_k_fold", settings.get("inner_k_fold", 10)),
                                    min=2,
                                    step=1,
                                    style={"width": "100%"},
                                ),
                            ],
                            style={
                                "flex": "1",
                                "minWidth": "280px",
                            },
                        ),
                        html.Div(
                            [
                                html.H5("Hyperparameter tuning", style={"marginTop": "0"}),
                                html.Label("Number of trials"),
                                dcc.Input(
                                    id="wizard-trials",
                                    type="number",
                                    value=form_data.get("trials", settings.get("trials", 64)),
                                    min=1,
                                    step=1,
                                    style={"width": "100%", "marginBottom": "12px"},
                                ),
                                html.Label("Tuning time limit (seconds)"),
                                dcc.Input(
                                    id="wizard-tuning-time-limit",
                                    type="number",
                                    value=form_data.get("tuning_time_limit", settings.get("tuning_time_limit", 1800)),
                                    min=1,
                                    step=1,
                                    style={"width": "100%"},
                                ),
                            ],
                            style={
                                "flex": "1",
                                "minWidth": "280px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "24px",
                        "alignItems": "flex-start",
                        "flexWrap": "wrap",
                    },
                ),
            ],
            style=STEP_CONTENT_STYLE,
        )

    elif step == 4:
        body = html.Div(
            [
                html.H4("Dataset"),
                html.P("Select the dataset to use for training."),
                html.Label("Dataset"),
                dcc.Dropdown(
                    id="wizard-dataset",
                    options=[
                        {"label": "ncmpss", "value": "ncmpss"},
                        {"label": "btry", "value": "btry"},
                        {"label": "phm2008", "value": "phm2008"},
                        {"label": "cmapss", "value": "cmapss"},
                        {"label": "phm", "value": "phm"},
                    ],
                    value=form_data.get("dataset", settings.get("dataset", "cmapss")),
                    clearable=False,
                    style={"width": "100%"},
                ),
            ],
            style=STEP_CONTENT_STYLE,
        )

    elif step == 5:
        effective = dict(settings)
        effective.update(form_data or {})

        slicer_output_map = {
            1: "Flatten all time-window features",
            2: "Average of each feature over the time window",
            3: "Standard deviation of each feature over the time window",
            4: "Average and standard deviation per feature",
            5: "Flattened features plus summary statistics",
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
                    _summary_row("Emergency maintenance penalty", effective.get("emergency_penalty")),
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
                _summary_row("Dataset", effective.get("dataset")),
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
                            html.H2("New model training"),
                            html.Button("✕", id="btn-close-wizard", n_clicks=0),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "marginBottom": "12px",
                            "flexShrink": 0,
                        },
                    ),
                    html.Div(
                        id="train-wizard-body",
                        style=WIZARD_BODY_SCROLL_STYLE,
                    ),
                    html.Div(
                        [
                            html.Button("Back", id="btn-wizard-back", n_clicks=0),
                            html.Div(
                                [
                                    html.Button("Next", id="btn-wizard-next", n_clicks=0),
                                    html.Button(
                                        "Start training",
                                        id="btn-wizard-finish",
                                        n_clicks=0,
                                        style={"marginLeft": "8px"},
                                    ),
                                ],
                            ),
                        ],
                        style=NAV_BTN_ROW_STYLE,
                    ),
                ],
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
        Output("train-wizard-form-store", "data", allow_duplicate=True),
        Input("btn-wizard-next", "n_clicks"),
        Input("btn-wizard-back", "n_clicks"),
        Input("btn-close-wizard", "n_clicks"),
        State("train-wizard-step", "data"),
        State("train-wizard-form-store", "data"),
        State("wizard-slice-window", "value"),
        State("wizard-slicer-output", "value"),
        State("wizard-instance-dependent-cost", "value"),
        State("wizard-cost-predictive", "value"),
        State("wizard-cost-reactive", "value"),
        State("wizard-early-penalty", "value"),
        State("wizard-late-penalty", "value"),
        State("wizard-emergency-penalty", "value"),
        State("wizard-lead-time", "value"),
        State("wizard-cost-weight", "value"),
        State("wizard-outer-k-fold", "value"),
        State("wizard-inner-k-fold", "value"),
        State("wizard-trials", "value"),
        State("wizard-tuning-time-limit", "value"),
        State("wizard-dataset", "value"),
        prevent_initial_call=True,
    )
    def persist_wizard_form_data(
        n_next,
        n_back,
        n_close,
        step,
        form_data,
        wizard_slice_window,
        wizard_slicer_output,
        wizard_instance_dependent_cost,
        wizard_cost_predictive,
        wizard_cost_reactive,
        wizard_early_penalty,
        wizard_late_penalty,
        wizard_emergency_penalty,
        wizard_lead_time,
        wizard_cost_weight,
        wizard_outer_k_fold,
        wizard_inner_k_fold,
        wizard_trials,
        wizard_tuning_time_limit,
        wizard_dataset,
    ):
        form_data = dict(form_data or {})
        step = step or 1

        instance_dependent = bool(
            wizard_instance_dependent_cost and True in wizard_instance_dependent_cost
        )

        if step == 1:
            form_data["slice_window"] = wizard_slice_window
            form_data["slicer_output"] = wizard_slicer_output

        elif step == 2:
            form_data["instance_dependent_cost"] = instance_dependent
            form_data["cost_predictive"] = wizard_cost_predictive
            form_data["cost_reactive"] = wizard_cost_reactive
            form_data["early_penalty"] = wizard_early_penalty
            form_data["late_penalty"] = wizard_late_penalty
            form_data["emergency_penalty"] = wizard_emergency_penalty
            form_data["lead_time"] = wizard_lead_time
            form_data["cost_weight"] = wizard_cost_weight

        elif step == 3:
            form_data["outer_k_fold"] = wizard_outer_k_fold
            form_data["inner_k_fold"] = wizard_inner_k_fold
            form_data["trials"] = wizard_trials
            form_data["tuning_time_limit"] = wizard_tuning_time_limit

        elif step == 4:
            form_data["dataset"] = wizard_dataset

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
        State("wizard-dataset", "value"),
        prevent_initial_call=True,
    )
    def finish_training_from_wizard(n_finish, s, form_data, wizard_dataset):
        if not n_finish or not s:
            raise PreventUpdate

        form_data = dict(form_data or {})
        s = dict(s or {})

        form_data["dataset"] = wizard_dataset or form_data.get("dataset", s.get("dataset"))
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
                "emergency_penalty",
                "lead_time",
                "cost_weight",
            ]:
                if key in form_data and form_data[key] is not None:
                    s[key] = form_data[key]

        assets_path = Path(MODELS_DIR)
        desired = settings_slug(s)
        desired_path = assets_path / desired

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
        Output("wizard-emergency-penalty", "disabled"),
        Output("wizard-lead-time", "disabled"),
        Output("wizard-cost-weight", "disabled"),
        Output("wizard-cost-predictive", "style"),
        Output("wizard-cost-reactive", "style"),
        Output("wizard-early-penalty", "style"),
        Output("wizard-late-penalty", "style"),
        Output("wizard-emergency-penalty", "style"),
        Output("wizard-lead-time", "style"),
        Output("wizard-cost-weight", "style"),
        Output("label-cost-predictive", "style"),
        Output("label-cost-reactive", "style"),
        Output("label-early-penalty", "style"),
        Output("label-late-penalty", "style"),
        Output("label-emergency-penalty", "style"),
        Output("label-lead-time", "style"),
        Output("label-cost-weight", "style"),
        Output("title-maintenance-cost", "style"),
        Output("title-fleet-characteristics", "style"),
        Input("wizard-instance-dependent-cost", "value"),
        prevent_initial_call=False,
    )
    def toggle_instance_dependent_cost_fields(instance_value):
        instance_dependent = bool(instance_value and True in instance_value)

        base_enabled = {
            "width": "100%",
            "marginBottom": "12px",
        }
        base_disabled = {
            "width": "100%",
            "marginBottom": "12px",
            "backgroundColor": "#F3F4F6",
            "color": "#9CA3AF",
            "cursor": "not-allowed",
        }

        enabled_last = {
            "width": "100%",
            "marginBottom": "0px",
        }
        disabled_last = {
            "width": "100%",
            "marginBottom": "0px",
            "backgroundColor": "#F3F4F6",
            "color": "#9CA3AF",
            "cursor": "not-allowed",
        }

        label_enabled = {}
        label_disabled = {"color": "#9CA3AF"}

        section_enabled = {"marginTop": "0"}
        section_disabled = {"marginTop": "0", "color": "#9CA3AF"}

        normal_style = base_disabled if instance_dependent else base_enabled
        last_style = disabled_last if instance_dependent else enabled_last
        label_style = label_disabled if instance_dependent else label_enabled
        section_style = section_disabled if instance_dependent else section_enabled

        return (
            instance_dependent,
            instance_dependent,
            instance_dependent,
            instance_dependent,
            instance_dependent,
            instance_dependent,
            instance_dependent,
            normal_style,
            normal_style,
            normal_style,
            normal_style,
            last_style,
            normal_style,
            last_style,
            label_style,
            label_style,
            label_style,
            label_style,
            label_style,
            label_style,
            label_style,
            section_style,
            section_style,
        )