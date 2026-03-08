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
    "minHeight": "520px",
    "maxHeight": "90vh",
    "overflowY": "auto",
    "backgroundColor": "white",
    "borderRadius": "16px",
    "boxShadow": "0 10px 30px rgba(0,0,0,0.2)",
    "padding": "24px",
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
}

STEP_CONTENT_STYLE = {
    "border": "1px solid #E5E7EB",
    "borderRadius": "12px",
    "padding": "20px",
    "marginTop": "12px",
    "backgroundColor": "#FAFAFA",
}


def render_training_step(step: int, settings: dict, form_data: dict, validation_csv: str):
    titles = {
        1: "Problem",
        2: "Cost function",
        3: "Training setup",
        4: "Data",
    }

    form_data = form_data or {}

    if step == 1:
        body = html.Div(
            [
                html.H4("Problem definition"),
                html.P("Describe the prediction task here."),
                html.Label("Problem name"),
                dcc.Input(
                    id="wizard-problem-name",
                    type="text",
                    value=form_data.get("problem_name", "RUL prediction"),
                    style={"width": "100%", "marginBottom": "12px"},
                ),
                html.Label("Target"),
                dcc.Input(
                    id="wizard-target-name",
                    type="text",
                    value=form_data.get("target_name", "rul"),
                    style={"width": "100%"},
                ),
            ],
            style=STEP_CONTENT_STYLE,
        )

    elif step == 2:
        body = html.Div(
            [
                html.H4("Cost function"),
                html.Div(f"Slice window: {settings.get('slice_window')}"),
                html.Div(f"Early penalty: {settings.get('early_penalty')}"),
                html.Div(f"Late penalty: {settings.get('late_penalty')}"),
                html.Div(f"Reactive cost: {settings.get('cost_reactive')}"),
                html.Div(f"Predictive cost: {settings.get('cost_predictive')}"),
            ],
            style=STEP_CONTENT_STYLE,
        )

    elif step == 3:
        body = html.Div(
            [
                html.H4("Training setup"),
                html.Label("Epochs"),
                dcc.Input(
                    id="wizard-epochs",
                    type="number",
                    value=form_data.get("epochs", 50),
                    min=1,
                    step=1,
                ),
                html.Br(),
                html.Br(),
                html.Label("Batch size"),
                dcc.Input(
                    id="wizard-batch-size",
                    type="number",
                    value=form_data.get("batch_size", 32),
                    min=1,
                    step=1,
                ),
            ],
            style=STEP_CONTENT_STYLE,
        )

    elif step == 4:
        body = html.Div(
            [
                html.H4("Data"),
                html.Label("Training file"),
                dcc.Input(
                    id="wizard-data-path",
                    type="text",
                    value=form_data.get("data_path", validation_csv),
                    style={"width": "100%"},
                ),
            ],
            style=STEP_CONTENT_STYLE,
        )

    else:
        body = html.Div("Unknown step", style=STEP_CONTENT_STYLE)

    return html.Div(
        [
            html.Div(f"Step {step} of 4", style=STEP_BADGE_STYLE),
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
                        },
                    ),
                    html.Div(id="train-wizard-body"),
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
            return min(4, step + 1)
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
        if step == 4:
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
        State("wizard-problem-name", "value"),
        State("wizard-target-name", "value"),
        State("wizard-epochs", "value"),
        State("wizard-batch-size", "value"),
        State("wizard-data-path", "value"),
        prevent_initial_call=True,
    )
    def persist_wizard_form_data(
        n_next,
        n_back,
        n_close,
        step,
        form_data,
        wizard_problem_name,
        wizard_target_name,
        wizard_epochs,
        wizard_batch_size,
        wizard_data_path,
    ):
        form_data = dict(form_data or {})
        step = step or 1

        if step == 1:
            form_data["problem_name"] = wizard_problem_name
            form_data["target_name"] = wizard_target_name
        elif step == 3:
            form_data["epochs"] = wizard_epochs
            form_data["batch_size"] = wizard_batch_size
        elif step == 4:
            form_data["data_path"] = wizard_data_path

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
        State("wizard-data-path", "value"),
        prevent_initial_call=True,
    )
    def finish_training_from_wizard(n_finish, s, form_data, wizard_data_path):
        if not n_finish or not s:
            raise PreventUpdate

        form_data = dict(form_data or {})
        form_data["data_path"] = wizard_data_path or form_data.get("data_path", VALIDATION_CSV)

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