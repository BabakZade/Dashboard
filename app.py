# app.py
import os
import gc

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from pages import (
    home,
    cost_function,
    data_simulator,
    rul_distribution,
    cost_sensitive_model,
    benchmark,
    maintenance_planning,
)

external_stylesheets = [
    dbc.themes.LUX,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css",
]

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets,
    assets_folder="assets",
)

server = app.server
app.title = "Cost-sensitive predictive maintenance"


@server.route("/health")
def health():
    return "ok", 200


ROUTES = {
    "/": ("Home", home.layout),
    "/cost-function": ("Cost function", cost_function.layout),
    "/data-simulator": ("Data Simulator", data_simulator.layout),
    "/rul-distribution": ("RUL distribution", rul_distribution.layout),
    "/cost-sensitive-model": ("Cost sensitive model", cost_sensitive_model.layout),
    "/maintenance-planning": ("Maintenance planning", maintenance_planning.layout),
    "/benchmark": ("Benchmark", benchmark.layout),
}

ICONS = {
    "/": "fa-solid fa-house-laptop",
    "/cost-function": "fa-solid fa-file-contract",
    "/data-simulator": "fa-solid fa-database",
    "/rul-distribution": "fa-solid fa-chart-area",
    "/cost-sensitive-model": "fa-solid fa-filter-circle-dollar",
    "/maintenance-planning": "fa-solid fa-screwdriver-wrench",
    "/benchmark": "fa-solid fa-chart-pie",
}

DATA_ROWS = benchmark.DATA_ROWS
OUT_ROWS = benchmark.OUT_ROWS


def clear_rul_distribution_memory():
    """
    Do not edit pages/rul_distribution.py, but when the user leaves that page,
    clear its module-level caches from here.
    """
    try:
        if hasattr(rul_distribution, "_ROLLING_CACHE"):
            rul_distribution._ROLLING_CACHE.clear()

        # Clear dataset cache too when leaving the page.
        # If you want faster reloads, comment out this block.
        if hasattr(rul_distribution, "_CACHE"):
            rul_distribution._CACHE.clear()

        gc.collect()
    except Exception:
        pass


def nav_link(label, href, icon_class):
    return dbc.NavLink(
        [
            html.I(className=icon_class, style={"marginRight": "10px"}),
            label,
        ],
        href=href,
        active="exact",
        className="mb-1 rounded",
    )


def sidebar_style(is_open: bool):
    return {
        "width": "280px" if is_open else "0px",
        "minWidth": "280px" if is_open else "0px",
        "maxWidth": "280px" if is_open else "0px",
        "transition": "all 0.25s ease",
        "overflowX": "hidden",
        "overflowY": "auto",
        "borderRight": "1px solid #dee2e6" if is_open else "none",
        "backgroundColor": "white",
        "height": "calc(100vh - 64px)",
        "position": "sticky",
        "top": "64px",
        "flexShrink": 0,
    }


def content_style():
    return {
        "flex": "1 1 auto",
        "minWidth": 0,
        "padding": "1rem",
        "transition": "all 0.25s ease",
    }


def serve_layout():
    return dbc.Container(
        fluid=True,
        className="px-0",
        children=[
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="menu_open", data=True),

            dcc.Store(id="bench_data_store", data=DATA_ROWS),
            dcc.Store(id="bench_out_store", data=OUT_ROWS),
            dcc.Store(id="shared-inputs", storage_type="session"),

            dbc.Navbar(
                dbc.Container(
                    fluid=True,
                    className="d-flex align-items-center",
                    children=[
                        dbc.Button(
                            html.I(className="fa-solid fa-bars"),
                            id="menu_btn",
                            n_clicks=0,
                            color="secondary",
                            outline=True,
                            className="me-3",
                        ),
                        html.Div(id="page_title", className="fw-bold fs-5"),
                    ],
                ),
                color="light",
                dark=False,
                sticky="top",
                className="border-bottom shadow-sm",
                style={"minHeight": "64px"},
            ),

            html.Div(
                id="app_shell",
                style={
                    "display": "flex",
                    "width": "100%",
                    "minHeight": "calc(100vh - 64px)",
                },
                children=[
                    html.Div(
                        id="sidebar",
                        style=sidebar_style(True),
                        children=[
                            html.Div(
                                [
                                    html.Div("Menu", className="fw-bold mb-3"),
                                    dbc.Nav(
                                        [
                                            nav_link("Home", "/", ICONS["/"]),
                                            nav_link("Cost function", "/cost-function", ICONS["/cost-function"]),
                                            nav_link("Data Simulator", "/data-simulator", ICONS["/data-simulator"]),
                                            nav_link("RUL distribution", "/rul-distribution", ICONS["/rul-distribution"]),
                                            nav_link("Cost sensitive model", "/cost-sensitive-model", ICONS["/cost-sensitive-model"]),
                                            nav_link("Maintenance planning", "/maintenance-planning", ICONS["/maintenance-planning"]),
                                            nav_link("Benchmark", "/benchmark", ICONS["/benchmark"]),
                                        ],
                                        vertical=True,
                                        pills=True,
                                        className="flex-column",
                                    ),
                                ],
                                className="p-3",
                            )
                        ],
                    ),

                    html.Div(
                        id="content",
                        style=content_style(),
                        children=[html.Div(id="page_content")],
                    ),
                ],
            ),
        ],
    )


app.layout = serve_layout


@app.callback(
    Output("menu_open", "data"),
    Input("menu_btn", "n_clicks"),
    State("menu_open", "data"),
    prevent_initial_call=True,
)
def toggle_menu(_, is_open):
    return not bool(is_open)


@app.callback(
    Output("sidebar", "style"),
    Input("menu_open", "data"),
)
def update_sidebar(is_open):
    return sidebar_style(bool(is_open))


@app.callback(
    Output("page_title", "children"),
    Input("url", "pathname"),
)
def set_title(pathname):
    if not pathname:
        pathname = "/"

    label = ROUTES.get(pathname, ("Unknown page", None))[0]
    icon_class = ICONS.get(pathname, "fa-solid fa-circle")

    return html.Div(
        [
            html.I(className=icon_class, style={"marginRight": "10px"}),
            label,
        ],
        className="d-flex align-items-center",
    )


@app.callback(
    Output("page_content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    if not pathname:
        pathname = "/"

    # If the user leaves the heavy page, clear its memory.
    if pathname != "/rul-distribution":
        clear_rul_distribution_memory()

    if pathname not in ROUTES:
        return dbc.Alert("404 — Page not found", color="warning")

    try:
        return ROUTES[pathname][1]()
    except Exception as e:
        return dbc.Alert(
            [
                html.H5("This page could not be loaded."),
                html.Div("The Python error was:"),
                html.Pre(str(e), style={"whiteSpace": "pre-wrap"}),
            ],
            color="danger",
        )


home.register_callbacks(app)
cost_function.register_callbacks(app)
data_simulator.register_callbacks(app)
rul_distribution.register_callbacks(app)
cost_sensitive_model.register_callbacks(app)
maintenance_planning.register_callbacks(app)
benchmark.register_callbacks(app)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)