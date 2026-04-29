# app.py
import os
import gc
from collections import OrderedDict

import numpy as np
import pandas as pd

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

# =============================================================================
# App setup
# =============================================================================

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


# =============================================================================
# Memory guard for the heavy RUL distribution page
# =============================================================================

RUL_DF_CACHE_ITEMS = int(os.environ.get("RUL_DF_CACHE_ITEMS", "1"))
RUL_ROLLING_CACHE_ITEMS = int(os.environ.get("RUL_ROLLING_CACHE_ITEMS", "1"))
RUL_MAX_POSTERIOR_SAMPLES = int(os.environ.get("RUL_MAX_POSTERIOR_SAMPLES", "100"))


class BoundedCache(OrderedDict):
    """
    Small LRU cache.

    This replaces the normal dictionaries inside pages/rul_distribution.py
    without editing that file.
    """

    def __init__(self, max_items=1):
        super().__init__()
        self.max_items = max(1, int(max_items))

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)

        while len(self) > self.max_items:
            self.popitem(last=False)

        gc.collect()


def compact_rolling_payload(rolling):
    """
    Reduce memory used by the rolling prediction dictionary.

    The biggest object is usually rul_samples_matrix, so this limits the
    posterior samples and downcasts numeric arrays.
    """
    if not isinstance(rolling, dict):
        return rolling

    compact = {}

    for key, value in rolling.items():
        if isinstance(value, pd.Series):
            value = value.to_numpy()

        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)

            if key == "rul_samples_matrix" and arr.ndim == 2:
                if arr.shape[1] > RUL_MAX_POSTERIOR_SAMPLES:
                    sample_idx = np.linspace(
                        0,
                        arr.shape[1] - 1,
                        RUL_MAX_POSTERIOR_SAMPLES,
                    ).astype(int)
                    arr = arr[:, sample_idx]

            if arr.dtype.kind == "f":
                arr = arr.astype(np.float32, copy=False)
            elif arr.dtype.kind in ("i", "u"):
                arr = arr.astype(np.int32, copy=False)
            elif arr.dtype.kind == "b":
                arr = arr.astype(bool, copy=False)

            compact[key] = arr
        else:
            compact[key] = value

    return compact


def install_rul_distribution_memory_guards():
    """
    Patch the RUL distribution module from app.py.

    This does not change pages/rul_distribution.py on disk.
    It only replaces its module-level caches at runtime.
    """

    # Limit dataset cache.
    if hasattr(rul_distribution, "_CACHE"):
        old_cache = getattr(rul_distribution, "_CACHE")

        if not isinstance(old_cache, BoundedCache):
            new_cache = BoundedCache(max_items=RUL_DF_CACHE_ITEMS)

            if isinstance(old_cache, dict):
                for key, value in old_cache.items():
                    new_cache[key] = value

            rul_distribution._CACHE = new_cache

    # Limit rolling prediction cache.
    if hasattr(rul_distribution, "_ROLLING_CACHE"):
        old_cache = getattr(rul_distribution, "_ROLLING_CACHE")

        if not isinstance(old_cache, BoundedCache):
            new_cache = BoundedCache(max_items=RUL_ROLLING_CACHE_ITEMS)

            if isinstance(old_cache, dict):
                for key, value in old_cache.items():
                    new_cache[key] = compact_rolling_payload(value)

            rul_distribution._ROLLING_CACHE = new_cache

    # Patch predict_machine_rolling so any new rolling output is compacted
    # before the page stores it.
    if hasattr(rul_distribution, "predict_machine_rolling"):
        if not getattr(rul_distribution, "_APP_MEMORY_PATCHED", False):
            original_predict_machine_rolling = rul_distribution.predict_machine_rolling

            def memory_safe_predict_machine_rolling(*args, **kwargs):
                rolling = original_predict_machine_rolling(*args, **kwargs)
                return compact_rolling_payload(rolling)

            rul_distribution.predict_machine_rolling = memory_safe_predict_machine_rolling
            rul_distribution._APP_MEMORY_PATCHED = True

    gc.collect()


def clear_rul_distribution_memory(clear_dataset_cache=True):
    """
    Clear heavy RUL distribution cache when leaving the page.
    """
    try:
        if hasattr(rul_distribution, "_ROLLING_CACHE"):
            rul_distribution._ROLLING_CACHE.clear()

        if clear_dataset_cache and hasattr(rul_distribution, "_CACHE"):
            rul_distribution._CACHE.clear()

        gc.collect()

    except Exception:
        pass


install_rul_distribution_memory_guards()


# =============================================================================
# Routes
# =============================================================================

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


# =============================================================================
# Layout helpers
# =============================================================================

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


# =============================================================================
# Main callbacks
# =============================================================================

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

    # If user leaves the heavy RUL page, clear its memory.
    if pathname != "/rul-distribution":
        clear_rul_distribution_memory(clear_dataset_cache=True)

    if pathname not in ROUTES:
        return dbc.Alert("404 — Page not found", color="warning")

    try:
        install_rul_distribution_memory_guards()
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


# =============================================================================
# Page callbacks
# =============================================================================

home.register_callbacks(app)
cost_function.register_callbacks(app)
data_simulator.register_callbacks(app)
rul_distribution.register_callbacks(app)
cost_sensitive_model.register_callbacks(app)
maintenance_planning.register_callbacks(app)
benchmark.register_callbacks(app)


# =============================================================================
# Local run
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)