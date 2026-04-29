# app.py
import os
import gc
from collections import OrderedDict

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None

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
# Memory monitor
# =============================================================================

def print_memory(label=""):
    """
    Print current Python process memory.
    Works locally and on Render if psutil is installed.
    """
    if psutil is None:
        print(f"[MEMORY] {label}: psutil is not installed", flush=True)
        return

    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEMORY] {label}: {mem_mb:.1f} MB", flush=True)


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
    print_memory("health check")
    return "ok", 200


print_memory("after imports and app creation")


# =============================================================================
# Memory guard for RUL distribution page
# =============================================================================

RUL_DF_CACHE_ITEMS = int(os.environ.get("RUL_DF_CACHE_ITEMS", "1"))
RUL_ROLLING_CACHE_ITEMS = int(os.environ.get("RUL_ROLLING_CACHE_ITEMS", "1"))
RUL_MAX_POSTERIOR_SAMPLES = int(os.environ.get("RUL_MAX_POSTERIOR_SAMPLES", "100"))


class BoundedCache(OrderedDict):
    """
    Small LRU cache.

    This replaces the normal dictionaries inside pages/rul_distribution.py
    without changing that file on disk.
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
    Reduce memory used by rolling prediction output.

    The largest object is usually rul_samples_matrix.
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

    This does not edit pages/rul_distribution.py.
    It only replaces its module-level caches at runtime.
    """
    try:
        # Limit dataframe cache.
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

        # Patch predict_machine_rolling so new rolling outputs are compacted
        # before the page stores them.
        if hasattr(rul_distribution, "predict_machine_rolling"):
            if not getattr(rul_distribution, "_APP_MEMORY_PATCHED", False):
                original_predict_machine_rolling = rul_distribution.predict_machine_rolling

                def memory_safe_predict_machine_rolling(*args, **kwargs):
                    print_memory("before predict_machine_rolling")
                    rolling = original_predict_machine_rolling(*args, **kwargs)
                    print_memory("after predict_machine_rolling before compact")
                    rolling = compact_rolling_payload(rolling)
                    print_memory("after compacting rolling payload")
                    return rolling

                rul_distribution.predict_machine_rolling = memory_safe_predict_machine_rolling
                rul_distribution._APP_MEMORY_PATCHED = True

        gc.collect()

    except Exception as e:
        print(f"[MEMORY GUARD ERROR] {e}", flush=True)


def clear_rul_distribution_memory(clear_dataset_cache=True):
    """
    Clear heavy RUL distribution caches.

    This helps when leaving the RUL page or when changing RUL inputs.
    """
    try:
        print_memory("before clearing RUL cache")

        if hasattr(rul_distribution, "_ROLLING_CACHE"):
            rul_distribution._ROLLING_CACHE.clear()

        if clear_dataset_cache and hasattr(rul_distribution, "_CACHE"):
            rul_distribution._CACHE.clear()

        gc.collect()
        print_memory("after clearing RUL cache")

    except Exception as e:
        print(f"[RUL CACHE CLEAR ERROR] {e}", flush=True)


install_rul_distribution_memory_guards()
print_memory("after installing RUL memory guards")


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

# Important:
# Do not put large benchmark data directly in app.layout.
# Large dcc.Store data can make /_dash-layout huge and crash Render.
BENCH_DATA_ROWS = []
BENCH_OUT_ROWS = []


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
    print_memory("before serve_layout")

    layout_obj = dbc.Container(
        fluid=True,
        className="px-0",
        children=[
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="menu_open", data=True),

            # Keep these light. Do not load large rows here.
            dcc.Store(id="bench_data_store", data=BENCH_DATA_ROWS),
            dcc.Store(id="bench_out_store", data=BENCH_OUT_ROWS),
            dcc.Store(id="shared-inputs", storage_type="session"),

            # Used only to trigger cache clearing from app.py.
            dcc.Store(id="rul_cache_clean_signal"),

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

    print_memory("after serve_layout")
    return layout_obj


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

    print_memory(f"before loading page {pathname}")

    # If user leaves the heavy RUL page, clear its memory.
    if pathname != "/rul-distribution":
        clear_rul_distribution_memory(clear_dataset_cache=True)

    if pathname not in ROUTES:
        print_memory(f"404 page {pathname}")
        return dbc.Alert("404 — Page not found", color="warning")

    try:
        install_rul_distribution_memory_guards()

        page = ROUTES[pathname][1]()

        print_memory(f"after loading page {pathname}")
        return page

    except Exception as e:
        print_memory(f"error while loading page {pathname}")
        return dbc.Alert(
            [
                html.H5("This page could not be loaded."),
                html.Div("The Python error was:"),
                html.Pre(str(e), style={"whiteSpace": "pre-wrap"}),
            ],
            color="danger",
        )


@app.callback(
    Output("rul_cache_clean_signal", "data"),
    Input("rul_split", "value"),
    Input("rul_machine_id", "value"),
    prevent_initial_call=True,
)
def clear_rul_cache_on_machine_change(split, machine_id):
    """
    This callback does not edit the RUL page.
    It clears the RUL cache when the machine/split changes so old rolling
    objects do not accumulate.
    """
    print_memory(f"before RUL input change clear split={split}, machine={machine_id}")

    # Keep dataset cache to avoid rereading CSV every time.
    # Clear only rolling prediction cache.
    clear_rul_distribution_memory(clear_dataset_cache=False)

    print_memory(f"after RUL input change clear split={split}, machine={machine_id}")

    return {
        "split": split,
        "machine_id": machine_id,
    }


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

print_memory("after registering callbacks")


# =============================================================================
# Local run
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print_memory("before app.run")
    app.run(debug=False, host="0.0.0.0", port=port)