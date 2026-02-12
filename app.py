# app.py
import importlib.util
import sys

REQUIRED = ["dash", "plotly", "numpy"]
missing = [p for p in REQUIRED if importlib.util.find_spec(p) is None]
if missing:
    print("❌ Missing required packages:", ", ".join(missing))
    print("Install with:")
    print(f"  {sys.executable} -m pip install " + " ".join(missing))
    raise SystemExit(1)

from dash import Dash, dcc, html, Input, Output, State

from pages import home, cost_function, data_simulator, rul_distribution, cost_sensitive_model, benchmark

external_stylesheets = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
]

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets
)
app.title = "Cost-sensitive predictive maintenance"

ROUTES = {
    "/": ("Home", home.layout),
    "/cost-function": ("Page 1 — Cost function", cost_function.layout),
    "/data-simulator": ("Page 2 — Data Simulator", data_simulator.layout),
    "/rul-distribution": ("Page 3 — RUL distribution", rul_distribution.layout),
    "/cost-sensitive-model": ("Page 4 — Cost sensitive model", cost_sensitive_model.layout),
    "/benchmark": ("Page 5 — Benchmark", benchmark.layout),
}

ICONS = {
    "/": "fa-solid fa-house",
    "/cost-function": "fa-solid fa-euro-sign",
    "/data-simulator": "fa-solid fa-flask",
    "/rul-distribution": "fa-solid fa-chart-column",
    "/cost-sensitive-model": "fa-solid fa-gears",
    "/benchmark": "fa-solid fa-chart-line",
}

DATA_ROWS = benchmark.DATA_ROWS
OUT_ROWS = benchmark.OUT_ROWS


def nav_link(label, href, icon_class):
    return dcc.Link(
        html.Div(
            [
                html.I(className=icon_class, style={"marginRight": "10px"}),
                label,
            ],
            style={"display": "flex", "alignItems": "center"},
        ),
        href=href,
        style={
            "display": "block",
            "padding": "10px 12px",
            "borderRadius": "10px",
            "textDecoration": "none",
            "color": "#111",
        },
    )


def sidebar_style(is_open: bool):
    return {
        "width": "270px" if is_open else "0px",
        "overflow": "hidden",
        "transition": "width 0.18s ease",
        "borderRight": "1px solid #eee",
        "backgroundColor": "white",
        "minHeight": "calc(100vh - 54px)",
        "flexShrink": 0,
    }


app.layout = html.Div(
    children=[
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="menu_open", data=False),

        dcc.Store(id="bench_data_store", data=DATA_ROWS),
        dcc.Store(id="bench_out_store", data=OUT_ROWS),

        html.Div(
            id="topbar",
            style={
                "position": "sticky",
                "top": 0,
                "zIndex": 2000,
                "backgroundColor": "white",
                "borderBottom": "1px solid #eee",
                "padding": "10px 12px",
            },
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "10px"},
                    children=[
                        html.Button(
                            "◫",
                            id="menu_btn",
                            n_clicks=0,
                            style={
                                "fontSize": "20px",
                                "padding": "8px 12px",
                                "borderRadius": "10px",
                                "border": "1px solid #ddd",
                                "background": "white",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div(id="page_title", style={"fontSize": "18px", "fontWeight": 800}),
                    ],
                )
            ],
        ),

        html.Div(
            id="shell",
            style={"display": "flex", "maxWidth": "1400px", "margin": "0 auto"},
            children=[
                html.Div(
                    id="sidebar",
                    style=sidebar_style(False),
                    children=[
                        html.Div(
                            style={"padding": "12px"},
                            children=[
                                html.Div("Menu", style={"fontWeight": 900, "marginBottom": "10px"}),
                                nav_link("Home", "/", ICONS["/"]),
                                nav_link("Cost function", "/cost-function", ICONS["/cost-function"]),
                                nav_link("Data Simulator", "/data-simulator", ICONS["/data-simulator"]),
                                nav_link("RUL distribution", "/rul-distribution", ICONS["/rul-distribution"]),
                                nav_link("Cost sensitive model", "/cost-sensitive-model", ICONS["/cost-sensitive-model"]),
                                nav_link("Benchmark", "/benchmark", ICONS["/benchmark"]),
                            ],
                        )
                    ],
                ),

                html.Div(
                    id="content",
                    style={"flex": 1, "padding": "16px", "minWidth": 0},
                    children=[html.Div(id="page_content")],
                ),
            ],
        ),
    ]
)


@app.callback(
    Output("menu_open", "data"),
    Input("menu_btn", "n_clicks"),
    State("menu_open", "data"),
    prevent_initial_call=True,
)
def toggle_menu(_, is_open):
    return not is_open


@app.callback(
    Output("menu_open", "data", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=True,
)
def close_menu_on_nav(_):
    return False


@app.callback(
    Output("sidebar", "style"),
    Input("menu_open", "data"),
)
def apply_sidebar(is_open):
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
        style={"display": "flex", "alignItems": "center"},
    )


@app.callback(
    Output("page_content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    if not pathname:
        pathname = "/"
    if pathname not in ROUTES:
        return html.Div("404 — Page not found")
    return ROUTES[pathname][1]()


home.register_callbacks(app)
cost_function.register_callbacks(app)
data_simulator.register_callbacks(app)
rul_distribution.register_callbacks(app)
cost_sensitive_model.register_callbacks(app)
benchmark.register_callbacks(app)

app.validation_layout = html.Div(
    [
        app.layout,
        home.layout(),
        cost_function.layout(),
        data_simulator.layout(),
        rul_distribution.layout(),
        cost_sensitive_model.layout(),
        benchmark.layout(),
    ]
)

if __name__ == "__main__":
    print("Starting Dash on http://127.0.0.1:8050/")
    app.run(debug=True, host="127.0.0.1", port=8050)
