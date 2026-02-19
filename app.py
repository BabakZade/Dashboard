import importlib.util
import subprocess
import sys
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from pages import home, cost_function, data_simulator, rul_distribution, cost_sensitive_model, benchmark
from pages.benchmark import DATA_ROWS, OUT_ROWS

# Check for missing required packages and install them
REQUIRED = {
    "dash": "dash",
    "plotly": "plotly",
    "numpy": "numpy",
    "dash_bootstrap_components": "dash-bootstrap-components",
}

missing = [mod for mod in REQUIRED if importlib.util.find_spec(mod) is None]

if missing:
    pkgs = [REQUIRED[m] for m in missing]
    print("❌ Missing required packages:", ", ".join(pkgs))
    print("Installing...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])
        print("✅ Installed:", ", ".join(pkgs))
    except Exception as e:
        print("❌ Install failed:", e)
        raise SystemExit(1)

    # re-check after install
    still_missing = [mod for mod in REQUIRED if importlib.util.find_spec(mod) is None]
    if still_missing:
        print("❌ Still missing after install:", ", ".join(still_missing))
        raise SystemExit(1)

# External stylesheets (including Font Awesome CDN)
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"  # Font Awesome 6 CDN
]


# Initialize the Dash app
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets,
    assets_folder="assets"
)
app.title = "Cost-sensitive predictive maintenance"

# Routes and icons for the sidebar menu
ROUTES = {
    "/": ("Home", home.layout),
    "/cost-function": ("Page 1 — Cost function", cost_function.layout),
    "/data-simulator": ("Page 2 — Data Simulator", data_simulator.layout),
    "/rul-distribution": ("Page 3 — RUL distribution", rul_distribution.layout),
    "/cost-sensitive-model": ("Page 4 — Cost sensitive model", cost_sensitive_model.layout),
    "/benchmark": ("Page 5 — Benchmark", benchmark.layout),
}

ICONS = {
    "/": "fa-solid fa-house-laptop",  # Home 
    "/cost-function": "fa-solid fa-file-contract", 
    "/data-simulator": "fa-solid fa-database",  # Flask
    "/rul-distribution": "fa-solid fa-chart-area",  # Line chart (alternative to fa-area-chart)
    "/cost-sensitive-model": "fa-solid fa-filter-circle-dollar",  # Gear icon (fa-gear can sometimes have issues)
    "/benchmark": "fa-solid fa-chart-pie",  # Pie chart
}

DATA_ROWS = benchmark.DATA_ROWS
OUT_ROWS = benchmark.OUT_ROWS

# Function to generate nav links with icons
def nav_link(label, href, icon_class):
    return dcc.Link(
        html.Div(
            [
                html.I(className=icon_class, style={"marginRight": "10px"}),  # Icon class applied
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

# Sidebar style
def sidebar_style(is_open: bool):
    return {
        "width": "270px" if is_open else "0px",
        "overflow": "hidden" if not is_open else "visible",
        "transition": "width 0.18s ease",
        "borderRight": "1px solid #eee",
        "backgroundColor": "white",
        "flexShrink": 0,
        "position": "sticky",
        "top": "54px",  # Must match topbar height
        "height": "calc(100vh - 54px)",
        "overflowY": "auto",
    }

# Layout of the main page
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

        # Shell for content and sidebar
        html.Div(
            id="shell",
            style={"display": "flex", "maxWidth": "1400px", "margin": "0 auto"},
            children=[
                html.Div(
                    id="sidebar",
                    style=sidebar_style(False),
                    children=[
                        html.Div("Menu", style={"fontWeight": 900, "marginBottom": "10px"}),
                        nav_link("Home", "/", ICONS["/"]),
                        nav_link("Cost function", "/cost-function", ICONS["/cost-function"]),
                        nav_link("Data Simulator", "/data-simulator", ICONS["/data-simulator"]),
                        nav_link("RUL distribution", "/rul-distribution", ICONS["/rul-distribution"]),
                        nav_link("Cost sensitive model", "/cost-sensitive-model", ICONS["/cost-sensitive-model"]),
                        nav_link("Benchmark", "/benchmark", ICONS["/benchmark"]),
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

# Callback to toggle menu visibility
@app.callback(
    Output("menu_open", "data"),
    Input("menu_btn", "n_clicks"),
    State("menu_open", "data"),
    prevent_initial_call=True,
)
def toggle_menu(_, is_open):
    return not is_open

# Callback to close the menu on navigation
@app.callback(
    Output("menu_open", "data", allow_duplicate=True),
    Input("url", "pathname"),
    prevent_initial_call=True,
)
def close_menu_on_nav(_):
    return False

# Apply sidebar style based on the menu state
@app.callback(
    Output("sidebar", "style"),
    Input("menu_open", "data"),
)
def apply_sidebar(is_open):
    return sidebar_style(bool(is_open))

# Update page title based on the URL
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

# Callback to render page content based on the URL
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

# Register callbacks for pages
home.register_callbacks(app)
cost_function.register_callbacks(app)
data_simulator.register_callbacks(app)
rul_distribution.register_callbacks(app)
cost_sensitive_model.register_callbacks(app)
benchmark.register_callbacks(app)

# Set validation layout for Dash
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

# Start the Dash app
if __name__ == "__main__":
    print("Starting Dash on http://127.0.0.1:8051/")
    app.run(debug=True, host="127.0.0.1", port=8051)
