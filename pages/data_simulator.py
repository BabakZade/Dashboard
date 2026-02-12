# pages/data_simulator.py
from dash import html

def layout():
    return html.Div(
        style={"border": "1px solid #ddd", "borderRadius": "12px", "padding": "16px"},
        children=[
            html.H3("Data Simulator", style={"marginTop": 0}),
            html.Div("Coming soon…", style={"opacity": 0.75}),
        ],
    )

def register_callbacks(app):
    return
