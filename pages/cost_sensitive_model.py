from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Define initial settings
settings = {
    "lead_time": 1,
    "emrgency_response_time": 0,
    "rul_thresh": 60,
    "slice_window": 7,
    "slice_shift": 7,
    "cost_weight": 1.0,
    "early_penalty": 1,
    "late_penalty": 10,
    "emergency_penalty": 10,
    "cost_reactive": 200,
    "cost_predictive": 20,
    "normalize": True,
    "slicer_output": 1
}

# Layout function - returns the layout of the page
def layout():
    return html.Div([
        # Gear icon to toggle settings visibility
        html.Div([
            html.I(className="fa fa-cogs", id="gear-icon", style={"fontSize": "30px", "cursor": "pointer", "color": "black", "marginRight": "20px"})
        ], style={"textAlign": "right", "padding": "10px"}),

        # Settings Panel (Initially visible)
        html.Div(
            id="settings-panel",
            children=[
                html.H3("Dataset Settings", style={'textAlign': 'center'}),
                html.Div([

                    # Lead time input box
                    html.Label('Lead Time (days):'),
                    html.Div([
                        html.Button("-", id="lead_time_decrease", n_clicks=0),
                        html.Div(id="lead_time_value", children=settings["lead_time"], style={"display": "inline-block", "padding": "5px", "border": "1px solid #ddd", "width": "50px", "textAlign": "center"}),
                        html.Button("+", id="lead_time_increase", n_clicks=0),
                    ], style={"display": "flex", "alignItems": "center"}),

                    # Emergency response time input box
                    html.Label('Emergency Response Time (days):'),
                    html.Div([
                        html.Button("-", id="emrgency_response_time_decrease", n_clicks=0),
                        html.Div(id="emrgency_response_time_value", children=settings["emrgency_response_time"], style={"display": "inline-block", "padding": "5px", "border": "1px solid #ddd", "width": "50px", "textAlign": "center"}),
                        html.Button("+", id="emrgency_response_time_increase", n_clicks=0),
                    ], style={"display": "flex", "alignItems": "center"}),

                    # RUL Threshold input box
                    html.Label('RUL Threshold:'),
                    html.Div([
                        html.Button("-", id="rul_thresh_decrease", n_clicks=0),
                        html.Div(id="rul_thresh_value", children=settings["rul_thresh"], style={"display": "inline-block", "padding": "5px", "border": "1px solid #ddd", "width": "50px", "textAlign": "center"}),
                        html.Button("+", id="rul_thresh_increase", n_clicks=0),
                    ], style={"display": "flex", "alignItems": "center"}),

                    # Slice window input box
                    html.Label('Slice Window (days):'),
                    html.Div([
                        html.Button("-", id="slice_window_decrease", n_clicks=0),
                        html.Div(id="slice_window_value", children=settings["slice_window"], style={"display": "inline-block", "padding": "5px", "border": "1px solid #ddd", "width": "50px", "textAlign": "center"}),
                        html.Button("+", id="slice_window_increase", n_clicks=0),
                    ], style={"display": "flex", "alignItems": "center"}),

                    # Slice shift input box
                    html.Label('Slice Shift (days):'),
                    html.Div([
                        html.Button("-", id="slice_shift_decrease", n_clicks=0),
                        html.Div(id="slice_shift_value", children=settings["slice_shift"], style={"display": "inline-block", "padding": "5px", "border": "1px solid #ddd", "width": "50px", "textAlign": "center"}),
                        html.Button("+", id="slice_shift_increase", n_clicks=0),
                    ], style={"display": "flex", "alignItems": "center"}),

                    # Cost Weight input box
                    html.Label('Cost Weight:'),
                    html.Div([
                        html.Button("-", id="cost_weight_decrease", n_clicks=0),
                        html.Div(id="cost_weight_value", children=settings["cost_weight"], style={"display": "inline-block", "padding": "5px", "border": "1px solid #ddd", "width": "50px", "textAlign": "center"}),
                        html.Button("+", id="cost_weight_increase", n_clicks=0),
                    ], style={"display": "flex", "alignItems": "center"}),

                    # Early penalty input box
                    html.Label('Early Penalty:'),
                    html.Div([
                        html.Button("-", id="early_penalty_decrease", n_clicks=0),
                        html.Div(id="early_penalty_value", children=settings["early_penalty"], style={"display": "inline-block", "padding": "5px", "border": "1px solid #ddd", "width": "50px", "textAlign": "center"}),
                        html.Button("+", id="early_penalty_increase", n_clicks=0),
                    ], style={"display": "flex", "alignItems": "center"}),

                    # More settings here...
                ], style={'width': '80%', 'padding': '30px', 'margin': 'auto'}),
            ],
            style={'display': 'none'}  # Initially hidden
        ),
        
        # Main area (currently empty for now)
        html.Div(id="main-area", style={"minHeight": "400px", "border": "1px solid #ddd", "marginTop": "20px"})
    ])

# Register callbacks for this page
def register_callbacks(app):
    # Toggle settings panel visibility
    @app.callback(
        Output('settings-panel', 'style'),
        Input('gear-icon', 'n_clicks'),
        prevent_initial_call=True
    )
   
    
    # Adjust the value boxes based on clicks (for all settings)
    @app.callback(
        Output('lead_time_value', 'children'),
        Output('emrgency_response_time_value', 'children'),
        Output('rul_thresh_value', 'children'),
        Output('slice_window_value', 'children'),
        Output('slice_shift_value', 'children'),
        Output('cost_weight_value', 'children'),
        Output('early_penalty_value', 'children'),
        Input('lead_time_increase', 'n_clicks'),
        Input('lead_time_decrease', 'n_clicks'),
        Input('emrgency_response_time_increase', 'n_clicks'),
        Input('emrgency_response_time_decrease', 'n_clicks'),
        Input('rul_thresh_increase', 'n_clicks'),
        Input('rul_thresh_decrease', 'n_clicks'),
        Input('slice_window_increase', 'n_clicks'),
        Input('slice_window_decrease', 'n_clicks'),
        Input('slice_shift_increase', 'n_clicks'),
        Input('slice_shift_decrease', 'n_clicks'),
        Input('cost_weight_increase', 'n_clicks'),
        Input('cost_weight_decrease', 'n_clicks'),
        Input('early_penalty_increase', 'n_clicks'),
        Input('early_penalty_decrease', 'n_clicks')
    )
    def adjust_values(*n_clicks):
        updated_values = settings.copy()
        
        # Increment or decrement logic for each setting
        if n_clicks[0] > n_clicks[1]:
            updated_values["lead_time"] += 1
        elif n_clicks[1] > n_clicks[0]:
            updated_values["lead_time"] -= 1

        if n_clicks[2] > n_clicks[3]:
            updated_values["emrgency_response_time"] += 1
        elif n_clicks[3] > n_clicks[2]:
            updated_values["emrgency_response_time"] -= 1

        if n_clicks[4] > n_clicks[5]:
            updated_values["rul_thresh"] += 1
        elif n_clicks[5] > n_clicks[4]:
            updated_values["rul_thresh"] -= 1

        if n_clicks[6] > n_clicks[7]:
            updated_values["slice_window"] += 1
        elif n_clicks[7] > n_clicks[6]:
            updated_values["slice_window"] -= 1

        if n_clicks[8] > n_clicks[9]:
            updated_values["slice_shift"] += 1
        elif n_clicks[9] > n_clicks[8]:
            updated_values["slice_shift"] -= 1

        if n_clicks[10] > n_clicks[11]:
            updated_values["cost_weight"] += 0.1
        elif n_clicks[11] > n_clicks[10]:
            updated_values["cost_weight"] -= 0.1

        if n_clicks[12] > n_clicks[13]:
            updated_values["early_penalty"] += 1
        elif n_clicks[13] > n_clicks[12]:
            updated_values["early_penalty"] -= 1

        # Return updated values to display
        return (
            updated_values["lead_time"],
            updated_values["emrgency_response_time"],
            updated_values["rul_thresh"],
            updated_values["slice_window"],
            updated_values["slice_shift"],
            updated_values["cost_weight"],
            updated_values["early_penalty"]
        )
