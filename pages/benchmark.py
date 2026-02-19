# pages/benchmark.py
from dash import dcc, html
from dash import dash_table
from dash import Input, Output
import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd
from dash import Dash

# Constants for cost calculation
X_MIN, X_MAX = 0, 30
C_PR = 2000
ALPHA = 100
C_RE = 10000
BETA = 500

# Record cost function (linear model)
def record_cost_linear(pred: float, true: float, leadtime: int) -> float:
    if pred <= true:
        return C_PR + ALPHA * (true - pred)
    diff = pred - true
    eff = float(np.minimum(diff, leadtime))
    return C_RE + BETA * eff


# Read data from CSV and return as a list of dictionaries (records)
def make_examples():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, 'assets', 'test.csv')

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    # Read the CSV data into a DataFrame with correct delimiter and decimal point
    df = pd.read_csv(file_path, sep=';', decimal=',')  # Ensure correct delimiter

    # Strip any extra spaces from column names
    df.columns = df.columns.str.strip()

    # Convert columns to numeric if necessary
    df['car_type'] = pd.to_numeric(df['car_type'], errors='coerce')
    df['failure_type'] = pd.to_numeric(df['failure_type'], errors='coerce')
    df['rul'] = pd.to_numeric(df['rul'], errors='coerce')  # Coerce any non-numeric values to NaN

    # Return the data as a list of dictionaries
    return df.to_dict('records')  # This is the list of dictionaries

# Process input rows to prepare them for the table
def process_input_rows(data_raw):
    
    out = []

    for r in data_raw:
        true = float(r["rul"])  # Accessing the "rul" value from the dictionary
        lt = int(r.get("leadtime", 0))  # Default to 0 if "leadtime" is missing

        # Use the actual rul value for both predictions
        pred_classic = true  # Use the true RUL for classic prediction
        pred_cost = true  # Use the true RUL for cost-sensitive prediction

        # MSE and cost calculations
        mse_classic = float((pred_classic - true) ** 2)
        mse_cost = float((pred_cost - true) ** 2)

        cost_classic = float(record_cost_linear(pred_classic, true, lt))
        cost_cost = float(record_cost_linear(pred_cost, true, lt))

        car = "van" if r.get("car_type", 0) == 0 else "truck"
        failure_type = "tire" if r.get("failure_type", 0) == 0 else "brake"

        out.append(
            dict(
                id=int(r.get("id", 0)),
                rul_true=true,
                car=car,
                failure=failure_type,
                leadtime=lt,
                route = r.get("route", 0),
                route_ratio = r.get("route_ratio", 0),
                speed = r.get("speed", 0),
                load = r.get("load", 0),

            )
        )
    return out  # Return the list of dictionaries

# Load data and generate output rows for the page
data_raw = make_examples()  # This line loads the data and stores it as DATA_ROWS
DATA_ROWS = process_input_rows(data_raw)  # Generate the output rows




# take the output prdicion from other pages and calculate teh output rows
# Function to generate output for Model 1 (Page 1)
def build_output_rows_page1(data_rows, input_preds, seed: int = 7):
    out = []
    for r, pred in zip(data_rows, input_preds):
        true = r["rul_true"]

        # Calculate MSE and cost for Model 1
        mse_model1 = float((pred - true) ** 2)
        cost_model1 = float(record_cost_linear(pred, true, r["leadtime"]))

        out.append(
            dict(
                id=r["id"],
                rul_true=true,
                pred_classic=pred,  # Match column name
                mse_classic=mse_model1,  # Rename to match column name
                cost_classic=cost_model1,  # Rename to match column name
            )
        )
    return out


# Function to generate output for Model 2 (Page 2)
def build_output_rows_page2(data_rows, input_preds, seed: int = 7):
    out = []
    for r, pred in zip(data_rows, input_preds):
        true = r["rul_true"]

        # Calculate MSE and cost for Model 2
        mse_model2 = float((pred - true) ** 2)
        cost_model2 = float(record_cost_linear(pred, true, r["leadtime"]))

        out.append(
            dict(
                id=r["id"],
                rul_true=true,
                pred_cost_sensitive=pred,  # Rename to match column name
                mse_cost_sensitive=mse_model2,  # Rename to match column name
                cost_cost_sensitive=cost_model2,  # Rename to match column name
            )
        )
    return out


# Function to generate random predictions for Model 1 and Model 2
def generate_random_predictions(data_rows, rul_column, seed: int = 7):
    rng = np.random.default_rng(seed)

    # Extract the actual RUL values from the data (assuming 'rul_column' contains the RUL)
    actual_rul =[]
    for r in data_raw:
        true = float(r[rul_column])  # Accessing the "rul" value from the dictionary
        actual_rul.append(true)
    # Generate random noise around the RUL values to simulate predictions
    model1_preds = actual_rul + rng.normal(loc=0, scale=5, size=len(data_rows))  # Model 1 predictions with noise
    model2_preds = actual_rul + rng.normal(loc=0, scale=5, size=len(data_rows))  # Model 2 predictions with noise
    
    return model1_preds, model2_preds

# Example usage
model1_preds, model2_preds = generate_random_predictions(DATA_ROWS, rul_column='rul', seed=7)



OUT_ROWS_page1 = build_output_rows_page1(DATA_ROWS, model1_preds)
OUT_ROWS_page2 = build_output_rows_page2(DATA_ROWS, model2_preds)

# Combine outputs from different models
# Combine outputs from different models (side by side columns)
def combine_output_rows(out_rows_page1, out_rows_page2):
    combined = []
    
    # Ensure both lists have the same length (this is expected since we're pairing corresponding rows)
    for row1, row2 in zip(out_rows_page1, out_rows_page2):
        combined_row = {**row1, **row2}  # Merge the two dictionaries (side by side)
        combined.append(combined_row)
    
    return combined


# Combine OUT_ROWS_page1 and OUT_ROWS_page2 side by side
OUT_ROWS = combine_output_rows(OUT_ROWS_page1, OUT_ROWS_page2)

# Generate the benchmark summary figure (MSE and cost)
def _summary_fig(output_rows):
    mse1 = np.mean([r["mse_classic"] for r in output_rows])
    mse2 = np.mean([r["mse_cost_sensitive"] for r in output_rows])
    c1 = np.mean([r["cost_classic"] for r in output_rows])
    c2 = np.mean([r["cost_cost_sensitive"] for r in output_rows])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Classic", "Cost-sensitive"], y=[mse1, mse2], name="Avg MSE"))
    fig.add_trace(go.Bar(x=["Classic", "Cost-sensitive"], y=[c1, c2], name="Avg Cost"))
    fig.update_layout(height=300, barmode="group", margin=dict(l=10, r=10, t=40, b=10))
    return fig


# Assuming DATA_ROWS and OUT_ROWS are defined outside the layout function
# For example:
# DATA_ROWS = make_examples()  # This line loads the data and stores it as DATA_ROWS
# OUT_ROWS = build_output_rows(DATA_ROWS, seed=7)  # Generate the output rows

# Layout of the page
def layout():
    # Directly use the pre-loaded DATA_ROWS and OUT_ROWS instead of re-reading the data
    global DATA_ROWS, OUT_ROWS  # Make sure the global variables are used
 
    # Define the columns based on the list of dictionary keys (from the first dictionary's keys)
    columns = [{"name": col, "id": col} for col in DATA_ROWS[0].keys()]  # Use DATA_ROWS instead of calling make_examples()

    return html.Div(
        children=[
            dcc.Tabs(
                id="bench_tabs",
                value="bench_data",
                children=[
                    dcc.Tab(
                        label="Data",
                        value="bench_data",
                        children=[
                            html.Div(
                                style={"border": "1px solid #eee", "borderRadius": "14px", "padding": "12px", "marginTop": "12px"},
                                children=[
                                    dash_table.DataTable(
                                        id="benchmark_data_table",
                                        columns=columns,  # Use column names from DATA_ROWS[0].keys()
                                        data=DATA_ROWS,  # Pass the pre-loaded DATA_ROWS directly
                                        page_size=8,
                                        sort_action="native",
                                        row_selectable="single",
                                        style_table={"overflowX": "auto"},
                                        style_cell={"fontFamily": "sans-serif", "fontSize": "13px", "padding": "6px"},
                                        style_header={"fontWeight": "800"},
                                    ),
                                ],
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Output",
                        value="bench_output",
                        children=[
                            html.Div(
                                style={"border": "1px solid #eee", "borderRadius": "14px", "padding": "12px", "marginTop": "12px"},
                                children=[
                                    dash_table.DataTable(
                                        id="benchmark_output_table",
                                        columns=[
                                            {"name": "id", "id": "id"},
                                            {"name": "classic", "id": "pred_classic", "type": "numeric", "format": {"specifier": ".1f"}},
                                            {"name": "cost-sensitive", "id": "pred_cost_sensitive", "type": "numeric", "format": {"specifier": ".1f"}},
                                            {"name": "mse classic", "id": "mse_classic", "type": "numeric", "format": {"specifier": ".1f"}},
                                            {"name": "mse cost", "id": "mse_cost_sensitive", "type": "numeric", "format": {"specifier": ".1f"}},
                                            {"name": "cost classic", "id": "cost_classic", "type": "numeric", "format": {"specifier": ".0f"}},
                                            {"name": "cost cost", "id": "cost_cost_sensitive", "type": "numeric", "format": {"specifier": ".0f"}},
                                        ],
                                        data=OUT_ROWS,  # Pass the pre-loaded OUT_ROWS directly
                                        page_size=8,
                                        sort_action="native",
                                        row_selectable="single",
                                        style_table={"overflowX": "auto"},
                                        style_cell={"fontFamily": "sans-serif", "fontSize": "13px", "padding": "6px"},
                                        style_header={"fontWeight": "800"},
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"marginTop": "12px", "border": "1px solid #eee", "borderRadius": "14px", "padding": "12px"},
                                children=[
                                    html.Div("Quick benchmark summary", style={"fontWeight": 800, "marginBottom": "8px"}),
                                    dcc.Graph(id="benchmark_summary_fig", config={"displayModeBar": False}),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ]
    )



# Callbacks
def register_callbacks(app):
    @app.callback(
        Output("benchmark_summary_fig", "figure"),
        Input("benchmark_output_table", "data"),  # Correct the Input here
    )
    def _update_summary(out_rows):
        out_rows = out_rows or []
        if not out_rows:
            return go.Figure()
        return _summary_fig(out_rows)



