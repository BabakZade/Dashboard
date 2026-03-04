# pages/maintenance_planning.py
from __future__ import annotations

import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import pyomo.environ as pyo

from datetime import date, timedelta
import full_calendar_component as fcc
from pathlib import Path


from dash.exceptions import PreventUpdate


import traceback
import plotly.graph_objects as go

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"  # pages/.. -> project root
    
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Optional: holidays (falls back to weekends-only if not installed)
# ---------------------------
try:
    import holidays as _holidays_lib  # pip install holidays
except Exception:
    _holidays_lib = None


# ---------------------------
# Pyomo model
# ---------------------------
def build_model(data: dict) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel("FleetMaintenancePlanning")

    # -----------------------
    # Sets
    # -----------------------
    m.I = pyo.Set(initialize=data["I"])
    m.F = pyo.Set(initialize=data["F"])
    m.D = pyo.Set(initialize=data["D"], ordered=True)
    m.IF = pyo.Set(initialize=[(i, f) for i in data["I"] for f in data["F"]])
    m.IFD = pyo.Set(initialize=[(i, f, d) for i in data["I"] for f in data["F"] for d in data["D"]])

    # -----------------------
    # Parameters
    # -----------------------
    m.c_setup = pyo.Param(m.I, initialize=data["c_setup"], within=pyo.NonNegativeReals)

    # penalty weights
    m.alpha = pyo.Param(m.IF, initialize=data["alpha"], within=pyo.NonNegativeReals)
    m.beta = pyo.Param(m.IF, initialize=data["beta"], within=pyo.NonNegativeReals)

    # mode costs (NEW)
    m.c_pr = pyo.Param(m.IF, initialize=data["c_pr"], within=pyo.NonNegativeReals)
    m.c_u_pr = pyo.Param(m.IF, initialize=data["c_u_pr"], within=pyo.NonNegativeReals)
    m.c_re = pyo.Param(m.IF, initialize=data["c_re"], within=pyo.NonNegativeReals)

    m.RUL = pyo.Param(m.IF, initialize=data["RUL"], within=pyo.NonNegativeReals)
    m.F_if = pyo.Param(m.IF, initialize=data["F_if"], within=pyo.Binary)
    m.LT = pyo.Param(m.I, initialize=data["LT"], within=pyo.NonNegativeReals)

    # daily capacity (NEW)
    m.R = pyo.Param(initialize=float(data["R"]), within=pyo.PositiveReals)

    # Big-M
    m.M = pyo.Param(initialize=float(data["M"]), within=pyo.PositiveReals)

    # Workday indicator (1=workday, 0=weekend/holiday)
    m.workday = pyo.Param(m.D, initialize=data["workday"], within=pyo.Binary)

    # -----------------------
    # Decision variables
    # -----------------------
    m.x = pyo.Var(m.IFD, within=pyo.Binary)             # schedule decision
    m.y = pyo.Var(m.I, m.D, within=pyo.Binary)          # setup day

    m.e = pyo.Var(m.IF, within=pyo.Binary)              # regular predictive
    m.l = pyo.Var(m.IF, within=pyo.Binary)              # reactive
    m.u = pyo.Var(m.IF, within=pyo.Binary)              # emergency predictive

    # penalty variables (NEW)
    m.phi = pyo.Var(m.IF, within=pyo.NonNegativeReals)  # varphi
    m.psi = pyo.Var(m.IF, within=pyo.NonNegativeReals)  # psi

    # -----------------------
    # Objective
    # -----------------------
    def obj_rule(m):
        setup_cost = sum(m.c_setup[i] * m.y[i, d] for i in m.I for d in m.D)

        penalty_cost = sum(
            m.alpha[i, f] * m.phi[i, f] + m.beta[i, f] * m.psi[i, f]
            for (i, f) in m.IF
        )

        mode_cost = sum(
            m.c_pr[i, f] * m.e[i, f]
            + m.c_u_pr[i, f] * m.u[i, f]
            + m.c_re[i, f] * m.l[i, f]
            for (i, f) in m.IF
        )

        return setup_cost + penalty_cost + mode_cost

    m.OBJ = pyo.Objective(
        rule=obj_rule,
        sense=pyo.minimize,
        doc=(
            "Minimize total maintenance planning cost = setup costs (sum c_setup*y) "
            "+ timing penalties (sum alpha*phi + beta*psi) "
            "+ mode costs (sum c_pr*e + c_u_pr*u + c_re*l)."
        ),
    )

    # -----------------------
    # Constraints
    # -----------------------

    # (1) Schedule: exactly one day if failure within horizon
    def schedule_rule(m, i, f):
        return sum(m.x[i, f, d] for d in m.D) == m.F_if[i, f]

    m.Schedule = pyo.Constraint(
        m.IF, rule=schedule_rule,
        doc="Scheduling within horizon: choose exactly one maintenance day for (i,f) if F_if=1, else choose none."
    )

    # (2) Link setup y to scheduling x  (sum_f x_ifd <= M * y_id)
    def y_link_rule(m, i, d):
        return sum(m.x[i, f, d] for f in m.F) <= m.M * m.y[i, d]

    m.YLink = pyo.Constraint(
        m.I, m.D, rule=y_link_rule,
        doc="Link setup to scheduled jobs: if any failure-type is scheduled on day d for asset i, then y_id must be 1."
    )

    # (3) Daily capacity: sum_i y_id <= R
    def cap_rule(m, d):
        return sum(m.y[i, d] for i in m.I) <= m.R

    m.Capacity = pyo.Constraint(
        m.D, rule=cap_rule,
        doc="Daily capacity: at most R assets can be initiated (y=1) on each day d."
    )

    # (4) Define penalty variable phi (tardiness/downtime)
    def phi_def_rule(m, i, f):
        if pyo.value(m.F_if[i, f]) < 0.5:
            return pyo.Constraint.Skip
        return sum(d * m.x[i, f, d] for d in m.D) - m.RUL[i, f] <= m.phi[i, f]

    m.PhiDef = pyo.Constraint(
        m.IF, rule=phi_def_rule,
        doc="Define phi (tardiness): phi_if upper-bounds (scheduled_day - RUL_if) for pairs with F_if=1."
    )

    # (5) Define penalty variable psi (earliness/unused RUL)
    def psi_def_rule(m, i, f):
        if pyo.value(m.F_if[i, f]) < 0.5:
            return pyo.Constraint.Skip
        return (m.RUL[i, f] - m.LT[i]) - sum(d * m.x[i, f, d] for d in m.D) <= m.psi[i, f]

    m.PsiDef = pyo.Constraint(
        m.IF, rule=psi_def_rule,
        doc="Define psi (earliness): psi_if upper-bounds ((RUL_if - LT_i) - scheduled_day) for pairs with F_if=1."
    )

    # (6) Big-M activation for reactive mode: phi_if <= M * l_if
    def reactive_activation_rule(m, i, f):
        if pyo.value(m.F_if[i, f]) < 0.5:
            return pyo.Constraint.Skip
        return m.phi[i, f] <= m.M * m.l[i, f]

    m.ReactiveActivation = pyo.Constraint(
        m.IF, rule=reactive_activation_rule,
        doc="Reactive activation: positive tardiness (phi) is allowed only if reactive mode l_if=1 (via big-M)."
    )

    # (7) Big-M activation for regular predictive mode: psi_if <= M * e_if
    def regular_activation_rule(m, i, f):
        if pyo.value(m.F_if[i, f]) < 0.5:
            return pyo.Constraint.Skip
        return m.psi[i, f] <= m.M * m.e[i, f]

    m.RegularActivation = pyo.Constraint(
        m.IF, rule=regular_activation_rule,
        doc="Regular predictive activation: positive earliness (psi) is allowed only if regular mode e_if=1 (via big-M)."
    )

    # (8) Exactly one mode: l + e + u = F_if
    def mode_rule(m, i, f):
        if pyo.value(m.F_if[i, f]) < 0.5:
            return pyo.Constraint.Skip
        return m.l[i, f] + m.e[i, f] + m.u[i, f] == m.F_if[i, f]

    m.ModeSelect = pyo.Constraint(
        m.IF, rule=mode_rule,
        doc="Mode selection: for each (i,f) with F_if=1, choose exactly one of {reactive, regular predictive, emergency predictive}."
    )

    # (9) Workday restriction: prevent scheduling on non-workdays
    def workday_rule(m, i, f, d):
        return m.x[i, f, d] <= m.workday[d]

    m.Workday = pyo.Constraint(
        m.IFD, rule=workday_rule,
        doc="Calendar feasibility: maintenance can only be scheduled on workdays (x_ifd <= workday_d)."
    )

    return m
def solve_instance(data: dict, solver_name: str) -> tuple[float, pd.DataFrame, pd.DataFrame, dict]:
    model = build_model(data)

    # Optional: export model for debugging
    

    # Write .lp
    lp_path = ASSETS_DIR / "maintenance_model.lp"
    model.write(str(lp_path), io_options={"symbolic_solver_labels": True})

    # Write a .txt copy so browsers reliably show it as text
    txt_path = ASSETS_DIR / "maintenance_model.txt"
    txt_path.write_text(lp_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    solver = pyo.SolverFactory(solver_name)
    if (solver is None) or (not solver.available(False)):
        raise RuntimeError(f"Solver '{solver_name}' not available. Install it (glpk/cbc/highs) or switch solver.")

    res = solver.solve(model, tee=True)

    term = str(res.solver.termination_condition).lower()
    if term not in {"optimal", "feasible"}:
        raise RuntimeError(f"Solver status: {res.solver.termination_condition}")

    rows = []
    for (i, f) in model.IF:
        if pyo.value(model.F_if[i, f]) < 0.5:
            continue

        chosen = None
        for d in model.D:
            if pyo.value(model.x[i, f, d]) > 0.5:
                chosen = int(d)
                break

        mode = (
            "Regular predictive" if pyo.value(model.e[i, f]) > 0.5 else
            "Reactive" if pyo.value(model.l[i, f]) > 0.5 else
            "Emergency predictive" if pyo.value(model.u[i, f]) > 0.5 else
            "None"
        )

        rows.append(
            {
                "asset_i": i,
                "failure_f": f,
                "scheduled_day_d": chosen,
                "RUL_if": float(pyo.value(model.RUL[i, f])),
                "LT_i": float(pyo.value(model.LT[i])),
                "phi_if": float(pyo.value(model.phi[i, f])),  # NEW
                "psi_if": float(pyo.value(model.psi[i, f])),  # NEW
                "mode": mode,
            }
        )

    df_plan = pd.DataFrame(rows).sort_values(["asset_i", "failure_f"], ignore_index=True)

    setup_rows = []
    for i in model.I:
        for d in model.D:
            if pyo.value(model.y[i, d]) > 0.5:
                setup_rows.append(
                    {"asset_i": i, "setup_day_d": int(d), "setup_cost": float(pyo.value(model.c_setup[i]))}
                )
    df_setup = pd.DataFrame(setup_rows).sort_values(["asset_i", "setup_day_d"], ignore_index=True)

    # --- cost components (UPDATED) ---
    setup_cost_val = float(
        sum(pyo.value(model.c_setup[i]) * pyo.value(model.y[i, d]) for i in model.I for d in model.D)
    )

    # penalties
    phi_cost_val = float(
        sum(pyo.value(model.alpha[i, f]) * pyo.value(model.phi[i, f]) for (i, f) in model.IF)
    )
    psi_cost_val = float(
        sum(pyo.value(model.beta[i, f]) * pyo.value(model.psi[i, f]) for (i, f) in model.IF)
    )
    penalty_cost_val = phi_cost_val + psi_cost_val

    # mode costs
    regular_mode_cost_val = float(
        sum(pyo.value(model.c_pr[i, f]) * pyo.value(model.e[i, f]) for (i, f) in model.IF)
    )
    emergency_mode_cost_val = float(
        sum(pyo.value(model.c_u_pr[i, f]) * pyo.value(model.u[i, f]) for (i, f) in model.IF)
    )
    reactive_mode_cost_val = float(
        sum(pyo.value(model.c_re[i, f]) * pyo.value(model.l[i, f]) for (i, f) in model.IF)
    )
    mode_cost_val = regular_mode_cost_val + emergency_mode_cost_val + reactive_mode_cost_val

    total_cost_val = setup_cost_val + penalty_cost_val + mode_cost_val

    summary = {
        "setup_cost": setup_cost_val,

        "phi_penalty_cost": phi_cost_val,
        "psi_penalty_cost": psi_cost_val,
        "penalty_cost": penalty_cost_val,

        "regular_mode_cost": regular_mode_cost_val,
        "emergency_mode_cost": emergency_mode_cost_val,
        "reactive_mode_cost": reactive_mode_cost_val,
        "mode_cost": mode_cost_val,

        "total_cost": total_cost_val,

        "n_regular": int(sum(1 for (i, f) in model.IF if pyo.value(model.e[i, f]) > 0.5)),
        "n_emergency": int(sum(1 for (i, f) in model.IF if pyo.value(model.u[i, f]) > 0.5)),
        "n_reactive": int(sum(1 for (i, f) in model.IF if pyo.value(model.l[i, f]) > 0.5)),
    }

    obj = float(pyo.value(model.OBJ))
    return obj, df_plan, df_setup, summary
# ---------------------------
# Data generator
# ---------------------------
def _base_data(
    n_assets: int = 100,
    horizon: int = 31,
    failure_types: list[str] | None = None,
    p_fail: float = 0.5,
    seed: int | None = 42,
    start_date: date | None = None,
    holiday_country: str = "BE",
    R: int = 10,  # NEW: daily capacity (max assets initiated per day)
) -> dict:
    rng = np.random.default_rng(seed)

    if failure_types is None:
        failure_types = ["motor", "battery"]

    I = [f"Fleet_{k+1}" for k in range(n_assets)]
    F = list(failure_types)
    D = list(range(1, horizon + 1))

    LT = {i: int(rng.integers(0, 7)) for i in I}

    setup_levels = np.arange(100, 501, 50)
    c_setup = {i: int(rng.choice(setup_levels)) for i in I}

    IF = [(i, f) for i in I for f in F]
    F_if = {k: int(rng.random() < p_fail) for k in IF}

    RUL = {}
    for (i, f) in IF:
        RUL[(i, f)] = int(rng.integers(1, horizon + 1)) if F_if[(i, f)] == 1 else horizon + 1

    # penalty weights (keep as before)
    alpha = {k: int(rng.integers(10, 51)) for k in IF}
    beta = {k: int(rng.integers(100, 201)) for k in IF}
    gamma = {k: int(rng.integers(50, 101)) for k in IF}

    # NEW: mode costs
    c_pr = {k: int(rng.integers(200, 501)) for k in IF}       # regular predictive
    c_u_pr = {k: int(rng.integers(500, 901)) for k in IF}     # emergency predictive
    c_re = {k: int(rng.integers(1500, 4001)) for k in IF}     # reactive

    # Ensure emergency predictive is (typically) more expensive than regular predictive
    for k in IF:
        if c_u_pr[k] <= c_pr[k]:
            c_u_pr[k] = c_pr[k] + int(rng.integers(50, 301))

    if start_date is None:
        start_date = date.today()

    hol_cal = None
    if _holidays_lib is not None:
        try:
            hol_cal = _holidays_lib.country_holidays(holiday_country)
        except Exception:
            hol_cal = None

    workday = {}
    holiday_name = {}
    for d in D:
        dt = start_date + timedelta(days=d - 1)
        is_weekend = dt.weekday() >= 5
        is_holiday = (hol_cal is not None) and (dt in hol_cal)

        workday[d] = 0 if (is_weekend or is_holiday) else 1
        holiday_name[d] = str(hol_cal.get(dt)) if is_holiday else ""

    return {
        "I": I,
        "F": F,
        "D": D,
        "M": float(horizon),  # big-M baseline
        "R": float(R),        # NEW capacity
        "LT": LT,
        "c_setup": c_setup,
        "RUL": RUL,
        "F_if": F_if,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "c_pr": c_pr,         # NEW
        "c_u_pr": c_u_pr,     # NEW
        "c_re": c_re,         # NEW
        "workday": workday,
        "start_date": start_date,
        "holiday_name": holiday_name,
    }





def layout():
    card_style = {"border": "1px solid #ddd", "borderRadius": "12px", "padding": "16px"}

    return html.Div(
        style=card_style,
        children=[
            html.H3("Maintenance Planning", style={"marginTop": 0}),

            # --- Top summary dialog ---
            html.Div(
                id="mp-summary",
                style={
                    "border": "1px solid #e5e5e5",
                    "borderRadius": "12px",
                    "padding": "12px 14px",
                    "display": "flex",
                    "gap": "14px",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "background": "#fafafa",
                    "marginBottom": "12px",
                },
                children=[
                    html.Div(
                        style={"display": "flex", "gap": "18px", "flexWrap": "wrap", "alignItems": "center"},
                        children=[
                            html.Div(["Total assets: ", html.B("-", id="mp-kpi-assets")]),
                            html.Div(["Assets with failures: ", html.B("-", id="mp-kpi-assets-fail")]),
                            html.Div(["Total failures: ", html.B("-", id="mp-kpi-failures")]),
                            html.Div(["Setup cost: ", html.B("-", id="mp-kpi-setup")]),
                            html.Div(["Penalty cost (α·φ + β·ψ): ", html.B("-", id="mp-kpi-penalty")]),
                            html.Div(["Mode cost: ", html.B("-", id="mp-kpi-mode")]),
                            html.Div(["Total cost: ", html.B("-", id="mp-kpi-total")]),
                            html.Div(["Status: ", html.B("-", id="mp-kpi-status")]),  # <--- NEW
                        ],
                    ),
                    html.Button(
                        "⚙",
                        id="mp-gear",
                        n_clicks=0,
                        title="Costs details",
                        style={
                            "border": "1px solid #ccc",
                            "borderRadius": "10px",
                            "background": "white",
                            "cursor": "pointer",
                            "width": "42px",
                            "height": "36px",
                            "fontSize": "18px",
                            "lineHeight": "18px",
                        },
                    ),
                ],
            ),

            # modal open state + content store
            dcc.Store(id="mp-cost-modal-open", data=False),
            dcc.Store(id="mp-last-summary", data=None),
            dcc.Store(id="mp-planner-controls", data={"R_override": None, "work_all_days": False}),
            

            # --- Costs modal (hidden by default) ---
            html.Div(
                id="mp-cost-modal",
                style={"display": "none"},
                children=[
                    html.Div(
                        style={
                            "position": "fixed",
                            "top": 0,
                            "left": 0,
                            "right": 0,
                            "bottom": 0,
                            "background": "rgba(0,0,0,0.35)",
                            "zIndex": 999,
                        }
                    ),
                    html.Div(
                        style={
                            "position": "fixed",
                            "top": "80px",
                            "left": "50%",
                            "transform": "translateX(-50%)",
                            "width": "min(720px, 92vw)",
                            "background": "white",
                            "borderRadius": "14px",
                            "border": "1px solid #ddd",
                            "boxShadow": "0 12px 40px rgba(0,0,0,0.18)",
                            "padding": "16px",
                            "zIndex": 1000,
                        },
                        children=[
                            html.Div(
                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                                children=[
                                    html.H4("Costs details", style={"margin": 0}),
                                    html.Button(
                                        "Close",
                                        id="mp-cost-close",
                                        n_clicks=0,
                                        style={
                                            "border": "1px solid #ccc",
                                            "borderRadius": "10px",
                                            "background": "white",
                                            "cursor": "pointer",
                                            "padding": "8px 12px",
                                        },
                                    ),
                                ],
                            ),
                            html.Hr(style={"margin": "12px 0"}),
                            html.Div(
                                style={
                                    "display": "flex",
                                    "gap": "18px",
                                    "flexWrap": "wrap",
                                    "alignItems": "center",
                                    "marginBottom": "10px",
                                    "padding": "10px",
                                    "border": "1px solid #eee",
                                    "borderRadius": "12px",
                                    "background": "#fafafa",
                                },
                                children=[
                                    # R stepper
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                        children=[
                                            html.Div("Daily capacity R:", style={"fontWeight": 600}),
                                            html.Div(
                                                style={
                                                    "display": "inline-flex",
                                                    "alignItems": "center",
                                                    "border": "1px solid #ddd",
                                                    "borderRadius": "999px",
                                                    "overflow": "hidden",
                                                    "background": "white",
                                                },
                                                children=[
                                                    html.Button("−", id="mp-R-minus", n_clicks=0,
                                                        style={"border": "none", "padding": "6px 12px", "cursor": "pointer", "background": "white", "fontSize": "16px"}
                                                    ),
                                                    html.Div(id="mp-R-value", children="-",
                                                        style={"padding": "6px 14px", "minWidth": "48px", "textAlign": "center"}
                                                    ),
                                                    html.Button("+", id="mp-R-plus", n_clicks=0,
                                                        style={"border": "none", "padding": "6px 12px", "cursor": "pointer", "background": "white", "fontSize": "16px"}
                                                    ),
                                                ],
                                            ),
                                            html.Button("Reset", id="mp-R-reset", n_clicks=0,
                                                style={"border": "1px solid #ddd", "borderRadius": "10px", "background": "white", "cursor": "pointer", "padding": "6px 10px"},
                                                title="Reset to default R from inputs",
                                            ),
                                        ],
                                    ),

                                    # Work all days toggle
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                        children=[
                                            html.Div("Work on holidays/weekends:", style={"fontWeight": 600}),
                                            dcc.Checklist(
                                                id="mp-work-all-days",
                                                options=[{"label": "Enabled", "value": "Y"}],
                                                value=[],
                                                style={"display": "flex"},
                                                inputStyle={"marginRight": "8px"},
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(id="mp-modal-status", style={"marginTop": "6px", "marginBottom": "8px"}),
                            html.Div(id="mp-cost-breakdown", style={"lineHeight": "1.9"}),
                            html.Hr(style={"margin": "12px 0"}),
                            html.Div(
                                style={"opacity": 0.8, "fontSize": "13px"},
                                children=[
                                    html.Div("Cost generation assumptions (from _base_data):"),
                                    html.Ul(
                                        style={"marginTop": "6px"},
                                        children=[
                                            html.Li("Setup cost levels: 100..500 step 50 (per asset)"),
                                            html.Li("Penalty weights: alpha=50..150, beta=50..150 (per asset × failure type)"),
                                            html.Li("Regular predictive mode cost c_pr: 200..500 (per asset × failure type)"),
                                            html.Li("Emergency predictive mode cost c_u-pr: 500..900 (forced > c_pr)"),
                                            html.Li("Reactive mode cost c_re: 1500..4000"),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            html.Div(
                [
                    "Reads inputs from the global store ",
                    html.Code("shared-inputs"),
                    " (saved on the Inputs page). Day 1 is the day you click Run Optimization.",
                ],
                style={"opacity": 0.75, "marginBottom": "12px"},
            ),

            html.Div(
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "end"},
                children=[
                    html.Div(
                        style={"minWidth": "220px"},
                        children=[
                            html.Label("Solver"),
                            dcc.Dropdown(
                                id="mp-solver",
                                options=[
                                    {"label": "HiGHS (highs)", "value": "highs"},
                                    {"label": "GLPK (glpk)", "value": "glpk"},
                                    {"label": "CBC (cbc)", "value": "cbc"},                                    
                                    {"label": "Gurobi (gurobi)", "value": "gurobi"},
                                ],
                                value="highs",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Button(
                        "Run Optimization",
                        id="mp-run",
                        n_clicks=0,
                        style={
                            "padding": "10px 14px",
                            "borderRadius": "10px",
                            "border": "1px solid #ccc",
                            "background": "white",
                            "cursor": "pointer",
                            "height": "40px",
                        },
                    ),
                    html.A(
                        "Open MIP (PDF)",
                        href="/assets/MIP.pdf",
                        target="_blank",  # opens in a new tab
                        style={"marginLeft": "8px"},
                    ),
                    html.Div(id="mp-after-run-links", style={"marginTop": "10px"}),
                                    ],
            ),

            html.Hr(style={"margin": "16px 0"}),

            dcc.Loading(
                type="default",
                children=[
                    html.Div(id="mp-status", style={"marginBottom": "10px"}),
                    html.Div(id="mp-obj", style={"fontWeight": 600, "marginBottom": "12px"}),

                    # --------- GRAPH 1: cashflow ----------
                    html.H4("Daily cash flow (cost by day)", style={"margin": "10px 0"}),
                    dcc.Graph(id="mp-cashflow-graph", figure={}, config={"displayModeBar": False}),

                    html.Hr(style={"margin": "16px 0"}),

                    # --------- GRAPH 2: stacked fleets ----------
                    html.H4("Daily planned fleets by failure-type (stacked)", style={"margin": "10px 0"}),
                    dcc.Graph(id="mp-stack-graph", figure={}, config={"displayModeBar": False}),

                    html.Hr(style={"margin": "16px 0"}),

                    html.H4("Maintenance Calendar", style={"margin": "10px 0"}),
                    html.Div(
                        style={"marginTop": "10px"},
                        children=[
                            fcc.FullCalendarComponent(
                                id="mp-calendar",
                                initialView="dayGridMonth",
                                initialDate=date.today().isoformat(),
                                headerToolbar={
                                    "left": "prev,next today",
                                    "center": "title",
                                    "right": "dayGridMonth,listWeek",
                                },
                                nowIndicator=True,
                                events=[],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )





def register_callbacks(app):

    def _fmt_money(x):
        try:
            return f"{float(x):,.2f}"
        except Exception:
            return "-"

    def _build_cashflow_figure(
        data: dict,
        df_plan: pd.DataFrame,
        df_setup: pd.DataFrame,
        baseline_counts: dict[int, int] | None = None,
    ) -> dict:
        days = list(map(int, data["D"]))
        day_cost = {int(d): 0.0 for d in days}

        # setup costs by setup_day_d
        if df_setup is not None and not df_setup.empty:
            for _, r in df_setup.iterrows():
                d = int(r["setup_day_d"])
                day_cost[d] += float(r.get("setup_cost", 0.0))

        # per (i,f) costs by scheduled_day_d:
        if df_plan is not None and not df_plan.empty:
            for _, r in df_plan.iterrows():
                d = int(r["scheduled_day_d"])
                i = r["asset_i"]
                f = r["failure_f"]

                phi = float(r.get("phi_if", 0.0))
                psi = float(r.get("psi_if", 0.0))

                penalty = float(data["alpha"][(i, f)]) * phi + float(data["beta"][(i, f)]) * psi

                mode = r.get("mode", "")
                if mode == "Regular predictive":
                    mode_cost = float(data["c_pr"][(i, f)])
                elif mode == "Emergency predictive":
                    mode_cost = float(data["c_u_pr"][(i, f)])
                elif mode == "Reactive":
                    mode_cost = float(data["c_re"][(i, f)])
                else:
                    mode_cost = 0.0

                day_cost[d] += penalty + mode_cost

        fig = go.Figure()

        # only add bars if there is something to show
        if (df_setup is not None and not df_setup.empty) or (df_plan is not None and not df_plan.empty):
            fig.add_bar(x=days, y=[day_cost[int(d)] for d in days], name="Optimized daily cost")

        # Baseline overlay: planned jobs per day (pre-optimizer)
        if baseline_counts is not None:
            fig.add_scatter(
                x=days,
                y=[baseline_counts.get(int(d), 0) for d in days],
                mode="lines+markers",
                name="Baseline planned jobs (pre-optimizer)",
                yaxis="y2",
            )

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Day",
            yaxis_title="Cost (cash flow)",
            barmode="group",
            legend=dict(orientation="h"),
        )

        # If baseline is shown, add a 2nd axis
        if baseline_counts is not None:
            fig.update_layout(
                yaxis2=dict(
                    title="Baseline planned jobs",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                )
            )

        return fig          


    def _build_stacked_fleets_figure(
        data: dict,
        df_plan: pd.DataFrame,
        baseline_counts: dict[int, int] | None = None,
    ) -> dict:
        days = list(map(int, data["D"]))
        failure_types = list(data["F"])

        fig = go.Figure()

        if df_plan is None or df_plan.empty:
            # still show baseline if available
            if baseline_counts is not None:
                fig.add_scatter(
                    x=days,
                    y=[baseline_counts.get(int(d), 0) for d in days],
                    mode="lines+markers",
                    name="Baseline planned jobs (pre-optimizer)",
                )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Day",
                yaxis_title="Number of fleets",
            )
            return fig

        # how many distinct failures each asset has (across horizon)
        asset_fail_count = df_plan.groupby("asset_i")["failure_f"].nunique().to_dict()

        only_f = {f: {d: 0 for d in days} for f in failure_types}
        multi_assets_per_day = {d: set() for d in days}

        for _, r in df_plan.iterrows():
            d = int(r["scheduled_day_d"])
            i = r["asset_i"]
            f = r["failure_f"]

            if asset_fail_count.get(i, 0) <= 1:
                only_f[f][d] += 1
            else:
                multi_assets_per_day[d].add(i)

        multi = {d: len(multi_assets_per_day[d]) for d in days}

        # Optimized stacked bars
        for f in failure_types:
            fig.add_bar(x=days, y=[only_f[f][d] for d in days], name=f"Optimized: only {f}")

        fig.add_bar(x=days, y=[multi[d] for d in days], name="Optimized: >1 failure")

        # Baseline overlay line (total planned jobs/day)
        if baseline_counts is not None:
            fig.add_scatter(
                x=days,
                y=[baseline_counts.get(int(d), 0) for d in days],
                mode="lines+markers",
                name="Baseline planned jobs (pre-optimizer)",
            )

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Day",
            yaxis_title="Number of fleets",
            barmode="stack",
            legend=dict(orientation="h"),
        )
        return fig
    
    def _parse_start_date(x):
        """Accept None/date/ISO string and return a date or None."""
        if x is None or x == "":
            return None
        if isinstance(x, date):
            return x
        if isinstance(x, str):
            return date.fromisoformat(x)  # expects "YYYY-MM-DD"
        return None

    def _shift_to_workday(d_int: int, workday: dict, horizon: int) -> int:
        """If day is not workday, move forward to next workday; if none, move backward."""
        d = int(d_int)
        d = max(1, min(horizon, d))

        # forward search
        dd = d
        while dd <= horizon and int(workday.get(dd, 1)) == 0:
            dd += 1
        if dd <= horizon:
            return dd

        # fallback: backward search
        dd = d
        while dd >= 1 and int(workday.get(dd, 1)) == 0:
            dd -= 1
        return max(1, dd)

    def _build_baseline_plan(data: dict) -> pd.DataFrame:
        """
        Baseline (pre-optimizer) plan:
        planned_day = max(RUL_if - LT_i, 0), then mapped to day index [1..horizon]
        and shifted to a workday.
        """
        horizon = len(data["D"])
        rows = []
        for (i, f), fif in data["F_if"].items():
            if int(fif) != 1:
                continue

            rul = int(data["RUL"][(i, f)])
            lt = int(data["LT"][i])

            planned_time = max(rul - lt, 0)  # <-- your requested formula

            # Map planned_time to scheduling day index in [1..horizon]
            # If planned_time=0, schedule at day 1 (earliest possible).
            planned_day = max(1, min(int(planned_time), horizon))

            # respect workday calendar
            planned_day = _shift_to_workday(planned_day, data["workday"], horizon)

            rows.append(
                {
                    "asset_i": i,
                    "failure_f": f,
                    "planned_time": int(planned_time),
                    "planned_day_d": int(planned_day),
                    "RUL_if": rul,
                    "LT_i": lt,
                }
            )

        df_base = pd.DataFrame(rows)
        if df_base.empty:
            return df_base

        return df_base.sort_values(["planned_day_d", "asset_i", "failure_f"], ignore_index=True)

    def _baseline_counts_by_day(data: dict, df_base: pd.DataFrame) -> dict[int, int]:
        days = list(map(int, data["D"]))
        counts = {d: 0 for d in days}
        if df_base is None or df_base.empty:
            return counts
        for d, g in df_base.groupby("planned_day_d"):
            counts[int(d)] = int(len(g))
        return counts

    def _baseline_costs(data: dict, df_base: pd.DataFrame) -> dict:
        """Estimate setup + penalty + mode costs for baseline plan (no solver)."""
        if df_base is None or df_base.empty:
            return {
                "setup_cost": 0.0,
                "phi_penalty_cost": 0.0,
                "psi_penalty_cost": 0.0,
                "penalty_cost": 0.0,
                "regular_mode_cost": 0.0,
                "emergency_mode_cost": 0.0,
                "reactive_mode_cost": 0.0,
                "mode_cost": 0.0,
                "total_cost": 0.0,
            }

        # Setup cost: charge once per asset that has any baseline event
        assets_with_any = set(df_base["asset_i"].tolist())
        setup_cost = float(sum(float(data["c_setup"][i]) for i in assets_with_any))

        phi_pen = 0.0
        psi_pen = 0.0
        reg_cost = 0.0
        react_cost = 0.0
        emer_cost = 0.0  # baseline rule won’t use emergency unless you decide to

        for _, r in df_base.iterrows():
            i = r["asset_i"]
            f = r["failure_f"]
            d0 = int(r["planned_day_d"])
            rul = int(r["RUL_if"])
            lt = int(r["LT_i"])

            phi = max(d0 - rul, 0)
            psi = max((rul - lt) - d0, 0)

            phi_pen += float(data["alpha"][(i, f)]) * float(phi)
            psi_pen += float(data["beta"][(i, f)]) * float(psi)

            # Baseline mode rule:
            # tardy => reactive, else regular predictive
            if phi > 0:
                react_cost += float(data["c_re"][(i, f)])
            else:
                reg_cost += float(data["c_pr"][(i, f)])

        penalty_cost = phi_pen + psi_pen
        mode_cost = reg_cost + react_cost + emer_cost
        total_cost = setup_cost + penalty_cost + mode_cost

        return {
            "setup_cost": setup_cost,
            "phi_penalty_cost": phi_pen,
            "psi_penalty_cost": psi_pen,
            "penalty_cost": penalty_cost,
            "regular_mode_cost": reg_cost,
            "emergency_mode_cost": emer_cost,
            "reactive_mode_cost": react_cost,
            "mode_cost": mode_cost,
            "total_cost": total_cost,
        }

    def _check_capacity_infeasible(df_base: pd.DataFrame, R: int) -> tuple[bool, dict]:
        """
        Returns (is_infeasible, details)
        details = {"max_day": int|None, "max_count": int, "days_over": {day:count}}
        """
        if df_base is None or df_base.empty:
            return False, {"max_day": None, "max_count": 0, "days_over": {}}

        counts = df_base.groupby("planned_day_d").size().to_dict()
        days_over = {int(d): int(c) for d, c in counts.items() if int(c) > int(R)}

        if not days_over:
            max_day = max(counts, key=counts.get) if counts else None
            max_count = int(counts[max_day]) if max_day is not None else 0
            return False, {"max_day": int(max_day) if max_day is not None else None, "max_count": max_count, "days_over": {}}

        # pick the worst day
        worst_day = max(days_over, key=days_over.get)
        return True, {"max_day": int(worst_day), "max_count": int(days_over[worst_day]), "days_over": days_over}

    # ---------------------------------------------------------
    # Run optimization + update KPIs + graphs + calendar + modal data
    # ---------------------------------------------------------
    @app.callback(
        Output("mp-status", "children"),
        Output("mp-obj", "children"),
        Output("mp-cashflow-graph", "figure"),
        Output("mp-stack-graph", "figure"),
        Output("mp-calendar", "events"),
        Output("mp-kpi-assets", "children"),
        Output("mp-kpi-assets-fail", "children"),
        Output("mp-kpi-failures", "children"),
        Output("mp-kpi-setup", "children"),
        Output("mp-kpi-penalty", "children"),
        Output("mp-kpi-mode", "children"),
        Output("mp-kpi-total", "children"),
        Output("mp-kpi-status", "children"),
        Output("mp-last-summary", "data"),
        Output("mp-after-run-links", "children"),
        Input("mp-run", "n_clicks"),
        Input("shared-inputs", "data"),
        Input("mp-planner-controls", "data"),   # <-- ADD THIS
        State("mp-solver", "value"),
        prevent_initial_call=False,
    )
    
    def run_optimization(n_clicks, shared_inputs,planner_controls, solver_name):
        try:
            shared_inputs = shared_inputs or {}
            planner_controls = planner_controls or {"R_override": None, "work_all_days": False}

            # --- Parse start_date safely ---
            parsed_start = _parse_start_date(shared_inputs.get("start_date", None))

            # --- build data FIRST (always) ---
            data = _base_data(
                n_assets=int(shared_inputs.get("n_assets", 100)),
                horizon=int(shared_inputs.get("horizon", 31)),
                failure_types=shared_inputs.get("failure_types", ["motor", "battery"]),
                p_fail=float(shared_inputs.get("p_fail", 0.5)),
                seed=int(shared_inputs.get("seed", 42)),
                holiday_country=str(shared_inputs.get("holiday_country", "BE")),
                R=int(shared_inputs.get("R", 10)),
                start_date=parsed_start,
            )


            # override R if user changed it
            if planner_controls.get("R_override") is not None:
                data["R"] = float(int(planner_controls["R_override"]))  # model expects float

            # work on holidays/weekends: make all days workdays
            if planner_controls.get("work_all_days"):
                data["workday"] = {d: 1 for d in data["D"]}

            # --- Baseline plan (always) ---
            df_base = _build_baseline_plan(data)
            base_counts = _baseline_counts_by_day(data, df_base)
            is_infeasible, cap_info = _check_capacity_infeasible(df_base, int(data["R"]))
            # baseline calendar events
            events = []
            if df_base is not None and not df_base.empty:
                for _, r in df_base.iterrows():
                    d = int(r["planned_day_d"])
                    start_dt = data["start_date"] + timedelta(days=d - 1)
                    events.append({
                        "title": f"Baseline · {r['asset_i']} · {r['failure_f']} (planned={r['planned_time']})",
                        "start": start_dt.isoformat(),
                        "allDay": True,
                        "display": "auto",
                    })

            # KPIs that can be shown even before optimization
            n_assets = len(data["I"])
            assets_with_fail = sum(
                1 for i in data["I"]
                if any(data["F_if"][(i, f)] == 1 for f in data["F"])
            )
            total_failures = sum(data["F_if"][(i, f)] for i in data["I"] for f in data["F"])
            
            # By default (baseline-only), keep costs as "-"
            baseline_summary = _baseline_costs(data, df_base)

            kpi_setup = _fmt_money(baseline_summary["setup_cost"])
            kpi_penalty = _fmt_money(baseline_summary["penalty_cost"])
            kpi_mode = _fmt_money(baseline_summary["mode_cost"])
            kpi_total = _fmt_money(baseline_summary["total_cost"])
            kpi_status = (html.Span("Infeasible", style={"color": "#b00020", "fontWeight": 700})
                if is_infeasible
                else html.Span("Feasible", style={"color": "#1bb31b", "fontWeight": 700})
            )
            # baseline graphs: no optimized plan yet
            empty_plan = pd.DataFrame()
            empty_setup = pd.DataFrame()
            fig_cash = _build_cashflow_figure(data, empty_plan, empty_setup, baseline_counts=base_counts)
            fig_stack = _build_stacked_fleets_figure(data, empty_plan, baseline_counts=base_counts)

            # Determine what triggered the callback
            trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else ""

            # If NOT triggered by the button, stop here (page load / inputs change)
            if trig != "mp-run":
                if is_infeasible:
                    worst_day = cap_info["max_day"]
                    worst_cnt = cap_info["max_count"]
                    status = html.Div(
                        [
                            html.B("Initial plan infeasible. "),
                            html.Span("Maintenance resource capacity exceeded. "),
                            html.Span(f"Capacity R = {int(data['R'])} per day, but day {worst_day} has {worst_cnt} maintenances. "),
                            html.Span("Adjust inputs or click Run Optimization to find a feasible schedule."),
                        ],
                        style={"color": "#b00020"},
                    )
                else:
                    status = html.Div(
                        [
                            html.Span("Initial plan generated (baseline cost estimate). "),
                            html.Span("Click Run Optimization to solve."),
                        ]
                    )
                return (
                    status,
                    "Objective value: - (baseline only)",
                    fig_cash,
                    fig_stack,
                    events,
                    str(n_assets),
                    str(assets_with_fail),
                    str(total_failures),
                    kpi_setup,
                    kpi_penalty,
                    kpi_mode,
                    kpi_total,
                    kpi_status, 
                    baseline_summary,
                    "",#  no links until optimization
                )

            # --- If triggered by Run button: solve optimizer and overlay results ---
            obj, df_plan, df_setup, summary = solve_instance(data, solver_name=solver_name)

            # costs
            kpi_setup = _fmt_money(summary.get("setup_cost", 0.0))
            kpi_penalty = _fmt_money(summary.get("penalty_cost", 0.0))
            kpi_mode = _fmt_money(summary.get("mode_cost", 0.0))
            kpi_total = _fmt_money(summary.get("total_cost", obj))

            # figures with optimized + baseline overlay
            fig_cash = _build_cashflow_figure(data, df_plan, df_setup, baseline_counts=base_counts)
            fig_stack = _build_stacked_fleets_figure(data, df_plan, baseline_counts=base_counts)

            # append optimized events
            if df_plan is not None and not df_plan.empty:
                for _, r in df_plan.iterrows():
                    d = r.get("scheduled_day_d", None)
                    if d is None or (isinstance(d, float) and pd.isna(d)):
                        continue
                    start_dt = data["start_date"] + timedelta(days=int(d) - 1)
                    events.append({
                        "title": f"Optimized · {r['asset_i']} · {r['failure_f']} ({r['mode']})",
                        "start": start_dt.isoformat(),
                        "allDay": True,
                        "display": "auto",
                    })

            status = html.Div([
                html.Span("Solved successfully. "),
                html.Span(f"Solver: {solver_name}. "),
                html.Span("Termination: optimal/feasible."),
            ])

            links = html.Div(
                style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap"},
                children=[
                    html.A(
                        "Open model",
                        href=f"/assets/maintenance_model.txt?v={n_clicks}",  # cache buster
                        target="_blank",
                    ),
                ],
            )

            return (
                status,
                f"Objective value: {float(obj):,.4f}",
                fig_cash,
                fig_stack,
                events,
                str(n_assets),
                str(assets_with_fail),
                str(total_failures),
                kpi_setup,
                kpi_penalty,
                kpi_mode,
                kpi_total,
                html.Span("Feasible", style={"color": "#1bb31b", "fontWeight": 700}),
                summary,
                links,
            )

        except Exception as e:
            tb = traceback.format_exc(limit=8)
            status = html.Div(
                [
                    html.B("Optimization failed: "),
                    html.Span(str(e)),
                    html.Details([html.Summary("Traceback"), html.Pre(tb)]),
                ],
                style={"color": "#b00020"},
            )
            empty_fig = go.Figure()
            return (
                status,
                "Objective value: -",
                empty_fig,
                empty_fig,
                [],
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                None,
                "",
            )

    # ----------------------------------------
    # Toggle modal open state (gear / close)
    # ----------------------------------------
    @app.callback(
        Output("mp-cost-modal-open", "data"),
        Input("mp-gear", "n_clicks"),
        Input("mp-cost-close", "n_clicks"),
        State("mp-cost-modal-open", "data"),
        prevent_initial_call=True,
    )
    def toggle_cost_modal(n_gear, n_close, is_open):
        is_open = bool(is_open)
        trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
        if trig == "mp-gear":
            return True
        if trig == "mp-cost-close":
            return False
        return is_open

    @app.callback(
        Output("mp-cost-modal", "style"),
        Input("mp-cost-modal-open", "data"),
    )
    def show_hide_modal(is_open):
        return {"display": "block"} if is_open else {"display": "none"}

    @app.callback(
        Output("mp-planner-controls", "data"),
        Output("mp-R-value", "children"),
        Input("mp-R-minus", "n_clicks"),
        Input("mp-R-plus", "n_clicks"),
        Input("mp-R-reset", "n_clicks"),
        Input("mp-work-all-days", "value"),
        State("mp-planner-controls", "data"),
        State("shared-inputs", "data"),
        prevent_initial_call=False,
    )
    def update_planner_controls(n_minus, n_plus, n_reset, work_all_days_value, controls, shared_inputs):
        controls = controls or {"R_override": None, "work_all_days": False}
        shared_inputs = shared_inputs or {}

        default_R = int(shared_inputs.get("R", 10))
        R_current = int(controls["R_override"]) if controls.get("R_override") is not None else default_R

        trig = dash.callback_context.triggered[0]["prop_id"].split(".")[0] if dash.callback_context.triggered else ""

        if trig == "mp-R-minus":
            R_current = max(1, R_current - 1)
            controls["R_override"] = R_current
        elif trig == "mp-R-plus":
            R_current = R_current + 1
            controls["R_override"] = R_current
        elif trig == "mp-R-reset":
            controls["R_override"] = None
            R_current = default_R

        controls["work_all_days"] = ("Y" in (work_all_days_value or []))

        return controls, str(R_current)
    

    @app.callback(
        Output("mp-modal-status", "children"),
        Input("mp-kpi-status", "children"),
    )
    def show_modal_status(kpi_status):
        if not kpi_status or kpi_status == "-":
            return html.Div("Status: -", style={"opacity": 0.7})

        return html.Div(
            [html.Span("Status: ", style={"fontWeight": 600}), kpi_status]
        )
    @app.callback(
        Output("mp-cost-breakdown", "children"),
        Input("mp-last-summary", "data"),
    )
    def render_cost_breakdown(summary):
        if not summary:
            return html.Div("Run optimization to see cost details.", style={"opacity": 0.75})

        return html.Div(
            [
                html.Div([html.B("Setup cost: "), _fmt_money(summary.get("setup_cost", 0.0))]),

                html.Div(style={"marginTop": "8px", "fontWeight": 600}, children="Penalty costs"),
                html.Div([html.Span("α·φ cost: "), html.B(_fmt_money(summary.get("phi_penalty_cost", 0.0)))]),
                html.Div([html.Span("β·ψ cost: "), html.B(_fmt_money(summary.get("psi_penalty_cost", 0.0)))]),
                html.Div([html.Span("Total penalty cost: "), html.B(_fmt_money(summary.get("penalty_cost", 0.0)))]),

                html.Div(style={"marginTop": "8px", "fontWeight": 600}, children="Mode costs"),
                html.Div([html.Span("Regular predictive (c_pr): "), html.B(_fmt_money(summary.get("regular_mode_cost", 0.0)))]),
                html.Div([html.Span("Emergency predictive (c_u-pr): "), html.B(_fmt_money(summary.get("emergency_mode_cost", 0.0)))]),
                html.Div([html.Span("Reactive (c_re): "), html.B(_fmt_money(summary.get("reactive_mode_cost", 0.0)))]),
                html.Div([html.Span("Total mode cost: "), html.B(_fmt_money(summary.get("mode_cost", 0.0)))]),

                html.Hr(style={"margin": "10px 0"}),

                html.Div([html.B("Total cost: "), _fmt_money(summary.get("total_cost", 0.0))]),
            ]
        )