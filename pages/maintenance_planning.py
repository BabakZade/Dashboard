# pages/maintenance_planning.py
from __future__ import annotations

import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import pyomo.environ as pyo

from datetime import date, timedelta
import full_calendar_component as fcc


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

    # Sets
    m.I = pyo.Set(initialize=data["I"])
    m.F = pyo.Set(initialize=data["F"])
    m.D = pyo.Set(initialize=data["D"], ordered=True)
    m.IF = pyo.Set(initialize=[(i, f) for i in data["I"] for f in data["F"]])
    m.IFD = pyo.Set(initialize=[(i, f, d) for i in data["I"] for f in data["F"] for d in data["D"]])

    # Params
    m.c_setup = pyo.Param(m.I, initialize=data["c_setup"], within=pyo.NonNegativeReals)
    m.alpha = pyo.Param(m.IF, initialize=data["alpha"], within=pyo.NonNegativeReals)
    m.beta = pyo.Param(m.IF, initialize=data["beta"], within=pyo.NonNegativeReals)
    m.gamma = pyo.Param(m.IF, initialize=data["gamma"], within=pyo.NonNegativeReals)
    m.RUL = pyo.Param(m.IF, initialize=data["RUL"], within=pyo.NonNegativeReals)
    m.F_if = pyo.Param(m.IF, initialize=data["F_if"], within=pyo.Binary)
    m.LT = pyo.Param(m.I, initialize=data["LT"], within=pyo.NonNegativeReals)

    # M big-M
    m.M = pyo.Param(initialize=float(data["M"]), within=pyo.PositiveReals)

    # Workday indicator (1=workday, 0=weekend/holiday) for each model day d
    #   workday_d ∈ {0,1}
    m.workday = pyo.Param(m.D, initialize=data["workday"], within=pyo.Binary)

    # Vars
    m.x = pyo.Var(m.IFD, within=pyo.Binary)      # schedule decision
    m.y = pyo.Var(m.I, m.D, within=pyo.Binary)   # setup day
    m.e = pyo.Var(m.IF, within=pyo.Binary)       # regular predictive
    m.l = pyo.Var(m.IF, within=pyo.Binary)       # reactive
    m.u = pyo.Var(m.IF, within=pyo.Binary)       # emergency predictive

    # Objective
    #   min  Σ_{i,d} c_setup_i * y_id  +  Σ_{i,f} ( alpha_if*e_if + gamma_if*u_if + beta_if*l_if )
    def obj_rule(m):
        setup_cost = sum(m.c_setup[i] * m.y[i, d] for i in m.I for d in m.D)
        mode_cost = sum(
            m.alpha[i, f] * m.e[i, f] + m.gamma[i, f] * m.u[i, f] + m.beta[i, f] * m.l[i, f]
            for (i, f) in m.IF
        )
        return setup_cost + mode_cost

    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # -----------------------
    # Constraints
    # -----------------------

    # (0) Daily total scheduling constraint
    #   Σ_{i∈I} Σ_{f∈F} x_ifd  ≤  |I|·|F|·workday_d     ∀ d
    nIF = len(data["I"]) * len(data["F"])

    def daily_total_rule(m, d):
        return sum(m.x[i, f, d] for i in m.I for f in m.F) <= nIF * m.workday[d]

    m.DailyTotal = pyo.Constraint(m.D, rule=daily_total_rule)

    # (1) Schedule (exactly one day if failure within horizon)
    #   Σ_{d∈D} x_ifd = F_if        ∀ (i,f)
    def schedule_rule(m, i, f):
        return sum(m.x[i, f, d] for d in m.D) == m.F_if[i, f]

    m.Schedule = pyo.Constraint(m.IF, rule=schedule_rule)

    # (2) Reactive timing constraint (only for F_if = 1)
    #   Σ_{d∈D} d·x_ifd  − RUL_if  ≤  M·l_if      ∀ (i,f) with F_if=1
    def reactive_rule(m, i, f):
        if pyo.value(m.F_if[i, f]) < 0.5:
            return pyo.Constraint.Skip
        return sum(d * m.x[i, f, d] for d in m.D) - m.RUL[i, f] <= m.M * m.l[i, f]

    m.Reactive = pyo.Constraint(m.IF, rule=reactive_rule)

    # (3) Regular predictive timing constraint (only for F_if = 1)
    #   (RUL_if − LT_i) − Σ_{d∈D} d·x_ifd  ≤  M·e_if      ∀ (i,f) with F_if=1
    def regular_pred_rule(m, i, f):
        if pyo.value(m.F_if[i, f]) < 0.5:
            return pyo.Constraint.Skip
        return (m.RUL[i, f] - m.LT[i]) - sum(d * m.x[i, f, d] for d in m.D) <= m.M * m.e[i, f]

    m.RegularPredictive = pyo.Constraint(m.IF, rule=regular_pred_rule)

    # (4) Mode selection (only for F_if = 1)
    #   l_if + e_if + u_if = F_if      ∀ (i,f) with F_if=1
    def mode_rule(m, i, f):
        if pyo.value(m.F_if[i, f]) < 0.5:
            return pyo.Constraint.Skip
        return m.l[i, f] + m.e[i, f] + m.u[i, f] == m.F_if[i, f]

    m.ModeSelect = pyo.Constraint(m.IF, rule=mode_rule)

    # (5) Link y to x (single constraint)
    #   Σ_{f∈F} x_ifd ≤ |F| · y_id     ∀ (i,d)
    # Meaning: if any x_ifd=1 then y_id must be 1 (setup paid)
    nF = len(data["F"])

    def y_link_rule(m, i, d):
        return sum(m.x[i, f, d] for f in m.F) <= nF * m.y[i, d]

    m.YLink = pyo.Constraint(m.I, m.D, rule=y_link_rule)

    return m


def solve_instance(data: dict, solver_name: str) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    model = build_model(data)

    # Optional: export model for debugging
    model.write("maintenance_model.lp", io_options={"symbolic_solver_labels": True})

    solver = pyo.SolverFactory(solver_name)
    if (solver is None) or (not solver.available(False)):
        raise RuntimeError(f"Solver '{solver_name}' not available. Install it (glpk/cbc/highs) or switch solver.")

    res = solver.solve(model, tee=True)  # prints solver log to console

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

    obj = float(pyo.value(model.OBJ))
    return obj, df_plan, df_setup


# ---------------------------
# Data generator (everything related to data lives here)
# ---------------------------
def _base_data(
    n_assets: int = 100,
    horizon: int = 31,
    failure_types: list[str] | None = None,
    p_fail: float = 0.5,
    seed: int | None = 42,
    start_date: date | None = None,      # Day 1 = run day
    holiday_country: str = "BE",         # Belgium by default
) -> dict:
    """
    Random instance generator + calendar workday creation.

    - |I| = n_assets
    - |D| = horizon, D = {1,...,horizon}
    - F_if ~ Bernoulli(p_fail)
    - If F_if=1 => RUL_if in [1, horizon]
      If F_if=0 => RUL_if = horizon+1 (outside horizon)
    - LT_i in [0,6]
    - c_setup_i in {100,150,...,500}
    - Costs satisfy: alpha < gamma << beta
    - M = horizon
    - workday[d] = 0 if weekend/holiday else 1, based on start_date (Day 1)
    """
    rng = np.random.default_rng(seed)

    if failure_types is None:
        failure_types = ["motor", "battery"]

    I = [f"Fleet_{k+1}" for k in range(n_assets)]
    F = list(failure_types)
    D = list(range(1, horizon + 1))

    # Lead time: 0..6
    LT = {i: int(rng.integers(0, 7)) for i in I}

    # Setup cost: 100..500 step 50
    setup_levels = np.arange(100, 501, 50)
    c_setup = {i: int(rng.choice(setup_levels)) for i in I}

    # Per (i,f): failure indicator and RUL
    IF = [(i, f) for i in I for f in F]
    F_if = {k: int(rng.random() < p_fail) for k in IF}

    RUL = {}
    for (i, f) in IF:
        if F_if[(i, f)] == 1:
            RUL[(i, f)] = int(rng.integers(1, horizon + 1))  # 1..horizon
        else:
            RUL[(i, f)] = horizon + 1

    # Costs: alpha < gamma << beta
    alpha = {k: int(rng.integers(50, 151)) for k in IF}
    gamma = {k: int(rng.integers(200, 401)) for k in IF}
    beta = {k: int(rng.integers(2000, 4001)) for k in IF}

    for k in IF:
        if alpha[k] >= gamma[k]:
            gamma[k] = alpha[k] + int(rng.integers(50, 151))

    # Day 1 = run day
    if start_date is None:
        start_date = date.today()

    # Holiday calendar (optional)
    hol_cal = None
    if _holidays_lib is not None:
        try:
            hol_cal = _holidays_lib.country_holidays(holiday_country)
        except Exception:
            hol_cal = None

    # workday[d] = 1 for workdays, 0 for weekends/holidays
    workday = {}
    holiday_name = {}  # for UI labels (not used by Pyomo)
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
        "M": float(horizon),
        "LT": LT,
        "c_setup": c_setup,
        "RUL": RUL,
        "F_if": F_if,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "workday": workday,             # ✅ used by Pyomo
        "start_date": start_date,       # ✅ used by calendar mapping
        "holiday_name": holiday_name,   # ✅ used by calendar background labels
    }


def layout():
    card_style = {"border": "1px solid #ddd", "borderRadius": "12px", "padding": "16px"}
    return html.Div(
        style=card_style,
        children=[
            html.H3("Maintenance Planning", style={"marginTop": 0}),
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
                                    {"label": "GLPK (glpk)", "value": "glpk"},
                                    {"label": "CBC (cbc)", "value": "cbc"},
                                    {"label": "HiGHS (highs)", "value": "highs"},
                                    {"label": "Gurobi (gurobi)", "value": "gurobi"},
                                ],
                                value="glpk",
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
                    dcc.Link("Go to Inputs page", href="/inputs", style={"marginLeft": "8px"}),
                ],
            ),

            html.Hr(style={"margin": "16px 0"}),

            dcc.Loading(
                type="default",
                children=[
                    html.Div(id="mp-status", style={"marginBottom": "10px"}),
                    html.Div(id="mp-obj", style={"fontWeight": 600, "marginBottom": "12px"}),

                    html.H4("Planned maintenance per asset and failure type", style={"margin": "10px 0"}),
                    dash_table.DataTable(
                        id="mp-plan-table",
                        columns=[
                            {"name": "asset_i", "id": "asset_i"},
                            {"name": "failure_f", "id": "failure_f"},
                            {"name": "scheduled_day_d", "id": "scheduled_day_d", "type": "numeric"},
                            {"name": "RUL_if", "id": "RUL_if", "type": "numeric"},
                            {"name": "LT_i", "id": "LT_i", "type": "numeric"},
                            {"name": "mode", "id": "mode"},
                        ],
                        data=[],
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={"padding": "8px", "fontFamily": "system-ui", "fontSize": "13px"},
                        style_header={"fontWeight": "600"},
                    ),

                    html.H4("Setup days (y_{id}=1)", style={"margin": "16px 0 10px"}),
                    dash_table.DataTable(
                        id="mp-setup-table",
                        columns=[
                            {"name": "asset_i", "id": "asset_i"},
                            {"name": "setup_day_d", "id": "setup_day_d", "type": "numeric"},
                            {"name": "setup_cost", "id": "setup_cost", "type": "numeric"},
                        ],
                        data=[],
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={"padding": "8px", "fontFamily": "system-ui", "fontSize": "13px"},
                        style_header={"fontWeight": "600"},
                    ),

                    html.Hr(style={"margin": "16px 0"}),

                    html.H4("Maintenance Calendar", style={"margin": "10px 0"}),
                    html.Div(
                        style={"marginTop": "10px"},
                        children=[
                            fcc.FullCalendarComponent(
                                id="mp-calendar",
                                initialView="dayGridMonth",
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
    @app.callback(
        Output("mp-status", "children"),
        Output("mp-obj", "children"),
        Output("mp-plan-table", "data"),
        Output("mp-setup-table", "data"),
        Output("mp-calendar", "events"),
        Input("mp-run", "n_clicks"),
        State("mp-solver", "value"),
        State("shared-inputs", "data"),
        prevent_initial_call=True,
    )
    def run_optimization(n_clicks, solver_name, shared_inputs):
        try:
            shared_inputs = shared_inputs or {}

            horizon = int(shared_inputs.get("horizon", 31))
            n_assets = int(shared_inputs.get("n_assets", 100))
            p_fail = float(shared_inputs.get("p_fail", 0.5))
            seed = int(shared_inputs.get("seed", 42))
            holiday_country = str(shared_inputs.get("holiday_country", "BE"))

            # Day 1 = the day the user clicks Run Optimization
            start_dt = date.today()

            # ✅ everything data-related is generated here (including workday & holidays)
            data = _base_data(
                n_assets=n_assets,
                horizon=horizon,
                p_fail=p_fail,
                seed=seed,
                start_date=start_dt,
                holiday_country=holiday_country,
            )

            obj, df_plan, df_setup = solve_instance(data, solver_name)

            # -------------------------
            # Calendar events:
            #  1) Background shading for weekends/holidays (based on workday[d])
            #  2) Maintenance events colored by daily share of total scheduled events
            # -------------------------
            total_events = int(len(df_plan))

            # Background events for all horizon days
            bg_events = []
            for d in data["D"]:
                dt = start_dt + timedelta(days=d - 1)
                start_str = dt.isoformat()
                end_str = (dt + timedelta(days=1)).isoformat()

                if data["workday"][d] == 0:
                    # Weekend or holiday
                    label = data.get("holiday_name", {}).get(d, "")
                    bg_events.append(
                        {
                            "start": start_str,
                            "end": end_str,
                            "allDay": True,
                            "display": "background",
                            "backgroundColor": "#f0f0f0" if not label else "#e7d7ff",
                            "title": label,
                        }
                    )

            # Maintenance events
            maint_events = []
            if total_events > 0:
                day_counts = df_plan.groupby("scheduled_day_d").size().to_dict()

                def color_for_pct(pct: float) -> str:
                    if pct > 20.0:
                        return "red"
                    if 10.0 <= pct <= 20.0:
                        return "gold"
                    return "green"

                for _, r in df_plan.iterrows():
                    d = int(r["scheduled_day_d"])
                    dt = start_dt + timedelta(days=d - 1)
                    start_str = dt.isoformat()
                    end_str = (dt + timedelta(days=1)).isoformat()

                    pct = 100.0 * float(day_counts.get(d, 0)) / float(total_events)
                    color = color_for_pct(pct)

                    maint_events.append(
                        {
                            "title": f'{r["asset_i"]} • {r["failure_f"]} ({r["mode"]})',
                            "start": start_str,
                            "end": end_str,
                            "allDay": True,
                            "backgroundColor": color,
                            "borderColor": color,
                            "textColor": ("white" if color == "red" else "black"),
                        }
                    )

            events = bg_events + maint_events

            status = html.Div(
                [
                    "✅ Solved using ",
                    html.B(solver_name),
                    ". Day 1 = ",
                    html.B(start_dt.isoformat()),
                    ". Weekends/holidays are shaded in the calendar (and blocked in optimization).",
                ],
                style={"color": "#0a7"},
            )

            return (
                status,
                f"Objective value (Min Z): {obj:,.2f}",
                df_plan.to_dict("records"),
                df_setup.to_dict("records"),
                events,
            )

        except Exception as ex:
            status = html.Div(["❌ Could not solve the model: ", str(ex)], style={"color": "#b00"})
            return status, "", [], [], []