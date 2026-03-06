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
import hashlib


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

import pyomo.environ as pyo

DEFAULT_PALETTE = [
    "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99",
    "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
]

# ---------------------------
# Pyomo model (WORKDAY-BASED DURATIONS)
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

    D_list = list(data["D"])
    D_set = set(D_list)
    D_min = min(D_list)
    D_max = max(D_list)

    # -----------------------
    # Parameters
    # -----------------------
    m.c_setup = pyo.Param(m.I, initialize=data["c_setup"], within=pyo.NonNegativeReals)

    # penalty weights
    m.alpha = pyo.Param(m.IF, initialize=data["alpha"], within=pyo.NonNegativeReals)
    m.beta = pyo.Param(m.IF, initialize=data["beta"], within=pyo.NonNegativeReals)

    # mode costs
    m.c_pr = pyo.Param(m.IF, initialize=data["c_pr"], within=pyo.NonNegativeReals)
    m.c_u_pr = pyo.Param(m.IF, initialize=data["c_u_pr"], within=pyo.NonNegativeReals)
    m.c_re = pyo.Param(m.IF, initialize=data["c_re"], within=pyo.NonNegativeReals)

    m.RUL = pyo.Param(m.IF, initialize=data["RUL"], within=pyo.NonNegativeReals)
    m.F_if = pyo.Param(m.IF, initialize=data["F_if"], within=pyo.Binary)
    m.LT = pyo.Param(m.I, initialize=data["LT"], within=pyo.NonNegativeReals)

    # durations (interpreted as WORKDAYS)
    m.tau_s = pyo.Param(m.I, initialize=data["tau_s"], within=pyo.PositiveIntegers)          # tau_i^s (workdays)
    m.tau_exec = pyo.Param(m.IF, initialize=data["tau_exec"], within=pyo.PositiveIntegers)  # tau_if  (workdays)

    # fleet daily capacity: max number of ONGOING EXECUTIONS per calendar day (counted in workday-time)
    m.R = pyo.Param(initialize=float(data["R"]), within=pyo.PositiveReals)

    # Big-M
    m.M = pyo.Param(initialize=float(data["M"]), within=pyo.PositiveReals)

    # Workday indicator (1=workday, 0=weekend/holiday)
    m.workday = pyo.Param(m.D, initialize=data["workday"], within=pyo.Binary)

    # -----------------------
    # Decision variables
    # -----------------------
    m.x = pyo.Var(m.IFD, within=pyo.Binary)  # x_ifd: execution starts on day d
    m.y = pyo.Var(m.IFD, within=pyo.Binary)  # y_ifd: setup starts on day d (associated to f)

    m.e = pyo.Var(m.IF, within=pyo.Binary)   # e_if: regular predictive
    m.l = pyo.Var(m.IF, within=pyo.Binary)   # l_if: reactive
    m.u = pyo.Var(m.IF, within=pyo.Binary)   # u_if: urgent predictive

    m.phi = pyo.Var(m.IF, within=pyo.NonNegativeReals)  # varphi_if
    m.psi = pyo.Var(m.IF, within=pyo.NonNegativeReals)  # psi_if

    # -----------------------
    # Precompute previous workday lookup
    # prev_wd[(d,k)] = k-th previous WORKDAY strictly before d, else None
    # Example: if 8,9 are weekend and d=10 then prev_wd[(10,2)] = 6
    # -----------------------
    workday_dict = data["workday"]  # raw dict day->0/1 for faster python checks

    prev_wd: dict[tuple[int, int], int | None] = {}

    def t1_working_days_before(the_day: int, t1: int, workday_dict: dict[int, int]) -> int | None:
        """
        Return the calendar day that is t1 WORKDAYS strictly before 'the_day'.
        Days are assumed to be 1..horizon.
        If not enough previous workdays exist, return None.
        """
        if t1 <= 0:
            return int(the_day)

        day = int(the_day)
        remaining = int(t1)

        while remaining > 0:
            day -= 1
            if day < 1:
                return None
            if int(workday_dict.get(day, 0)) == 1:
                remaining -= 1

        return day

   

    # -----------------------
    # Objective
    # -----------------------
    def obj_rule(m):
        # Minimize: setup + penalties + mode costs
        # setup_cost = sum_{i,f,d} c_setup[i] * y_ifd
        setup_cost = sum(m.c_setup[i] * m.y[i, f, d] for i in m.I for f in m.F for d in m.D)

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

    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # -----------------------
    # Constraints
    # -----------------------

    # (C1) Scheduling within horizon:
    #   sum_{d in D} x_{i,f,d} = F_if(i,f)   for all (i,f)
    def schedule_x_rule(m, i, f):
        return sum(m.x[i, f, d] for d in m.D) == m.F_if[i, f]
    m.ScheduleX = pyo.Constraint(m.IF, rule=schedule_x_rule)

    #   sum_{d in D} y_{i,f,d} <= F_if(i,f)   for all (i,f)
    def schedule_y_rule(m, i, f):
        return sum(m.y[i, f, d] for d in m.D) <= m.F_if[i, f]
    m.ScheduleY = pyo.Constraint(m.IF, rule=schedule_y_rule)

    # (C10) Workday restriction for execution starts:
    #   x_{i,f,d} <= workday_d   for all (i,f,d)
    def workday_rule(m, i, f, d):
        return m.x[i, f, d] <= m.workday[d]

    m.Workday = pyo.Constraint(m.IFD, rule=workday_rule)

    # (C11) Workday restriction for setup starts:
    #   y_{i,f,d} <= workday_d   for all (i,f,d)
    def setup_workday_rule(m, i, f, d):
        return m.y[i, f, d] <= m.workday[d]

    m.SetupWorkday = pyo.Constraint(m.IFD, rule=setup_workday_rule)

    # (C2) Link (setup-before OR continuation on previous WORKDAY):
    #   x_{i,f,d} <= y_{i,f, prev_wd(d, tau_s(i)) } + 1{asset i executing at prev_wd(d,1) via some f2 != f}
    #
    # Interpretation:
    # - setup must start tau_s(i) WORKDAYS before day d, OR
    # - asset was already executing another fault on the previous WORKDAY (continuation of visit).
    def link_rule(m, i, f, d):
        # Skip non-workdays (x is already forced to 0 by Workday constraint)
        if int(pyo.value(m.workday[d])) == 0:
            return pyo.Constraint.Skip

        # ---- setup term: y_{if, d - tau_s(i)} in WORKDAYS
        tau_s = int(pyo.value(m.tau_s[i]))
        setup_day = t1_working_days_before(d, tau_s, workday_dict)

        setup_ok = 0
        if setup_day is not None and setup_day in D_set:
            setup_ok = m.y[i, f, setup_day]

        # ---- continuation term: sum_{f2 != f} x_{i f2, d - tau_exec(i,f2)} in WORKDAYS
        cont_ok = 0
        for f2 in m.F:
            if f2 == f:
                continue

            tau2 = int(pyo.value(m.tau_exec[i, f2]))
            maint_day = t1_working_days_before(d, tau2, workday_dict)

            if maint_day is not None and maint_day in D_set:
                cont_ok += m.x[i, f2, maint_day]

        return m.x[i, f, d] <= setup_ok + cont_ok

    m.Link = pyo.Constraint(m.IFD, rule=link_rule)

    # (C3) Continuity (no overlap per asset-day; WORKDAY durations):
    #   ongoing_exec(i,d) + ongoing_setup(i,d) <= 1   for all i,d
    #
    # ongoing_exec(i,d) = sum_{f} sum_{d0} x_{i,f,d0} * 1{d is within first tau_exec(i,f) WORKDAYS after d0}
    # ongoing_setup(i,d) = sum_{f} sum_{d0} y_{i,f,d0} * 1{d is within first tau_s(i) WORKDAYS after d0}
    def continuity_rule(m, i, d):
        ongoing_exec = 0
        for f in m.F:
            tau = int(pyo.value(m.tau_exec[i, f]))  # workdays
            for d0 in m.D:
                if d0 > d:
                    continue

                wd_count = 0
                t = d0
                while t <= d:
                    if int(pyo.value(m.workday[t])) == 1:
                        wd_count += 1
                    t += 1

                if 1 <= wd_count <= tau:
                    ongoing_exec += m.x[i, f, d0]

        ongoing_setup = 0
        tau_s = int(pyo.value(m.tau_s[i]))  # workdays
        for f in m.F:
            for d0 in m.D:
                if d0 > d:
                    continue

                wd_count = 0
                t = d0
                while t <= d:
                    if int(pyo.value(m.workday[t])) == 1:
                        wd_count += 1
                    t += 1

                if 1 <= wd_count <= tau_s:
                    ongoing_setup += m.y[i, f, d0]

        return ongoing_exec + ongoing_setup <= 1

    m.Continuity = pyo.Constraint(
        m.I, m.D, rule=continuity_rule,
        doc="Continuity (workday-based): at most one ongoing activity per asset-day; durations counted in workdays."
    )

    # (C4) Fleet capacity (ongoing executions across all assets; WORKDAY durations):
    #   sum_{i,f} sum_{d0} x_{i,f,d0} * 1{d is within first tau_exec(i,f) WORKDAYS after d0} <= R   for all d
    def fleet_capacity_rule(m, d):
        expr = 0
        for i in m.I:
            for f in m.F:
                tau = int(pyo.value(m.tau_exec[i, f]))  # workdays
                for d0 in m.D:
                    if d0 > d:
                        continue

                    wd_count = 0
                    t = d0
                    while t <= d:
                        if int(pyo.value(m.workday[t])) == 1:
                            wd_count += 1
                        t += 1

                    if 1 <= wd_count <= tau:
                        expr += m.x[i, f, d0]
        for i in m.I:
            for f in m.F:
                tau_s = int(pyo.value(m.tau_s[i]))  # workdays
                for d0 in m.D:
                    if d0 > d:
                        continue

                    wd_count = 0
                    t = d0
                    while t <= d:
                        if int(pyo.value(m.workday[t])) == 1:
                            wd_count += 1
                        t += 1

                    if 1 <= wd_count <= tau_s:
                        expr += m.y[i, f, d0]

        return expr <= m.R

    m.FleetCapacity = pyo.Constraint(m.D, rule=fleet_capacity_rule)

    # (C5) Define phi (lateness / downtime):
    #   sum_{d in D} d*x_{i,f,d} - RUL_{i,f} <= phi_{i,f}
    def phi_def_rule(m, i, f):
        if pyo.value(m.F_if[i, f]) == 0:
            return pyo.Constraint.Skip
        return sum(d * m.x[i, f, d] for d in m.D) - m.RUL[i, f] <= m.phi[i, f]

    m.PhiDef = pyo.Constraint(m.IF, rule=phi_def_rule)

    # (C6) Define psi (earliness / unused RUL):
    #   (RUL_{i,f} - LT_i) - sum_{d in D} d*x_{i,f,d} <= psi_{i,f}
    def psi_def_rule(m, i, f):
        if pyo.value(m.F_if[i, f]) == 0:
            return pyo.Constraint.Skip
        return (m.RUL[i, f] - m.LT[i]) - sum(d * m.x[i, f, d] for d in m.D) <= m.psi[i, f]

    m.PsiDef = pyo.Constraint(m.IF, rule=psi_def_rule)

    # (C7) Big-M activation for reactive mode:
    #   phi_{i,f} <= M * l_{i,f}
    def reactive_activation_rule(m, i, f):
        if pyo.value(m.F_if[i, f]) == 0:
            return pyo.Constraint.Skip
        return m.phi[i, f] <= m.M * m.l[i, f]

    m.ReactiveActivation = pyo.Constraint(m.IF, rule=reactive_activation_rule)

    # (C8) Big-M activation for regular predictive mode:
    #   psi_{i,f} <= M * e_{i,f}
    def regular_activation_rule(m, i, f):
        if pyo.value(m.F_if[i, f]) == 0:
            return pyo.Constraint.Skip
        return m.psi[i, f] <= m.M * m.e[i, f]

    m.RegularActivation = pyo.Constraint(m.IF, rule=regular_activation_rule)

    # (C9) Mode selection:
    #   l_{i,f} + e_{i,f} + u_{i,f} = F_if(i,f)
    def mode_rule(m, i, f):
        return m.l[i, f] + m.e[i, f] + m.u[i, f] == m.F_if[i, f]

    m.ModeSelect = pyo.Constraint(m.IF, rule=mode_rule)

    return m


def solve_instance(data: dict, solver_name: str) -> tuple[float, pd.DataFrame, pd.DataFrame, dict]:
    model = build_model(data)

    # --- export model for debugging ---
    lp_path = ASSETS_DIR / "maintenance_model.lp"
    model.write(str(lp_path), io_options={"symbolic_solver_labels": True})

    txt_path = ASSETS_DIR / "maintenance_model.txt"
    txt_path.write_text(lp_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    # --- solve ---
    solver = pyo.SolverFactory(solver_name)
    if (solver is None) or (not solver.available(False)):
        raise RuntimeError(
            f"Solver '{solver_name}' not available. Install it (glpk/cbc/highs) or switch solver."
        )

    res = solver.solve(model, tee=True)

    tc = res.solver.termination_condition
    tc_str = str(tc).lower()

    # Accept optimal OR feasible (some solvers return 'feasible' for MIP when stopped early)
    if not (("optimal" in tc_str) or ("feasible" in tc_str)):
        raise RuntimeError(f"Solver status: {tc}")

    # --- helper: safe value extraction ---
    def val(x, default=0.0):
        v = pyo.value(x, exception=False)
        return default if v is None else v

    # --- plan dataframe ---
    plan_cols = [
        "asset_i", "failure_f", "scheduled_day_d",
        "RUL_if", "LT_i", "phi_if", "psi_if", "mode"
    ]

    rows = []
    # Some quick diagnostics about IF / F_if
    n_if = 0
    n_active_if = 0

    for (i, f) in model.IF:
        
        n_if += 1
        if val(model.F_if[i, f], default=0.0) < 0.5:
            continue
        n_active_if += 1

        chosen = None
        for d in model.D:
            if val(model.x[i, f, d], default=0.0) > 0.5:
                chosen = int(d)
                break

        mode = (
            "Regular predictive" if val(model.e[i, f], 0.0) > 0.5 else
            "Reactive" if val(model.l[i, f], 0.0) > 0.5 else
            "Emergency predictive" if val(model.u[i, f], 0.0) > 0.5 else
            "None"
        )

        rows.append({
            "asset_i": i,
            "failure_f": f,
            "scheduled_day_d": chosen,
            "RUL_if": float(val(model.RUL[i, f], default=0.0)),
            "LT_i": float(val(model.LT[i], default=0.0)),
            "phi_if": float(val(model.phi[i, f], default=0.0)),
            "psi_if": float(val(model.psi[i, f], default=0.0)),
            "mode": mode,
        })

    df_plan = pd.DataFrame(rows, columns=plan_cols)
    if not df_plan.empty:
        df_plan = df_plan.sort_values(["asset_i", "failure_f"], ignore_index=True)

    # --- setup dataframe ---
    setup_cols = ["asset_i", "failure_f", "setup_day_d", "setup_cost"]
    setup_rows = []

    for i in model.I:
        for f in model.F:
            for d in model.D:
                if val(model.y[i, f, d], default=0.0) > 0.5:
                    setup_rows.append({
                        "asset_i": i,
                        "failure_f": f,
                        "setup_day_d": int(d),
                        "setup_cost": float(val(model.c_setup[i], default=0.0)),
                    })

    df_setup = pd.DataFrame(setup_rows, columns=setup_cols)
    if not df_setup.empty:
        df_setup = df_setup.sort_values(["asset_i", "setup_day_d", "failure_f"], ignore_index=True)

    # --- cost components ---
    setup_cost_val = float(
        sum(
            val(model.c_setup[i], 0.0) * val(model.y[i, f, d], 0.0)
            for i in model.I
            for f in model.F
            for d in model.D
        )
    )

    phi_cost_val = float(
        sum(val(model.alpha[i, f], 0.0) * val(model.phi[i, f], 0.0) for (i, f) in model.IF)
    )
    psi_cost_val = float(
        sum(val(model.beta[i, f], 0.0) * val(model.psi[i, f], 0.0) for (i, f) in model.IF)
    )
    penalty_cost_val = phi_cost_val + psi_cost_val

    regular_mode_cost_val = float(
        sum(val(model.c_pr[i, f], 0.0) * val(model.e[i, f], 0.0) for (i, f) in model.IF)
    )
    emergency_mode_cost_val = float(
        sum(val(model.c_u_pr[i, f], 0.0) * val(model.u[i, f], 0.0) for (i, f) in model.IF)
    )
    reactive_mode_cost_val = float(
        sum(val(model.c_re[i, f], 0.0) * val(model.l[i, f], 0.0) for (i, f) in model.IF)
    )
    mode_cost_val = regular_mode_cost_val + emergency_mode_cost_val + reactive_mode_cost_val

    total_cost_val = setup_cost_val + penalty_cost_val + mode_cost_val

    summary = {
        "termination_condition": str(tc),

        # helpful diagnostics
        "n_if_pairs": int(n_if),
        "n_active_if_pairs": int(n_active_if),

        "setup_cost": setup_cost_val,

        "phi_penalty_cost": phi_cost_val,
        "psi_penalty_cost": psi_cost_val,
        "penalty_cost": penalty_cost_val,

        "regular_mode_cost": regular_mode_cost_val,
        "emergency_mode_cost": emergency_mode_cost_val,
        "reactive_mode_cost": reactive_mode_cost_val,
        "mode_cost": mode_cost_val,

        "total_cost": total_cost_val,

        "n_regular": int(sum(1 for (i, f) in model.IF if val(model.e[i, f], 0.0) > 0.5)),
        "n_emergency": int(sum(1 for (i, f) in model.IF if val(model.u[i, f], 0.0) > 0.5)),
        "n_reactive": int(sum(1 for (i, f) in model.IF if val(model.l[i, f], 0.0) > 0.5)),
    }

    obj_val = val(model.OBJ, default=None)
    if obj_val is None:
        # extremely rare if solver reports feasible but no values were loaded
        raise RuntimeError("Solver returned feasible/optimal but objective value is None (solution not loaded).")

    obj = float(obj_val)
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
    R: int = 10,  # daily capacity
    # Durations (days)
    tau_s_range: tuple[int, int] = (1, 3),      # tau_i^s
    tau_exec_range: tuple[int, int] = (1, 5),   # tau_if
) -> dict:
    rng = np.random.default_rng(seed)

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if failure_types is None:
        failure_types = ["motor", "battery"]
    if not (0.0 <= p_fail <= 1.0):
        raise ValueError("p_fail must be in [0, 1]")
    if tau_s_range[0] <= 0 or tau_exec_range[0] <= 0:
        raise ValueError("Durations must be >= 1 day")
    if tau_s_range[0] > tau_s_range[1] or tau_exec_range[0] > tau_exec_range[1]:
        raise ValueError("Invalid duration range")

    # Sets
    I = [f"Fleet_{k+1}" for k in range(n_assets)]
    F = list(failure_types)
    D = list(range(1, horizon + 1))  # days are already numeric (use d directly)

    # Lead time
    LT = {i: int(rng.integers(0, 7)) for i in I}

    # Setup duration tau_i^s
    tau_s = {i: int(rng.integers(tau_s_range[0], tau_s_range[1] + 1)) for i in I}

    # Setup costs
    setup_levels = np.arange(100, 501, 50)
    c_setup = {i: int(rng.choice(setup_levels)) for i in I}

    # Asset-failure pairs
    IF = [(i, f) for i in I for f in F]

    # In-horizon indicator F_if
    F_if = {(i, f): int(rng.random() < p_fail) for (i, f) in IF}

    # RUL (days)
    RUL: dict[tuple[str, str], int] = {}
    for (i, f) in IF:
        RUL[(i, f)] = int(rng.integers(1, horizon + 1)) if F_if[(i, f)] == 1 else horizon + 1

    # Execution duration tau_if
    tau_exec = {(i, f): int(rng.integers(tau_exec_range[0], tau_exec_range[1] + 1)) for (i, f) in IF}

    # Penalty weights
    alpha = {(i, f): int(rng.integers(10, 51)) for (i, f) in IF}
    beta = {(i, f): int(rng.integers(100, 201)) for (i, f) in IF}

    # Mode costs
    c_pr = {(i, f): int(rng.integers(200, 501)) for (i, f) in IF}       # regular predictive
    c_u_pr = {(i, f): int(rng.integers(500, 901)) for (i, f) in IF}     # urgent predictive
    c_re = {(i, f): int(rng.integers(1500, 4001)) for (i, f) in IF}     # reactive

    # Ensure urgent predictive is typically more expensive than regular predictive
    for k in IF:
        if c_u_pr[k] <= c_pr[k]:
            c_u_pr[k] = c_pr[k] + int(rng.integers(50, 301))

    # Calendar/workdays
    if start_date is None:
        start_date = date.today()

    hol_cal = None
    if _holidays_lib is not None:
        try:
            hol_cal = _holidays_lib.country_holidays(holiday_country)
        except Exception:
            hol_cal = None

    workday: dict[int, int] = {}
    holiday_name: dict[int, str] = {}
    for d in D:
        dt = start_date + timedelta(days=d - 1)
        is_weekend = dt.weekday() >= 5
        is_holiday = (hol_cal is not None) and (dt in hol_cal)

        workday[d] = 0 if (is_weekend or is_holiday) else 1
        holiday_name[d] = str(hol_cal.get(dt)) if is_holiday else ""

    # Big-M baseline
    M = float(horizon)

    return {
        "I": I,
        "F": F,
        "D": D,
        "M": M,
        "R": float(R),
        "LT": LT,
        "tau_s": tau_s,           # tau_i^s
        "tau_exec": tau_exec,     # tau_if
        "c_setup": c_setup,
        "RUL": RUL,
        "F_if": F_if,
        "alpha": alpha,
        "beta": beta,
        "c_pr": c_pr,
        "c_u_pr": c_u_pr,
        "c_re": c_re,
        "workday": workday,
        "start_date": start_date,
        "holiday_name": holiday_name,
    }


def build_base_plan(data: dict) -> tuple[float, pd.DataFrame, pd.DataFrame, dict]:
    """
    Baseline plan with durations + setup:
    - For each asset: pick earliest planned failure as anchor.
    - Schedule ONE setup before the first execution.
    - Schedule all failures for that asset back-to-back (ignore capacity).
    - Return (obj, df_plan, df_setup, summary) like solve_instance().
    """

    days = sorted(list(map(int, data["D"])))
    if not days:
        raise ValueError("data['D'] is empty")

    horizon = len(days)
    day_min, day_max = days[0], days[-1]
    workday = data["workday"]  # dict day->0/1

    # -----------------------------
    # helpers (workday-based)
    # -----------------------------
    def _shift_to_workday(d_int: int) -> int:
        """If day is not workday, move forward to next workday; if none, move backward."""
        d = int(d_int)
        d = max(1, min(horizon, d))

        dd = d
        while dd <= horizon and int(workday.get(dd, 1)) == 0:
            dd += 1
        if dd <= horizon:
            return dd

        dd = d
        while dd >= 1 and int(workday.get(dd, 1)) == 0:
            dd -= 1
        return max(1, dd)

    def _t1_working_days_before(the_day: int, t1: int) -> int | None:
        """Return calendar day that is t1 WORKDAYS strictly before the_day; None if doesn't exist."""
        if t1 <= 0:
            return int(the_day)

        day = int(the_day)
        remaining = int(t1)
        while remaining > 0:
            day -= 1
            if day < 1:
                return None
            if int(workday.get(day, 0)) == 1:
                remaining -= 1
        return day

    def _end_day_after_tau_workdays(start_day: int, tau: int) -> int:
        """
        Returns the calendar day index of the first day AFTER completing tau workdays,
        starting from start_day (assumed workday). Clipped to horizon+1.
        """
        if tau <= 0:
            return max(1, min(horizon + 1, int(start_day)))

        d = max(1, min(horizon, int(start_day)))
        wd_seen = 0
        while d <= horizon and wd_seen < tau:
            if int(workday.get(d, 0)) == 1:
                wd_seen += 1
            d += 1
        return min(horizon + 1, d)  # day AFTER finishing

    # -----------------------------
    # Build baseline schedule
    # -----------------------------
    plan_rows = []
    setup_rows = []

    # Pre-collect active failures per asset
    F = list(data["F"])
    I = list(data["I"])

    for i in I:
        active_fs = [f for f in F if int(data["F_if"].get((i, f), 0)) == 1]
        if not active_fs:
            continue

        lt = int(data["LT"][i])
        tau_s_i = int(data["tau_s"][i])

        # compute baseline "desired day" per failure, choose earliest as anchor
        candidates = []
        for f in active_fs:
            rul = int(data["RUL"][(i, f)])
            planned_time = max(rul - lt, 0)
            planned_day = max(1, min(int(planned_time), horizon))
            planned_day = _shift_to_workday(planned_day)
            candidates.append((planned_day, planned_time, f))

        candidates.sort(key=lambda x: (x[0], str(x[2])))
        first_planned_day, first_planned_time, f_first = candidates[0]

        # schedule setup before first execution (workdays-before)
        setup_start = _t1_working_days_before(first_planned_day, tau_s_i)
        if setup_start is None:
            # not enough history -> start at first available workday
            setup_start = _shift_to_workday(1)
        else:
            setup_start = _shift_to_workday(setup_start)

        setup_rows.append({
            "asset_i": i,
            "failure_f": f_first,  # associate setup with anchor failure
            "setup_day_d": int(setup_start),
            "setup_cost": float(data["c_setup"][i]),
        })

        # execution starts after setup completes, next workday
        exec_start_candidate = _end_day_after_tau_workdays(setup_start, tau_s_i)
        if exec_start_candidate > horizon:
            # if setup pushes beyond horizon, just clamp (still returns something)
            exec_start_candidate = horizon
        current_start = _shift_to_workday(exec_start_candidate)

        # schedule failures back-to-back in the chosen order (earliest planned day first)
        for planned_day, planned_time, f in candidates:
            start_d = int(current_start)
            if start_d < 1:
                start_d = 1
            if start_d > horizon:
                start_d = horizon
            start_d = _shift_to_workday(start_d)

            tau_exec_if = int(data["tau_exec"][(i, f)])
            end_after = _end_day_after_tau_workdays(start_d, tau_exec_if)

            rul = int(data["RUL"][(i, f)])
            # phi / psi definitions consistent with your baseline_costs
            phi = max(start_d - rul, 0)
            psi = max((rul - lt) - start_d, 0)

            mode = "Reactive" if phi > 0 else "Regular predictive"

            plan_rows.append({
                "asset_i": i,
                "failure_f": f,
                "scheduled_day_d": int(start_d),
                "RUL_if": float(rul),
                "LT_i": float(lt),
                "phi_if": float(phi),
                "psi_if": float(psi),
                "mode": mode,
            })

            # next task starts the day after this one completes
            next_start = end_after
            if next_start > horizon:
                next_start = horizon
            current_start = _shift_to_workday(next_start)

    # Ensure stable schema even if empty
    plan_cols = ["asset_i", "failure_f", "scheduled_day_d", "RUL_if", "LT_i", "phi_if", "psi_if", "mode"]
    setup_cols = ["asset_i", "failure_f", "setup_day_d", "setup_cost"]

    df_plan = pd.DataFrame(plan_rows, columns=plan_cols)
    if not df_plan.empty:
        df_plan = df_plan.sort_values(["asset_i", "failure_f"], ignore_index=True)

    df_setup = pd.DataFrame(setup_rows, columns=setup_cols)
    if not df_setup.empty:
        df_setup = df_setup.sort_values(["asset_i", "setup_day_d", "failure_f"], ignore_index=True)

    # -----------------------------
    # Costs + "objective"
    # -----------------------------
    setup_cost_val = float(df_setup["setup_cost"].sum()) if not df_setup.empty else 0.0

    phi_cost_val = 0.0
    psi_cost_val = 0.0
    regular_mode_cost_val = 0.0
    emergency_mode_cost_val = 0.0  # baseline uses none
    reactive_mode_cost_val = 0.0

    if not df_plan.empty:
        for _, r in df_plan.iterrows():
            i = r["asset_i"]
            f = r["failure_f"]
            phi = float(r["phi_if"])
            psi = float(r["psi_if"])

            phi_cost_val += float(data["alpha"][(i, f)]) * phi
            psi_cost_val += float(data["beta"][(i, f)]) * psi

            if r["mode"] == "Reactive":
                reactive_mode_cost_val += float(data["c_re"][(i, f)])
            else:
                regular_mode_cost_val += float(data["c_pr"][(i, f)])

    penalty_cost_val = float(phi_cost_val + psi_cost_val)
    mode_cost_val = float(regular_mode_cost_val + emergency_mode_cost_val + reactive_mode_cost_val)
    total_cost_val = float(setup_cost_val + penalty_cost_val + mode_cost_val)

    summary = {
        "termination_condition": "baseline",

        "n_if_pairs": int(len(data["I"]) * len(data["F"])),
        "n_active_if_pairs": int(sum(int(v) for v in data["F_if"].values())),

        "setup_cost": setup_cost_val,

        "phi_penalty_cost": float(phi_cost_val),
        "psi_penalty_cost": float(psi_cost_val),
        "penalty_cost": penalty_cost_val,

        "regular_mode_cost": float(regular_mode_cost_val),
        "emergency_mode_cost": float(emergency_mode_cost_val),
        "reactive_mode_cost": float(reactive_mode_cost_val),
        "mode_cost": mode_cost_val,

        "total_cost": total_cost_val,

        "n_regular": int((df_plan["mode"] == "Regular predictive").sum()) if not df_plan.empty else 0,
        "n_emergency": 0,
        "n_reactive": int((df_plan["mode"] == "Reactive").sum()) if not df_plan.empty else 0,
    }

    obj = total_cost_val
    return obj, df_plan, df_setup, summary

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

                    html.H4("Gantt", style={"margin": "10px 0"}),
                    dcc.Graph(id="mp-gantt-graph", figure={}, config={"displayModeBar": False}),

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

    def build_color_config(data: dict, palette: list[str] | None = None):
        """
        Uses ONLY colors from `palette`.

        Returns:
        fault_colors: dict[failure_type -> hex]
        setup_color: hex
        exec_default_color: hex (fallback if f missing)
        multi_fault_color: hex
        """
        if palette is None or len(palette) == 0:
            raise ValueError("palette must be a non-empty list of hex colors")

        failure_types = list(data.get("F", []))
        nF = len(failure_types)
        nP = len(palette)

        # 1) fault colors (use as many as needed; cycle if palette shorter)
        fault_colors = {}
        for idx, f in enumerate(failure_types):
            fault_colors[f] = palette[idx % nP]

        # 2) special colors (take subsequent palette positions)
        # Prefer distinct colors by stepping forward if we collide.
        used = set(fault_colors.values())

        def pick_color(start_idx: int) -> str:
            # Try a full loop to find a not-yet-used color
            for k in range(nP):
                c = palette[(start_idx + k) % nP]
                if c not in used:
                    used.add(c)
                    return c
            # If all colors are used (small palette), just cycle
            c = palette[start_idx % nP]
            used.add(c)
            return c

        setup_color = pick_color(nF + 0)
        exec_default_color = pick_color(nF + 1)
        multi_fault_color = pick_color(nF + 2)

        return fault_colors, setup_color, exec_default_color, multi_fault_color


    def _fmt_money(x):
        try:
            return f"{float(x):,.2f}"
        except Exception:
            return "-"

    def _build_cashflow_figure(
        data: dict,
        df_plan: pd.DataFrame | None,
        df_setup: pd.DataFrame | None,
        baseline_counts: dict[int, int] | None = None,
        *,
        exec_color: str = "#1f77b4",
        baseline_color: str = "#7f7f7f",
    ):
        days = sorted(list(map(int, data["D"])))
        if not days:
            raise ValueError("data['D'] is empty")

        day_min, day_max = days[0], days[-1]
        day_cost = {int(d): 0.0 for d in days}

        tau_s = data.get("tau_s", {})          # expects tau_s[i] (workdays)
        tau_exec = data.get("tau_exec", {})    # expects tau_exec[(i,f)] (workdays)

        def _add_cost_over_window(start_day: int, duration: int, total_cost: float):
            """Spread total_cost uniformly over [start_day, start_day+duration-1], clipped to horizon."""
            if start_day is None:
                return
            d0 = int(start_day)
            dur = int(duration) if duration is not None else 0
            if dur <= 0:
                if day_min <= d0 <= day_max:
                    day_cost[d0] += float(total_cost)
                return

            d1 = d0 + dur - 1
            win_start = max(d0, day_min)
            win_end = min(d1, day_max)
            if win_start > win_end:
                return

            n = win_end - win_start + 1
            per_day = float(total_cost) / float(n) if n > 0 else 0.0
            for d in range(win_start, win_end + 1):
                day_cost[d] += per_day

        # -----------------------
        # 1) setup costs
        # -----------------------
        if df_setup is not None and not df_setup.empty:
            for _, r in df_setup.iterrows():
                i = r["asset_i"]
                d = int(r["setup_day_d"])
                setup_cost = float(r.get("setup_cost", 0.0))
                dur = int(tau_s.get(i, 0))  # no default_setup_duration used
                _add_cost_over_window(d, dur, setup_cost)

        # -----------------------
        # 2) execution-related costs (penalty + mode)
        # -----------------------
        if df_plan is not None and not df_plan.empty:
            for _, r in df_plan.iterrows():
                d = r.get("scheduled_day_d", None)
                if d is None or pd.isna(d):
                    continue
                d = int(d)

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

                total_exec_cost = penalty + mode_cost
                dur = int(tau_exec.get((i, f), 0))
                _add_cost_over_window(d, dur, total_exec_cost)

        # -----------------------
        # Build figure
        # -----------------------
        fig = go.Figure()

        has_bars = any(abs(day_cost[int(d)]) > 1e-12 for d in days)
        if has_bars:
            fig.add_bar(
                x=days,
                y=[day_cost[int(d)] for d in days],
                name="Optimized daily cost (spread)",
                marker=dict(color=exec_color),
            )

        if baseline_counts is not None:
            fig.add_scatter(
                x=days,
                y=[baseline_counts.get(int(d), 0) for d in days],
                mode="lines+markers",
                name="Baseline planned jobs (pre-optimizer)",
                yaxis="y2",
                line=dict(color=baseline_color),
                marker=dict(color=baseline_color),
            )

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Day",
            yaxis_title="Cost (cash flow)",
            barmode="group",
            legend=dict(orientation="h"),
        )

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


   
    def _build_gantt_figure(
        data: dict,
        df_plan: pd.DataFrame | None,
        df_setup: pd.DataFrame | None,
        *,
        setup_color: str = "#ff7f0e",
        exec_color: str = "#1f77b4",
    ) -> dict:
        days = sorted(list(map(int, data["D"])))
        if not days:
            raise ValueError("data['D'] is empty")
        day_min, day_max = days[0], days[-1]

        tau_s = data.get("tau_s", {})
        tau_exec = data.get("tau_exec", {})

        tasks = []

        if df_setup is not None and not df_setup.empty:
            for _, r in df_setup.iterrows():
                i = r["asset_i"]
                f = r.get("failure_f", None)
                start = r.get("setup_day_d", None)
                if start is None or pd.isna(start):
                    continue
                start = int(start)

                dur = int(tau_s.get(i, 0))
                if dur <= 0:
                    dur = 1
                end = start + dur - 1

                s = max(start, day_min)
                e = min(end, day_max)
                if s > e:
                    continue

                tasks.append({"asset_i": i, "kind": "Setup", "failure_f": f, "start": s, "end": e})

        if df_plan is not None and not df_plan.empty:
            for _, r in df_plan.iterrows():
                i = r["asset_i"]
                f = r["failure_f"]
                start = r.get("scheduled_day_d", None)
                if start is None or pd.isna(start):
                    continue
                start = int(start)

                dur = int(tau_exec.get((i, f), 0))
                if dur <= 0:
                    dur = 1
                end = start + dur - 1

                s = max(start, day_min)
                e = min(end, day_max)
                if s > e:
                    continue

                tasks.append({"asset_i": i, "kind": "Exec", "failure_f": f, "start": s, "end": e})

        fig = go.Figure()

        if not tasks:
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Day",
                yaxis_title=None,
                xaxis=dict(range=[day_min - 0.5, day_max + 0.5]),
            )
            return fig

        df_tasks = pd.DataFrame(tasks)
        assets = sorted(df_tasks["asset_i"].unique().tolist())

        # Setup trace
        sub = df_tasks[df_tasks["kind"] == "Setup"]
        if not sub.empty:
            fig.add_trace(
                go.Bar(
                    y=sub["asset_i"],
                    x=(sub["end"] - sub["start"] + 1),
                    base=sub["start"],
                    orientation="h",
                    name="Setup",
                    marker=dict(color=setup_color),
                    customdata=sub[["failure_f", "start", "end"]].to_numpy(),
                    hovertemplate=(
                        "Asset: %{y}<br>"
                        "Type: Setup<br>"
                        "Failure: %{customdata[0]}<br>"
                        "Start: %{customdata[1]}<br>"
                        "End: %{customdata[2]}<extra></extra>"
                    ),
                )
            )

        # Exec trace
        sub = df_tasks[df_tasks["kind"] == "Exec"]
        if not sub.empty:
            fig.add_trace(
                go.Bar(
                    y=sub["asset_i"],
                    x=(sub["end"] - sub["start"] + 1),
                    base=sub["start"],
                    orientation="h",
                    name="Exec",
                    marker=dict(color=exec_color),
                    customdata=sub[["failure_f", "start", "end"]].to_numpy(),
                    hovertemplate=(
                        "Asset: %{y}<br>"
                        "Type: Exec<br>"
                        "Failure: %{customdata[0]}<br>"
                        "Start: %{customdata[1]}<br>"
                        "End: %{customdata[2]}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Day",
            yaxis_title=None,
            barmode="overlay",
            legend=dict(orientation="h"),
            xaxis=dict(range=[day_min - 0.5, day_max + 0.5], dtick=1),
            yaxis=dict(
                categoryorder="array",
                categoryarray=assets,
                showticklabels=False,
                ticks="",
                showgrid=False,
            ),
        )
        return fig

    def _build_stacked_fleets_figure(
        data: dict,
        df_plan: pd.DataFrame,
        baseline_counts: dict[int, int] | None = None,
        *,
        fault_colors: dict[str, str] | None = None,
        multi_fault_color: str = "#9467bd",
        baseline_color: str = "#7f7f7f",
    ) -> dict:
        days = list(map(int, data["D"]))
        failure_types = list(data["F"])

        # fallback if not provided
        fault_colors = fault_colors or {}

        fig = go.Figure()

        if df_plan is None or df_plan.empty:
            if baseline_counts is not None:
                fig.add_scatter(
                    x=days,
                    y=[baseline_counts.get(int(d), 0) for d in days],
                    mode="lines+markers",
                    name="Baseline planned jobs (pre-optimizer)",
                    line=dict(color=baseline_color),
                    marker=dict(color=baseline_color),
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
            d = r.get("scheduled_day_d", None)
            if d is None or pd.isna(d):
                continue
            d = int(d)

            i = r["asset_i"]
            f = r["failure_f"]

            if asset_fail_count.get(i, 0) <= 1:
                # protect against missing day keys (if horizon mismatch)
                if d in only_f.get(f, {}):
                    only_f[f][d] += 1
            else:
                if d in multi_assets_per_day:
                    multi_assets_per_day[d].add(i)

        multi = {d: len(multi_assets_per_day[d]) for d in days}

        # Optimized stacked bars (colored by failure type)
        for f in failure_types:
            fig.add_bar(
                x=days,
                y=[only_f[f][d] for d in days],
                name=f"Optimized: only {f}",
                marker=dict(color=fault_colors.get(f)),  # if None, plotly auto-colors
            )

        # multi-failure bar (custom color)
        fig.add_bar(
            x=days,
            y=[multi[d] for d in days],
            name="Optimized: >1 failure",
            marker=dict(color=multi_fault_color),
        )

        # Baseline overlay line (total planned jobs/day)
        if baseline_counts is not None:
            fig.add_scatter(
                x=days,
                y=[baseline_counts.get(int(d), 0) for d in days],
                mode="lines+markers",
                name="Baseline planned jobs (pre-optimizer)",
                line=dict(color=baseline_color),
                marker=dict(color=baseline_color),
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

    def _check_capacity_infeasible(
        df_base: pd.DataFrame,
        R: int,
        workday: dict[int, int],
        horizon: int,
        tau_exec_col: str = "tau_exec",      # column name in df_base, if present
        start_col: str = "planned_day_d",    # start day column
    ) -> tuple[bool, dict]:
        """
        Capacity check consistent with the Pyomo constraint:
        For each day d: (# of ONGOING executions on day d) <= R
        where task durations are measured in WORKDAYS (weekends/holidays skipped).

        Returns (is_infeasible, details)
        details = {"max_day": int|None, "max_count": int, "days_over": {day:count}}
        """
        if df_base is None or df_base.empty:
            return False, {"max_day": None, "max_count": 0, "days_over": {}}

        # ensure we only consider valid numeric days
        df = df_base.copy()
        df = df[pd.notnull(df[start_col])]
        if df.empty:
            return False, {"max_day": None, "max_count": 0, "days_over": {}}

        # Precompute the list of workdays in the horizon for fast stepping
        D = list(range(1, int(horizon) + 1))
        workdays_in_horizon = [d for d in D if int(workday.get(d, 0)) == 1]

        # Helper: given a start day d0 and duration tau (workdays), return covered calendar days
        def covered_days(d0: int, tau: int) -> list[int]:
            if tau <= 0:
                return []
            if int(workday.get(d0, 0)) != 1:
                # if starts must be workdays, treat as covering nothing (or raise)
                return []
            covered = []
            wd_seen = 0
            t = d0
            while t <= horizon and wd_seen < tau:
                if int(workday.get(t, 0)) == 1:
                    wd_seen += 1
                    covered.append(t)
                t += 1
            return covered

        # Count ongoing executions per day
        ongoing_count: dict[int, int] = {d: 0 for d in range(1, horizon + 1)}

        for _, row in df.iterrows():
            d0 = int(row[start_col])

            # duration: read from column if exists; otherwise assume 1 (or change default)
            if tau_exec_col in df.columns and pd.notnull(row[tau_exec_col]):
                tau = int(row[tau_exec_col])
            else:
                tau = 1

            for d in covered_days(d0, tau):
                ongoing_count[d] += 1

        # Find violations
        days_over = {d: c for d, c in ongoing_count.items() if c > int(R)}

        if not days_over:
            # report max utilization day
            max_day = max(ongoing_count, key=ongoing_count.get) if ongoing_count else None
            max_count = int(ongoing_count[max_day]) if max_day is not None else 0
            return False, {"max_day": int(max_day) if max_day is not None else None, "max_count": max_count, "days_over": {}}

        worst_day = max(days_over, key=days_over.get)
        return True, {"max_day": int(worst_day), "max_count": int(days_over[worst_day]), "days_over": {int(k): int(v) for k, v in days_over.items()}}
    
    def _end_day_after_tau_workdays(start_day: int, tau: int, workday: dict[int, int], horizon: int) -> int:
        """
        Returns the calendar day index of the first day AFTER completing tau workdays,
        starting from start_day (which is assumed to be a workday).
        Example: if start_day=6 (Fri), tau=3 => covers 6,7,10, so end_day = 11 (day after 10).
        """
        if tau <= 0:
            return max(1, min(horizon + 1, start_day))

        d = int(start_day)
        d = max(1, min(horizon, d))

        wd_seen = 0
        while d <= horizon and wd_seen < tau:
            if int(workday.get(d, 0)) == 1:
                wd_seen += 1
            d += 1

        # d is now the first day AFTER we counted tau workdays (or horizon+1 if it ran out)
        return min(horizon + 1, d)

    # ---------------------------------------------------------
    # Run optimization + update KPIs + graphs + calendar + modal data
    # ---------------------------------------------------------
    @app.callback(
        Output("mp-status", "children"),
        Output("mp-obj", "children"),
        Output("mp-cashflow-graph", "figure"),
        Output("mp-stack-graph", "figure"),
        Output("mp-gantt-graph", "figure"),
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
                R=int(shared_inputs.get("R", 30)),
                start_date=parsed_start,

                # NEW (durations in WORKDAYS)
                tau_s_range=tuple(shared_inputs.get("tau_s_range", (1, 3))),
                tau_exec_range=tuple(shared_inputs.get("tau_exec_range", (1, 5))),
            )
            fault_colors, setup_color, exec_color, multi_fault_color = build_color_config(data, DEFAULT_PALETTE)


            # override R if user changed it
            if planner_controls.get("R_override") is not None:
                data["R"] = float(int(planner_controls["R_override"]))  # model expects float

            # work on holidays/weekends: make all days workdays
            if planner_controls.get("work_all_days"):
                data["workday"] = {d: 1 for d in data["D"]}

            # --- Baseline plan (always) ---
            obj_base, df_plan_base, df_setup_base, summary_base = build_base_plan(data)

            # ---- baseline counts by day (use execution STARTS) ----
            days = list(map(int, data["D"]))
            base_counts = {d: 0 for d in days}
            if df_plan_base is not None and not df_plan_base.empty:
                for d, g in df_plan_base.groupby("scheduled_day_d"):
                    if pd.isna(d):
                        continue
                    base_counts[int(d)] = int(len(g))

            # ---- capacity check (optional): build a df_base-like table from baseline plan ----
            # _check_capacity_infeasible expects:
            #   start_col planned_day_d and a tau_exec column
            df_base_like = pd.DataFrame()
            if df_plan_base is not None and not df_plan_base.empty:
                df_base_like = df_plan_base.copy()
                df_base_like["planned_day_d"] = df_base_like["scheduled_day_d"]

                # attach tau_exec per (i,f)
                def _tau_exec_lookup(row):
                    i = row["asset_i"]
                    f = row["failure_f"]
                    return int(data["tau_exec"].get((i, f), 1))  # safe fallback for check only

                df_base_like["tau_exec"] = df_base_like.apply(_tau_exec_lookup, axis=1)

            is_infeasible, cap_info = _check_capacity_infeasible(
                df_base=df_base_like,
                R=int(data["R"]),
                workday=data["workday"],
                horizon=len(data["D"]),
                tau_exec_col="tau_exec",
                start_col="planned_day_d",
            )

            events = []
            horizon = len(data["D"])

            # (A) baseline EXEC events
            if df_plan_base is not None and not df_plan_base.empty:
                for _, r in df_plan_base.iterrows():
                    d0 = r.get("scheduled_day_d", None)
                    if d0 is None or (isinstance(d0, float) and pd.isna(d0)):
                        continue

                    d0 = int(d0)
                    i = r["asset_i"]
                    f = r["failure_f"]
                    tau = int(data["tau_exec"].get((i, f), 1))

                    start_dt = data["start_date"] + timedelta(days=d0 - 1)
                    end_day = _end_day_after_tau_workdays(d0, tau, data["workday"], horizon)
                    end_dt = data["start_date"] + timedelta(days=end_day - 1)

                    # choose exec color (per-fault, fallback to exec_color)
                    exec_bg = fault_colors.get(f, exec_color)

                    events.append({
                        "title": f"Baseline · {i} · {f} ({r.get('mode','')})",
                        "start": start_dt.isoformat(),
                        "end": end_dt.isoformat(),
                        "allDay": True,
                        "display": "auto",
                        "backgroundColor": exec_bg,
                        "borderColor": exec_bg,
                        "textColor": "#0e0d0d",
                    })

            # (B) baseline SETUP events
            if df_setup_base is not None and not df_setup_base.empty:
                for _, r in df_setup_base.iterrows():
                    d0 = r.get("setup_day_d", None)
                    if d0 is None or (isinstance(d0, float) and pd.isna(d0)):
                        continue

                    d0 = int(d0)
                    i = r["asset_i"]
                    tau = int(data["tau_s"].get(i, 1))

                    start_dt = data["start_date"] + timedelta(days=d0 - 1)
                    end_day = _end_day_after_tau_workdays(d0, tau, data["workday"], horizon)
                    end_dt = data["start_date"] + timedelta(days=end_day - 1)

                    events.append({
                        "title": f"Baseline · {i} · Setup",
                        "start": start_dt.isoformat(),
                        "end": end_dt.isoformat(),
                        "allDay": True,
                        "display": "auto",
                        "backgroundColor": setup_color,
                        "borderColor": setup_color,
                        "textColor": "#ffffff",
                    })
            # ---- KPIs that can be shown even before optimization ----
            n_assets = len(data["I"])
            assets_with_fail = sum(
                1 for i in data["I"]
                if any(data["F_if"][(i, f)] == 1 for f in data["F"])
            )
            total_failures = sum(data["F_if"][(i, f)] for i in data["I"] for f in data["F"])

            # Baseline KPIs now come straight from summary_base (already includes setup + penalties + mode)
            kpi_setup = _fmt_money(summary_base.get("setup_cost", 0.0))
            kpi_penalty = _fmt_money(summary_base.get("penalty_cost", 0.0))
            kpi_mode = _fmt_money(summary_base.get("mode_cost", 0.0))
            kpi_total = _fmt_money(summary_base.get("total_cost", obj_base))

            kpi_status = (
                html.Span("Infeasible", style={"color": "#b00020", "fontWeight": 700})
                if is_infeasible
                else html.Span("Feasible", style={"color": "#1bb31b", "fontWeight": 700})
            )

            # ---- baseline graphs: now show BASELINE plan (not empty) ----
            # cashflow uses df_plan + df_setup (and spreads across durations)
            fig_cash = _build_cashflow_figure(data, df_plan_base, df_setup_base, baseline_counts=base_counts, 
                                              exec_color=exec_color, baseline_color=multi_fault_color )

            # stack graph uses df_plan start days (bars) + optional baseline overlay (line)
            # if you want the overlay line to represent baseline starts as well, pass baseline_counts=None
            fig_stack = _build_stacked_fleets_figure(data, df_plan_base, baseline_counts=base_counts, 
                                                     fault_colors=fault_colors, multi_fault_color=multi_fault_color, baseline_color=multi_fault_color)

            # gantt: show baseline schedule too (setup+exec)
            fig_gantt = _build_gantt_figure(data, df_plan_base, df_setup_base, 
                                            setup_color=setup_color, exec_color=exec_color,)

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
                            html.Span(
                                f"Capacity R = {int(data['R'])} concurrent executions, "
                                f"but day {worst_day} has {worst_cnt} ongoing executions. "
                            ),
                        ],
                        style={"color": "#b00020"},
                    )
                else:
                    status = html.Div(
                        [
                            html.Span("Initial plan generated (baseline). "),
                            html.Span("Click Run Optimization to solve."),
                        ]
                    )

                return (
                    status,
                    f"Objective value: {float(obj_base):,.4f} (baseline)",
                    fig_cash,
                    fig_stack,
                    fig_gantt,
                    events,
                    str(n_assets),
                    str(assets_with_fail),
                    str(total_failures),
                    kpi_setup,
                    kpi_penalty,
                    kpi_mode,
                    kpi_total,
                    kpi_status,
                    summary_base,   # store baseline summary
                    "",             # no links until optimization
                )
            # --- If triggered by Run button: solve optimizer and overlay results ---
            obj, df_plan, df_setup, summary = solve_instance(data, solver_name=solver_name)

            # costs
            kpi_setup = _fmt_money(summary.get("setup_cost", 0.0))
            kpi_penalty = _fmt_money(summary.get("penalty_cost", 0.0))
            kpi_mode = _fmt_money(summary.get("mode_cost", 0.0))
            kpi_total = _fmt_money(summary.get("total_cost", obj))

            # figures with optimized + baseline overlay
            fig_cash = _build_cashflow_figure(data, df_plan, df_setup, baseline_counts=base_counts, 
                                              exec_color=exec_color,    baseline_color=multi_fault_color)
            fig_stack = _build_stacked_fleets_figure(data, df_plan, baseline_counts=base_counts, 
                                                     fault_colors=fault_colors, multi_fault_color=multi_fault_color, baseline_color=multi_fault_color)
            fig_gantt = _build_gantt_figure(data, df_plan, df_setup, 
                                            setup_color=setup_color, exec_color=exec_color,)

            # append optimized events
            # append optimized events (execution spans tau_exec WORKDAYS)
            horizon = len(data["D"])
            events = []

            if df_plan is not None and not df_plan.empty:
                for _, r in df_plan.iterrows():
                    d0 = r.get("scheduled_day_d", None)
                    if d0 is None or (isinstance(d0, float) and pd.isna(d0)):
                        continue

                    d0 = int(d0)
                    i = r["asset_i"]
                    f = r["failure_f"]
                    tau = int(data["tau_exec"].get((i, f), 1))

                    start_dt = data["start_date"] + timedelta(days=d0 - 1)
                    end_day = _end_day_after_tau_workdays(d0, tau, data["workday"], horizon)
                    end_dt = data["start_date"] + timedelta(days=end_day - 1)

                    exec_bg = fault_colors.get(f, exec_color)

                    events.append({
                        "title": f"Optimized · {i} · {f} ({r.get('mode','')})",
                        "start": start_dt.isoformat(),
                        "end": end_dt.isoformat(),
                        "allDay": True,
                        "display": "auto",

                        # colors
                        "backgroundColor": exec_bg,
                        "borderColor": exec_bg,
                        "textColor": "#ffffff",

                        # optional: make optimized visually “stronger”
                        # (FullCalendar supports this via extendedProps + eventDidMount if you want,
                        # but many wrappers also pass this through directly)
                        # "classNames": ["optimized-event"],
                    })

            if df_setup is not None and not df_setup.empty:
                for _, r in df_setup.iterrows():
                    d0 = r.get("setup_day_d", None)
                    if d0 is None or (isinstance(d0, float) and pd.isna(d0)):
                        continue

                    d0 = int(d0)
                    i = r["asset_i"]
                    tau = int(data["tau_s"].get(i, 1))

                    start_dt = data["start_date"] + timedelta(days=d0 - 1)
                    end_day = _end_day_after_tau_workdays(d0, tau, data["workday"], horizon)
                    end_dt = data["start_date"] + timedelta(days=end_day - 1)

                    events.append({
                        "title": f"Optimized · {i} · Setup",
                        "start": start_dt.isoformat(),
                        "end": end_dt.isoformat(),
                        "allDay": True,
                        "display": "auto",

                        # setup color
                        "backgroundColor": setup_color,
                        "borderColor": setup_color,
                        "textColor": "#ffffff",
                    })

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
            status = html.Div([
                html.Span("Solved successfully. "),
                html.Span(f"Solver: {solver_name}. "),
                html.Span("Termination: optimal/feasible."),
            ])
            return (
                status,
                f"Objective value: {float(obj):,.4f}",
                fig_cash,
                fig_stack,
                fig_gantt,
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