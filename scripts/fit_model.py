"""Offline model fitting script.

Outputs written to assets/:
    model_trace.nc   – ArviZ InferenceData (NetCDF4)
    scalers.npz      – scaler mean/scale arrays (no scikit-learn needed to load)
"""

import pathlib
import sys

# Allow importing from Dashboard root
DASHBOARD_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(DASHBOARD_ROOT))

import arviz as az
import numpy as np
import nutpie
import pymc as pm
import pandas as pd
from sklearn.preprocessing import StandardScaler

from create_dset import (
    create_dset,
    build_survival_dataset,
)

ASSETS = DASHBOARD_ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load training data
# ---------------------------------------------------------------------------
TRAIN_PATH = DASHBOARD_ROOT / "assets" / "simulated_dataset" / "train"
print(f"Loading training data from {TRAIN_PATH} ...")
dset = create_dset(TRAIN_PATH)
print(f"  Loaded {len(dset):,} rows across {dset['machine_id'].nunique()} machines")

# ---------------------------------------------------------------------------
# 2. Fit scalers
# ---------------------------------------------------------------------------
scaler_fixed = StandardScaler()
scaler_fixed.fit(dset[["car_type", "region", "route"]].to_numpy(dtype=np.float32))

scaler_dynamic = StandardScaler()
scaler_dynamic.fit(dset[["route_ratio", "speed", "load"]].to_numpy(dtype=np.float32))

# Save as plain numpy arrays (no scikit-learn needed to reload)
np.savez(
    ASSETS / "scalers.npz",
    fixed_mean=scaler_fixed.mean_,
    fixed_scale=scaler_fixed.scale_,
    dynamic_mean=scaler_dynamic.mean_,
    dynamic_scale=scaler_dynamic.scale_,
)
print("Saved scalers.npz")

# ---------------------------------------------------------------------------
# 3. Build survival datasets
# ---------------------------------------------------------------------------
print("Building survival datasets ...")
tire_surv  = build_survival_dataset(dset, "Tires")
brake_surv = build_survival_dataset(dset, "Brakes")

tire_surv["x_fixed"]   = scaler_fixed.transform(tire_surv["x_fixed"]).astype(np.float32)
tire_surv["x_dynamic"] = scaler_dynamic.transform(tire_surv["x_dynamic"]).astype(np.float32)
brake_surv["x_fixed"]  = scaler_fixed.transform(brake_surv["x_fixed"]).astype(np.float32)
brake_surv["x_dynamic"]= scaler_dynamic.transform(brake_surv["x_dynamic"]).astype(np.float32)

mean_ttf_tire  = float(tire_surv["t"][tire_surv["event"] == 1].mean())
mean_ttf_brake = float(brake_surv["t"][brake_surv["event"] == 1].mean())
print(f"  Mean tire TTF: {mean_ttf_tire:.1f}  |  brake TTF: {mean_ttf_brake:.1f}")

n_fixed   = tire_surv["x_fixed"].shape[1]
n_dynamic = tire_surv["x_dynamic"].shape[1]

# ---------------------------------------------------------------------------
# 4. Define and fit the PyMC model (same as create_dset.py)
# ---------------------------------------------------------------------------
print("Building PyMC model ...")
with pm.Model() as model:
    x_f_tire  = pm.Data("x_f_tire",  tire_surv["x_fixed"])
    x_d_tire  = pm.Data("x_d_tire",  tire_surv["x_dynamic"])
    t_tire    = pm.Data("t_tire",    tire_surv["t"])
    ev_tire   = pm.Data("ev_tire",   tire_surv["event"].astype(np.float32))

    x_f_brake = pm.Data("x_f_brake", brake_surv["x_fixed"])
    x_d_brake = pm.Data("x_d_brake", brake_surv["x_dynamic"])
    t_brake   = pm.Data("t_brake",   brake_surv["t"])
    ev_brake  = pm.Data("ev_brake",  brake_surv["event"].astype(np.float32))

    beta_fixed_tire    = pm.Normal("beta_fixed_tire",    mu=0, sigma=1, shape=(n_fixed,))
    beta_dynamic_tire  = pm.Normal("beta_dynamic_tire",  mu=0, sigma=1, shape=(n_dynamic,))
    beta_fixed_brake   = pm.Normal("beta_fixed_brake",   mu=0, sigma=1, shape=(n_fixed,))
    beta_dynamic_brake = pm.Normal("beta_dynamic_brake", mu=0, sigma=1, shape=(n_dynamic,))

    alpha_tire  = pm.TruncatedNormal("alpha_tire",  mu=2, sigma=1, lower=0.5)
    alpha_brake = pm.TruncatedNormal("alpha_brake", mu=2, sigma=1, lower=0.5)

    beta0_tire  = pm.TruncatedNormal("beta0_tire",  mu=mean_ttf_tire,  sigma=mean_ttf_tire  / 2, lower=1.0)
    beta0_brake = pm.TruncatedNormal("beta0_brake", mu=mean_ttf_brake, sigma=mean_ttf_brake / 2, lower=1.0)

    lp_tire  = pm.math.dot(x_f_tire,  beta_fixed_tire)  + pm.math.dot(x_d_tire,  beta_dynamic_tire)
    lp_brake = pm.math.dot(x_f_brake, beta_fixed_brake) + pm.math.dot(x_d_brake, beta_dynamic_brake)

    t_tire_safe  = pm.math.clip(t_tire,  1e-6, 1e10)
    t_brake_safe = pm.math.clip(t_brake, 1e-6, 1e10)

    log_haz_tire  = pm.math.log(alpha_tire  / beta0_tire)  + (alpha_tire  - 1) * pm.math.log(t_tire_safe  / beta0_tire)
    log_haz_brake = pm.math.log(alpha_brake / beta0_brake) + (alpha_brake - 1) * pm.math.log(t_brake_safe / beta0_brake)

    cum_haz_tire  = pm.math.exp(lp_tire)  * (t_tire_safe  / beta0_tire)  ** alpha_tire
    cum_haz_brake = pm.math.exp(lp_brake) * (t_brake_safe / beta0_brake) ** alpha_brake

    pm.Potential("tire_lik",
        pm.math.sum(ev_tire  * (lp_tire  + log_haz_tire)  - cum_haz_tire))
    pm.Potential("brake_lik",
        pm.math.sum(ev_brake * (lp_brake + log_haz_brake) - cum_haz_brake))

print("Running ADVI warm start ...")
with model:
    approx = pm.fit(n=10_000, method="advi", progressbar=True)
    advi_mean = approx.mean.eval()

print("Sampling (nutpie / JAX) ...")
compiled = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="jax")
trace = nutpie.sample(compiled, init_mean=advi_mean)

# ---------------------------------------------------------------------------
# 5. Save trace
# ---------------------------------------------------------------------------
out_path = str(ASSETS / "model_trace.nc")
az.to_netcdf(trace, out_path)
print(f"Saved trace → {out_path}")
print("Done. You can now start the Dashboard.")
