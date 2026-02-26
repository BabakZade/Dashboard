"""Pure-NumPy inference utilities for the Weibull PH survival model.

At runtime the Dashboard only needs:
  - arviz  (load NetCDF trace)
  - numpy / pandas  (already present)
  - scikit-learn is NOT required – scalers are stored as plain npz arrays.
"""

from __future__ import annotations

import os
from typing import Optional

import arviz as az
import numpy as np
import pandas as pd

_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)
_TRACE_PATH   = os.path.join(_ASSETS_DIR, "model_trace.nc")
_SCALERS_PATH = os.path.join(_ASSETS_DIR, "scalers.npz")

# Module-level caches -------------------------------------------------------
_TRACE   = None
_SCALERS = None
_PRED_CACHE: dict[tuple, dict] = {}   # (split, machine_id, failure_type) → result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def model_is_fitted() -> bool:
    return os.path.exists(_TRACE_PATH) and os.path.exists(_SCALERS_PATH)


def load_artifacts():
    """Load (and cache) trace + scaler params.  Returns (trace, scalers_dict)."""
    global _TRACE, _SCALERS
    if _TRACE is None:
        _TRACE = az.from_netcdf(_TRACE_PATH)
    if _SCALERS is None:
        _SCALERS = dict(np.load(_SCALERS_PATH))
    return _TRACE, _SCALERS


# ---------------------------------------------------------------------------
# Time-series utilities
# ---------------------------------------------------------------------------

def compute_time_since_last_event(
    time_values: np.ndarray, event_indicator: np.ndarray
) -> np.ndarray:
    tsle = np.empty_like(time_values, dtype=np.float32)
    last_event_time = time_values[0]
    for i in range(len(time_values)):
        if event_indicator[i]:
            last_event_time = time_values[i]
        tsle[i] = max(float(time_values[i] - last_event_time), 1e-6)
    return tsle


def compute_time_to_failure(
    time_values: np.ndarray, failure_indicator: np.ndarray
) -> np.ndarray:
    ttf = np.full(len(time_values), np.nan, dtype=np.float32)
    next_failure_time = np.inf
    for i in range(len(time_values) - 1, -1, -1):
        if failure_indicator[i]:
            next_failure_time = time_values[i]
            ttf[i] = 0.0
        elif np.isfinite(next_failure_time):
            ttf[i] = float(next_failure_time - time_values[i])
    return ttf


def _ttf_one_draw(
    H_d: np.ndarray,        # (n_steps+1,) cumulative hazard, H[0]=0
    ext_time: np.ndarray,   # (n_steps+1,) cumulative time,   t[0]=0
    h_d: np.ndarray,        # (n_steps,)   instantaneous hazard
    dt: np.ndarray,         # (n_steps,)   time-step sizes
    threshold: float,
    horizon: float,
) -> np.ndarray:
    """TTF for *all* start positions for a single posterior draw.

    Complexity: O(n_steps · log n_steps) via searchsorted — fully vectorised,
    no Python loops over timesteps.
    """
    n_steps = len(h_d)
    targets = H_d[:n_steps] + threshold          # (n_steps,) target cum-hazard

    # First index j in H_d where H_d[j] >= target[i]
    j_cross = np.searchsorted(H_d, targets, side="left")   # (n_steps,)

    # Valid if j_cross is strictly after start and within range
    valid = (j_cross > np.arange(n_steps)) & (j_cross <= n_steps)

    # Safe index into the n_steps+1 array
    j_s = np.where(valid, j_cross, 1)            # fallback → 1 to avoid oob
    j_step = np.clip(j_s - 1, 0, n_steps - 1)   # step index (0-based into h_d)

    h_at  = np.maximum(h_d[j_step], 1e-9)
    H_pre = np.where(j_s >= 1, H_d[np.maximum(j_s - 1, 0)], 0.0)
    rem   = np.maximum(targets - H_pre, 0.0)
    partial = np.minimum(rem / h_at, dt[j_step])
    t_pre = np.where(j_s >= 1, ext_time[np.maximum(j_s - 1, 0)], 0.0)

    # Subtract each start position's own time so the result is TTF (time
    # remaining from position i), not the absolute crossing time.
    start_times = ext_time[:n_steps]
    remaining_horizon = horizon - start_times          # time left to end of segment
    return np.where(valid, t_pre + partial - start_times, remaining_horizon).astype(np.float32)


def _compute_ttf_quantile_from_hazard(
    hazard_draws: np.ndarray, time_values: np.ndarray, quantile: float
) -> np.ndarray:
    """TTF quantile across posterior draws.

    Now loops over draws (≈400) rather than timesteps (≈4 000), with each
    iteration fully vectorised — ~10× faster than the original approach.
    """
    n_draws, n_steps = hazard_draws.shape
    if n_steps == 0:
        return np.array([], dtype=np.float32)
    if n_steps == 1:
        return np.full(1, np.inf, dtype=np.float32)

    dt = np.diff(time_values.astype(np.float32))
    pos = dt[dt > 0]
    last_dt = float(np.median(pos)) if pos.size > 0 else 1.0
    dt = np.append(dt, last_dt).astype(np.float32)

    ext_time = np.concatenate([[0.0], np.cumsum(dt)]).astype(np.float32)  # (n_steps+1,)
    horizon  = float(ext_time[-1])
    threshold = float(-np.log(np.clip(1.0 - quantile, 1e-9, 1.0)))

    # Pre-compute cumulative hazard for all draws at once: (n_draws, n_steps+1)
    cum_H = np.zeros((n_draws, n_steps + 1), dtype=np.float64)
    cum_H[:, 1:] = np.cumsum(hazard_draws.astype(np.float64) * dt[None, :], axis=1)

    ttf_draws = np.empty((n_draws, n_steps), dtype=np.float32)
    for d in range(n_draws):
        ttf_draws[d] = _ttf_one_draw(
            cum_H[d], ext_time, hazard_draws[d], dt, threshold, horizon
        )

    return np.percentile(ttf_draws, quantile * 100.0, axis=0).astype(np.float32)


def compute_ttf_with_reset(
    hazard_draws: np.ndarray,
    time_values: np.ndarray,
    event_indicator: np.ndarray,
    quantile: float,
) -> np.ndarray:
    n_steps = len(time_values)
    ttf_q = np.full(n_steps, np.inf, dtype=np.float32)
    segment_starts = [0] + [i for i in range(n_steps) if event_indicator[i]]

    for seg_start in segment_starts:
        seg_end = n_steps
        seg_times = (time_values[seg_start:seg_end] - time_values[seg_start] + 1e-6).astype(np.float32)
        seg_h = hazard_draws[:, seg_start:seg_end]
        ttf_q[seg_start:seg_end] = _compute_ttf_quantile_from_hazard(seg_h, seg_times, quantile)

    return ttf_q


def compute_ttf_with_projected_tsle(
    time_values: np.ndarray,
    tsle: np.ndarray,
    lp: np.ndarray,
    b0: np.ndarray,
    alp: np.ndarray,
    fault_indicator: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """TTF quantile using a causal, no-future-reset TSLE projection.

    For each segment starting at `seg_start`, TSLE is projected forward as:
        tsle_proj[j] = tsle[seg_start] + (time[j] - time[seg_start])

    This eliminates leakage from future failure events baked into the
    pre-computed hazard array, while preserving the observed TSLE at the
    start of each segment.
    """
    n_steps = len(time_values)
    ttf_q = np.full(n_steps, np.inf, dtype=np.float32)
    segment_starts = [0] + [i for i in range(n_steps) if fault_indicator[i]]

    for seg_start in segment_starts:
        seg_times = time_values[seg_start:]
        tsle_at_start = float(tsle[seg_start])

        # Project TSLE monotonically forward — no resets after seg_start
        tsle_proj = (tsle_at_start + (seg_times - seg_times[0])).astype(np.float64)  # (seg_len,)

        # Recompute lam0 with projected TSLE: (draws, seg_len)
        lam0_proj = (alp / b0) * (tsle_proj[None, :] / b0) ** (alp - 1)
        lam_proj = (np.exp(lp[:, seg_start:]) * lam0_proj + 1e-9).astype(np.float32)

        seg_times_rel = (seg_times - seg_times[0] + 1e-6).astype(np.float32)
        ttf_q[seg_start:] = _compute_ttf_quantile_from_hazard(lam_proj, seg_times_rel, quantile)

    return ttf_q


# ---------------------------------------------------------------------------
# Prediction entry point
# ---------------------------------------------------------------------------

FAILURE_TYPE_CODES = {"Tires": 1, "Brakes": 2}


def predict_machine(
    machine_df: pd.DataFrame,
    failure_type: str = "Tires",
    cache_key: tuple | None = None,
) -> Optional[dict]:
    """Compute TTF quantiles and failure probability for a single machine.

    Parameters
    ----------
    machine_df  : rows for one machine (car_type and failure_type already int-encoded).
    failure_type: "Tires" | "Brakes"
    cache_key   : optional hashable key; if provided, results are memoised.
    """
    if cache_key is not None and cache_key in _PRED_CACHE:
        return _PRED_CACHE[cache_key]

    if not model_is_fitted():
        return None

    trace, scalers = load_artifacts()
    post = trace.posterior

    ft_key  = failure_type.lower().rstrip("s")   # "tire" or "brake"
    ft_code = FAILURE_TYPE_CODES[failure_type]

    machine_df = machine_df.sort_values("time").reset_index(drop=True)
    time_values = machine_df["time"].to_numpy(dtype=np.float32)

    x_fixed_raw   = machine_df[["car_type", "region", "route"]].to_numpy(dtype=np.float32)
    x_dynamic_raw = machine_df[["route_ratio", "speed", "load"]].to_numpy(dtype=np.float32)

    x_fixed   = (x_fixed_raw   - scalers["fixed_mean"])   / scalers["fixed_scale"]
    x_dynamic = (x_dynamic_raw - scalers["dynamic_mean"]) / scalers["dynamic_scale"]

    bf  = post[f"beta_fixed_{ft_key}"].values.reshape(-1, x_fixed.shape[1])
    bd  = post[f"beta_dynamic_{ft_key}"].values.reshape(-1, x_dynamic.shape[1])
    b0  = post[f"beta0_{ft_key}"].values.reshape(-1, 1)
    alp = post[f"alpha_{ft_key}"].values.reshape(-1, 1)

    fault_indicator = (machine_df["failure_type"] == ft_code).astype(int).to_numpy()
    tsle = compute_time_since_last_event(time_values, fault_indicator)

    lp   = bf @ x_fixed.T + bd @ x_dynamic.T           # (draws, T)
    lam0 = (alp / b0) * (tsle[None, :] / b0) ** (alp - 1)
    lam  = np.exp(lp) * lam0 + 1e-9                    # (draws, T)

    p_draws   = 1.0 - np.exp(-lam)
    prob_mean = p_draws.mean(axis=0)
    prob_lo   = np.percentile(p_draws,  5, axis=0)
    prob_hi   = np.percentile(p_draws, 95, axis=0)

    ttf_05   = compute_ttf_with_projected_tsle(time_values, tsle, lp, b0, alp, fault_indicator, 0.05)
    ttf_50   = compute_ttf_with_projected_tsle(time_values, tsle, lp, b0, alp, fault_indicator, 0.50)
    ttf_95   = compute_ttf_with_projected_tsle(time_values, tsle, lp, b0, alp, fault_indicator, 0.95)
    ttf_true = compute_time_to_failure(time_values, fault_indicator.astype(bool))

    result = dict(
        time=time_values,
        ttf_05=ttf_05, ttf_50=ttf_50, ttf_95=ttf_95, ttf_true=ttf_true,
        prob_mean=prob_mean, prob_lo=prob_lo, prob_hi=prob_hi,
        fault_indicator=fault_indicator,
    )

    if cache_key is not None:
        _PRED_CACHE[cache_key] = result

    return result