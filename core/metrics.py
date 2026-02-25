# metrics.py
from __future__ import annotations

from typing import Callable, Dict, Any
import numpy as np


# ---- ONLY the keys you want returned by compute_all ----
# (neg keys are placed right after their related "whole" KPI)
keys = [
    "mse",
    "cost",

    # overall: <=min (whole then conservative<=0)
    "error_within_0_min_pct",
    "error_within_0_min_pct_neg",

    # overall: min..max (whole then conservative<=0)
    "error_within_min_max_pct",
    "error_within_min_max_pct_neg",

    # overall: >=max (whole then conservative<=0)
    "error_largerthan_max_pct",
    "error_largerthan_max_pct_neg",

    # critical: <=min (whole then conservative<=0)
    "error_within_0_min_pct_critical",
    "error_within_0_min_pct_critical_neg",

    # critical: min..max (whole then conservative<=0)
    "error_within_min_max_pct_critical",
    "error_within_min_max_pct_critical_neg",

    # critical: >=max (whole then conservative<=0)
    "error_largerthan_max_pct_critical",
    "error_largerthan_max_pct_critical_neg",
]


def _to_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


def _signed_err(y_true, y_pred) -> np.ndarray:
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    return yp - yt  # err <= 0 => conservative or exact


def mse(y_true, y_pred) -> float:
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    return float(np.mean((yp - yt) ** 2))


def cost(y_true, y_pred, *, cost_fn: Callable[[np.ndarray, np.ndarray], float]) -> float:
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    return float(cost_fn(yt, yp))


# -----------------------------
# Absolute buckets (whole)
# -----------------------------
def error_within_0_min_pct(y_true, y_pred, *, min_err: float) -> float:
    """Percent of instances where |err| <= min_err."""
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    ae = np.abs(yp - yt)
    return float(np.mean(ae <= min_err) * 100.0)


def error_within_min_max_pct(y_true, y_pred, *, min_err: float, max_err: float) -> float:
    """Percent of instances where min_err < |err| < max_err."""
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    ae = np.abs(yp - yt)
    return float(np.mean((ae > min_err) & (ae < max_err)) * 100.0)


def error_largerthan_max_pct(y_true, y_pred, *, max_err: float) -> float:
    """Percent of instances where |err| >= max_err."""
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    ae = np.abs(yp - yt)
    return float(np.mean(ae >= max_err) * 100.0)


# ---------------------------------------------------
# Conservative-only buckets (err <= 0), SIGNED bands
#   small:   -min_err < err <= 0
#   mid:     -max_err < err <= -min_err
#   large:   err <= -max_err
# ---------------------------------------------------
def error_within_0_min_pct_neg(y_true, y_pred, *, min_err: float) -> float:
    """Percent where -min_err < err <= 0 (conservative or exact, small)."""
    err = _signed_err(y_true, y_pred)
    return float(np.mean((err > -min_err) & (err <= 0.0)) * 100.0)


def error_within_min_max_pct_neg(y_true, y_pred, *, min_err: float, max_err: float) -> float:
    """Percent where -max_err < err <= -min_err (conservative, moderate)."""
    err = _signed_err(y_true, y_pred)
    return float(np.mean((err > -max_err) & (err <= -min_err)) * 100.0)


def error_largerthan_max_pct_neg(y_true, y_pred, *, max_err: float) -> float:
    """Percent where err <= -max_err (conservative, large)."""
    err = _signed_err(y_true, y_pred)
    return float(np.mean(err <= -max_err) * 100.0)


def compute_all(
    y_true,
    y_pred,
    *,
    min_err: float = 2.0,
    max_err: float = 10.0,
    critical_true_lt: float = 10.0,
    cost_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> Dict[str, Any]:
    """
    Returns ONLY the KPIs listed in `keys`, with neg metrics placed right after their related KPI.

    Convention:
      err = y_pred - y_true
      conservative-only KPIs count err <= 0 (includes exact matches).
    """
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")

    out: Dict[str, Any] = {
        "mse": mse(yt, yp),
        "cost": None,

        # overall: <=min (whole then conservative<=0)
        "error_within_0_min_pct": error_within_0_min_pct(yt, yp, min_err=min_err),
        "error_within_0_min_pct_neg": error_within_0_min_pct_neg(yt, yp, min_err=min_err),

        # overall: min..max (whole then conservative<=0)
        "error_within_min_max_pct": error_within_min_max_pct(yt, yp, min_err=min_err, max_err=max_err),
        "error_within_min_max_pct_neg": error_within_min_max_pct_neg(yt, yp, min_err=min_err, max_err=max_err),

        # overall: >=max (whole then conservative<=0)
        "error_largerthan_max_pct": error_largerthan_max_pct(yt, yp, max_err=max_err),
        "error_largerthan_max_pct_neg": error_largerthan_max_pct_neg(yt, yp, max_err=max_err),
    }

    if cost_fn is not None:
        out["cost"] = cost(yt, yp, cost_fn=cost_fn)

    # critical subset (true < critical_true_lt)
    crit_mask = yt < critical_true_lt
    yt_c, yp_c = yt[crit_mask], yp[crit_mask]

    if yt_c.size == 0:
        out["error_within_0_min_pct_critical"] = None
        out["error_within_0_min_pct_critical_neg"] = None

        out["error_within_min_max_pct_critical"] = None
        out["error_within_min_max_pct_critical_neg"] = None

        out["error_largerthan_max_pct_critical"] = None
        out["error_largerthan_max_pct_critical_neg"] = None
    else:
        out["error_within_0_min_pct_critical"] = error_within_0_min_pct(yt_c, yp_c, min_err=min_err)
        out["error_within_0_min_pct_critical_neg"] = error_within_0_min_pct_neg(yt_c, yp_c, min_err=min_err)

        out["error_within_min_max_pct_critical"] = error_within_min_max_pct(yt_c, yp_c, min_err=min_err, max_err=max_err)
        out["error_within_min_max_pct_critical_neg"] = error_within_min_max_pct_neg(yt_c, yp_c, min_err=min_err, max_err=max_err)

        out["error_largerthan_max_pct_critical"] = error_largerthan_max_pct(yt_c, yp_c, max_err=max_err)
        out["error_largerthan_max_pct_critical_neg"] = error_largerthan_max_pct_neg(yt_c, yp_c, max_err=max_err)

    # return ONLY the requested keys (no extras)
    return {k: out.get(k, None) for k in keys}


def metric_labels(min_err: float, max_err: float, critical_true_lt: float) -> dict[str, str]:
    # labels ONLY for kept keys, with neg right after related KPI
    return {
        "mse": "MSE (days²)",
        "cost": "Mean cost",

        "error_within_0_min_pct": f"Accuracy (|err| ≤ {min_err:g})",
        "error_within_0_min_pct_neg": f"Conservative (−{min_err:g} < err ≤ 0)",

        "error_within_min_max_pct": f"Acceptable ({min_err:g} < |err| < {max_err:g})",
        "error_within_min_max_pct_neg": f"Conservative (−{max_err:g} < err ≤ −{min_err:g})",

        "error_largerthan_max_pct": f"Large error (|err| ≥ {max_err:g})",
        "error_largerthan_max_pct_neg": f"Conservative (err ≤ −{max_err:g})",

        "error_within_0_min_pct_critical": f"Critical accuracy (true < {critical_true_lt:g}, |err| ≤ {min_err:g})",
        "error_within_0_min_pct_critical_neg": f"Critical conservative (true < {critical_true_lt:g}, −{min_err:g} < err ≤ 0)",

        "error_within_min_max_pct_critical": f"Critical acceptable (true < {critical_true_lt:g}, {min_err:g} < |err| < {max_err:g})",
        "error_within_min_max_pct_critical_neg": f"Critical conservative (true < {critical_true_lt:g}, −{max_err:g} < err ≤ −{min_err:g})",

        "error_largerthan_max_pct_critical": f"Critical large error (true < {critical_true_lt:g}, |err| ≥ {max_err:g})",
        "error_largerthan_max_pct_critical_neg": f"Critical conservative (true < {critical_true_lt:g}, err ≤ −{max_err:g})",
    }