# metrics.py
from __future__ import annotations

from typing import Callable, Dict, Any
import numpy as np


def _to_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


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


def error_within_0_min_pct(y_true, y_pred, *, min_err: float) -> float:
    """
    Percent of instances where |error| <= min_err.
    (Your 'min' is the epsilon threshold.)
    """
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    ae = np.abs(yp - yt)
    return float(np.mean(ae <= min_err) * 100.0)


def error_within_min_max_pct(y_true, y_pred, *, min_err: float, max_err: float) -> float:
    """
    Percent of instances where min_err < |error| < max_err.
    """
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    ae = np.abs(yp - yt)
    return float(np.mean((ae > min_err) & (ae < max_err)) * 100.0)


def error_largerthan_max_pct(y_true, y_pred, *, max_err: float) -> float:
    """
    Percent of instances where |error| >= max_err.
    """
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    ae = np.abs(yp - yt)
    return float(np.mean(ae >= max_err) * 100.0)


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
    Returns a dict with exactly these keys:

    Overall:
      - mse
      - cost
      - error_within_0_min_pct
      - error_within_min_max_pct
      - error_largerthan_max_pct

    Critical subset (true RUL < critical_true_lt):
      - error_within_0_min_pct_critical
      - error_within_min_max_pct_critical
      - error_largerthan_max_pct_critical

    Notes:
      - All *_pct metrics are 0..100.
      - If critical subset is empty, critical metrics return None.
      - mse/cost are computed on ALL data (as you requested).
    """
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")

    # overall metrics
    out: Dict[str, Any] = {
        "mse": mse(yt, yp),
        "cost": None,
        "error_within_0_min_pct": error_within_0_min_pct(yt, yp, min_err=min_err),
        "error_within_min_max_pct": error_within_min_max_pct(yt, yp, min_err=min_err, max_err=max_err),
        "error_largerthan_max_pct": error_largerthan_max_pct(yt, yp, max_err=max_err),
    }

    if cost_fn is not None:
        out["cost"] = cost(yt, yp, cost_fn=cost_fn)

    # critical subset (true RUL < 10)
    crit_mask = yt < critical_true_lt
    yt_c, yp_c = yt[crit_mask], yp[crit_mask]

    if yt_c.size == 0:
        out["error_within_0_min_pct_critical"] = None
        out["error_within_min_max_pct_critical"] = None
        out["error_largerthan_max_pct_critical"] = None
    else:
        out["error_within_0_min_pct_critical"] = error_within_0_min_pct(yt_c, yp_c, min_err=min_err)
        out["error_within_min_max_pct_critical"] = error_within_min_max_pct(yt_c, yp_c, min_err=min_err, max_err=max_err)
        out["error_largerthan_max_pct_critical"] = error_largerthan_max_pct(yt_c, yp_c, max_err=max_err)

    # (optional) include thresholds so other pages know how these metrics were computed
    out["min_err"] = float(min_err)
    out["max_err"] = float(max_err)
    out["critical_true_lt"] = float(critical_true_lt)

    return out