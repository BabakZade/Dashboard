# metrics.py
from __future__ import annotations

from typing import Callable, Dict, Any
import numpy as np


# Only the metrics you want returned by compute_all
keys = [
    "mse",
    "cost",
    "accuracy",
    "critical_accuracy",
    "conservative_pct",
    "critical_conservative_pct",
    "epsilon",
    "critical_true_lt",
]


def _to_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


def _check_same_length(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt, yp = _to_1d(y_true), _to_1d(y_pred)
    if yt.size != yp.size:
        raise ValueError("y_true and y_pred must have the same length")
    return yt, yp


def _signed_err(y_true, y_pred) -> np.ndarray:
    yt, yp = _check_same_length(y_true, y_pred)
    return yp - yt  # err <= 0 => conservative or exact


def mse(y_true, y_pred) -> float:
    """
    Mean Squared Error.

    Measures the average squared difference between prediction and true value.
    Lower is better.
    """
    yt, yp = _check_same_length(y_true, y_pred)
    return float(np.mean((yp - yt) ** 2))


def cost(
    y_true,
    y_pred,
    *,
    cost_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """
    User-defined cost.

    The caller provides a custom cost function with signature:
        cost_fn(y_true_array, y_pred_array) -> float
    """
    yt, yp = _check_same_length(y_true, y_pred)
    return float(cost_fn(yt, yp))


def accuracy(y_true, y_pred, *, epsilon: float) -> float:
    """
    Accuracy within an error tolerance epsilon.

    A prediction is counted as accurate if:
        |pred - true| <= epsilon

    Returns percentage in [0, 100].
    """
    yt, yp = _check_same_length(y_true, y_pred)
    ae = np.abs(yp - yt)
    return float(np.mean(ae <= epsilon) * 100.0)


def conservative_pct(y_true, y_pred) -> float:
    """
    Percentage of conservative predictions.

    Conservative means:
        pred - true <= 0

    So the prediction is either exact or an under-prediction.
    Returns percentage in [0, 100].
    """
    err = _signed_err(y_true, y_pred)
    return float(np.mean(err <= 0.0) * 100.0)


def compute_all(
    y_true,
    y_pred,
    *,
    epsilon: float = 2.0,
    critical_true_lt: float = 10.0,
    cost_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> Dict[str, Any]:
    """
    Compute the selected metrics only.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.
    epsilon : float, default=2.0
        Error tolerance used for accuracy:
            |pred - true| <= epsilon
    critical_true_lt : float, default=10.0
        Critical time window threshold.
        Any sample with true value < critical_true_lt is considered critical.
    cost_fn : callable or None, default=None
        Optional custom cost function:
            cost_fn(y_true_array, y_pred_array) -> float

    Returns
    -------
    dict
        Dictionary containing only:
        - mse
        - cost
        - accuracy
        - critical_accuracy
        - conservative_pct
        - critical_conservative_pct
        - epsilon
        - critical_true_lt
    """
    yt, yp = _check_same_length(y_true, y_pred)

    out: Dict[str, Any] = {
        "mse": mse(yt, yp),
        "cost": None,
        "accuracy": accuracy(yt, yp, epsilon=epsilon),
        "conservative_pct": conservative_pct(yt, yp),
        "epsilon": float(epsilon),
        "critical_true_lt": float(critical_true_lt),
    }

    if cost_fn is not None:
        out["cost"] = cost(yt, yp, cost_fn=cost_fn)

    # Critical subset: true < critical_true_lt
    crit_mask = yt < critical_true_lt
    yt_c, yp_c = yt[crit_mask], yp[crit_mask]

    if yt_c.size == 0:
        out["critical_accuracy"] = None
        out["critical_conservative_pct"] = None
    else:
        out["critical_accuracy"] = accuracy(yt_c, yp_c, epsilon=epsilon)
        out["critical_conservative_pct"] = conservative_pct(yt_c, yp_c)

    return {k: out.get(k, None) for k in keys}


def metric_labels(epsilon: float, critical_true_lt: float) -> dict[str, str]:
    """
    Human-readable labels for reporting.
    """
    return {
        "mse": "MSE",
        "cost": "Mean cost",
        "accuracy": f"Accuracy (|pred - true| ≤ {epsilon:g})",
        "critical_accuracy": (
            f"Critical accuracy (true < {critical_true_lt:g} and |pred - true| ≤ {epsilon:g})"
        ),
        "conservative_pct": "Conservative predictions (pred - true ≤ 0)",
        "critical_conservative_pct": (
            f"Critical conservative predictions (true < {critical_true_lt:g} and pred - true ≤ 0)"
        ),
    }


def metric_descriptions(epsilon: float, critical_true_lt: float) -> dict[str, str]:
    """
    Longer descriptions for UI, logs, reports, or documentation.
    """
    return {
        "mse": (
            "Mean Squared Error. It measures the average squared difference "
            "between predicted and true values. Lower is better."
        ),
        "cost": (
            "Mean custom cost computed using the user-provided cost function."
        ),
        "accuracy": (
            f"Percentage of predictions considered accurate under tolerance epsilon={epsilon:g}, "
            f"where a prediction is accurate if |pred - true| ≤ {epsilon:g}."
        ),
        "critical_accuracy": (
            f"Percentage of accurate predictions within the critical subset, "
            f"defined as samples where true < {critical_true_lt:g}. "
            f"Accuracy still uses |pred - true| ≤ {epsilon:g}."
        ),
        "conservative_pct": (
            "Percentage of predictions that are conservative, meaning pred - true ≤ 0. "
            "This includes exact predictions and under-predictions."
        ),
        "critical_conservative_pct": (
            f"Percentage of conservative predictions within the critical subset "
            f"(true < {critical_true_lt:g}), where conservative means pred - true ≤ 0."
        ),
    }