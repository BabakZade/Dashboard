"""CBM (Condition-Based Maintenance) RUL prediction module.

Implements a Bayesian exponential degradation model for Remaining Useful Life
prediction, based on the approach in scripts/CBM_RUL_prediction2.ipynb.

Uses fast importance resampling (RUL_prediction_resample2 / predict_segment_rolling)
for online RUL prediction.
"""

import numpy as np

# ============================================================================
# Hardcoded training posterior statistics
# (from fitting exp_model on assets/CBM_dataset/train/complete_df_train.csv)
# Recalibrate by running the training notebook if needed.
# ============================================================================

# Bounds on a0, a1 from the posterior (HalfNormal(sigma=0.01) prior)
MIN_A0 = 1e-5
MAX_A0 = 0.05
MIN_A1 = 1e-5
MAX_A1 = 0.10

# Posterior mean and covariance of (a0, a1) across all training segments
MEAN_A01 = np.array([0.005, 0.045])
COV_A01 = np.array([[1e-5, -5e-6],
                     [-5e-6, 5e-4]])

# Observation noise
SIGMA_OBS_MEAN = 0.42
SIGMA_OBS_STD = 0.05

# Symptom model constants (temp, vib power-law mappings)
SIGMA_TEMP = 4.5
SIGMA_VIB = 0.2575
SIGMA_L_CRIT = 1.0

# Critical degradation level
L_CRIT = 10.0

# Valid t_crit range (from RUL histogram of training data)
T_CRIT_LO = 65.0
T_CRIT_HI = 135.0

# Number of Monte Carlo samples
N_SAMPLES = 10_000
N_RESAMPLE = 1_000


# ============================================================================
# Fast importance-resampling predictor
# ============================================================================

def RUL_prediction_resample2(
    t: np.ndarray,
    L: np.ndarray,
    vib: np.ndarray,
    temp: np.ndarray,
    t_predict: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Importance-resampled RUL prediction using degradation + symptoms.

    Parameters
    ----------
    t : observed time steps (segment-relative)
    L : observed degradation values
    vib : observed vibration levels
    temp : observed brake temperatures
    t_predict : full time grid to predict degradation on

    Returns
    -------
    Lpred_full : (len(t_predict), N_RESAMPLE) predicted degradation curves
    RUL_pred   : (N_RESAMPLE,) resampled RUL values
    """
    L_crit = L_CRIT + SIGMA_L_CRIT * np.random.randn(N_SAMPLES)
    a0 = np.random.uniform(MIN_A0, MAX_A0, size=N_SAMPLES)
    a1 = np.random.uniform(MIN_A1, MAX_A1, size=N_SAMPLES)

    t_crit = np.log(1.0 + L_crit * (a1 / a0)) / a1
    mask_valid = (t_crit > T_CRIT_LO) & (t_crit < T_CRIT_HI)
    a0 = a0[mask_valid]
    a1 = a1[mask_valid]
    t_crit = t_crit[mask_valid]

    if len(a0) < 10:
        # Not enough valid samples — return NaNs
        return (
            np.full((len(t_predict), N_RESAMPLE), np.nan),
            np.full(N_RESAMPLE, np.nan),
        )

    # Predicted degradation at observed times
    Lpred = (a0[None, :] / a1[None, :]) * (np.exp(a1[None, :] * t[:, None]) - 1.0)
    temp_pred = 40.0 * (Lpred / 10.0) ** 1.5
    vib_pred = 2.5 * (Lpred / 10.0) ** 2.0

    RUL_sample = np.maximum(t_crit - t[-1], 0.0)

    # Log-likelihood weights from temperature + vibration
    temp_err = temp_pred - temp[:, None]
    vib_err = vib_pred - vib[:, None]

    logp = np.zeros(len(a0))
    logp += np.sum(-np.log(SIGMA_TEMP) - (temp_err / SIGMA_TEMP) ** 2 / 2, axis=0)
    logp += np.sum(-np.log(SIGMA_VIB) - (vib_err / SIGMA_VIB) ** 2 / 2, axis=0)
    logp -= np.max(logp)

    w = np.exp(logp)
    w_sum = np.sum(w)
    if w_sum == 0 or not np.isfinite(w_sum):
        w = np.ones(len(a0)) / len(a0)
    else:
        w /= w_sum

    idx = np.random.choice(len(a0), size=N_RESAMPLE, p=w)

    Lpred_full = (a0[None, idx] / a1[None, idx]) * (
        np.exp(a1[None, idx] * t_predict[:, None]) - 1.0
    )
    RUL_pred = RUL_sample[idx]

    return Lpred_full, RUL_pred


# ============================================================================
# Rolling RUL prediction (for maintenance cost analysis)
# ============================================================================

def predict_segment_rolling(
    time_seg: np.ndarray,
    obs_seg: np.ndarray,
    temp_seg: np.ndarray,
    vib_seg: np.ndarray,
) -> dict:
    """Online rolling RUL prediction using the fast resampler.

    At each timestep k, observes data [0..k] and predicts RUL.

    Returns dict with:
        time, rul_q05, rul_q50, rul_q95, rul_samples_matrix
    where rul_samples_matrix has shape (n_steps, N_RESAMPLE).
    """
    n_steps = len(time_seg)
    rul_q05 = np.full(n_steps, np.nan)
    rul_q50 = np.full(n_steps, np.nan)
    rul_q95 = np.full(n_steps, np.nan)
    rul_samples_matrix = np.full((n_steps, N_RESAMPLE), np.nan)

    for k in range(n_steps):
        t_obs = time_seg[: k + 1]
        L_obs = obs_seg[: k + 1]
        temp_obs = temp_seg[: k + 1]
        vib_obs = vib_seg[: k + 1]

        _, rul_samples = RUL_prediction_resample2(
            t=t_obs, L=L_obs, temp=temp_obs, vib=vib_obs, t_predict=time_seg,
        )
        rul_samples_matrix[k, :] = rul_samples
        valid = np.isfinite(rul_samples) & (rul_samples >= 0.0)
        if np.any(valid):
            rul_q05[k] = np.quantile(rul_samples[valid], 0.05)
            rul_q50[k] = np.quantile(rul_samples[valid], 0.50)
            rul_q95[k] = np.quantile(rul_samples[valid], 0.95)

    return {
        "time": time_seg,
        "rul_q05": rul_q05,
        "rul_q50": rul_q50,
        "rul_q95": rul_q95,
        "rul_samples_matrix": rul_samples_matrix,
    }


# ============================================================================
# Machine-level rolling prediction (across all segments)
# ============================================================================

def predict_machine_rolling(machine_df, segment_ids=None):
    """Run rolling RUL prediction for all segments of a machine.

    Returns a dict with arrays aligned to the full machine timeline:
        time, rul_q05, rul_q50, rul_q95, rul_true, fault_indicator
    """
    machine_df = machine_df.sort_values("time").reset_index(drop=True)
    time_all = machine_df["time"].to_numpy(dtype=float)
    rul_all = machine_df["rul"].to_numpy(dtype=float)
    n = len(time_all)

    rul_q05 = np.full(n, np.nan)
    rul_q50 = np.full(n, np.nan)
    rul_q95 = np.full(n, np.nan)
    rul_samples_matrix = np.full((n, N_RESAMPLE), np.nan)

    if segment_ids is None:
        segment_ids = sorted(machine_df["segment_id"].unique())

    for seg_id in segment_ids:
        seg_mask = machine_df["segment_id"] == seg_id
        seg_df = machine_df.loc[seg_mask].sort_values("time").reset_index(drop=True)
        seg_indices = machine_df.index[seg_mask].to_numpy()

        t_min = seg_df["time"].min()
        time_seg = (seg_df["time"] - t_min).to_numpy(dtype=float)
        obs_seg = seg_df["degra_level_observed"].to_numpy(dtype=float)
        temp_seg = seg_df["brake_temperature"].to_numpy(dtype=float)
        vib_seg = seg_df["vibration_level"].to_numpy(dtype=float)

        rolling = predict_segment_rolling(time_seg, obs_seg, temp_seg, vib_seg)

        rul_q05[seg_indices] = rolling["rul_q05"]
        rul_q50[seg_indices] = rolling["rul_q50"]
        rul_q95[seg_indices] = rolling["rul_q95"]
        rul_samples_matrix[seg_indices, :] = rolling["rul_samples_matrix"]

    # Fault indicator: RUL reaches 1 (end of segment = failure)
    fault_indicator = np.zeros(n, dtype=int)
    for seg_id in segment_ids:
        seg_mask = machine_df["segment_id"] == seg_id
        seg_indices = machine_df.index[seg_mask].to_numpy()
        if len(seg_indices) > 0:
            fault_indicator[seg_indices[-1]] = 1

    return {
        "time": time_all,
        "rul_q05": rul_q05,
        "rul_q50": rul_q50,
        "rul_q95": rul_q95,
        "rul_samples_matrix": rul_samples_matrix,
        "rul_true": rul_all,
        "fault_indicator": fault_indicator,
    }
