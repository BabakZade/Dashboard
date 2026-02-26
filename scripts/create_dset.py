import pathlib

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import nutpie
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler


VEHICLE_TYPES = {"Vans": 0, "Trucks": 1}

FAILURE_TYPES = {np.nan: 0,"Tires": 1, "Brakes": 2}


def compute_time_since_last_event(time_values: np.ndarray, event_indicator: np.ndarray) -> np.ndarray:
    """Compute elapsed time since the last observed event at each timestep.

    At and immediately after an event, the clock resets to near zero.

    Parameters
    ----------
    time_values : (n_steps,) absolute timestamps
    event_indicator : (n_steps,) 1 where a failure was observed, else 0

    Returns
    -------
    (n_steps,) float32 – time elapsed since the most recent previous event,
    or since the start of the series if no event has yet occurred.
    """
    tsle = np.empty_like(time_values)
    last_event_time = time_values[0]
    for i in range(len(time_values)):
        if event_indicator[i]:
            last_event_time = time_values[i]
        tsle[i] = max(time_values[i] - last_event_time, 1e-6)
    return tsle.astype(np.float32)


def compute_ttf_with_reset(
    hazard_draws: np.ndarray,
    time_values: np.ndarray,
    event_indicator: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """Compute TTF quantile, restarting the prediction after each observed event.

    Between successive events the series is split into segments and
    compute_ttf_quantile_from_hazard is called on each independently,
    so the cumulative hazard integrator resets at each observed failure.

    Parameters
    ----------
    hazard_draws : (n_draws, n_steps)
    time_values  : (n_steps,)  absolute timestamps
    event_indicator : (n_steps,) 1 where a failure was observed
    quantile     : float in (0, 1)

    Returns
    -------
    (n_steps,) float32
    """
    n_steps = len(time_values)
    ttf_q = np.full(n_steps, np.inf, dtype=np.float32)

    # Segment boundaries: start at 0, restart after every event
    segment_starts = [0] + [i for i in range(n_steps) if event_indicator[i]]

    for seg_start in segment_starts:
        # Find end of this segment (next event after seg_start, exclusive)
        next_events = [i for i in range(seg_start + 1, n_steps) if event_indicator[i]]
        seg_end = next_events[0] + 1 if next_events else n_steps  # inclusive of next event

        seg_times = time_values[seg_start:seg_end]
        # Reset clock to be relative within the segment
        seg_times_relative = (seg_times - seg_times[0] + 1e-6).astype(np.float32)
        seg_hazard = hazard_draws[:, seg_start:seg_end]

        seg_ttf = compute_ttf_quantile_from_hazard(seg_hazard, seg_times_relative, quantile)
        ttf_q[seg_start:seg_end] = seg_ttf

    return ttf_q


def compute_time_to_failure(time_values: np.ndarray, failure_indicator: np.ndarray) -> np.ndarray:
    """Compute time-to-failure at each timestep using the next observed failure."""

    ttf = np.full(time_values.shape[0], np.nan, dtype=np.float32)
    next_failure_time = np.inf

    for i in range(time_values.shape[0] - 1, -1, -1):
        if failure_indicator[i]:
            next_failure_time = time_values[i]
            ttf[i] = 0.0
        elif np.isfinite(next_failure_time):
            ttf[i] = next_failure_time - time_values[i]

    return ttf


def compute_ttf_quantile_from_hazard(
    hazard_draws: np.ndarray, time_values: np.ndarray, quantile: float
) -> np.ndarray:
    """Compute TTF quantile from time-varying hazard draws along a trajectory.

    hazard_draws shape: (n_draws, n_steps)
    time_values shape: (n_steps,)
    """

    n_steps = time_values.shape[0]
    if n_steps == 0:
        return np.array([], dtype=np.float32)

    if n_steps == 1:
        return np.full(1, np.inf, dtype=np.float32)

    dt = np.diff(time_values)
    positive_dt = dt[dt > 0]
    last_dt = np.median(positive_dt) if positive_dt.size > 0 else 1.0
    dt = np.append(dt, last_dt).astype(np.float32)

    threshold = -np.log(np.clip(1.0 - quantile, 1e-9, 1.0))
    ttf_q = np.full(n_steps, np.inf, dtype=np.float32)

    for start_idx in range(n_steps):
        local_hazard = hazard_draws[:, start_idx:]
        local_dt = dt[start_idx:]
        incremental_hazard = local_hazard * local_dt[None, :]
        cumulative_hazard = np.cumsum(incremental_hazard, axis=1)

        reached = cumulative_hazard >= threshold
        has_crossing = reached.any(axis=1)
        horizon_ttf = np.sum(local_dt)
        if not has_crossing.any():
            ttf_q[start_idx] = horizon_ttf
            continue

        first_cross_idx = np.argmax(reached, axis=1)
        cumulative_time = np.cumsum(local_dt)

        crossing_time = np.full(local_hazard.shape[0], np.inf, dtype=np.float32)
        valid_rows = np.where(has_crossing)[0]

        for row in valid_rows:
            cross_col = first_cross_idx[row]
            hazard_level = max(local_hazard[row, cross_col], 1e-9)
            hazard_before = cumulative_hazard[row, cross_col - 1] if cross_col > 0 else 0.0
            remaining_hazard = max(threshold - hazard_before, 0.0)
            partial_dt = remaining_hazard / hazard_level
            partial_dt = min(partial_dt, local_dt[cross_col])
            time_before = cumulative_time[cross_col - 1] if cross_col > 0 else 0.0
            crossing_time[row] = time_before + partial_dt

        finite_crossing = crossing_time[np.isfinite(crossing_time)]
        if finite_crossing.size > 0:
            ttf_q[start_idx] = np.percentile(finite_crossing, quantile * 100.0)
        else:
            ttf_q[start_idx] = horizon_ttf

    return ttf_q

def build_survival_dataset(
    dset: pd.DataFrame,
    failure_type_key: str,
) -> dict[str, np.ndarray]:
    """Convert time-series data into survival format (inter-event intervals).

    For each machine, extracts intervals between successive failures of the given
    type. The final interval (from the last event to the last observation) is
    right-censored. Mean covariate values over each interval are used.

    Parameters
    ----------
    dset : pd.DataFrame
        Time-series dataset sorted by machine_id and time.
    failure_type_key : str
        Key into FAILURE_TYPES, e.g. "Tires" or "Brakes".

    Returns
    -------
    dict with arrays:
        t         : (N,) float32  – interval durations
        event     : (N,) int32    – 1=observed failure, 0=right-censored
        x_fixed   : (N, F) float32 – mean fixed covariates over the interval
        x_dynamic : (N, D) float32 – mean dynamic covariates over the interval
    """
    failure_code = FAILURE_TYPES[failure_type_key]
    records: list[dict] = []

    for _, machine_df in dset.groupby("machine_id"):
        machine_df = machine_df.sort_values("time").reset_index(drop=True)
        times = machine_df["time"].to_numpy(dtype=np.float32)
        is_failure = (machine_df["failure_type"] == failure_code).to_numpy()
        failure_indices = np.where(is_failure)[0]

        # Build (start_idx, end_idx, event) intervals
        prev_idx = 0
        for f_idx in failure_indices:
            if f_idx > prev_idx:
                chunk = machine_df.iloc[prev_idx : f_idx + 1]
                duration = float(times[f_idx] - times[prev_idx])
                if duration > 0.0:
                    records.append({
                        "t": duration,
                        "event": 1,
                        "x_fixed": chunk[["car_type", "region", "route"]].mean().to_numpy(dtype=np.float32),
                        "x_dynamic": chunk[["route_ratio", "speed", "load"]].mean().to_numpy(dtype=np.float32),
                    })
            prev_idx = f_idx

        # Final right-censored interval
        last_idx = len(times) - 1
        if prev_idx < last_idx:
            chunk = machine_df.iloc[prev_idx : last_idx + 1]
            duration = float(times[last_idx] - times[prev_idx])
            if duration > 0.0:
                records.append({
                    "t": duration,
                    "event": 0,
                    "x_fixed": chunk[["car_type", "region", "route"]].mean().to_numpy(dtype=np.float32),
                    "x_dynamic": chunk[["route_ratio", "speed", "load"]].mean().to_numpy(dtype=np.float32),
                })

    return {
        "t":         np.array([r["t"]         for r in records], dtype=np.float32),
        "event":     np.array([r["event"]     for r in records], dtype=np.int32),
        "x_fixed":   np.stack([r["x_fixed"]   for r in records]).astype(np.float32),
        "x_dynamic": np.stack([r["x_dynamic"] for r in records]).astype(np.float32),
    }


def create_dset(path : pathlib.Path = pathlib.Path("KULData/simulated_dataset/train")) -> pd.DataFrame:
    """Create a dataset from the given path.

    Parameters
    ----------
    path (pathlib.Path):
        The path to the dataset.

    Returns
    -------
    pd.DataFrame:
        The created dataset.
    """

    frames = []

    for file in path.glob("*.csv"):
        if file.name.find("failure") != -1:
            continue

        # Read the CSV file into a DataFrame
        one_file_raw_data = pd.read_csv(file, index_col=None, header=0)

        one_file_raw_data["car_type"] = one_file_raw_data["car_type"].map(VEHICLE_TYPES)
        one_file_raw_data["failure_type"] = one_file_raw_data["failure_type"].map(FAILURE_TYPES)

        original_machine_id = one_file_raw_data["machine_id"].astype(int)
        one_file_raw_data["machine_id"] = (
            one_file_raw_data["car_type"].astype(int) * 1_000 + original_machine_id
        )

        frames.append(one_file_raw_data)

    if not frames:
        return pd.DataFrame()

    dset = pd.concat(frames, ignore_index=True)

    dset.sort_values(by=["machine_id", "time"], inplace=True)

    # Return the created dataset
    return dset

if __name__ == "__main__":
    dset = create_dset()

    # Fit scalers on the full training set
    scaler_fixed = StandardScaler()
    scaler_fixed.fit(dset[["car_type", "region", "route"]].to_numpy(dtype=np.float32))

    scaler_dynamic = StandardScaler()
    scaler_dynamic.fit(dset[["route_ratio", "speed", "load"]].to_numpy(dtype=np.float32))

    # Build survival datasets (inter-event intervals per machine)
    tire_surv  = build_survival_dataset(dset, "Tires")
    brake_surv = build_survival_dataset(dset, "Brakes")

    # Scale covariates
    tire_surv["x_fixed"]    = scaler_fixed.transform(tire_surv["x_fixed"]).astype(np.float32)
    tire_surv["x_dynamic"]  = scaler_dynamic.transform(tire_surv["x_dynamic"]).astype(np.float32)
    brake_surv["x_fixed"]   = scaler_fixed.transform(brake_surv["x_fixed"]).astype(np.float32)
    brake_surv["x_dynamic"] = scaler_dynamic.transform(brake_surv["x_dynamic"]).astype(np.float32)

    # Estimate mean observed inter-failure time for informative beta0 prior
    mean_ttf_tire  = float(tire_surv["t"][tire_surv["event"] == 1].mean()) if (tire_surv["event"] == 1).any() else 100.0
    mean_ttf_brake = float(brake_surv["t"][brake_surv["event"] == 1].mean()) if (brake_surv["event"] == 1).any() else 100.0
    print(f"Mean observed tire TTF: {mean_ttf_tire:.1f}, brake TTF: {mean_ttf_brake:.1f}")

    n_fixed   = tire_surv["x_fixed"].shape[1]
    n_dynamic = tire_surv["x_dynamic"].shape[1]

    with pm.Model() as model:
        # --- Tire survival submodel ---
        x_f_tire = pm.Data("x_f_tire", tire_surv["x_fixed"])
        x_d_tire = pm.Data("x_d_tire", tire_surv["x_dynamic"])
        t_tire   = pm.Data("t_tire",   tire_surv["t"])
        ev_tire  = pm.Data("ev_tire",  tire_surv["event"].astype(np.float32))

        # --- Brake survival submodel ---
        x_f_brake = pm.Data("x_f_brake", brake_surv["x_fixed"])
        x_d_brake = pm.Data("x_d_brake", brake_surv["x_dynamic"])
        t_brake   = pm.Data("t_brake",   brake_surv["t"])
        ev_brake  = pm.Data("ev_brake",  brake_surv["event"].astype(np.float32))

        beta_fixed_tire    = pm.Normal("beta_fixed_tire",    mu=0, sigma=1, shape=(n_fixed,))
        beta_dynamic_tire  = pm.Normal("beta_dynamic_tire",  mu=0, sigma=1, shape=(n_dynamic,))
        beta_fixed_brake   = pm.Normal("beta_fixed_brake",   mu=0, sigma=1, shape=(n_fixed,))
        beta_dynamic_brake = pm.Normal("beta_dynamic_brake", mu=0, sigma=1, shape=(n_dynamic,))

        # Weibull shape (> 1 → wear-out failure, < 1 → infant mortality)
        alpha_tire  = pm.TruncatedNormal("alpha_tire",  mu=2, sigma=1, lower=0.5)
        alpha_brake = pm.TruncatedNormal("alpha_brake", mu=2, sigma=1, lower=0.5)

        # Weibull scale ~ mean inter-failure time
        beta0_tire  = pm.TruncatedNormal("beta0_tire",  mu=mean_ttf_tire,  sigma=mean_ttf_tire / 2,  lower=1.0)
        beta0_brake = pm.TruncatedNormal("beta0_brake", mu=mean_ttf_brake, sigma=mean_ttf_brake / 2, lower=1.0)

        # Linear predictors (log relative hazard)
        lp_tire  = pm.math.dot(x_f_tire,  beta_fixed_tire)  + pm.math.dot(x_d_tire,  beta_dynamic_tire)
        lp_brake = pm.math.dot(x_f_brake, beta_fixed_brake) + pm.math.dot(x_d_brake, beta_dynamic_brake)

        # Weibull PH log-likelihood (combined for events and censored in one expression):
        #   log L_i = event_i * log λ(t_i) + log S(t_i)
        #           = event_i * [lp_i + log(α/β₀) + (α-1)*log(t_i/β₀)]  -  exp(lp_i) * (t_i/β₀)^α
        t_tire_safe  = pm.math.clip(t_tire,  1e-6, 1e10)
        t_brake_safe = pm.math.clip(t_brake, 1e-6, 1e10)

        log_haz_tire  = pm.math.log(alpha_tire  / beta0_tire)  + (alpha_tire  - 1) * pm.math.log(t_tire_safe  / beta0_tire)
        log_haz_brake = pm.math.log(alpha_brake / beta0_brake) + (alpha_brake - 1) * pm.math.log(t_brake_safe / beta0_brake)

        cum_haz_tire  = pm.math.exp(lp_tire)  * (t_tire_safe  / beta0_tire)  ** alpha_tire
        cum_haz_brake = pm.math.exp(lp_brake) * (t_brake_safe / beta0_brake) ** alpha_brake

        log_lik_tire  = pm.math.sum(ev_tire  * (lp_tire  + log_haz_tire)  - cum_haz_tire)
        log_lik_brake = pm.math.sum(ev_brake * (lp_brake + log_haz_brake) - cum_haz_brake)

        pm.Potential("tire_lik",  log_lik_tire)
        pm.Potential("brake_lik", log_lik_brake)

    with model:
        print("Running ADVI warm start...")
        approx = pm.fit(n=10_000, method="advi", progressbar=True)
        advi_mean = approx.mean.eval()

    compiled = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="jax")
    trace = nutpie.sample(compiled, init_mean=advi_mean)

    graph = pm.model_to_graphviz(model)
    graph.render(filename="model_graph.pnh", directory="KULData", format="png")

    az.plot_trace(
        trace,
        var_names=[
            "beta_fixed_tire",
            "beta_dynamic_tire",
            "beta_fixed_brake",
            "beta_dynamic_brake",
            "alpha_tire",
            "beta0_tire",
            "alpha_brake",
            "beta0_brake",
        ],
    )

    az.plot_energy(trace)

    az.plot_posterior(
        trace,
        var_names=[
            "beta_fixed_tire",
            "beta_dynamic_tire",
            "beta_fixed_brake",
            "beta_dynamic_brake",
            "alpha_tire",
            "beta0_tire",
            "alpha_brake",
            "beta0_brake",
        ],
    )

    plt.show()

    test_dset = create_dset(path=pathlib.Path("KULData/simulated_dataset/test"))

    selected_machine_id = int(test_dset["machine_id"].iloc[0])
    machine_data = test_dset[test_dset["machine_id"] == selected_machine_id]
    x_fixed_future = machine_data[["car_type", "region", "route"]].to_numpy(dtype=np.float32)
    x_dynamic_future = machine_data[["route_ratio", "speed", "load"]].to_numpy(dtype=np.float32)

    x_fixed_future = scaler_fixed.transform(x_fixed_future).astype(np.float32, copy=False)
    x_dynamic_future = scaler_dynamic.transform(x_dynamic_future).astype(np.float32, copy=False)

    posterior = trace.posterior
    beta_fixed_tire_draws = posterior["beta_fixed_tire"].values.reshape(-1, x_fixed_future.shape[1])
    beta_dynamic_tire_draws = posterior["beta_dynamic_tire"].values.reshape(-1, x_dynamic_future.shape[1])
    lambda_0_tire_scale_draws = posterior["beta0_tire"].values.reshape(-1, 1)
    lambda_0_tire_shape_draws = posterior["alpha_tire"].values.reshape(-1, 1)

    beta_fixed_brake_draws = posterior["beta_fixed_brake"].values.reshape(-1, x_fixed_future.shape[1])
    beta_dynamic_brake_draws = posterior["beta_dynamic_brake"].values.reshape(-1, x_dynamic_future.shape[1])
    lambda_0_brake_scale_draws = posterior["beta0_brake"].values.reshape(-1, 1)
    lambda_0_brake_shape_draws = posterior["alpha_brake"].values.reshape(-1, 1)

    # Compute linear predictors
    linear_predictor_tire_future = (
        beta_fixed_tire_draws @ x_fixed_future.T + beta_dynamic_tire_draws @ x_dynamic_future.T
    )
    linear_predictor_brake_future = (
        beta_fixed_brake_draws @ x_fixed_future.T + beta_dynamic_brake_draws @ x_dynamic_future.T
    )

    # Compute Weibull baseline hazard at each time point (matching the model definition)
    # Use time-since-last-failure so the Weibull clock resets after each observed event
    ground_truth_tire = (machine_data["failure_type"] == FAILURE_TYPES["Tires"]).astype(int).to_numpy()
    ground_truth_brake = (machine_data["failure_type"] == FAILURE_TYPES["Brakes"]).astype(int).to_numpy()

    abs_time = machine_data["time"].to_numpy(dtype=np.float32)
    tsle_tire  = compute_time_since_last_event(abs_time, ground_truth_tire)
    tsle_brake = compute_time_since_last_event(abs_time, ground_truth_brake)

    # lambda_0(t) = (shape/scale) * (t/scale)^(shape-1)  where t = time since last failure
    lambda_0_tire_draws = (
        (lambda_0_tire_shape_draws / lambda_0_tire_scale_draws) *
        (tsle_tire[None, :] / lambda_0_tire_scale_draws) ** (lambda_0_tire_shape_draws - 1)
    )
    lambda_0_brake_draws = (
        (lambda_0_brake_shape_draws / lambda_0_brake_scale_draws) *
        (tsle_brake[None, :] / lambda_0_brake_scale_draws) ** (lambda_0_brake_shape_draws - 1)
    )

    # Combine baseline hazard with covariate effects
    lambda_tire_future = np.exp(linear_predictor_tire_future) * lambda_0_tire_draws + 1e-9
    lambda_brake_future = np.exp(linear_predictor_brake_future) * lambda_0_brake_draws + 1e-9

    p_tire_mean = (1.0 - np.exp(-lambda_tire_future)).mean(axis=0)
    p_brake_mean = (1.0 - np.exp(-lambda_brake_future)).mean(axis=0)
    p_tire_lo  = np.percentile(1.0 - np.exp(-lambda_tire_future),  5, axis=0)
    p_tire_hi  = np.percentile(1.0 - np.exp(-lambda_tire_future), 95, axis=0)
    p_brake_lo = np.percentile(1.0 - np.exp(-lambda_brake_future),  5, axis=0)
    p_brake_hi = np.percentile(1.0 - np.exp(-lambda_brake_future), 95, axis=0)

    time_values = abs_time
    # TTF prediction also resets after each observed failure
    tire_ttf_05 = compute_ttf_with_reset(lambda_tire_future,  time_values, ground_truth_tire,  quantile=0.05)
    tire_ttf_50 = compute_ttf_with_reset(lambda_tire_future,  time_values, ground_truth_tire,  quantile=0.50)
    tire_ttf_95 = compute_ttf_with_reset(lambda_tire_future,  time_values, ground_truth_tire,  quantile=0.95)
    brake_ttf_05 = compute_ttf_with_reset(lambda_brake_future, time_values, ground_truth_brake, quantile=0.05)
    brake_ttf_50 = compute_ttf_with_reset(lambda_brake_future, time_values, ground_truth_brake, quantile=0.50)
    brake_ttf_95 = compute_ttf_with_reset(lambda_brake_future, time_values, ground_truth_brake, quantile=0.95)
    ground_truth_tire_ttf = compute_time_to_failure(time_values, ground_truth_tire.astype(bool))
    ground_truth_brake_ttf = compute_time_to_failure(time_values, ground_truth_brake.astype(bool))

    valid_tire = np.isfinite(ground_truth_tire_ttf) & np.isfinite(tire_ttf_50)
    if valid_tire.any():
        tire_mae = np.mean(np.abs(tire_ttf_50[valid_tire] - ground_truth_tire_ttf[valid_tire]))
        print(f"Machine {selected_machine_id} Tire TTF MAE (median prediction): {tire_mae:.3f}")

    valid_brake = np.isfinite(ground_truth_brake_ttf) & np.isfinite(brake_ttf_50)
    if valid_brake.any():
        brake_mae = np.mean(np.abs(brake_ttf_50[valid_brake] - ground_truth_brake_ttf[valid_brake]))
        print(f"Machine {selected_machine_id} Brake TTF MAE (median prediction): {brake_mae:.3f}")

    # plt.figure(figsize=(10, 6))
    # plt.plot(np.arange(len(p_fault_mean)), p_fault_mean, marker='o', linestyle='-', linewidth=2, label='Predicted Fault Probability')
    # plt.plot(np.arange(len(ground_truth_fault)), ground_truth_fault, marker='s', linestyle='--', linewidth=2, label='Ground Truth Fault Indicator')
    # plt.xlabel('Time Step')
    # plt.ylabel('Fault Probability / Indicator')
    # plt.title('Predicted Fault Risk vs Ground Truth')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    failure_times_tire  = time_values[ground_truth_tire.astype(bool)]
    failure_times_brake = time_values[ground_truth_brake.astype(bool)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].fill_between(time_values, tire_ttf_05, tire_ttf_95, alpha=0.25, label='90% CI')
    axes[0].plot(time_values, tire_ttf_50, linewidth=2, label='Pred Tire TTF (Median)')
    axes[0].plot(time_values, ground_truth_tire_ttf, marker='s', linestyle='-.', linewidth=2, label='Actual Tire TTF')
    for ft in failure_times_tire:
        axes[0].axvline(ft, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Tire TTF')
    axes[0].set_title('Tire Time-to-Failure: Predicted vs Actual (red lines = observed failures)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].fill_between(time_values, brake_ttf_05, brake_ttf_95, alpha=0.25, label='90% CI')
    axes[1].plot(time_values, brake_ttf_50, linewidth=2, label='Pred Brake TTF (Median)')
    axes[1].plot(time_values, ground_truth_brake_ttf, marker='s', linestyle='-.', linewidth=2, label='Actual Brake TTF')
    for ft in failure_times_brake:
        axes[1].axvline(ft, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Brake TTF')
    axes[1].set_title('Brake Time-to-Failure: Predicted vs Actual (red lines = observed failures)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].fill_between(time_values, p_tire_lo, p_tire_hi, alpha=0.25, label='90% CI')
    axes[0].plot(time_values, p_tire_mean, linewidth=2, label='Pred Tire Fault Prob (Mean)')
    axes[0].plot(time_values, ground_truth_tire, marker='s', linestyle='--', linewidth=2, label='Actual Tire Fault')
    for ft in failure_times_tire:
        axes[0].axvline(ft, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Probability / Indicator')
    axes[0].set_title('Tire Fault Risk vs Ground Truth (red lines = observed failures)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].fill_between(time_values, p_brake_lo, p_brake_hi, alpha=0.25, label='90% CI')
    axes[1].plot(time_values, p_brake_mean, linewidth=2, label='Pred Brake Fault Prob (Mean)')
    axes[1].plot(time_values, ground_truth_brake, marker='s', linestyle='--', linewidth=2, label='Actual Brake Fault')
    for ft in failure_times_brake:
        axes[1].axvline(ft, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Probability / Indicator')
    axes[1].set_title('Brake Fault Risk vs Ground Truth (red lines = observed failures)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()