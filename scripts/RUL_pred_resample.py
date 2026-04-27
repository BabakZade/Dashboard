import numpy as np

sigma_obs_mean = 0.42 #np.mean(idata_exp.posterior["sigma_obs"].values.flatten())
sigma_temp_mean = 4.5 #np.mean(idata_exp.posterior["sigma_temp"].values.flatten())
sigma_vib_mean = 0.2575 #np.mean(idata_exp.posterior["sigma_vib"].values.flatten
sigma_L_crit = 1.0
min_a0, max_a0 = 0.012, 0.102
min_a1, max_a1 = 0.0017, 0.0323

def RUL_prediction_resample(t, L, vib, temp, t_predict):
    # np.random.seed(42)
    # a01 = np.random.multivariate_normal(mean_a01, cov_matrix, size=10000)
    # a0 = a01[:, 0]
    # a1 = a01[:, 1] #np.clip(a01[:, 1], min_a1, max_a1)
    L_crit = 10.0 + sigma_L_crit * np.random.randn(10000)
    a0 = np.random.uniform(min_a0, max_a0, size=10000)
    a1 = np.random.uniform(min_a1, max_a1, size=10000)
    t_crit = np.log(1.0 + L_crit * (a1 / a0)) / a1
    mask_valid = (t_crit > 65) & (t_crit < 135)
    a0 = a0[mask_valid]
    a1 = a1[mask_valid]
    t_crit = t_crit[mask_valid]

    Lpred = (a0[None, :] / a1[None, :]) * (np.exp(a1[None, :] * t[:, None]) - 1.0)
    temp_pred = 40.0 * (Lpred/10.0)**1.5
    vib_pred = 2.5 * (Lpred/10.0)**2.
    
    RUL_sample = np.maximum(t_crit - t[-1], 0.0)
    Lerr = Lpred - L[:, None]
    temp_err = temp_pred - temp[:, None]
    vib_err = vib_pred - vib[:, None]
    logp_corr = np.zeros(len(a0))
    # logp_corr += np.sum(-np.log(sigma_obs_mean) - (Lerr / sigma_obs_mean)**2 / 2, axis=0).flatten()
    logp_corr += np.sum(-np.log(sigma_temp_mean) - (temp_err / sigma_temp_mean)**2 / 2, axis=0).flatten()
    logp_corr += np.sum(-np.log(sigma_vib_mean) - (vib_err / sigma_vib_mean)**2 / 2, axis=0).flatten()

    # Stabilize weights to avoid numerical underflow/overflow
    logp_corr = logp_corr - np.max(logp_corr)
    w = np.exp(logp_corr)
    w = w / np.sum(w)

    index_resampled = np.random.choice(len(a0), size=1000, p=w)
    Lpred_full = (a0[None, index_resampled] / a1[None, index_resampled]) * (
        np.exp(a1[None, index_resampled] * t_predict[:, None]) - 1.0
    )

    RUL_pred = RUL_sample[index_resampled]
    return Lpred_full, RUL_pred