import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr


def subject_bootstrap_ci(
    targets,
    preds,
    subject_ids,
    n_bootstrap=2000,
    ci=95,
    random_state=42,
):
    """
    Bootstrap confidence intervals for cross-validated predictions
    using subject-level resampling (correct for CV).

    Parameters
    ----------
    targets : array-like (N,)
        Ground truth values (pooled OOF targets)

    preds : array-like (N,)
        Predictions aligned with targets (pooled OOF predictions)

    subject_ids : array-like (N,)
        Subject identifier for each sample

    n_bootstrap : int
        Number of bootstrap resamples

    ci : float
        Confidence interval width (e.g. 95)

    random_state : int
        RNG seed

    Returns
    -------
    dict
        metrics with point estimate + CI
    """

    rng = np.random.default_rng(random_state)

    targets = np.asarray(targets)
    preds = np.asarray(preds)
    subject_ids = np.asarray(subject_ids)

    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    boot_mae = []
    boot_rmse = []
    boot_r2 = []
    boot_r = []

    for _ in range(n_bootstrap):
        sampled_subjects = rng.choice(unique_subjects, size=n_subjects, replace=True)
        idx = np.isin(subject_ids, sampled_subjects)

        t = targets[idx]
        p = preds[idx]

        if len(np.unique(t)) < 2:
            continue

        boot_mae.append(mean_absolute_error(t, p))
        boot_rmse.append(np.sqrt(mean_squared_error(t, p)))
        boot_r2.append(r2_score(t, p))
        boot_r.append(pearsonr(t, p)[0])

    alpha = (100 - ci) / 2
    def interval(x):
        return np.round(np.percentile(x, [alpha, 100 - alpha]), 3)

    results = {
        "MAE": {"mean": mean_absolute_error(targets, preds), "ci": interval(boot_mae)},
        "RMSE": {"mean": np.sqrt(mean_squared_error(targets, preds)), "ci": interval(boot_rmse)},
        "R2": {"mean": r2_score(targets, preds), "ci": interval(boot_r2)},
        "Pearson r": {"mean": pearsonr(targets, preds)[0], "ci": interval(boot_r)},
    }

    return results
