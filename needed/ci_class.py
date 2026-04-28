import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def subject_bootstrap_ci_class(
    targets,
    preds,
    subject_ids,
    n_bootstrap=2000,
    ci=95,
    random_state=42,
):
    """
    Bootstrap confidence intervals for cross-validated classification predictions
    using subject-level resampling (correct for CV).

    Parameters
    ----------
    targets : array-like (N,)
        Ground truth labels (pooled OOF targets)

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
        metrics with point estimate + CI (Accuracy, F1, Precision, Recall)
    """

    rng = np.random.default_rng(random_state)

    targets = np.asarray(targets)
    preds = np.asarray(preds)
    subject_ids = np.asarray(subject_ids)

    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    boot_acc = []
    boot_f1_w = []
    boot_f1_m = []
    boot_prec_w = []
    boot_prec_m = []
    boot_rec_w = []
    boot_rec_m = []

    for _ in range(n_bootstrap):
        sampled_subjects = rng.choice(unique_subjects, size=n_subjects, replace=True)
        idx = np.isin(subject_ids, sampled_subjects)

        t = targets[idx]
        p = preds[idx]

        if len(t) < 2 or len(np.unique(t)) < 2:
            continue

        boot_acc.append(accuracy_score(t, p))
        boot_f1_w.append(f1_score(t, p, average="weighted", zero_division=0))
        boot_f1_m.append(f1_score(t, p, average="macro", zero_division=0))
        boot_prec_w.append(
            precision_score(t, p, average="weighted", zero_division=0)
        )
        boot_prec_m.append(precision_score(t, p, average="macro", zero_division=0))
        boot_rec_w.append(recall_score(t, p, average="weighted", zero_division=0))
        boot_rec_m.append(recall_score(t, p, average="macro", zero_division=0))

    alpha = (100 - ci) / 2

    def interval(x):
        return np.round(np.percentile(x, [alpha, 100 - alpha]), 3)

    results = {
        "Accuracy": {
            "mean": accuracy_score(targets, preds),
            "ci": interval(boot_acc),
        },
        "F1 (weighted)": {
            "mean": f1_score(targets, preds, average="weighted", zero_division=0),
            "ci": interval(boot_f1_w),
        },
        "F1 (macro)": {
            "mean": f1_score(targets, preds, average="macro", zero_division=0),
            "ci": interval(boot_f1_m),
        },
        "Precision (weighted)": {
            "mean": precision_score(
                targets, preds, average="weighted", zero_division=0
            ),
            "ci": interval(boot_prec_w),
        },
        "Precision (macro)": {
            "mean": precision_score(
                targets, preds, average="macro", zero_division=0
            ),
            "ci": interval(boot_prec_m),
        },
        "Recall (weighted)": {
            "mean": recall_score(
                targets, preds, average="weighted", zero_division=0
            ),
            "ci": interval(boot_rec_w),
        },
        "Recall (macro)": {
            "mean": recall_score(
                targets, preds, average="macro", zero_division=0
            ),
            "ci": interval(boot_rec_m),
        },
    }

    return results
