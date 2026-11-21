"""AUROC, accuracy at a fixed threshold, and bootstrap confidence intervals."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y, s):
    return float(roc_auc_score(y, s))


def acc_at(y, s, thr):
    y, s = np.asarray(y), np.asarray(s)
    return float(np.mean((s >= thr).astype(int) == y))


def common_threshold(y_ctrl, s_ctrl):
    """Threshold that maximises accuracy on the control source; then applied to every source."""
    s = np.unique(np.asarray(s_ctrl))
    # candidates: every observed score plus one above the max (predict all negative)
    cands = np.append(s, s.max() + 1e-9)
    accs = [acc_at(y_ctrl, s_ctrl, t) for t in cands]
    return float(cands[int(np.argmax(accs))])


def bootstrap_ci(fn, y, s, n=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of fn(y, s). Resamples that lose a class are skipped."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], s[idx]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_diff(fn, y, s_a, s_b, n=1000, seed=0, alpha=0.05):
    """Point estimate and CI of fn(y, s_a) - fn(y, s_b) resampling the same units for both."""
    y, s_a, s_b = map(np.asarray, (y, s_a, s_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(fn(y[idx], s_a[idx]) - fn(y[idx], s_b[idx]))
    point = fn(y, s_a) - fn(y, s_b)
    return float(point), float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))
