import numpy as np

from rhw.metrics import acc_at, auroc, bootstrap_ci, common_threshold, paired_bootstrap_diff

Y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
PERFECT = Y * 0.8 + 0.1


def test_perfect_scorer():
    assert auroc(Y, PERFECT) == 1.0


def test_constant_scorer():
    assert auroc(Y, np.full(len(Y), 0.3)) == 0.5


def test_bootstrap_contains_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = y * 0.3 + rng.random(200)
    lo, hi = bootstrap_ci(auroc, y, s, n=300, seed=0)
    assert lo <= auroc(y, s) <= hi


def test_common_threshold():
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.6, 0.9]
    thr = common_threshold(y, s)
    assert 0.4 < thr <= 0.6
    assert acc_at(y, s, thr) == 1.0


def test_threshold_all_negative():
    # when every unit is negative the best threshold predicts nothing positive
    thr = common_threshold([0, 0, 0], [0.2, 0.5, 0.9])
    assert acc_at([0, 0, 0], [0.2, 0.5, 0.9], thr) == 1.0


def test_paired_diff_zero_for_same_scores():
    point, lo, hi = paired_bootstrap_diff(auroc, Y, PERFECT, PERFECT, n=100)
    assert point == lo == hi == 0.0
