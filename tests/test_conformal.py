import math

import numpy as np
import pytest

from plotscreen import conformal


def test_thresholds_are_order_statistics():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] * 2, dtype=float)
    labels = np.array([False] * 10 + [True] * 10)
    thr = conformal.calibrate(scores, labels, alpha=0.2, beta=0.2)
    # 10 positives: floor(0.2 * 11) = 2, so the 2nd smallest positive score.
    assert thr.t_clear == pytest.approx(0.2)
    # 10 negatives: 2nd largest negative score.
    assert thr.t_flag == pytest.approx(0.9)


def test_too_few_positives_never_clears():
    scores = np.array([0.1, 0.2, 0.9])
    labels = np.array([False, False, True])
    thr = conformal.calibrate(scores, labels, alpha=0.05, beta=0.05)
    assert thr.t_clear == -math.inf and thr.t_flag == math.inf
    assert set(conformal.decide(np.array([0.0, 0.5, 1.0]), thr)) == {conformal.REVIEW}


def test_non_finite_scores_go_to_review():
    thr = conformal.Thresholds(0.3, 0.7, 0.05, 0.05, 100, 100)
    out = conformal.decide(np.array([0.1, np.nan, 0.9, np.inf]), thr)
    assert list(out) == [conformal.CLEAR, conformal.REVIEW, conformal.FLAG, conformal.REVIEW]


def test_conflict_fails_closed():
    # When the model separates well the thresholds cross. A score between them
    # satisfies both rules and must be flagged, not cleared.
    thr = conformal.Thresholds(t_clear=0.8, t_flag=0.2, alpha=0.05, beta=0.05, n_pos=100, n_neg=100)
    assert conformal.decide(np.array([0.5]), thr)[0] == conformal.FLAG


def test_calibrate_rejects_nan():
    with pytest.raises(ValueError):
        conformal.calibrate(np.array([0.1, np.nan]), np.array([True, False]), 0.1, 0.1)


@pytest.mark.parametrize("alpha", [0.02, 0.05, 0.1])
def test_error_rates_hold_on_average(alpha):
    """The guarantee is distribution free: check it with a deliberately poor score."""
    rng = np.random.default_rng(0)
    missed, wrongly_flagged = [], []
    for _ in range(300):
        labels = rng.random(1200) < 0.15
        scores = rng.normal(labels * 0.8, 1.0)  # heavy overlap between classes
        cal, test = slice(0, 600), slice(600, None)
        thr = conformal.calibrate(scores[cal], labels[cal], alpha, alpha)
        ev = conformal.evaluate(conformal.decide(scores[test], thr), labels[test])
        missed.append(ev["false_clear_rate"])
        wrongly_flagged.append(ev["false_flag_rate"])
    assert np.mean(missed) <= alpha + 0.005
    assert np.mean(wrongly_flagged) <= alpha + 0.005


def test_guarantee_survives_a_change_in_prevalence():
    rng = np.random.default_rng(1)
    missed = []
    for _ in range(300):
        cal_y = rng.random(800) < 0.3
        test_y = rng.random(800) < 0.03  # deforestation ten times rarer when deployed
        thr = conformal.calibrate(rng.normal(cal_y * 1.5, 1.0), cal_y, 0.05, 0.05)
        ev = conformal.evaluate(conformal.decide(rng.normal(test_y * 1.5, 1.0), thr), test_y)
        if ev["n_deforested"]:
            missed.append(ev["false_clear"] / ev["n_deforested"])
    assert np.mean(missed) <= 0.06


def test_evaluate_counts():
    decisions = np.array(["clear", "clear", "flag", "review", "flag"], dtype=object)
    labels = np.array([False, True, True, True, False])
    ev = conformal.evaluate(decisions, labels)
    assert ev["false_clear"] == 1 and ev["false_flag"] == 1
    assert ev["false_clear_rate"] == pytest.approx(1 / 3)
    assert ev["review_share"] == pytest.approx(0.2)
    assert ev["flag_precision"] == pytest.approx(0.5)


def test_upper_bound():
    assert conformal.upper_bound(0, 100) == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-6)
    assert conformal.upper_bound(5, 100) > 0.05
    assert conformal.upper_bound(3, 0) == 1.0


def test_confidence_mode_is_stricter_and_keeps_its_promise():
    rng = np.random.default_rng(2)
    above_plain, above_sure, review_plain, review_sure = [], [], [], []
    for _ in range(300):
        labels = rng.random(3000) < 0.2
        scores = rng.normal(labels * 2.0, 1.0)
        cal, test = slice(0, 1000), slice(1000, None)
        plain = conformal.calibrate(scores[cal], labels[cal], 0.05, 0.05)
        sure = conformal.calibrate(scores[cal], labels[cal], 0.05, 0.05, confidence=0.9)
        assert sure.t_clear <= plain.t_clear and sure.t_flag >= plain.t_flag
        # error rate on the population, not on a finite test set
        from scipy.stats import norm

        above_plain.append(norm.cdf(plain.t_clear - 2.0) > 0.05)
        above_sure.append(norm.cdf(sure.t_clear - 2.0) > 0.05)
        review_plain.append((conformal.decide(scores[test], plain) == conformal.REVIEW).mean())
        review_sure.append((conformal.decide(scores[test], sure) == conformal.REVIEW).mean())
    assert 0.3 < np.mean(above_plain) < 0.7  # the average guarantee overshoots about half the time
    assert np.mean(above_sure) <= 0.15  # the confident one almost never does
    assert np.mean(review_sure) >= np.mean(review_plain)


def test_allowed_rank():
    assert conformal.allowed_rank(99, 0.05, None) == 5
    assert conformal.allowed_rank(99, 0.05, 0.95) < 5
    assert conformal.allowed_rank(10, 0.05, 0.95) == 0
    with pytest.raises(ValueError):
        conformal.calibrate(np.array([0.1, 0.9]), np.array([False, True]), 0.1, 0.1, confidence=1.5)


def test_effective_sample_size():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(100), 10)
    independent = rng.random(1000) < 0.2
    assert conformal.effective_sample_size(independent, groups) > 700
    copied = np.repeat(rng.random(100) < 0.2, 10)  # every plot in a block is the same
    assert conformal.effective_sample_size(copied, groups) <= 110
    assert conformal.effective_sample_size(independent, np.arange(1000)) == 1000
    assert conformal.effective_sample_size(np.zeros(1000), groups) == 1000


def test_clustered_calibration_is_more_careful():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(200), 10)
    block_effect = np.repeat(rng.normal(0, 1, 200), 10)
    labels = np.repeat(rng.random(200) < 0.3, 10)
    scores = labels * 2.0 + block_effect + rng.normal(0, 0.3, 2000)
    plain = conformal.calibrate(scores, labels, 0.05, 0.05, confidence=0.95)
    careful = conformal.calibrate(scores, labels, 0.05, 0.05, confidence=0.95, groups=groups)
    assert careful.n_eff_pos < careful.n_pos / 3
    assert careful.t_clear <= plain.t_clear and careful.t_flag >= plain.t_flag
