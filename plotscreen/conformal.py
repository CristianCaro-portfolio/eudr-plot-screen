"""Three-way triage with class-conditional split conformal thresholds.

Every plot gets a score. Two thresholds are learned from a calibration set:

    score <  t_clear  -> clear   (treated as deforestation-free)
    score >  t_flag   -> flag    (treated as deforested)
    otherwise         -> review  (a person looks at it)

t_clear is set from the scores of deforested calibration plots so that a new
deforested plot is cleared with probability at most alpha. t_flag is set from
the scores of clean calibration plots so that a new clean plot is flagged with
probability at most beta. Both statements hold for any score function as long
as calibration and new plots are exchangeable within each class, and they do
not depend on how common deforestation is.

That guarantee is an average over calibration sets: about half of them end up
slightly above the target. With `confidence` set, the thresholds are moved so
that the error rate of this particular calibration stays under the target with
that probability, at the price of a wider review zone.

Neighbouring plots are not independent, so a thousand plots carry less
information than a thousand independent draws. With `groups` (the spatial block
of each plot) the calibration size is first shrunk by the Kish design effect.

When the two rules disagree the plot is flagged: the pipeline fails closed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import beta as beta_dist

CLEAR, REVIEW, FLAG = "clear", "review", "flag"


@dataclass
class Thresholds:
    t_clear: float
    t_flag: float
    alpha: float
    beta: float
    n_pos: int
    n_neg: int
    confidence: float | None = None
    n_eff_pos: int | None = None
    n_eff_neg: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def effective_sample_size(indicator: np.ndarray, groups: np.ndarray) -> int:
    """Number of independent observations a clustered sample is worth.

    Uses the Kish design effect 1 + (m - 1) * icc, with the intraclass
    correlation estimated by one-way analysis of variance and m the size of the
    cluster an average observation sits in.
    """
    z = np.asarray(indicator, dtype=np.float64)
    n = len(z)
    _, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    n_groups = len(sizes)
    if n_groups < 2 or n_groups == n or z.var() == 0:
        return n
    means = np.bincount(inverse, weights=z, minlength=n_groups) / sizes
    between = float((sizes * (means - z.mean()) ** 2).sum()) / (n_groups - 1)
    within = float(((z - means[inverse]) ** 2).sum()) / (n - n_groups)
    m0 = (n - (sizes**2).sum() / n) / (n_groups - 1)
    denominator = between + (m0 - 1) * within
    icc = max((between - within) / denominator, 0.0) if denominator > 0 else 0.0
    design_effect = 1 + ((sizes**2).sum() / n - 1) * icc
    return max(int(n / design_effect), 1)


def allowed_rank(n: int, rate: float, confidence: float | None, n_eff: int | None = None) -> int:
    """How many calibration scores may sit on the wrong side of the threshold.

    With the threshold at the k-th order statistic of n calibration scores, the
    error rate on new plots follows a Beta(k, n - k + 1) distribution. Without
    `confidence` k is chosen so its mean is at most `rate`. With it, k is the
    largest value for which the error rate is at most `rate` with probability
    `confidence`. When the sample is clustered the search runs on `n_eff`
    observations and the resulting fraction is applied to the n real ones.
    """
    k = math.floor(rate * (n + 1))
    if confidence is None:
        return k
    size = min(n_eff, n) if n_eff else n
    k_eff = math.floor(rate * (size + 1))
    while k_eff >= 1 and beta_dist.cdf(rate, k_eff, size - k_eff + 1) < confidence:
        k_eff -= 1
    return min(k, math.floor(k_eff * n / size)) if size else 0


def calibrate(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    beta: float,
    confidence: float | None = None,
    groups: np.ndarray | None = None,
) -> Thresholds:
    """Learn the two thresholds from calibration plots."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    if not np.isfinite(scores).all():
        raise ValueError("calibration scores contain NaN or inf")
    if confidence is not None and not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    order_pos, order_neg = np.argsort(scores[labels]), np.argsort(scores[~labels])
    pos, neg = scores[labels][order_pos], scores[~labels][order_neg]
    n_pos, n_neg = len(pos), len(neg)

    # Plain conformal ranks: a new deforested plot ranks uniformly among the
    # n_pos + 1 values, so it falls strictly below the k-th smallest with
    # probability k / (n_pos + 1). Same on the other side for clean plots.
    k_pos = math.floor(alpha * (n_pos + 1))
    k_neg = math.floor(beta * (n_neg + 1))

    n_eff_pos = n_eff_neg = None
    if confidence is not None:
        if groups is not None:
            groups = np.asarray(groups)
            wrong_pos = np.arange(n_pos) < max(k_pos, 1)  # would be cleared at the plain threshold
            wrong_neg = np.arange(n_neg) >= n_neg - max(k_neg, 1)  # would be flagged
            n_eff_pos = effective_sample_size(wrong_pos, groups[labels][order_pos]) if n_pos else 0
            n_eff_neg = effective_sample_size(wrong_neg, groups[~labels][order_neg]) if n_neg else 0
        k_pos = allowed_rank(n_pos, alpha, confidence, n_eff_pos)
        k_neg = allowed_rank(n_neg, beta, confidence, n_eff_neg)

    t_clear = pos[k_pos - 1] if k_pos >= 1 else -math.inf
    t_flag = neg[n_neg - k_neg] if k_neg >= 1 else math.inf
    return Thresholds(float(t_clear), float(t_flag), alpha, beta, n_pos, n_neg, confidence, n_eff_pos, n_eff_neg)


def decide(scores: np.ndarray, thr: Thresholds) -> np.ndarray:
    """Clear, flag or send to review. Non-finite scores go to review."""
    scores = np.asarray(scores, dtype=np.float64)
    out = np.full(scores.shape, REVIEW, dtype=object)
    finite = np.isfinite(scores)
    out[finite & (scores < thr.t_clear)] = CLEAR
    out[finite & (scores > thr.t_flag)] = FLAG  # applied last on purpose
    return out


def evaluate(decisions: np.ndarray, labels: np.ndarray) -> dict:
    """Error rates and workload of a triage on labelled plots."""
    labels = np.asarray(labels, dtype=bool)
    n, n_pos, n_neg = len(labels), int(labels.sum()), int((~labels).sum())
    cleared, flagged = decisions == CLEAR, decisions == FLAG
    false_clear = int((cleared & labels).sum())
    false_flag = int((flagged & ~labels).sum())
    return {
        "n": n,
        "n_deforested": n_pos,
        "false_clear_rate": false_clear / n_pos if n_pos else float("nan"),
        "false_flag_rate": false_flag / n_neg if n_neg else float("nan"),
        "false_clear": false_clear,
        "false_flag": false_flag,
        "clear_share": float(cleared.mean()) if n else float("nan"),
        "flag_share": float(flagged.mean()) if n else float("nan"),
        "review_share": float((decisions == REVIEW).mean()) if n else float("nan"),
        "flag_precision": float((flagged & labels).sum() / flagged.sum()) if flagged.any() else float("nan"),
        "clear_purity": float((cleared & ~labels).sum() / cleared.sum()) if cleared.any() else float("nan"),
    }


def upper_bound(errors: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound for an error rate."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    return float(beta_dist.ppf(confidence, errors + 1, n - errors))
