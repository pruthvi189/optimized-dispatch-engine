"""Paired-comparison statistics for the experiment runner.

Differences are stored as ``adaptive - immediate``. Direction convention:
- ``METRIC_DIRECTIONS`` maps each metric to ``higher_is_better`` or
  ``lower_is_better``.
- On a lower-is-better metric, a negative mean difference means Adaptive is
  better.

Primary inference is a bootstrap percentile CI plus a sign-flip permutation
p-value, both implemented in numpy (chunked to bound transient memory).
The paired t-test (via ``scipy.stats.t.sf``) is a corroborating check.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as _scipy_stats

METRIC_DIRECTIONS = {
    "on_time_rate": "higher_is_better",
    "avg_delivery_min": "lower_is_better",
    "avg_late_min": "lower_is_better",
    "p95_delivery_min": "lower_is_better",
    "avg_order_wait_min": "lower_is_better",
    "avg_rider_wait_kitchen_min": "lower_is_better",
    "cost_score": "lower_is_better",
}


@dataclass
class PairedDiffStats:
    """Statistical summary of paired differences (adaptive - immediate)."""

    n: int
    mean_diff: float
    std_diff: float
    se_diff: float
    ci95_low: float
    ci95_high: float
    t_stat: float
    p_value_permutation: float
    p_value_ttest: float
    cohens_dz: float
    significant: bool
    direction: str
    method: str

    def to_dict(self):
        return asdict(self)


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, clamped to [0, 1]."""
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denom
    half = z * np.sqrt(max(0.0, p * (1 - p) / trials + z2 / (4 * trials * trials))) / denom
    return (max(0.0, float(centre - half)), min(1.0, float(centre + half)))


def paired_difference_stats(
    diffs,
    n_resamples: int = 9999,
    seed: int = 12345,
    direction: str = "lower_is_better",
    alpha: float = 0.05,
    batch: int = 250,
) -> PairedDiffStats:
    """Paired-difference statistics on ``diffs`` (adaptive - immediate).

    - Bootstrap percentile CI (chunked over ``batch`` to keep transient
      memory bounded).
    - Sign-flip permutation p-value (two-sided; conservative
      ``(count + 1) / (n_resamples + 1)`` estimator).
    - Paired t-test corroboration via ``scipy.stats.t.sf``.
    - Cohen's d_z effect size (mean / sample std of the differences).
    """
    x = np.asarray(diffs, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("paired_difference_stats requires at least one difference")
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0

    if n == 1:
        return PairedDiffStats(
            n=1, mean_diff=mean, std_diff=0.0, se_diff=0.0,
            ci95_low=mean, ci95_high=mean, t_stat=0.0,
            p_value_permutation=1.0, p_value_ttest=1.0, cohens_dz=0.0,
            significant=False, direction=direction, method="degenerate (n=1)",
        )

    rng = np.random.default_rng(seed)

    draws = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, batch):
        k = min(batch, n_resamples - start)
        idx = rng.integers(0, n, size=(k, n))
        draws[start:start + k] = np.mean(x[idx], axis=1)
    draws.sort()
    lo_i = int(np.floor((alpha / 2) * n_resamples))
    hi_i = int(np.ceil((1 - alpha / 2) * n_resamples)) - 1
    ci_low = float(draws[lo_i])
    ci_high = float(draws[hi_i])

    centered = x - mean
    c = centered[centered != 0.0]
    m = c.size
    if m:
        flips = rng.integers(0, 2, size=(n_resamples, m)).astype(bool)
        signs = np.where(flips, 1.0, -1.0)
        perm_means = np.mean(signs * c[None, :], axis=1)
        count = int(np.sum(np.abs(perm_means) >= np.abs(mean)))
        p_perm = float((count + 1) / (n_resamples + 1))
    else:
        p_perm = 1.0

    if se > 0.0:
        t = mean / se
        p_t = float(_scipy_stats.t.sf(abs(t), df=n - 1) * 2.0)
    else:
        t = 0.0
        p_t = 1.0

    cohens_dz = mean / std if std > 0.0 else 0.0
    significant = (ci_low > 0.0) or (ci_high < 0.0)

    return PairedDiffStats(
        n=n, mean_diff=mean, std_diff=std, se_diff=se,
        ci95_low=ci_low, ci95_high=ci_high, t_stat=t,
        p_value_permutation=p_perm, p_value_ttest=p_t, cohens_dz=cohens_dz,
        significant=significant, direction=direction,
        method="bootstrap-ci + sign-flip permutation (scipy t-test corroboration)",
    )
