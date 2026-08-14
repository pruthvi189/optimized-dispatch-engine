"""Tests for paired-difference statistics (Task 5)."""

import numpy as np
import pytest

from dispatch.statistics import (
    METRIC_DIRECTIONS,
    PairedDiffStats,
    paired_difference_stats,
    wilson_ci,
)


def test_no_effect_is_not_significant():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0.0, 1.0, size=200)
    s = paired_difference_stats(diffs, n_resamples=999, seed=1)
    assert not s.significant
    assert 0.0 < s.p_value_permutation <= 1.0
    assert abs(s.mean_diff) < 0.3


def test_positive_effect_significant_lower_is_better():
    rng = np.random.default_rng(0)
    diffs = rng.normal(2.0, 1.0, size=200)
    s = paired_difference_stats(diffs, n_resamples=999, seed=1, direction="lower_is_better")
    # Positive = adaptive worse on lower-is-better, CI excludes 0.
    assert s.significant
    assert s.ci95_low > 0.0
    assert s.mean_diff > 0.0
    assert s.p_value_permutation < 0.01
    assert s.p_value_ttest < 0.01
    assert s.cohens_dz > 0.0


def test_negative_effect_significant_upper_is_better():
    rng = np.random.default_rng(0)
    # For a higher-is-better metric, a negative mean means adaptive is worse.
    diffs = rng.normal(-1.5, 1.0, size=300)
    s = paired_difference_stats(diffs, n_resamples=999, seed=1, direction="higher_is_better")
    assert s.significant
    assert s.ci95_high < 0.0


def test_ci_contains_true_mean():
    rng = np.random.default_rng(123)
    diffs = rng.normal(1.0, 2.0, size=100)
    s = paired_difference_stats(diffs, n_resamples=1999, seed=7)
    assert s.ci95_low <= s.mean_diff <= s.ci95_high
    assert s.n == 100
    assert s.mean_diff == pytest.approx(np.mean(diffs))


def test_seeded_reproducibility():
    diffs = [0.5, 1.0, -2.0, 3.0, 0.0, 1.5, -0.5, 2.0]
    a = paired_difference_stats(diffs, n_resamples=499, seed=99)
    b = paired_difference_stats(diffs, n_resamples=499, seed=99)
    assert a.to_dict() == b.to_dict()


def test_single_observation_degrades_gracefully():
    s = paired_difference_stats([0.5], seed=1)
    assert isinstance(s, PairedDiffStats)
    assert s.n == 1
    assert not s.significant
    assert s.ci95_low == s.ci95_high == 0.5
    assert s.p_value_permutation == 1.0


def test_empty_raises():
    with pytest.raises(ValueError):
        paired_difference_stats([])


def test_cohens_dz_sign_matches_mean():
    rng = np.random.default_rng(5)
    diffs = rng.normal(-0.8, 1.0, size=150)
    s = paired_difference_stats(diffs, n_resamples=499, seed=3)
    assert (s.cohens_dz < 0) == (s.mean_diff < 0)


def test_direction_conventions():
    assert METRIC_DIRECTIONS["on_time_rate"] == "higher_is_better"
    for k in ("avg_delivery_min", "avg_late_min", "p95_delivery_min",
              "avg_order_wait_min", "avg_rider_wait_kitchen_min", "cost_score"):
        assert METRIC_DIRECTIONS[k] == "lower_is_better"


def test_wilson_ci_bounds_and_contains_rate():
    lo, hi = wilson_ci(950, 1000)
    assert 0.0 < lo < hi < 1.0
    assert lo < 0.95 < hi
    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo2, hi2 = wilson_ci(1000, 1000)
    assert lo2 < 1.0 and hi2 == 1.0
    lo3, hi3 = wilson_ci(0, 1000)
    assert lo3 == 0.0 and hi3 > 0.0


def test_wilson_ci_contains_observed_rate():
    for trials, successes in [(10, 3), (100, 50), (5, 5), (7, 0)]:
        lo, hi = wilson_ci(successes, trials)
        assert lo <= successes / trials <= hi
