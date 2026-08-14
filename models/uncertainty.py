"""Split-conformal prediction intervals.

An 80% prediction interval is calibrated on a HOLD-OUT calibration split
(train/calibration/test) using the absolute residuals of the point model, so the
reported coverage is an honest out-of-sample statement — no tuning on the test
set, and no in-sample coverage search.

The interval is centered on the point prediction (prep_mean ± qhat), which makes
the dispatch mean and the interval consistent, unlike the previous quantile
midpoint widening that could shift the interval away from the forecast the
dispatch policy actually uses.
"""

import numpy as np


def fit_calibration(model, X_calib, y_calib, nominal=0.80):
    """Return the conformal quantile qhat from absolute residuals on a held-out
    calibration set.

    With the finite-sample correction the resulting interval [pred - qhat,
    pred + qhat] has marginal coverage >= nominal on new data (assuming
    exchangeability between calibration and future data). qhat is a single
    scalar width added to the point prediction in both directions.
    """
    residuals = np.abs(np.asarray(y_calib) - np.asarray(model.predict(X_calib)))
    n = len(residuals)
    if n == 0:
        raise ValueError("calibration set is empty")
    alpha = 1.0 - nominal
    level = min(1.0, (n + 1) / n * (1.0 - alpha))
    return float(max(np.quantile(residuals, level), 0.0))


def conformal_interval(prep_mean, qhat):
    """Interval centered on the point prediction: [mean - qhat, mean + qhat]."""
    low = max(0.0, float(prep_mean) - qhat)
    high = float(prep_mean) + qhat
    return low, high


def interval_empirical_coverage(y_true, y_low, y_high):
    """Fraction of true values inside the predicted interval."""
    y_low = np.asarray(y_low)
    y_high = np.asarray(y_high)
    inside = (y_true >= y_low) & (y_true <= y_high)
    return float(np.mean(inside))


def interval_width_std(y_low, y_high, y_train):
    """Normalize interval width by training residual std for tier thresholds."""
    residuals_std = float(np.std(y_train))
    widths = np.asarray(y_high) - np.asarray(y_low)
    return float(np.mean(widths)), max(residuals_std, 1e-9)


def uncertainty_tier(width, width_std, train_std):
    """Map mean interval width to a Low/Med/High risk tier (Phase 3 input)."""
    ratio = width / train_std
    if ratio < 0.5:
        return "low"
    if ratio < 1.0:
        return "medium"
    return "high"


def evaluate_uncertainty(model, X_test, y_test, train_std, qhat, nominal=0.80):
    """Report honest coverage/width/tier of the conformal interval on a held-out
    test set (never used during fitting or calibration)."""
    preds = np.asarray(model.predict(X_test))
    lows = np.maximum(preds - qhat, 0.0)
    highs = preds + qhat
    coverage = interval_empirical_coverage(y_test, lows, highs)
    width = float(np.mean(highs - lows))
    tier = uncertainty_tier(width, train_std, train_std)
    return {
        "nominal_coverage": nominal,
        "coverage": coverage,
        "mean_interval_width_min": round(width, 3),
        "train_std_min": round(float(train_std), 3),
        "uncertainty_tier": tier,
        "calibration_quantile": round(qhat, 3),
    }
