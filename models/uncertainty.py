import numpy as np
from sklearn.model_selection import KFold

from .train import train_quantile, HAS_LIGHTGBM

ALPHA_LOW = 0.10
ALPHA_HIGH = 0.90


def fit_quantiles(X_train, y_train, nominal=0.80):
    """Fit low (10th) and high (90th) percentile LightGBM regressors,
    then calibrate a widening multiplier via cross-validated out-of-fold
    predictions so the interval generalizes to ~nominal coverage."""
    q_low = train_quantile(X_train, y_train, ALPHA_LOW)
    q_high = train_quantile(X_train, y_train, ALPHA_HIGH)
    if q_low is None or q_high is None:
        return q_low, q_high, 1.0

    cov = interval_empirical_coverage(
        y_train, q_low.predict(X_train), q_high.predict(X_train)
    )
    if cov <= 0.0:
        return q_low, q_high, 1.0
    if cov >= nominal:
        return q_low, q_high, 1.0

    oof_low = np.full_like(y_train, np.nan, dtype=float)
    oof_high = np.full_like(y_train, np.nan, dtype=float)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kfold.split(X_train):
        lo = train_quantile(X_train[train_idx], y_train[train_idx], ALPHA_LOW)
        hi = train_quantile(X_train[train_idx], y_train[train_idx], ALPHA_HIGH)
        if lo is None or hi is None:
            return q_low, q_high, 1.0
        oof_low[val_idx] = lo.predict(X_train[val_idx])
        oof_high[val_idx] = hi.predict(X_train[val_idx])

    mid = (oof_low + oof_high) / 2.0
    half = np.maximum((oof_high - oof_low) / 2.0, 1e-6)

    factor = 1.0
    for _ in range(50):
        low = mid - factor * half
        high = mid + factor * half
        if interval_empirical_coverage(y_train, low, high) >= nominal:
            break
        factor *= 1.1
    return q_low, q_high, factor


def predict_interval(q_low, q_high, X, factor=1.0):
    low = q_low.predict(X)
    high = q_high.predict(X)
    if factor != 1.0:
        mid = (low + high) / 2.0
        half = np.maximum((high - low) / 2.0, 1e-6) * factor
        low, high = mid - half, mid + half
    return low, high


def interval_empirical_coverage(y_true, y_low, y_high):
    """Fraction of true values inside the predicted interval."""
    inside = (y_true >= y_low) & (y_true <= y_high)
    return float(np.mean(inside))


def interval_width_std(y_low, y_high, y_train):
    """Normalize interval width by training residual std for tier thresholds."""
    residuals_std = float(np.std(y_train))
    widths = y_high - y_low
    return float(np.mean(widths)), max(residuals_std, 1e-9)


def uncertainty_tier(width, width_std, train_std):
    """Map mean interval width to a Low/Med/High risk tier (Phase 3 input)."""
    ratio = width / train_std
    if ratio < 0.5:
        return "low"
    if ratio < 1.0:
        return "medium"
    return "high"


def evaluate_uncertainty(q_low, q_high, X_test, y_test, y_train, factor=1.0):
    if q_low is None or q_high is None:
        return None
    y_low, y_high = predict_interval(q_low, q_high, X_test, factor=factor)
    coverage = interval_empirical_coverage(y_test, y_low, y_high)
    width, train_std = interval_width_std(y_low, y_high, y_train)
    tier = uncertainty_tier(width, train_std, train_std)
    return {
        "coverage": coverage,
        "mean_interval_width_min": round(width, 3),
        "train_std_min": round(train_std, 3),
        "uncertainty_tier": tier,
        "calibration_factor": factor,
    }
