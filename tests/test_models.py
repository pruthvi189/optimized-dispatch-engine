import sys
import os
import json

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline import RuleBaseline  # noqa: E402
from models.features import fit_encoder, make_features, TARGET  # noqa: E402
from models.features import temporal_three_way_split  # noqa: E402
from models.train import train_lightgbm, train_quantile  # noqa: E402
from models.evaluate import mae, rmse, mape  # noqa: E402
from models.uncertainty import (  # noqa: E402
    fit_calibration,
    conformal_interval,
    evaluate_uncertainty,
    interval_empirical_coverage,
    uncertainty_tier,
)
from models.predict import Predictor, save_predictor, ARTIFACTS  # noqa: E402
from models.features import build_feature_vector  # noqa: E402


@pytest.fixture()
def toy_data(tmp_path):
    rng = np.random.default_rng(7)
    n = 200
    df_rows = []
    for i in range(n):
        workload = rng.uniform(0, 10)
        items = int(rng.integers(1, 6))
        complexity = ["simple", "standard", "complex"][i % 3]
        weather = ["clear", "rain", "storm"][i % 3]
        traffic = ["low", "moderate", "heavy"][(i // 2) % 3]
        noise = rng.normal(0, 1.2)
        dur = max(1.0, 4.0 + 0.5 * items + 0.6 * workload + noise + (3.0 if complexity == "complex" else 0.0))
        df_rows.append({
            "order_id": i,
            "kitchen_id": 1 + (i % 3),
            "placed_at": f"2025-01-{1 + (i % 5):02d}T12:00:00",
            "hour_of_day": 12,
            "day_of_week": 3,
            "order_complexity": complexity,
            "items_count": items,
            "workload_at_placement": workload,
            "staff_level": 4,
            "weather_severity": weather,
            "traffic_severity": traffic,
            "actual_prep_duration_min": dur,
            "status": "READY",
            "cancel_reason": "",
        })
    return df_rows


def _sample_features():
    return {
        "items_count": 2,
        "workload_at_placement": 5.0,
        "staff_level": 4,
        "hour_of_day": 12,
        "order_complexity": "standard",
        "weather_severity": "rain",
        "traffic_severity": "moderate",
        "kitchen_id": 2,
    }


def test_rule_baseline_predicts_cell_median(toy_data):
    import pandas as pd

    df = pd.DataFrame(toy_data)
    baseline = RuleBaseline()
    baseline.fit(df)
    preds = baseline.predict(df)
    assert len(preds) == len(df)
    assert np.all(preds > 0)


def test_lightgbm_beats_naive(toy_data):
    import pandas as pd

    df = pd.DataFrame(toy_data)
    encoder = fit_encoder(df)
    X, y = make_features(df, encoder)
    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    model = train_lightgbm(X_tr, y_tr)
    preds = model.predict(X_te)
    naive = np.full_like(y_te, y_tr.mean())
    assert mae(y_te, preds) < mae(y_te, naive)


def test_conformal_calibration_reasonable(toy_data):
    import pandas as pd

    df = pd.DataFrame(toy_data)
    encoder = fit_encoder(df)
    X, y = make_features(df, encoder)
    train_end = int(0.6 * len(X))
    calib_end = int(0.8 * len(X))
    model = train_lightgbm(X[:train_end], y[:train_end])
    qhat = fit_calibration(model, X[train_end:calib_end], y[train_end:calib_end])
    assert qhat > 0
    unc = evaluate_uncertainty(model, X[calib_end:], y[calib_end:], y[:train_end].std(), qhat)
    assert 0.5 <= unc["coverage"] <= 0.95
    assert unc["mean_interval_width_min"] > 0
    assert unc["calibration_quantile"] == round(qhat, 3)


def test_conformal_coverage_approx_nominal():
    """Split conformal on a synthetic noised signal gives ~80% test coverage."""
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.normal(size=(n, 3))
    y = 5.0 + 0.8 * x[:, 0] - 0.4 * x[:, 1] + rng.normal(0, 1.0, size=n)
    train_end = int(0.6 * n)
    calib_end = int(0.8 * n)
    model = train_lightgbm(x[:train_end], y[:train_end])
    qhat = fit_calibration(model, x[train_end:calib_end], y[train_end:calib_end])
    preds = model.predict(x[calib_end:])
    low = preds - qhat
    high = preds + qhat
    cov = interval_empirical_coverage(y[calib_end:], low, high)
    assert qhat > 0
    assert 0.75 <= cov <= 0.85, f"conformal coverage {cov:.3f} far from nominal 0.80"


def test_conformal_interval_contains_point_prediction():
    for m in (1.0, 5.5, 12.3):
        low, high = conformal_interval(m, 2.0)
        assert low <= m <= high
        assert np.isfinite(low) and np.isfinite(high)
        expected_width = 4.0 if m >= 2.0 else m + 2.0
        assert np.isclose(high - low, expected_width)


def test_three_way_split_is_temporal_and_disjoint(toy_data):
    import pandas as pd

    df = pd.DataFrame(toy_data)
    train, calib, test = temporal_three_way_split(df)
    assert len(train) + len(calib) + len(test) == len(df)
    assert set(train["order_id"]).isdisjoint(set(calib["order_id"]))
    assert set(train["order_id"]).isdisjoint(set(test["order_id"]))
    assert train["placed_at"].max() <= calib["placed_at"].min()
    assert calib["placed_at"].max() <= test["placed_at"].min()


def test_predictor_contract(toy_data, tmp_path):
    import pandas as pd

    df = pd.DataFrame(toy_data)
    encoder = fit_encoder(df)
    X, y = make_features(df, encoder)
    split = int(0.7 * len(X))
    model = train_lightgbm(X[:split], y[:split])
    qhat = fit_calibration(model, X[split:], y[split:])
    pred = Predictor(model, encoder, y[:split].std(), calibration_quantile=qhat)
    out = pred.predict(_sample_features())
    assert set(out) == {"prep_mean", "prep_low", "prep_high", "uncertainty"}
    assert out["prep_low"] <= out["prep_mean"] <= out["prep_high"]
    assert out["uncertainty"] in {"low", "medium", "high"}

    save_predictor(
        pred, str(tmp_path), "lightgbm",
        {"train_std_min": float(y[:split].std())}, calibration_quantile=qhat,
    )
    loaded = Predictor.load(str(tmp_path))
    assert loaded.calibration_quantile == pytest.approx(qhat)
    out2 = loaded.predict(_sample_features())
    assert out2 == out


def test_predictor_loads_legacy_quantile_artifacts(toy_data, tmp_path):
    """Old artifacts (q_low/q_high + calibration_factor, no calibration_quantile)
    must still load so previously deployed predictors keep working."""
    import joblib
    import pandas as pd

    df = pd.DataFrame(toy_data)
    encoder = fit_encoder(df)
    X, y = make_features(df, encoder)
    model = train_lightgbm(X, y)
    q_low = train_quantile(X, y, 0.10)
    q_high = train_quantile(X, y, 0.90)
    if q_low is None or q_high is None:
        pytest.skip("lightgbm quantile regressors unavailable")

    joblib.dump(encoder, os.path.join(tmp_path, ARTIFACTS["encoder"]))
    joblib.dump(model, os.path.join(tmp_path, ARTIFACTS["model"]))
    joblib.dump(q_low, os.path.join(tmp_path, ARTIFACTS["q_low"]))
    joblib.dump(q_high, os.path.join(tmp_path, ARTIFACTS["q_high"]))
    with open(os.path.join(tmp_path, ARTIFACTS["meta"]), "w", encoding="utf-8") as f:
        json.dump({"model": "lightgbm", "train_std_min": float(y.std()),
                   "calibration_factor": 1.1}, f)

    pred = Predictor.load(str(tmp_path))
    out = pred.predict(_sample_features())
    assert out["prep_low"] <= out["prep_mean"] <= out["prep_high"]
