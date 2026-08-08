import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline import RuleBaseline  # noqa: E402
from models.features import fit_encoder, make_features, TARGET  # noqa: E402
from models.train import train_lightgbm, train_quantile  # noqa: E402
from models.evaluate import mae, rmse, mape  # noqa: E402
from models.uncertainty import (  # noqa: E402
    fit_quantiles,
    evaluate_uncertainty,
    interval_empirical_coverage,
    uncertainty_tier,
)
from models.predict import Predictor, save_predictor  # noqa: E402
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


def test_quantile_intervals_reasonable(toy_data):
    import pandas as pd

    df = pd.DataFrame(toy_data)
    encoder = fit_encoder(df)
    X, y = make_features(df, encoder)
    split = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    q_low, q_high, factor = fit_quantiles(X_tr, y_tr)
    assert q_low is not None and q_high is not None
    assert factor >= 1.0
    unc = evaluate_uncertainty(q_low, q_high, X_te, y_te, y_tr, factor=factor)
    assert 0.5 <= unc["coverage"] <= 0.95
    assert unc["mean_interval_width_min"] > 0


def test_predictor_contract(toy_data, tmp_path):
    import pandas as pd

    df = pd.DataFrame(toy_data)
    encoder = fit_encoder(df)
    X, y = make_features(df, encoder)
    model = train_lightgbm(X, y)
    q_low, q_high, factor = fit_quantiles(X, y)
    pred = Predictor(model, q_low, q_high, encoder, y.std(), calibration_factor=factor)
    out = pred.predict({
        "items_count": 2,
        "workload_at_placement": 5.0,
        "staff_level": 4,
        "hour_of_day": 12,
        "order_complexity": "standard",
        "weather_severity": "rain",
        "traffic_severity": "moderate",
        "kitchen_id": 2,
    })
    assert set(out) == {"prep_mean", "prep_low", "prep_high", "uncertainty"}
    assert out["prep_low"] <= out["prep_mean"] <= out["prep_high"]
    assert out["uncertainty"] in {"low", "medium", "high"}

    save_predictor(pred, tmp_path, "lightgbm",
                   {"train_std_min": float(y.std())}, calibration_factor=factor)
    loaded = Predictor.load(str(tmp_path))
    out2 = loaded.predict({
        "items_count": 2,
        "workload_at_placement": 5.0,
        "staff_level": 4,
        "hour_of_day": 12,
        "order_complexity": "standard",
        "weather_severity": "rain",
        "traffic_severity": "moderate",
        "kitchen_id": 2,
    })
    assert out2 == out
