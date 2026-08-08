import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.features import (  # noqa: E402
    pool_orders,
    add_cyclical_hour,
    fit_encoder,
    make_features,
    build_feature_vector,
    temporal_split,
    TARGET,
)


@pytest.fixture()
def sample_orders():
    return [
        {
            "order_id": i,
            "kitchen_id": 1,
            "placed_at": f"2025-01-01T{10 + (i % 8):02d}:00:00",
            "hour_of_day": 10 + (i % 8),
            "day_of_week": 3,
            "order_complexity": "simple" if i % 2 == 0 else "standard",
            "items_count": (i % 4) + 1,
            "workload_at_placement": float(i % 6),
            "staff_level": 4,
            "weather_severity": "clear",
            "traffic_severity": "normal",
            "actual_prep_duration_min": 5.0 + (i % 7),
            "status": "ready",
            "cancel_reason": "",
        }
        for i in range(20)
    ]


def test_pool_orders(sample_orders, tmp_path):
    import pandas as pd

    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    pd.DataFrame(sample_orders).to_csv(p1, index=False)
    pd.DataFrame(sample_orders).to_csv(p2, index=False)
    df = pool_orders([str(p1), str(p2)])
    assert len(df) == 20
    assert TARGET in df.columns


def test_add_cyclical_hour(sample_orders):
    import pandas as pd

    df = add_cyclical_hour(pd.DataFrame(sample_orders))
    sin2 = (df["hour_sin"] ** 2 + df["hour_cos"] ** 2).to_numpy()
    assert np.allclose(sin2, 1.0, atol=1e-9)


def test_temporal_split_keeps_order(sample_orders):
    import pandas as pd

    df = pd.DataFrame(sample_orders)
    train, test = temporal_split(df, frac=0.8)
    assert set(train.index).isdisjoint(set(test.index))
    latest_train = train["placed_at"].max()
    earliest_test = test["placed_at"].min()
    assert latest_train <= earliest_test


def test_encoder_roundtrip(sample_orders):
    import pandas as pd

    df = pd.DataFrame(sample_orders)
    encoder = fit_encoder(df)
    X_cat = encoder.transform(df[["order_complexity", "weather_severity", "traffic_severity", "kitchen_id"]])
    assert X_cat.shape == (20, 4)
    assert len(encoder.categories_) == 4


def test_make_features_shapes(sample_orders):
    import pandas as pd

    df = pd.DataFrame(sample_orders)
    encoder = fit_encoder(df)
    X, y = make_features(df, encoder)
    assert X.shape == (20, 9)
    assert y.shape == (20,)


def test_build_feature_vector_matches_training_shape(sample_orders):
    import pandas as pd

    df = pd.DataFrame(sample_orders)
    encoder = fit_encoder(df)
    X_row = build_feature_vector(
        encoder,
        items_count=3,
        workload=2.0,
        staff_level=4,
        hour_of_day=12,
        order_complexity="standard",
        weather_severity="clear",
        traffic_severity="normal",
        kitchen_id=1,
    )
    X_train, _ = make_features(df, encoder)
    assert X_row.shape[1] == X_train.shape[1]


def test_unknown_category_handled(sample_orders):
    import pandas as pd

    df = pd.DataFrame(sample_orders)
    encoder = fit_encoder(df)
    X_row = build_feature_vector(
        encoder,
        items_count=1,
        workload=0.0,
        staff_level=4,
        hour_of_day=12,
        order_complexity="complex",
        weather_severity="rain",
        traffic_severity="normal",
        kitchen_id=99,
    )
    assert X_row.shape == (1, 9)
