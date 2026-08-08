import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

TARGET = "actual_prep_duration_min"
KEEP_STATUS = ("ready", "completed")

CATEGORICAL_FEATURES = ["order_complexity", "weather_severity", "traffic_severity", "kitchen_id"]
NUMERIC_FEATURES = ["items_count", "workload_at_placement", "staff_level", "hour_sin", "hour_cos"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

VALID_KITCHEN_IDS = [1, 2, 3]
VALID_COMPLEXITY = ["simple", "standard", "complex"]
VALID_WEATHER = ["clear", "rain", "storm"]
VALID_TRAFFIC = ["low", "moderate", "heavy"]


def pool_orders(csv_paths):
    """Concatenate orders.csv files, dedupe by order_id, keep trainable rows."""
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise ValueError("no order CSVs provided")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="order_id", keep="first")
    df = df[df["status"].isin(KEEP_STATUS)]
    df = df.dropna(subset=[TARGET])
    return df.reset_index(drop=True)


def add_cyclical_hour(df):
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
    return df


def fit_encoder(df):
    enc = OrdinalEncoder(
        categories=[
            VALID_COMPLEXITY, VALID_WEATHER, VALID_TRAFFIC, [str(k) for k in VALID_KITCHEN_IDS]
        ],
        dtype=int,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    enc.fit(df[CATEGORICAL_FEATURES])
    return enc


def make_features(df, encoder):
    """Build the model matrix X (features) and target y from a pooled orders frame."""
    return make_X(df, encoder), df[TARGET].to_numpy(dtype=float)


def make_X(df, encoder):
    """Build only the model matrix X (features) from a pooled orders frame."""
    df = add_cyclical_hour(df)
    X_cat = encoder.transform(df[CATEGORICAL_FEATURES])
    X_num = df[NUMERIC_FEATURES].to_numpy(dtype=float)
    return np.hstack([X_num, X_cat])


def build_feature_vector(encoder, items_count, workload, staff_level, hour_of_day,
                         order_complexity, weather_severity, traffic_severity, kitchen_id):
    """Build a single-row model input matching the training feature order."""
    row = {
        "items_count": items_count,
        "workload_at_placement": workload,
        "staff_level": staff_level,
        "hour_of_day": int(hour_of_day),
        "order_complexity": order_complexity,
        "weather_severity": weather_severity,
        "traffic_severity": traffic_severity,
        "kitchen_id": int(kitchen_id),
    }
    df = pd.DataFrame([row])
    return make_X(df, encoder)


def temporal_split(df, frac=0.8):
    """Split strictly by placed_at time to avoid temporal leakage."""
    df = df.sort_values("placed_at").reset_index(drop=True)
    n = len(df)
    cutoff = int(n * frac)
    train = df.iloc[:cutoff].copy()
    test = df.iloc[cutoff:].copy()
    assert train["placed_at"].max() <= test["placed_at"].min(), "temporal leakage in split"
    return train, test
