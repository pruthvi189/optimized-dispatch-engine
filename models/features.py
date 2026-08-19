import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

TARGET = "actual_prep_duration_min"
KEEP_STATUS = ("ready", "completed")

CATEGORICAL_FEATURES = ["order_complexity", "weather_severity", "kitchen_id"]
NUMERIC_FEATURES = ["items_count", "workload_at_placement", "staff_level", "hour_sin", "hour_cos"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

VALID_KITCHEN_IDS = [1, 2, 3]
VALID_COMPLEXITY = ["simple", "standard", "complex"]
VALID_WEATHER = ["clear", "rain", "storm"]


def pool_orders(csv_paths):
    """Concatenate orders.csv files, dedupe by (run_id, order_id), keep trainable rows.

    order_id resets to 1 at the start of every simulation run, so an order's
    identity is (run_id, order_id). Deduping by order_id alone collapses
    independent runs onto each other. Runs carrying a run_id column are keyed on
    both columns; legacy CSVs without the column fall back to order_id-only dedup.
    """
    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise ValueError("no order CSVs provided")
    df = pd.concat(frames, ignore_index=True)
    if "run_id" in df.columns:
        df = df.drop_duplicates(subset=["run_id", "order_id"], keep="first")
    else:
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
            VALID_COMPLEXITY, VALID_WEATHER, [str(k) for k in VALID_KITCHEN_IDS]
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
                         order_complexity, weather_severity, kitchen_id):
    """Build a single-row model input matching the training feature order.

    traffic_severity is deliberately NOT a prep-model feature: the prep-time
    simulator draws prep duration from weather/workload/staffing only, so
    including traffic would ask the model to learn signal that does not exist.
    """
    row = {
        "items_count": items_count,
        "workload_at_placement": workload,
        "staff_level": staff_level,
        "hour_of_day": int(hour_of_day),
        "order_complexity": order_complexity,
        "weather_severity": weather_severity,
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


def temporal_three_way_split(df, train_frac=0.7, calib_frac=0.15):
    """Strictly time-ordered train/calibration/test split for conformal calibration.

    The calibration set is held out from training so prediction-interval coverage
    can be tuned on it instead of the test set (which stays untouched for the
    final evaluation). Order boundaries never overlap in time.
    """
    df = df.sort_values("placed_at").reset_index(drop=True)
    n = len(df)
    train_cut = int(n * train_frac)
    calib_cut = train_cut + int(n * calib_frac)
    train = df.iloc[:train_cut].copy()
    calib = df.iloc[train_cut:calib_cut].copy()
    test = df.iloc[calib_cut:].copy()
    assert train["placed_at"].max() <= calib["placed_at"].min(), "temporal leakage train->calib"
    assert calib["placed_at"].max() <= test["placed_at"].min(), "temporal leakage calib->test"
    return train, calib, test


def temporal_four_way_split(df, train_frac=0.6, val_frac=0.15, calib_frac=0.15):
    """Strictly time-ordered train/validation/calibration/test split.

    Model selection happens on the validation split and conformal calibration
    on the calibration split, so neither selection nor calibration is tuned on
    the test set. The test split stays untouched for the final evaluation.
    """
    df = df.sort_values("placed_at").reset_index(drop=True)
    n = len(df)
    train_cut = int(n * train_frac)
    val_cut = train_cut + int(n * val_frac)
    calib_cut = val_cut + int(n * calib_frac)
    train = df.iloc[:train_cut].copy()
    val = df.iloc[train_cut:val_cut].copy()
    calib = df.iloc[val_cut:calib_cut].copy()
    test = df.iloc[calib_cut:].copy()
    assert train["placed_at"].max() <= val["placed_at"].min(), "temporal leakage train->val"
    assert val["placed_at"].max() <= calib["placed_at"].min(), "temporal leakage val->calib"
    assert calib["placed_at"].max() <= test["placed_at"].min(), "temporal leakage calib->test"
    return train, val, calib, test
