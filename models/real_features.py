"""Feature engineering for real-world Bangalore food-delivery data.

Maps the Zomato/Courier dataset columns to the model interface expected by
models/train.py.  The target is delivery_time_min (total end-to-end), NOT
prep time — the real dataset has no prep/delivery split.

Available features (derived from real data):
    weather_severity  — clear / rain / storm
    traffic_severity  — low / moderate / heavy
    order_complexity  — simple / standard / complex
    hour_sin, hour_cos — cyclical hour encoding
    distance_km       — restaurant-to-customer distance

NOT available (synthetic-only, do not fabricate):
    items_count, workload_at_placement, staff_level, kitchen_id

The feature matrix order is: [numeric..., categorical...] matching the
existing models/features.py convention.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

TARGET = "delivery_time_min"

CATEGORICAL_FEATURES = ["order_complexity", "weather_severity", "traffic_severity"]
NUMERIC_FEATURES = ["distance_km", "hour_sin", "hour_cos"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

VALID_COMPLEXITY = ["simple", "standard", "complex"]
VALID_WEATHER = ["clear", "rain", "storm"]
VALID_TRAFFIC = ["low", "moderate", "heavy"]

# ── Normalisation maps (from raw string values) ───────────────────────
WEATHER_MAP = {
    "sunny": "clear",
    "cloudy": "clear",
    "fog": "rain",
    "windy": "clear",
    "stormy": "storm",
    "sandstorms": "storm",
}

TRAFFIC_MAP = {
    "low": "low",
    "medium": "moderate",
    "high": "heavy",
    "jam": "heavy",
}

COMPLEXITY_MAP = {
    "snack": "simple",
    "drinks": "simple",
    "meal": "standard",
    "buffet": "complex",
}


def load_bangalore(csv_path: str) -> pd.DataFrame:
    """Load the pre-extracted Bangalore CSV and add model columns."""
    df = pd.read_csv(csv_path)

    # Drop rows with missing target or key features
    df = df.dropna(subset=[TARGET, "hour_of_day", "Weather_conditions",
                           "Road_traffic_density", "Type_of_order", "distance_km"])

    # Map categorical columns to model vocabulary
    df["weather_severity"] = df["Weather_conditions"].str.lower().map(WEATHER_MAP)
    df["traffic_severity"] = df["Road_traffic_density"].str.lower().map(TRAFFIC_MAP)
    df["order_complexity"] = df["Type_of_order"].str.lower().map(COMPLEXITY_MAP)

    # Drop any rows where mapping produced NaN (unknown category)
    df = df.dropna(subset=["weather_severity", "traffic_severity", "order_complexity"])

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)

    return df.reset_index(drop=True)


def fit_encoder(df: pd.DataFrame) -> OrdinalEncoder:
    """Fit an OrdinalEncoder on the categorical columns."""
    enc = OrdinalEncoder(
        categories=[VALID_COMPLEXITY, VALID_WEATHER, VALID_TRAFFIC],
        dtype=int,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    enc.fit(df[CATEGORICAL_FEATURES])
    return enc


def make_features(df: pd.DataFrame, encoder: OrdinalEncoder):
    """Build model matrix X and target vector y."""
    return make_X(df, encoder), df[TARGET].to_numpy(dtype=float)


def make_X(df: pd.DataFrame, encoder: OrdinalEncoder) -> np.ndarray:
    """Build model matrix X from a DataFrame."""
    X_cat = encoder.transform(df[CATEGORICAL_FEATURES])
    X_num = df[NUMERIC_FEATURES].to_numpy(dtype=float)
    return np.hstack([X_num, X_cat])


def build_feature_vector(encoder: OrdinalEncoder, distance_km: float,
                         hour_of_day: float, order_complexity: str,
                         weather_severity: str, traffic_severity: str) -> np.ndarray:
    """Build a single-row model input for inference."""
    row = {
        "distance_km": distance_km,
        "hour_of_day": float(hour_of_day),
        "order_complexity": order_complexity,
        "weather_severity": weather_severity,
        "traffic_severity": traffic_severity,
    }
    df = pd.DataFrame([row])
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
    return make_X(df, encoder)
