"""
Phase 1: Preprocess the Zomato food-delivery dataset for Bangalore.

Extracts Bangalore/Bengaluru records from the multi-city dataset,
normalizes columns, and produces a clean CSV ready for future
calibration and feature engineering.

Input:  data/external/zomato_cleaned.csv (from HuggingFace allenborochin/zomato_delivery_EDA)
Output: data/external/bangalore_orders.csv

Usage:
    python scripts/prepare_real_data.py
"""
import os
import sys
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "external")
RAW_PATH = os.path.join(DATA_DIR, "zomato_cleaned.csv")
OUT_PATH = os.path.join(DATA_DIR, "bangalore_orders.csv")

# ── City mapping ──────────────────────────────────────────────────────
# Delivery_person_ID encodes the city in its first 7 characters.
# Example: BANGRES17DEL01 → BANGRES → Bangalore
# Maps the 6-letter city stem (first 6 chars) to city name.
# Delivery_person_ID format: CITYRES##DEL##  e.g. BANGRES17DEL01
CITY_STEM_TO_CITY = {
    "BANGRE": "bangalore",
    "MUMRES": "mumbai",
    "PUNERE": "pune",
    "CHENRE": "chennai",
    "HYDRES": "hyderabad",
    "COIMBR": "coimbatore",
    "INDORE": "indore",
    "RANCHI": "ranchi",
    "KOLRES": "kolkata",
    "KOCRES": "kochi",
    "SURRES": "surat",
    "MYSRES": "mysore",
    "VADRES": "vadodara",
    "LUDHRE": "ludhiana",
    "AURGRE": "aurangabad",
    "KNPRES": "kanpur",
    "DEHRES": "delhi",
    "GOARES": "goa",
    "ALHRES": "allahabad",
    "AGRRES": "agra",
    "BHPRES": "bhopal",
    "JAPRES": "jaipur",
}

# ── Normalization maps ────────────────────────────────────────────────
WEATHER_MAP = {
    "sunny": "clear",
    "cloudy": "clear",
    "fog": "rain",       # fog ~ reduced visibility ≈ mild weather penalty
    "windy": "clear",
    "stormy": "storm",
    "sandstorms": "storm",
}

TRAFFIC_MAP = {
    "low": 0.0,
    "medium": 0.5,
    "high": 0.8,
    "jam": 1.0,
}

ORDER_TYPE_TO_COMPLEXITY = {
    "snack": "simple",
    "drinks": "simple",
    "meal": "standard",
    "buffet": "complex",
}


def load_and_normalize(path):
    """Load raw CSV and add normalized columns."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")

    # Extract city from Delivery_person_ID stem (first 6 chars)
    # Format: CITYRES##DEL##  e.g. BANGRES17DEL01 → stem BANGRE → bangalore
    df["city_stem"] = df["Delivery_person_ID"].str[:6]
    df["city"] = df["city_stem"].map(CITY_STEM_TO_CITY)

    unknown = df["city"].isna().sum()
    if unknown:
        print(f"  Warning: {unknown} rows have unknown city prefix")
    return df


def extract_bangalore(df):
    """Filter to Bangalore-only rows."""
    blr = df[df["city"] == "bangalore"].copy()
    print(f"Bangalore subset: {len(blr):,} rows")
    return blr


def parse_order_time(df):
    """Parse Time_Orderd into hour_of_day (0-23).

    Formats seen: 'HH:MM' and decimal like 0.458333333 (= fraction of day).
    """
    def _parse(t):
        t = str(t).strip()
        if t == "nan" or t == "null":
            return np.nan
        if ":" in t:
            parts = t.split(":")
            h, m = int(parts[0]), int(parts[1])
            return h + m / 60.0
        try:
            return float(t) * 24.0
        except ValueError:
            return np.nan

    df["hour_of_day"] = df["Time_Orderd"].apply(_parse)
    return df


def add_derived_columns(df):
    """Add columns that map to the existing ML pipeline's expectations."""
    # Cyclical hour encoding (matches models/features.py)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)

    # Weather severity mapping
    df["weather_severity"] = df["Weather_conditions"].str.lower().map(WEATHER_MAP)

    # Traffic severity (0-1 scale)
    df["traffic_severity"] = df["Road_traffic_density"].str.lower().map(TRAFFIC_MAP)

    # Order complexity mapping
    df["order_complexity"] = df["Type_of_order"].str.lower().map(ORDER_TYPE_TO_COMPLEXITY)

    # Delivery time (target variable)
    df["delivery_time_min"] = df["Time_taken (min)"]

    return df


def quality_report(df):
    """Print a quality summary of the extracted data."""
    print("\n" + "=" * 60)
    print("QUALITY REPORT — Bangalore Orders")
    print("=" * 60)

    print(f"\nRows:          {len(df):,}")
    print(f"Columns:       {len(df.columns)}")
    print(f"Date range:    {df['Order_Date'].min()} to {df['Order_Date'].max()}")
    print(f"Unique dates:  {df['Order_Date'].nunique()}")
    print(f"Unique riders: {df['Delivery_person_ID'].nunique()}")

    print("\n--- Missing values ---")
    for c in df.columns:
        n = df[c].isna().sum()
        if n > 0:
            print(f"  {c:35s} {n:5d} ({100*n/len(df):.1f}%)")

    print("\n--- Numeric distributions ---")
    for c in ["delivery_time_min", "distance_km", "hour_of_day",
              "Delivery_person_Age", "Delivery_person_Ratings"]:
        if c in df.columns:
            s = df[c].dropna()
            print(f"  {c:35s} mean={s.mean():.1f}  std={s.std():.1f}  "
                  f"min={s.min():.1f}  p50={s.median():.1f}  max={s.max():.1f}")

    print("\n--- Categorical distributions ---")
    for c in ["weather_severity", "order_complexity", "Type_of_vehicle",
              "Festival", "delivery_speed"]:
        if c in df.columns:
            print(f"  {c}:")
            vc = df[c].value_counts()
            for v, cnt in vc.items():
                print(f"    {v:20s} {cnt:5d} ({100*cnt/len(df):.1f}%)")

    print("\n--- Delivery time by weather ---")
    print(df.groupby("weather_severity")["delivery_time_min"]
          .agg(["mean", "median", "count"]).to_string())

    print("\n--- Delivery time by traffic ---")
    print(df.groupby("Road_traffic_density")["delivery_time_min"]
          .agg(["mean", "median", "count"]).to_string())

    print("\n--- Delivery time by festival ---")
    print(df.groupby("Festival")["delivery_time_min"]
          .agg(["mean", "median", "count"]).to_string())


def main():
    if not os.path.exists(RAW_PATH):
        print(f"ERROR: Raw dataset not found at {RAW_PATH}")
        print("Download from: https://huggingface.co/datasets/allenborochin/zomato_delivery_EDA")
        sys.exit(1)

    # Step 1: Load and normalize
    df = load_and_normalize(RAW_PATH)

    # Step 2: Extract Bangalore
    blr = extract_bangalore(df)

    # Step 3: Parse timestamps
    blr = parse_order_time(blr)

    # Step 4: Add derived columns
    blr = add_derived_columns(blr)

    # Step 5: Quality report
    quality_report(blr)

    # Step 6: Save
    out_cols = [
        "ID", "Delivery_person_ID", "city",
        "Order_Date", "Time_Orderd", "hour_of_day", "hour_sin", "hour_cos",
        "Restaurant_latitude", "Restaurant_longitude",
        "Delivery_location_latitude", "Delivery_location_longitude",
        "distance_km", "delivery_speed",
        "Weather_conditions", "weather_severity",
        "Road_traffic_density", "traffic_severity",
        "Vehicle_condition", "Type_of_vehicle",
        "Type_of_order", "order_complexity",
        "multiple_deliveries", "Festival",
        "Delivery_person_Age", "Delivery_person_Ratings",
        "delivery_time_min",
    ]
    blr_out = blr[[c for c in out_cols if c in blr.columns]].copy()
    blr_out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(blr_out):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
