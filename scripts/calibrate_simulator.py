"""Calibrate simulator distributions from real Bangalore food-delivery data.

Reads bangalore_orders.csv and computes empirical distributions for:
    - Hour-of-day demand curve
    - Delivery time by weather/traffic/complexity
    - Distance distribution
    - Weather frequency distribution
    - Traffic frequency distribution
    - Fleet mix (vehicle types)

Output: config/bangalore_calibration.json — a structured file documenting
REAL-DATA-DERIVED parameters vs SYNTHETIC ASSUMPTIONS that remain.

This script does NOT modify the simulator code or configs.  It produces
a reference document for human review before any config changes are applied.

Usage:
    python scripts/calibrate_simulator.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "external", "bangalore_orders.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "config", "bangalore_calibration.json")


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Parse hour
    def _parse_hour(t):
        t = str(t).strip()
        if ":" in t:
            h, m = t.split(":")
            return int(h) + int(m) / 60.0
        try:
            return float(t) * 24.0
        except ValueError:
            return np.nan
    df["hour_parsed"] = df["Time_Orderd"].apply(_parse_hour)
    df["hour_int"] = df["hour_parsed"].apply(lambda x: int(x) if pd.notna(x) else None)
    return df


def calibrate_demand_curve(df):
    """Hour-of-day demand curve (orders per hour, normalised to 24h)."""
    counts = df["hour_int"].value_counts().sort_index()
    # Normalise to a 1-day scale (data spans ~36 days)
    n_days = df["Order_Date"].nunique()
    curve = [0] * 24
    for h, c in counts.items():
        if pd.notna(h) and 0 <= int(h) <= 23:
            curve[int(h)] = round(c / n_days, 1)
    return curve


def calibrate_delivery_times(df):
    """Delivery time distributions by conditions."""
    result = {}
    for col, name in [("Weather_conditions", "by_weather"),
                       ("Road_traffic_density", "by_traffic"),
                       ("Type_of_order", "by_order_type"),
                       ("Type_of_vehicle", "by_vehicle")]:
        groups = {}
        for val, grp in df.groupby(col):
            times = grp["delivery_time_min"]
            groups[str(val)] = {
                "count": int(len(grp)),
                "mean": round(float(times.mean()), 2),
                "std": round(float(times.std()), 2),
                "median": round(float(times.median()), 1),
                "p10": round(float(times.quantile(0.10)), 1),
                "p25": round(float(times.quantile(0.25)), 1),
                "p75": round(float(times.quantile(0.75)), 1),
                "p90": round(float(times.quantile(0.90)), 1),
                "min": round(float(times.min()), 1),
                "max": round(float(times.max()), 1),
            }
        result[name] = groups
    return result


def calibrate_distance(df):
    """Distance distribution."""
    d = df["distance_km"]
    return {
        "count": int(len(d)),
        "mean": round(float(d.mean()), 2),
        "std": round(float(d.std()), 2),
        "median": round(float(d.median()), 2),
        "p10": round(float(d.quantile(0.10)), 2),
        "p25": round(float(d.quantile(0.25)), 2),
        "p75": round(float(d.quantile(0.75)), 2),
        "p90": round(float(d.quantile(0.90)), 2),
        "min": round(float(d.min()), 2),
        "max": round(float(d.max()), 2),
    }


def calibrate_weather_freq(df):
    """Weather condition frequency."""
    vc = df["Weather_conditions"].value_counts()
    total = len(df)
    return {str(k): round(v / total, 4) for k, v in vc.items()}


def calibrate_traffic_freq(df):
    """Traffic density frequency."""
    vc = df["Road_traffic_density"].value_counts()
    total = len(df)
    return {str(k): round(v / total, 4) for k, v in vc.items()}


def calibrate_fleet_mix(df):
    """Vehicle type distribution."""
    vc = df["Type_of_vehicle"].value_counts()
    total = len(df)
    return {str(k): round(v / total, 4) for k, v in vc.items()}


def compare_with_simulator(bangalore_cal, default_config, df):
    """Document differences between real data and simulator defaults."""
    comparisons = {}

    # Demand curve
    sim_curve = list(default_config.get("demand_curve", [0]*24))
    real_curve = bangalore_cal["demand_curve"]
    comparisons["demand_curve"] = {
        "real": real_curve,
        "simulator": sim_curve,
        "note": "REAL-DATA-DERIVED: Hourly demand from 36 days of Bangalore orders",
    }

    # Distance
    sim_range = default_config.get("orders", {}).get("distance_range_km", [1.0, 3.5])
    real_dist = bangalore_cal["distance"]
    comparisons["distance_km"] = {
        "real_range": [real_dist["p10"], real_dist["p90"]],
        "simulator_range": sim_range,
        "note": "REAL-DATA-DERIVED: Distance from restaurant to customer (1.6-20.2 km in real data)",
    }

    # Delivery time by weather
    comparisons["weather_effect"] = {
        "real": {},
        "simulator_prep_factor": {"clear": 1.0, "rain": 1.1, "storm": 1.2},
        "note": "REAL-DATA-DERIVED: Delivery time deltas from real Bangalore weather conditions",
    }
    weather_means = {}
    for w, stats in bangalore_cal["delivery_times"]["by_weather"].items():
        weather_means[w] = stats["mean"]
    baseline = weather_means.get("Sunny", 25.0)
    for w, m in weather_means.items():
        comparisons["weather_effect"]["real"][w] = {
            "mean_min": m,
            "factor_vs_sunny": round(m / baseline, 3) if baseline > 0 else None,
        }

    # Traffic effect
    comparisons["traffic_effect"] = {
        "real": {},
        "simulator_traffic_factor": {"low": 1.0, "moderate": 1.2, "heavy": 1.55},
        "note": "REAL-DATA-DERIVED: Delivery time deltas from real Bangalore traffic density",
    }
    traffic_means = {}
    for t, stats in bangalore_cal["delivery_times"]["by_traffic"].items():
        traffic_means[t] = stats["mean"]
    baseline_t = traffic_means.get("Low", 22.0)
    for t, m in traffic_means.items():
        comparisons["traffic_effect"]["real"][t] = {
            "mean_min": m,
            "factor_vs_low": round(m / baseline_t, 3) if baseline_t > 0 else None,
        }

    # Fleet mix
    comparisons["fleet_mix"] = {
        "real": bangalore_cal["fleet_mix"],
        "simulator": "Not modelled (all riders assumed identical)",
        "note": "REAL-DATA-DERIVED: 59% motorcycle, 34% scooter, 7% electric",
    }

    # Festival effect
    festival_data = {}
    for fest, grp in df.groupby("Festival"):
        times = grp["delivery_time_min"]
        festival_data[str(fest)] = {
            "count": int(len(grp)),
            "mean": round(float(times.mean()), 2),
        }
    comparisons["festival_effect"] = {
        "real": festival_data,
        "simulator": "Not modelled",
        "note": "REAL-DATA-DERIVED: Festival causes +69% delivery time increase (26 -> 44 min)",
    }

    return comparisons


def main():
    print("=" * 60)
    print("Simulator Calibration from Real Bangalore Data")
    print("=" * 60)

    df = load_data()
    print(f"Loaded {len(df):,} rows")

    # Compute all calibrations
    cal = {
        "demand_curve": calibrate_demand_curve(df),
        "distance": calibrate_distance(df),
        "delivery_times": calibrate_delivery_times(df),
        "weather_frequency": calibrate_weather_freq(df),
        "traffic_frequency": calibrate_traffic_freq(df),
        "fleet_mix": calibrate_fleet_mix(df),
    }

    # Load simulator defaults for comparison
    sys.path.insert(0, PROJECT_ROOT)
    from simulation.scenarios import DEFAULT_CONFIG
    comparisons = compare_with_simulator(cal, DEFAULT_CONFIG, df)

    # Build output document
    output = {
        "_metadata": {
            "description": "Bangalore calibration data derived from real Zomato/Courier dataset",
            "source": "data/external/bangalore_orders.csv",
            "rows": len(df),
            "unique_dates": int(df["Order_Date"].nunique()),
            "unique_riders": int(df["Delivery_person_ID"].nunique()),
            "date_range": f"{df['Order_Date'].min()} to {df['Order_Date'].max()}",
        },
        "calibrated_distributions": cal,
        "simulator_comparison": comparisons,
        "remaining_synthetic_assumptions": [
            "Prep time model (BASE_PREP_RANGES in kitchen.py) - no real prep time data available",
            "Kitchen workload/staffing dynamics - no real data available",
            "Kitchen count and staff levels - synthetic parameter",
            "Rider count and speed - synthetic parameter (22 km/h assumed)",
            "Cancellation rates - synthetic parameter",
            "Weather state transitions (Markov chain probabilities) - synthetic parameter",
            "Traffic spike probability - synthetic parameter",
            "Rider hub-to-kitchen distance - synthetic parameter",
            "Pickup time at kitchen - synthetic parameter (0.5 min assumed)",
            "Order items count distribution - synthetic parameter",
        ],
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved calibration to {OUT_PATH}")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY CALIBRATION FINDINGS")
    print("=" * 60)

    print("\n--- Demand curve (orders/hour) ---")
    for h, rate in enumerate(cal["demand_curve"]):
        bar = "#" * int(rate)
        print(f"  {h:2d}:00  {rate:5.1f}  {bar}")

    print("\n--- Delivery time by weather ---")
    for w, stats in cal["delivery_times"]["by_weather"].items():
        print(f"  {w:12s}  mean={stats['mean']:5.1f}  n={stats['count']}")

    print("\n--- Delivery time by traffic ---")
    for t, stats in cal["delivery_times"]["by_traffic"].items():
        print(f"  {t:12s}  mean={stats['mean']:5.1f}  n={stats['count']}")

    print("\n--- Distance ---")
    d = cal["distance"]
    print(f"  mean={d['mean']:.1f} km, p10={d['p10']:.1f}, p90={d['p90']:.1f}, range=[{d['min']:.1f}, {d['max']:.1f}]")

    print("\n--- Fleet mix ---")
    for v, pct in cal["fleet_mix"].items():
        print(f"  {v:20s}  {pct*100:.1f}%")

    print("\n--- Festival effect ---")
    for f, stats in comparisons["festival_effect"]["real"].items():
        print(f"  Festival={f}:  mean={stats['mean']:.1f} min  (n={stats['count']})")

    print("\n--- Remaining synthetic assumptions ---")
    for a in output["remaining_synthetic_assumptions"]:
        print(f"  - {a}")


if __name__ == "__main__":
    main()
