"""Train prep-time prediction models on real Bangalore food-delivery data.

Uses the existing models/train.py, models/evaluate.py, and models/uncertainty.py
infrastructure.  Target = delivery_time_min (total end-to-end from the real
dataset).  Feature set differs from the synthetic model (no items_count,
workload, staff_level, kitchen_id — those are synthetic-only).

Artifacts are saved to artifacts_real/ (separate from the production
artifacts/ directory, which stays untouched).

Usage:
    python scripts/train_real_data.py
    python scripts/train_real_data.py --nominal 0.80 --seed 42
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

# ── Project paths ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.real_features import (
    load_bangalore, fit_encoder, make_features, make_X,
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET,
)
from models.train import train_all, train_quantile
from models.evaluate import evaluate_models, format_results_table, mae, mape, rmse
from models.uncertainty import fit_calibration, evaluate_uncertainty


def temporal_split(df, train_frac=0.7, calib_frac=0.15):
    """Chronological train/calibration/test split using Order_Date + hour_of_day.

    The real data has Order_Date (DD-MM-YYYY) and hour_of_day.  We sort by
    (Order_Date, hour_of_day) and split chronologically to prevent leakage.
    """
    df = df.copy()
    # Parse date for sorting
    df["_date_sort"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")
    df["_sort_key"] = df["_date_sort"] + pd.to_timedelta(df["hour_of_day"], unit="h")
    df = df.sort_values("_sort_key").reset_index(drop=True)

    n = len(df)
    train_cut = int(n * train_frac)
    calib_cut = train_cut + int(n * calib_frac)

    train = df.iloc[:train_cut].copy()
    calib = df.iloc[train_cut:calib_cut].copy()
    test = df.iloc[calib_cut:].copy()

    # Verify temporal ordering
    assert train["_sort_key"].max() <= calib["_sort_key"].min(), "temporal leakage train->calib"
    assert calib["_sort_key"].max() <= test["_sort_key"].min(), "temporal leakage calib->test"

    train = train.drop(columns=["_date_sort", "_sort_key"])
    calib = calib.drop(columns=["_date_sort", "_sort_key"])
    test = test.drop(columns=["_date_sort", "_sort_key"])
    return train, calib, test


def save_real_artifacts(model, encoder, qhat, nominal, train_std, out_dir):
    """Save model artifacts in the same structure as the synthetic model."""
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(encoder, os.path.join(out_dir, "encoder.joblib"))
    joblib.dump(model, os.path.join(out_dir, "model.joblib"))

    meta = {
        "model": "lightgbm",
        "target": TARGET,
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "calibration_method": "split_conformal",
        "calibration_quantile": float(qhat),
        "nominal_coverage": nominal,
        "train_std_min": float(train_std),
        "data_source": "real_bangalore_zomato",
        "created": "phase2_real_data",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved artifacts to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train on real Bangalore delivery data")
    parser.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "external", "bangalore_orders.csv"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "artifacts_real"))
    parser.add_argument("--nominal", type=float, default=0.80, help="Nominal prediction-interval coverage")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── 1. Load data ───────────────────────────────────────────────────
    print("=" * 60)
    print("Phase 2: Train on Real Bangalore Data")
    print("=" * 60)
    df = load_bangalore(args.data)
    print(f"\nLoaded {len(df):,} rows from {args.data}")
    print(f"Target: {TARGET}")
    print(f"Features: {ALL_FEATURES}")

    # ── 2. Chronological split ─────────────────────────────────────────
    train_df, calib_df, test_df = temporal_split(df, train_frac=0.70, calib_frac=0.15)
    print(f"\nTemporal split: train={len(train_df):,} / calib={len(calib_df):,} / test={len(test_df):,}")
    print(f"  Train date range: {train_df['Order_Date'].min()} -> {train_df['Order_Date'].max()}")
    print(f"  Calib date range: {calib_df['Order_Date'].min()} -> {calib_df['Order_Date'].max()}")
    print(f"  Test date range:  {test_df['Order_Date'].min()} -> {test_df['Order_Date'].max()}")

    # ── 3. Fit encoder on training data ────────────────────────────────
    encoder = fit_encoder(train_df)
    X_train, y_train = make_features(train_df, encoder)
    X_calib, y_calib = make_features(calib_df, encoder)
    X_test, y_test = make_features(test_df, encoder)
    print(f"\nFeature matrix shape: X_train={X_train.shape}, X_calib={X_calib.shape}, X_test={X_test.shape}")

    # ── 4. Train all models ────────────────────────────────────────────
    print("\n--- Training models ---")
    t0 = time.time()
    models = train_all(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s")

    # ── 5. Evaluate on validation (calibration set acts as validation) ─
    print("\n--- Model comparison (on calibration set) ---")
    calib_rows = evaluate_models(models, X_calib, y_calib)
    print(format_results_table(calib_rows))

    # ── 6. Select best model ───────────────────────────────────────────
    best_name = min(calib_rows, key=lambda r: r["mae"])["model"]
    best_model = models[best_name]
    print(f"\nBest model: {best_name}")

    # ── 7. Conformal calibration on calib set ──────────────────────────
    print(f"\n--- Conformal calibration (nominal={args.nominal}) ---")
    qhat = fit_calibration(best_model, X_calib, y_calib, nominal=args.nominal)
    train_std = float(np.std(y_train))
    print(f"  qhat = {qhat:.3f}")
    print(f"  train_std = {train_std:.3f}")

    # ── 8. Evaluate on held-out test set ───────────────────────────────
    print("\n--- Test set evaluation ---")
    test_metrics = evaluate_uncertainty(best_model, X_test, y_test, train_std, qhat, nominal=args.nominal)
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # Point metrics on test
    y_pred = best_model.predict(X_test)
    test_mae = mae(y_test, y_pred)
    test_mape = mape(y_test, y_pred)
    test_rmse = rmse(y_test, y_pred)
    print(f"\n  Point metrics:")
    print(f"    MAE:  {test_mae:.3f} min")
    print(f"    MAPE: {test_mape:.2f}%")
    print(f"    RMSE: {test_rmse:.3f} min")

    # ── 9. Distribution of target ──────────────────────────────────────
    print("\n--- Target distribution (delivery_time_min) ---")
    for split_name, y in [("train", y_train), ("calib", y_calib), ("test", y_test)]:
        print(f"  {split_name}: mean={np.mean(y):.2f}, std={np.std(y):.2f}, "
              f"min={np.min(y):.1f}, p50={np.median(y):.1f}, max={np.max(y):.1f}")

    # ── 10. Save artifacts ─────────────────────────────────────────────
    print("\n--- Saving artifacts ---")
    save_real_artifacts(best_model, encoder, qhat, args.nominal, train_std, args.out)

    # ── 11. Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Data source:     Real Bangalore food-delivery (Zomato/Courier)")
    print(f"  Training rows:   {len(train_df):,}")
    print(f"  Features:        {ALL_FEATURES}")
    print(f"  Target:          {TARGET} (total delivery time, NOT prep time)")
    print(f"  Best model:      {best_name}")
    print(f"  Test MAE:        {test_mae:.3f} min")
    print(f"  Test MAPE:       {test_mape:.2f}%")
    print(f"  Test RMSE:       {test_rmse:.3f} min")
    print(f"  Coverage:        {test_metrics['coverage']:.3f} (nominal={args.nominal})")
    print(f"  Interval width:  {test_metrics['mean_interval_width_min']:.3f} min")
    print(f"  Artifacts:       {args.out}")
    print(f"  NOTE: This model predicts DELIVERY TIME (end-to-end), not prep time.")
    print(f"        The production model (artifacts/) predicts prep time from synthetic data.")
    print("=" * 60)


if __name__ == "__main__":
    main()
