import argparse
import json
import os
import glob

from models.features import pool_orders, temporal_split, fit_encoder, make_features
from models.baseline import RuleBaseline
from models.train import train_all
from models.evaluate import evaluate_models, format_results_table
from models.uncertainty import fit_quantiles, evaluate_uncertainty
from models.predict import Predictor, save_predictor


def load_training_data(data_dir):
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"no order CSVs found in {data_dir}")
    return pool_orders(csvs), csvs


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate prep-time prediction models.")
    parser.add_argument("--data-dir", default="data/train", help="Directory of pooled orders.csv files")
    parser.add_argument("--out", default="artifacts", help="Artifacts output directory")
    parser.add_argument("--split-frac", type=float, default=0.8, help="Temporal train/test fraction")
    args = parser.parse_args()

    df, sources = load_training_data(args.data_dir)
    print(f"Loaded {len(df)} orders from {len(sources)} files")

    train_df, test_df = temporal_split(df, frac=args.split_frac)
    print(f"Temporal split: train={len(train_df)}, test={len(test_df)}")

    encoder = fit_encoder(train_df)
    X_train, y_train = make_features(train_df, encoder)
    X_test, y_test = make_features(test_df, encoder)

    baseline = RuleBaseline().fit(train_df)
    y_pred_base = baseline.predict(test_df)
    from models.evaluate import mae, mape, rmse
    base_metrics = {
        "model": "rule_baseline",
        "mae": mae(y_test, y_pred_base),
        "mape": mape(y_test, y_pred_base),
        "rmse": rmse(y_test, y_pred_base),
    }

    models = train_all(X_train, y_train)
    rows = evaluate_models(models, X_test, y_test)
    rows.insert(0, base_metrics)

    print("\n=== Model comparison ===")
    print(format_results_table(rows))

    best = min((r for r in rows if r["model"] != "rule_baseline"), key=lambda r: r["mae"], default=None)
    if best is None:
        raise RuntimeError("no ML model trained")

    q_low, q_high, calib_factor = fit_quantiles(X_train, y_train)
    unc = evaluate_uncertainty(q_low, q_high, X_test, y_test, y_train, factor=calib_factor)
    if unc:
        print(f"\nUncertainty (LightGBM quantiles): {unc}")

    train_std = float(y_train.std())
    predictor = Predictor(
        model=models[best["model"]],
        q_low=q_low,
        q_high=q_high,
        encoder=encoder,
        train_std=train_std,
        calibration_factor=calib_factor,
    )

    results = {
        "best_model": best["model"],
        "models": rows,
        "uncertainty": unc,
        "train_std_min": round(train_std, 3),
        "train_size": len(train_df),
        "test_size": len(test_df),
    }
    save_predictor(predictor, args.out, best["model"], results, calibration_factor=calib_factor)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nArtifacts saved to: {os.path.abspath(args.out)}")
    print(f"Selected production model: {best['model']} (MAE={best['mae']:.3f})")

    sample = test_df.iloc[0]
    features = {
        "items_count": int(sample["items_count"]),
        "workload_at_placement": int(sample["workload_at_placement"]),
        "staff_level": int(sample["staff_level"]),
        "hour_of_day": int(sample["hour_of_day"]),
        "order_complexity": sample["order_complexity"],
        "weather_severity": sample["weather_severity"],
        "traffic_severity": sample["traffic_severity"],
        "kitchen_id": int(sample["kitchen_id"]),
    }
    print(f"\nPredictor smoke (actual={sample['actual_prep_duration_min']:.2f} min):")
    print(predictor.predict(features))


if __name__ == "__main__":
    main()
