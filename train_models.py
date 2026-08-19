import argparse
import json
import os
import glob

from models.features import pool_orders, temporal_four_way_split, fit_encoder, make_features
from models.baseline import RuleBaseline
from models.train import train_all
from models.evaluate import evaluate_models, format_results_table, mae, mape, rmse
from models.uncertainty import fit_calibration, evaluate_uncertainty
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
    parser.add_argument("--train-frac", type=float, default=0.6, help="Temporal train fraction")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Temporal validation fraction")
    parser.add_argument("--calib-frac", type=float, default=0.15, help="Temporal calibration fraction")
    parser.add_argument("--nominal", type=float, default=0.80, help="Nominal prediction-interval coverage")
    args = parser.parse_args()

    df, sources = load_training_data(args.data_dir)
    print(f"Loaded {len(df)} orders from {len(sources)} files")

    train_df, val_df, calib_df, test_df = temporal_four_way_split(
        df, args.train_frac, args.val_frac, args.calib_frac
    )
    print(f"Temporal split (train/val/calib/test): {len(train_df)}/{len(val_df)}/{len(calib_df)}/{len(test_df)}")

    encoder = fit_encoder(train_df)
    X_train, y_train = make_features(train_df, encoder)
    X_val, y_val = make_features(val_df, encoder)
    X_calib, y_calib = make_features(calib_df, encoder)
    X_test, y_test = make_features(test_df, encoder)

    baseline = RuleBaseline().fit(train_df)
    y_pred_base = baseline.predict(test_df)
    base_metrics = {
        "model": "rule_baseline",
        "mae": mae(y_test, y_pred_base),
        "mape": mape(y_test, y_pred_base),
        "rmse": rmse(y_test, y_pred_base),
    }

    models = train_all(X_train, y_train)
    val_rows = evaluate_models(models, X_val, y_val)

    print("\n=== Model comparison (on validation split, for selection) ===")
    print(format_results_table(val_rows))

    best = min(
        (r for r in val_rows if r["model"] != "rule_baseline"),
        key=lambda r: r["mae"],
        default=None,
    )
    if best is None:
        raise RuntimeError("no ML model trained")

    best_model = models[best["model"]]
    train_std = float(y_train.std())
    qhat = fit_calibration(best_model, X_calib, y_calib, nominal=args.nominal)
    print(f"\nCalibration (split conformal, held-out): qhat={qhat:.3f} min on {len(calib_df)} calibration rows")

    test_pred = best_model.predict(X_test)
    test_metrics = {
        "model": best["model"],
        "mae": mae(y_test, test_pred),
        "mape": mape(y_test, test_pred),
        "rmse": rmse(y_test, test_pred),
    }

    unc = evaluate_uncertainty(
        best_model, X_test, y_test, train_std, qhat, nominal=args.nominal
    )
    print(f"\nFinal test evaluation (untouched test split): {unc}")
    print(f"Baseline on test: MAE={base_metrics['mae']:.3f} RMSE={base_metrics['rmse']:.3f} MAPE={base_metrics['mape']:.2f}%")

    predictor = Predictor(
        model=best_model,
        encoder=encoder,
        train_std=train_std,
        calibration_quantile=qhat,
    )

    results = {
        "best_model": best["model"],
        "model_comparison_validation": val_rows,
        "baseline_test": base_metrics,
        "test_metrics": test_metrics,
        "uncertainty": unc,
        "calibration": {
            "method": "split_conformal",
            "nominal_coverage": args.nominal,
            "calibration_quantile": round(qhat, 3),
        },
        "train_std_min": round(train_std, 3),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "calib_size": len(calib_df),
        "test_size": len(test_df),
    }
    save_predictor(predictor, args.out, best["model"], results, calibration_quantile=qhat)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nArtifacts saved to: {os.path.abspath(args.out)}")
    print(f"Selected production model: {best['model']} (validation MAE={best['mae']:.3f}, test MAE={test_metrics['mae']:.3f})")

    sample = test_df.iloc[0]
    features = {
        "items_count": int(sample["items_count"]),
        "workload_at_placement": int(sample["workload_at_placement"]),
        "staff_level": int(sample["staff_level"]),
        "hour_of_day": int(sample["hour_of_day"]),
        "order_complexity": sample["order_complexity"],
        "weather_severity": sample["weather_severity"],
        "kitchen_id": int(sample["kitchen_id"]),
    }
    print(f"\nPredictor smoke (actual={sample['actual_prep_duration_min']:.2f} min):")
    print(predictor.predict(features))


if __name__ == "__main__":
    main()
