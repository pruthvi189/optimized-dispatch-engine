import time

import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def latency_ms(model, X, n_repeats=50):
    """Approximate per-sample inference latency on a single-row slice."""
    sample = X[:1]
    model.predict(sample)
    start = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(sample)
    elapsed = time.perf_counter() - start
    return (elapsed / n_repeats) * 1000.0


def evaluate_model(model, X_test, y_test):
    if model is None:
        return None
    y_pred = model.predict(X_test)
    return {
        "mae": mae(y_test, y_pred),
        "mape": mape(y_test, y_pred),
        "rmse": rmse(y_test, y_pred),
        "latency_ms": latency_ms(model, X_test),
    }


def evaluate_models(models, X_test, y_test, feature_names=None):
    """Return a list of {model, metric} dicts for all fitted models."""
    rows = []
    for name, model in models.items():
        if model is None:
            continue
        metrics = evaluate_model(model, X_test, y_test)
        if metrics is not None:
            rows.append({"model": name, **metrics})
    return rows


def format_results_table(rows):
    df = pd.DataFrame(rows)
    df["mape"] = df["mape"].round(2)
    df["mae"] = df["mae"].round(3)
    df["rmse"] = df["rmse"].round(3)
    df["latency_ms"] = df["latency_ms"].round(3)
    return df.to_string(index=False)
