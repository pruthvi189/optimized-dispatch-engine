import json
import os

import joblib
import numpy as np

from .features import (
    build_feature_vector, fit_encoder, make_features,
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
)
from .uncertainty import uncertainty_tier, conformal_interval

ARTIFACTS = {
    "encoder": "encoder.joblib",
    "model": "model.joblib",
    "q_low": "q_low.joblib",
    "q_high": "q_high.joblib",
    "meta": "meta.json",
}


class Predictor:
    """In-process prep-time predictor. Prepares the Phase 3 contract.

    Preferred mode: split-conformal calibration (``calibration_quantile`` in the
    artifact meta). Legacy artifacts (``q_low.joblib``/``q_high.joblib`` +
    ``calibration_factor``, no ``calibration_quantile``) are still loaded so the
    previously deployed predictor keeps working.
    """

    def __init__(self, model, encoder, train_std, calibration_quantile=None,
                 q_low=None, q_high=None, calibration_factor=1.0):
        self.model = model
        self.encoder = encoder
        self.train_std = train_std
        self.calibration_quantile = calibration_quantile
        self.q_low = q_low
        self.q_high = q_high
        self.calibration_factor = calibration_factor

    @classmethod
    def load(cls, artifacts_dir):
        def _p(name):
            return joblib.load(os.path.join(artifacts_dir, name))

        with open(os.path.join(artifacts_dir, ARTIFACTS["meta"]), "r", encoding="utf-8") as f:
            meta = json.load(f)
        qhat = meta.get("calibration_quantile")
        q_low = q_high = None
        if qhat is None:
            q_low_path = os.path.join(artifacts_dir, ARTIFACTS["q_low"])
            q_high_path = os.path.join(artifacts_dir, ARTIFACTS["q_high"])
            if os.path.exists(q_low_path) and os.path.exists(q_high_path):
                q_low = joblib.load(q_low_path)
                q_high = joblib.load(q_high_path)
        return cls(
            model=_p(ARTIFACTS["model"]),
            encoder=_p(ARTIFACTS["encoder"]),
            train_std=meta.get("train_std_min", 1.0),
            calibration_quantile=qhat,
            q_low=q_low,
            q_high=q_high,
            calibration_factor=meta.get("calibration_factor", 1.0),
        )

    def predict(self, features: dict) -> dict:
        X = build_feature_vector(
            self.encoder,
            items_count=features["items_count"],
            workload=features["workload_at_placement"],
            staff_level=features["staff_level"],
            hour_of_day=features["hour_of_day"],
            order_complexity=features["order_complexity"],
            weather_severity=features["weather_severity"],
            kitchen_id=features["kitchen_id"],
        )
        prep_mean = float(self.model.predict(X)[0])
        if self.calibration_quantile is not None:
            low, high = conformal_interval(prep_mean, self.calibration_quantile)
            tier = uncertainty_tier(high - low, self.train_std, self.train_std)
        elif self.q_low is not None and self.q_high is not None:
            low = float(self.q_low.predict(X)[0])
            high = float(self.q_high.predict(X)[0])
            if self.calibration_factor != 1.0:
                mid = (low + high) / 2.0
                half = max((high - low) / 2.0, 1e-6) * self.calibration_factor
                low, high = mid - half, mid + half
            low = max(0.0, low)
            tier = uncertainty_tier(high - low, self.train_std, self.train_std)
        else:
            low = high = prep_mean
            tier = "medium"
        return {
            "prep_mean": round(prep_mean, 2),
            "prep_low": round(low, 2),
            "prep_high": round(high, 2),
            "uncertainty": tier,
        }


def save_predictor(predictor, artifacts_dir, model_name, results, calibration_quantile=0.0):
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(predictor.encoder, os.path.join(artifacts_dir, ARTIFACTS["encoder"]))
    joblib.dump(predictor.model, os.path.join(artifacts_dir, ARTIFACTS["model"]))
    meta = {
        "model": model_name,
        "train_std_min": results.get("train_std_min", 1.0),
        "calibration_method": "split_conformal",
        "calibration_quantile": float(calibration_quantile),
        "nominal_coverage": results.get("nominal_coverage", 0.80),
        "feature_names": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "created": "phase2",
    }
    with open(os.path.join(artifacts_dir, ARTIFACTS["meta"]), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
