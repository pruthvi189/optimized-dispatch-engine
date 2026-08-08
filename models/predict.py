import json
import os

import joblib
import numpy as np

from .features import (
    build_feature_vector, fit_encoder, make_features,
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
)
from .uncertainty import uncertainty_tier, predict_interval

ARTIFACTS = {
    "encoder": "encoder.joblib",
    "model": "model.joblib",
    "q_low": "q_low.joblib",
    "q_high": "q_high.joblib",
    "meta": "meta.json",
}


class Predictor:
    """In-process prep-time predictor. Prepares the Phase 3 contract."""

    def __init__(self, model, q_low, q_high, encoder, train_std, calibration_factor=1.0):
        self.model = model
        self.q_low = q_low
        self.q_high = q_high
        self.encoder = encoder
        self.train_std = train_std
        self.calibration_factor = calibration_factor

    @classmethod
    def load(cls, artifacts_dir):
        def _p(name):
            return joblib.load(os.path.join(artifacts_dir, name))

        with open(os.path.join(artifacts_dir, ARTIFACTS["meta"]), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(
            model=_p(ARTIFACTS["model"]),
            q_low=_p(ARTIFACTS["q_low"]),
            q_high=_p(ARTIFACTS["q_high"]),
            encoder=_p(ARTIFACTS["encoder"]),
            train_std=meta.get("train_std_min", 1.0),
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
            traffic_severity=features["traffic_severity"],
            kitchen_id=features["kitchen_id"],
        )
        prep_mean = float(self.model.predict(X)[0])
        if self.q_low is not None and self.q_high is not None:
            low, high = predict_interval(self.q_low, self.q_high, X, factor=self.calibration_factor)
            low = float(low[0])
            high = float(high[0])
            width = high - low
            tier = uncertainty_tier(width, self.train_std, self.train_std)
        else:
            low = high = prep_mean
            tier = "medium"
        return {
            "prep_mean": round(prep_mean, 2),
            "prep_low": round(low, 2),
            "prep_high": round(high, 2),
            "uncertainty": tier,
        }


def save_predictor(predictor, artifacts_dir, model_name, results, calibration_factor=1.0):
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(predictor.encoder, os.path.join(artifacts_dir, ARTIFACTS["encoder"]))
    joblib.dump(predictor.model, os.path.join(artifacts_dir, ARTIFACTS["model"]))
    joblib.dump(predictor.q_low, os.path.join(artifacts_dir, ARTIFACTS["q_low"]))
    joblib.dump(predictor.q_high, os.path.join(artifacts_dir, ARTIFACTS["q_high"]))
    meta = {
        "model": model_name,
        "train_std_min": results.get("train_std_min", 1.0),
        "calibration_factor": calibration_factor,
        "feature_names": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "created": "phase2",
    }
    with open(os.path.join(artifacts_dir, ARTIFACTS["meta"]), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
