import numpy as np
import pandas as pd


class RuleBaseline:
    """Interpretable median-prep-by-cell baseline. Fit on training data only."""

    def __init__(self):
        self.lookup = None
        self.global_median = None

    def fit(self, df):
        """df: raw (encoded) train frame with target column."""
        df = df.copy()
        df["_cell"] = list(zip(df["order_complexity"], df["weather_severity"]))
        medians = df.groupby("_cell")["actual_prep_duration_min"].median()
        self.lookup = medians.to_dict()
        self.global_median = float(df["actual_prep_duration_min"].median())
        return self

    def predict(self, df):
        cells = list(zip(df["order_complexity"], df["weather_severity"]))
        return np.array(
            [self.lookup.get(cell, self.global_median) for cell in cells],
            dtype=float,
        )
