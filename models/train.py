import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

_HP = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": 42,
    "n_jobs": -1,
}


def train_lightgbm(X_train, y_train):
    if not HAS_LIGHTGBM:
        return None
    model = LGBMRegressor(**_HP, verbose=-1)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    if not HAS_XGBOOST:
        return None
    model = XGBRegressor(**_HP, verbosity=0)
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train):
    """Experimental: sklearn MLP on scaled features. Never blocks selection."""
    try:
        from sklearn.exceptions import ConvergenceWarning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    n_iter_no_change=20,
                )),
            ])
            model.fit(X_train, y_train)
        return model
    except Exception:
        return None


def train_quantile(X_train, y_train, alpha):
    """Train a LightGBM quantile regressor for prediction intervals."""
    if not HAS_LIGHTGBM:
        return None
    model = LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=_HP["max_depth"],
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_all(X_train, y_train):
    return {
        "lightgbm": train_lightgbm(X_train, y_train),
        "xgboost": train_xgboost(X_train, y_train),
        "mlp": train_mlp(X_train, y_train),
    }
