"""Shared helpers: build a scenario config for the API."""

import os

from simulation import load_scenario

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PREDICTOR_DIR = os.path.join(REPO_ROOT, "artifacts")


def build_config(scenario="normal", seed=42, policy="adaptive", days=1):
    """Same config shape as run_dispatch.py, resolved against repo-root paths."""
    config = load_scenario(scenario, seed=seed)
    config["days"] = days
    config["dispatch"]["enabled"] = True
    config["dispatch"]["default_policy"] = policy
    config["dispatch"]["predictor_dir"] = DEFAULT_PREDICTOR_DIR
    return config
