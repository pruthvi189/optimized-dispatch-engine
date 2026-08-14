import copy
import os

import yaml

DEFAULT_CONFIG = {
    "seed": 42,
    "days": 1,
    "drain_timeout_min": 240,  # extra sim minutes allowed to finish in-flight orders
    "kitchens": {"count": 3, "staff_level": 3},
    "riders": {"count": 10, "speed_kmh": 22.0},
    "demand_multiplier": 1.0,
    "orders": {
        "items_weights": [0.5, 0.3, 0.2],
        "distance_range_km": [1.0, 3.5],
    },
    "prep": {
        "workload_factor_per_order": 0.08,
        "staff_threshold": 3,
        "staffing_factor": 1.25,
        "clip": [2.0, 25.0],
    },
    "travel": {
        "traffic_factor": 1.0,
        "weather_factor": 1.0,
    },
    "weather": {
        "start_severity": "clear",
        "duration_min": [30, 90],
        "transitions": {
            "clear": {"clear": 0.85, "rain": 0.12, "storm": 0.03},
            "rain": {"rain": 0.7, "clear": 0.25, "storm": 0.05},
            "storm": {"storm": 0.6, "rain": 0.3, "clear": 0.1},
        },
    },
    "traffic": {
        "spike_prob_per_min": 0.0005,
    },
    "cancellation_rates": {
        "customer_cancel_per_min": 0.0005,
        "rider_cancel_per_order": 0.02,
        "kitchen_failure_per_min": 0.0002,
    },
}


def _load_dispatch_defaults() -> dict:
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "dispatch.yaml"
    )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


DEFAULT_CONFIG["dispatch"] = _load_dispatch_defaults()

SCENARIO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "scenarios")


def deep_merge(base, override):
    """Recursively merge override dict into base dict (mutates base)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_scenario(name: str, seed: int | None = None) -> dict:
    path = os.path.join(SCENARIO_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scenario '{name}' not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        override = yaml.safe_load(f)
    config = deep_merge(copy.deepcopy(DEFAULT_CONFIG), override or {})
    if seed is not None:
        config["seed"] = seed
    return config
