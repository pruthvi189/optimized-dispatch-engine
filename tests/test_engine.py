import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import SimulationEngine, load_scenario  # noqa: E402


def _run(scenario, seed=42, days=1, out="data/test_run"):
    config = load_scenario(scenario, seed=seed)
    config["days"] = days
    engine = SimulationEngine(config, out_dir=out, scenario_name=scenario)
    return engine, engine.run()


def test_engine_runs(tmp_path):
    engine, summary = _run("normal", seed=42, out=str(tmp_path))
    assert summary["orders_placed"] > 0
    assert summary["orders_prepared"] > 0
    assert os.path.exists(os.path.join(str(tmp_path), "event_log.csv"))
    assert os.path.exists(os.path.join(str(tmp_path), "orders.csv"))


def test_lunch_rush_more_orders(tmp_path):
    _, normal = _run("normal", seed=42, out=str(tmp_path / "n"))
    _, rush = _run("lunch_rush", seed=42, out=str(tmp_path / "r"))
    assert rush["orders_placed"] > normal["orders_placed"]


def test_same_seed_same_arrivals(tmp_path):
    """Weather changes must not perturb the arrival stream (per-component RNGs)."""
    _, normal = _run("normal", seed=42, out=str(tmp_path / "n"))
    _, rain = _run("rain", seed=42, out=str(tmp_path / "r"))
    assert normal["orders_placed"] == rain["orders_placed"]


def test_rain_increases_prep(tmp_path):
    _, normal = _run("normal", seed=42, out=str(tmp_path / "n"))
    _, rain = _run("rain", seed=42, out=str(tmp_path / "r"))
    assert rain["avg_prep_min"] > normal["avg_prep_min"]


def test_low_staffing_increases_prep(tmp_path):
    _, normal = _run("normal", seed=42, out=str(tmp_path / "n"))
    _, low = _run("low_staffing", seed=42, out=str(tmp_path / "l"))
    assert low["avg_prep_min"] > normal["avg_prep_min"]
