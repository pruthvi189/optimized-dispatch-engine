import hashlib
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import SimulationEngine, load_scenario  # noqa: E402
from simulation.rng import make_rng, spawn_streams  # noqa: E402


@pytest.fixture(scope="module")
def normal_summary(tmp_path_factory):
    out = tmp_path_factory.mktemp("normal")
    config = load_scenario("normal", seed=42)
    engine = SimulationEngine(config, out_dir=str(out), scenario_name="normal")
    return engine.run(), str(out)


def test_same_seed_is_deterministic(tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    for out in (out1, out2):
        config = load_scenario("normal", seed=42)
        engine = SimulationEngine(config, out_dir=str(out), scenario_name="normal")
        engine.run()
    hash1 = hashlib.sha256((out1 / "event_log.csv").read_bytes()).hexdigest()
    hash2 = hashlib.sha256((out2 / "event_log.csv").read_bytes()).hexdigest()
    assert hash1 == hash2


def test_different_seed_differs(tmp_path):
    outs = {}
    for seed in (1, 2):
        out = tmp_path / f"seed{seed}"
        config = load_scenario("normal", seed=seed)
        engine = SimulationEngine(config, out_dir=str(out), scenario_name="normal")
        engine.run()
        outs[seed] = hashlib.sha256((out / "event_log.csv").read_bytes()).hexdigest()
    assert outs[1] != outs[2]


def test_spawn_streams_independent():
    s1 = spawn_streams(42)
    s2 = spawn_streams(42)
    assert s1["arrivals"].random() == s2["arrivals"].random()
    assert s1["weather"].random() == s2["weather"].random()
    assert s1["arrivals"].random() != s1["weather"].random()


def test_orders_csv_schema(normal_summary):
    _, out = normal_summary
    df = pd.read_csv(os.path.join(out, "orders.csv"))
    expected = {
        "order_id", "kitchen_id", "placed_at", "hour_of_day", "day_of_week",
        "order_complexity", "items_count", "workload_at_placement", "staff_level",
        "weather_severity", "traffic_severity", "actual_prep_duration_min",
        "status", "cancel_reason",
    }
    assert expected <= set(df.columns)
    assert df["actual_prep_duration_min"].notna().sum() > 0
