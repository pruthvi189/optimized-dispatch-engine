import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.scenarios import load_scenario, DEFAULT_CONFIG  # noqa: E402


def test_load_normal():
    config = load_scenario("normal")
    assert config["seed"] == 42
    assert config["days"] == 1
    assert config["demand_multiplier"] == 1.0


def test_scenarios_exist():
    for name in ("normal", "lunch_rush", "rain", "low_staffing", "traffic_spike"):
        config = load_scenario(name)
        assert isinstance(config, dict)
        assert "seed" in config and "days" in config


def test_unknown_scenario():
    with pytest.raises(FileNotFoundError):
        load_scenario("does_not_exist")


def test_seed_override():
    config = load_scenario("normal", seed=7)
    assert config["seed"] == 7


def test_lunch_rush_multiplier():
    config = load_scenario("lunch_rush")
    assert config["demand_multiplier"] == 2.0


def test_low_staffing_staff_level():
    config = load_scenario("low_staffing")
    assert config["kitchens"]["staff_level"] == 2


def test_scenarios_merge_with_defaults():
    config = load_scenario("lunch_rush")
    assert config["riders"]["count"] == DEFAULT_CONFIG["riders"]["count"]
