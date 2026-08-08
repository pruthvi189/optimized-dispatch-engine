import sys
import os

import simpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.environment import WeatherGenerator, TrafficGenerator  # noqa: E402
from simulation.entities import WeatherSeverity, TrafficSeverity  # noqa: E402
from simulation.rng import spawn_streams  # noqa: E402


def _config(weather_start="clear"):
    return {
        "weather": {
            "start_severity": weather_start,
            "duration_min": [5, 10],
            "transitions": {
                "clear": {"clear": 0.7, "rain": 0.3, "storm": 0.0},
                "rain": {"rain": 0.7, "clear": 0.3, "storm": 0.0},
                "storm": {"storm": 0.7, "rain": 0.3, "clear": 0.0},
            },
        },
        "traffic": {"spike_prob_per_min": 0.5},
    }


def test_weather_starts_clear():
    env = simpy.Environment()
    rng = spawn_streams(0)["weather"]
    w = WeatherGenerator(env, rng, _config("clear"))
    assert w.current_severity() == WeatherSeverity.CLEAR


def test_weather_starts_rain():
    env = simpy.Environment()
    rng = spawn_streams(0)["weather"]
    w = WeatherGenerator(env, rng, _config("rain"))
    assert w.current_severity() == WeatherSeverity.RAIN


def test_weather_changes_over_time():
    env = simpy.Environment()
    rng = spawn_streams(0)["weather"]
    w = WeatherGenerator(env, rng, _config("clear"))
    env.run(until=200)
    assert w.state.changed_at > 0


def test_weather_prep_factor():
    env = simpy.Environment()
    rng = spawn_streams(0)["weather"]
    w = WeatherGenerator(env, rng, _config("clear"))
    assert w.prep_factor() == 1.0
    w.state.severity = WeatherSeverity.RAIN
    assert w.prep_factor() > 1.0


def test_traffic_generator():
    env = simpy.Environment()
    rng = spawn_streams(0)["traffic"]
    t = TrafficGenerator(env, rng, _config())
    env.run(until=300)
    assert t.current_severity() in (TrafficSeverity.LOW, TrafficSeverity.MODERATE, TrafficSeverity.HEAVY)
