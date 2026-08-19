import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.kitchen import sample_prep_time  # noqa: E402
from simulation.scenarios import load_scenario  # noqa: E402
from simulation.entities import Order, Kitchen, OrderComplexity  # noqa: E402


class FakeWeather:
    def prep_factor(self):
        return 1.0


class FakeTraffic:
    def prep_factor(self):
        return 1.0


def _order(complexity, items, workload):
    return Order(
        order_id=1, kitchen_id=1, placed_at=0.0, items=items,
        complexity=complexity, distance_km=2.0,
        workload_at_placement=workload,
    )


def test_prep_time_in_bounds():
    rng = np.random.default_rng(0)
    config = load_scenario("normal", seed=0)
    kitchen = Kitchen(kitchen_id=1, staff_level=3)
    weather = FakeWeather()
    traffic = FakeTraffic()
    for _ in range(200):
        duration = sample_prep_time(
            rng, _order(OrderComplexity.SIMPLE, 1, 0), kitchen, weather, traffic, config
        )
        assert 2.0 <= duration <= 25.0


def test_workload_increases_prep():
    rng = np.random.default_rng(0)
    config = load_scenario("normal", seed=0)
    weather = FakeWeather()
    traffic = FakeTraffic()

    low_kitchen = Kitchen(kitchen_id=1, staff_level=3)
    high_kitchen = Kitchen(kitchen_id=2, staff_level=3)
    low_kitchen.current_orders = [None]
    high_kitchen.current_orders = [None] * 8

    lows = [sample_prep_time(rng, _order(OrderComplexity.SIMPLE, 1, 0), low_kitchen, weather, traffic, config) for _ in range(100)]
    highs = [sample_prep_time(rng, _order(OrderComplexity.SIMPLE, 1, 0), high_kitchen, weather, traffic, config) for _ in range(100)]
    assert sum(highs) / len(highs) > sum(lows) / len(lows)


def test_low_staffing_increases_prep():
    rng = np.random.default_rng(0)
    config = load_scenario("normal", seed=0)
    weather = FakeWeather()
    traffic = FakeTraffic()

    well_staffed = Kitchen(kitchen_id=1, staff_level=4)
    under_staffed = Kitchen(kitchen_id=2, staff_level=2)

    a = [sample_prep_time(rng, _order(OrderComplexity.SIMPLE, 1, 0), well_staffed, weather, traffic, config) for _ in range(100)]
    b = [sample_prep_time(rng, _order(OrderComplexity.SIMPLE, 1, 0), under_staffed, weather, traffic, config) for _ in range(100)]
    assert sum(b) / len(b) > sum(a) / len(a)


def _contention_config():
    from simulation.scenarios import load_scenario

    config = load_scenario("normal", seed=7)
    config["days"] = 1
    config["dispatch"]["enabled"] = True
    config["dispatch"]["default_policy"] = "immediate"
    config["drain_timeout_min"] = 300
    config["cancellation_rates"]["customer_cancel_per_min"] = 0.0
    config["cancellation_rates"]["kitchen_failure_per_min"] = 0.0
    # One prep slot per kitchen -> forced queueing under load.
    config["kitchens"]["staff_level"] = 1
    config["kitchens"]["count"] = 1
    return config


def test_prep_timestamps_separate_queue_from_prep(tmp_path):
    """prep_started_at must be set only after the resource is acquired, so
    kitchen_queue (prep_started_at - entered_kitchen_at) is pure queue time."""
    from simulation import SimulationEngine
    from simulation.entities import OrderStatus

    config = _contention_config()
    engine = SimulationEngine(config, out_dir=str(tmp_path / "k"), scenario_name="normal")
    engine.run()

    completed = [o for o in engine.orders if o.status == OrderStatus.COMPLETED]
    assert len(completed) > 1, "expected a non-trivial set of completed orders"

    for o in completed:
        assert o.entered_kitchen_at is not None
        assert o.prep_started_at is not None
        assert o.prep_finished_at is not None
        # Prep starts no earlier than queue entry.
        assert o.prep_started_at >= o.entered_kitchen_at
        # Actual prep duration equals the measured prep window exactly.
        assert o.prep_finished_at - o.prep_started_at == pytest.approx(o.actual_prep_duration_min, abs=1e-9)

    # With a single prep slot, at least one order must have queued.
    assert any(o.prep_started_at > o.entered_kitchen_at for o in completed)
