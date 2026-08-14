import sys
import os

import numpy as np

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
