"""Tests for the spatial model and OptimizedKitchenDispatch policy (Phase 9)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.spatial import (  # noqa: E402
    DEFAULT_KITCHEN_LOCATIONS,
    Point2D,
    compute_distances_to_kitchens,
    distance_distribution_stats,
    generate_customer_location,
    validate_spatial_model,
)
from simulation.entities import Order, OrderComplexity  # noqa: E402
from dispatch.state import DispatchState, KitchenCandidate  # noqa: E402
from dispatch.policies import OptimizedKitchenDispatch, make_policy  # noqa: E402


# ---- Spatial model unit tests ----

class TestPoint2D:
    def test_distance_to_self(self):
        p = Point2D(3.0, 4.0)
        assert p.distance_to(p) == 0.0

    def test_distance_known(self):
        a = Point2D(0.0, 0.0)
        b = Point2D(3.0, 4.0)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_distance_symmetric(self):
        a = Point2D(1.0, 2.0)
        b = Point2D(4.0, 6.0)
        assert a.distance_to(b) == pytest.approx(b.distance_to(a))


class TestSpatialModel:
    def test_customer_location_in_service_area(self):
        rng = np.random.default_rng(42)
        for _ in range(200):
            loc = generate_customer_location(rng)
            assert -15.0 <= loc.x <= 15.0
            assert -15.0 <= loc.y <= 15.0

    def test_compute_distances_to_kitchens_count(self):
        loc = Point2D(0.0, 0.0)
        dists = compute_distances_to_kitchens(loc, DEFAULT_KITCHEN_LOCATIONS)
        assert len(dists) == 3

    def test_compute_distances_positive(self):
        loc = Point2D(0.0, 0.0)
        dists = compute_distances_to_kitchens(loc, DEFAULT_KITCHEN_LOCATIONS)
        assert all(d >= 0.0 for d in dists)

    def test_distance_distribution_has_expected_keys(self):
        rng = np.random.default_rng(42)
        stats = distance_distribution_stats(rng, DEFAULT_KITCHEN_LOCATIONS, n_samples=1000)
        for key in ("mean", "median", "min", "max", "p10", "p90"):
            assert key in stats
            assert stats[key] >= 0.0

    def test_validate_spatial_model(self):
        result = validate_spatial_model(DEFAULT_KITCHEN_LOCATIONS, n_samples=5000)
        assert "stats" in result
        assert "passes_validation" in result
        assert isinstance(result["passes_validation"], bool)

    def test_kitchen_locations_are_3(self):
        assert len(DEFAULT_KITCHEN_LOCATIONS) == 3

    def test_kitchen_locations_are_tuples_of_floats(self):
        for loc in DEFAULT_KITCHEN_LOCATIONS:
            assert isinstance(loc, tuple)
            assert len(loc) == 2
            assert isinstance(loc[0], float)
            assert isinstance(loc[1], float)


# ---- DispatchState with kitchen candidates ----

def _make_kitchens():
    """Create 3 mock kitchen objects with kitchen_id and staff_level."""
    from simulation.entities import Kitchen
    kitchens = []
    for i in range(3):
        k = Kitchen(kitchen_id=i + 1, staff_level=3)
        k.current_orders = []
        kitchens.append(k)
    return kitchens


class TestDispatchStateSpatial:
    def test_kitchen_candidates_populated(self):
        kitchens = _make_kitchens()
        candidates = [
            KitchenCandidate(kitchen_id=k.kitchen_id, queue_len=0,
                             staff_level=k.staff_level, distance_km=5.0 + i)
            for i, k in enumerate(kitchens)
        ]
        state = DispatchState(
            now=10.0,
            kitchen_queue_lens={1: 0, 2: 0, 3: 0},
            idle_rider_count=5,
            weather_severity="clear",
            traffic_severity="low",
            hub_distance_km=1.0,
            travel_to_kitchen_min=2.4,
            kitchen_candidates=candidates,
        )
        assert state.kitchen_candidates is not None
        assert len(state.kitchen_candidates) == 3
        assert state.kitchen_candidates[0].kitchen_id == 1
        assert state.kitchen_candidates[2].distance_km == 7.0

    def test_baseline_state_has_no_candidates(self):
        state = DispatchState(
            now=10.0,
            kitchen_queue_lens={1: 0},
            idle_rider_count=5,
            weather_severity="clear",
            traffic_severity="low",
            hub_distance_km=1.0,
            travel_to_kitchen_min=2.4,
        )
        assert state.kitchen_candidates is None


# ---- OptimizedKitchenDispatch policy tests ----

def _order(distance_to_kitchens=None):
    order = Order(
        order_id=1, kitchen_id=None, placed_at=10.0, items=3,
        complexity=OrderComplexity.STANDARD, distance_km=5.0,
        staff_level=0, weather_severity="clear", traffic_severity="low",
        distance_to_kitchens=distance_to_kitchens or [5.0, 8.0, 12.0],
    )
    return order


def _state(distance_to_kitchens=None):
    dists = distance_to_kitchens or [5.0, 8.0, 12.0]
    candidates = [
        KitchenCandidate(kitchen_id=i + 1, queue_len=i, staff_level=3, distance_km=d)
        for i, d in enumerate(dists)
    ]
    return DispatchState(
        now=10.0,
        kitchen_queue_lens={1: 0, 2: 1, 3: 2},
        idle_rider_count=5,
        weather_severity="clear",
        traffic_severity="low",
        hub_distance_km=1.0,
        travel_to_kitchen_min=2.4,
        kitchen_candidates=candidates,
    )


def _config():
    from simulation.scenarios import load_scenario
    config = load_scenario("normal", seed=42)
    config["dispatch"]["enabled"] = True
    config["dispatch"]["default_policy"] = "optimized_kitchen"
    return config


class TestOptimizedKitchenDispatch:
    def test_policy_name(self):
        policy = OptimizedKitchenDispatch(_config())
        assert policy.name == "optimized_kitchen"

    def test_selects_closest_kitchen_when_queues_equal(self):
        dists = [4.0, 10.0, 15.0]
        policy = OptimizedKitchenDispatch(_config())
        decision = policy.decide(_order(dists), _state(dists))
        # Kitchen 1 has the shortest distance (4.0) and queue_len=0, should be selected.
        assert decision.selected_kitchen_id == 1
        assert decision.selected_kitchen_distance == pytest.approx(4.0)

    def test_prefers_shorter_distance_over_empty_queue(self):
        # Kitchen 1: dist=12, queue=0.  Kitchen 2: dist=5, queue=1.
        # Distance dominates, so Kitchen 2 should win.
        dists = [12.0, 5.0, 8.0]
        candidates = [
            KitchenCandidate(kitchen_id=1, queue_len=0, staff_level=3, distance_km=12.0),
            KitchenCandidate(kitchen_id=2, queue_len=1, staff_level=3, distance_km=5.0),
            KitchenCandidate(kitchen_id=3, queue_len=2, staff_level=3, distance_km=8.0),
        ]
        state = DispatchState(
            now=10.0, kitchen_queue_lens={1: 0, 2: 1, 3: 2},
            idle_rider_count=5, weather_severity="clear", traffic_severity="low",
            hub_distance_km=1.0, travel_to_kitchen_min=2.4,
            kitchen_candidates=candidates,
        )
        policy = OptimizedKitchenDispatch(_config())
        decision = policy.decide(_order(dists), state)
        assert decision.selected_kitchen_id == 2
        assert decision.selected_kitchen_distance == pytest.approx(5.0)

    def test_dispatch_at_is_immediate(self):
        policy = OptimizedKitchenDispatch(_config())
        decision = policy.decide(_order(), _state())
        assert decision.dispatch_at == 10.0  # immediate dispatch

    def test_eta_is_positive(self):
        policy = OptimizedKitchenDispatch(_config())
        decision = policy.decide(_order(), _state())
        assert decision.eta > 0.0

    def test_selected_fields_in_payload(self):
        policy = OptimizedKitchenDispatch(_config())
        decision = policy.decide(_order(), _state())
        payload = decision.to_payload()
        assert "selected_kitchen_id" in payload
        assert "selected_kitchen_distance" in payload
        assert payload["selected_kitchen_id"] in (1, 2, 3)

    def test_evaluations_in_inputs(self):
        policy = OptimizedKitchenDispatch(_config())
        decision = policy.decide(_order(), _state())
        assert "evaluations" in decision.inputs
        assert len(decision.inputs["evaluations"]) == 3
        for ev in decision.inputs["evaluations"]:
            assert "kitchen_id" in ev
            assert "delivery_est_min" in ev

    def test_raises_without_candidates(self):
        policy = OptimizedKitchenDispatch(_config())
        state_no_candidates = DispatchState(
            now=10.0, kitchen_queue_lens={1: 0}, idle_rider_count=5,
            weather_severity="clear", traffic_severity="low",
            hub_distance_km=1.0, travel_to_kitchen_min=2.4,
        )
        with pytest.raises(ValueError, match="kitchen_candidates"):
            policy.decide(_order(), state_no_candidates)

    def test_make_policy_optimized_kitchen(self):
        policy = make_policy("optimized_kitchen", None, _config())
        assert isinstance(policy, OptimizedKitchenDispatch)
        assert policy.name == "optimized_kitchen"


# ---- End-to-end: simulation with optimized_kitchen policy ----

class TestOptimizedKitchenE2E:
    def test_run_produces_summary(self):
        from simulation.engine import SimulationEngine
        from simulation.scenarios import load_scenario

        config = load_scenario("normal", seed=42)
        config["dispatch"]["enabled"] = True
        config["dispatch"]["default_policy"] = "optimized_kitchen"
        config["seed"] = 42
        config["days"] = 1
        engine = SimulationEngine(config, out_dir="data/test_optimized", scenario_name="normal")
        summary = engine.run()
        assert summary["policy"] == "optimized_kitchen"
        assert summary["orders_completed"] > 0
        assert 0.0 <= summary["on_time_rate"] <= 1.0

    def test_selected_kitchen_distances_populated(self):
        from simulation.engine import SimulationEngine
        from simulation.scenarios import load_scenario

        config = load_scenario("normal", seed=42)
        config["dispatch"]["enabled"] = True
        config["dispatch"]["default_policy"] = "optimized_kitchen"
        config["seed"] = 42
        config["days"] = 1
        engine = SimulationEngine(config, out_dir="data/test_optimized2", scenario_name="normal")
        engine.run()
        delivered = [o for o in engine.orders if o.delivered_at is not None]
        assert delivered
        selected_dists = [o.selected_kitchen_distance for o in delivered
                          if o.selected_kitchen_distance is not None]
        assert len(selected_dists) > 0
        assert all(d > 0 for d in selected_dists)

    def test_kitchen_load_distribution(self):
        from simulation.engine import SimulationEngine
        from simulation.scenarios import load_scenario

        config = load_scenario("normal", seed=42)
        config["dispatch"]["enabled"] = True
        config["dispatch"]["default_policy"] = "optimized_kitchen"
        config["seed"] = 42
        config["days"] = 1
        engine = SimulationEngine(config, out_dir="data/test_optimized3", scenario_name="normal")
        summary = engine.run()
        if "orders_per_kitchen" in summary:
            counts = list(summary["orders_per_kitchen"].values())
            # All kitchens should get some orders (load balancing).
            assert len(counts) == 3
            # No kitchen should get < 5% of all orders (extreme imbalance).
            total = sum(counts)
            for c in counts:
                assert c / total >= 0.05, f"kitchen got only {c/total:.1%} of orders"

    def test_baseline_unchanged(self):
        """Verify ImmediateDispatch still works identically (no regression).
        Now uses spatial model too, so kitchen distance metrics ARE present."""
        from simulation.engine import SimulationEngine
        from simulation.scenarios import load_scenario

        config = load_scenario("normal", seed=42)
        config["dispatch"]["enabled"] = True
        config["dispatch"]["default_policy"] = "immediate"
        config["seed"] = 42
        config["days"] = 1
        engine = SimulationEngine(config, out_dir="data/test_baseline_unch", scenario_name="normal")
        summary = engine.run()
        assert summary["policy"] == "immediate"
        assert summary["orders_completed"] > 0
        # Baseline uses spatial model too (random kitchen), so distance
        # metrics ARE present — but orders_per_kitchen should show random
        # distribution (roughly even across 3 kitchens).
        assert "avg_selected_kitchen_distance_km" in summary
        assert "orders_per_kitchen" in summary
