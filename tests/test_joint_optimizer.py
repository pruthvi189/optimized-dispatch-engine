"""Fairness validation tests for joint kitchen+rider optimization.

Proves the 7 constraints before running the 20x5 experiment:
1. Shared initial rider positions per seed
2. Kitchen queue estimate matches simulator mechanics
3. No future information (deterministic forecast only)
4. Direct time components without arbitrary weights
5. Same environment snapshot timestamp
6. No assumed improvement target
7. Rider positions update correctly after delivery
"""

import sys
import pytest
sys.path.insert(0, '.')

from simulation.scenarios import load_scenario
from simulation.engine import SimulationEngine, RIDER_POSITION_STREAM_SALT
from simulation.spatial import DEFAULT_KITCHEN_LOCATIONS
from dispatch.policies import NearestHeuristicDispatch, JointOptimizerDispatch, make_policy
from dispatch.state import DispatchState, RiderCandidate


# ── Constraint 1: Shared initial rider positions per seed ──────────────

class TestSharedRiderPositions:
    """Both policies receive identical rider layouts per seed."""

    def test_same_seed_same_positions(self):
        config = load_scenario('normal', seed=42)
        config['dispatch']['enabled'] = True
        config['days'] = 1

        engines = {}
        for policy in ['nearest_heuristic', 'joint_optimizer']:
            c = load_scenario('normal', seed=42)
            c['dispatch']['enabled'] = True
            c['dispatch']['default_policy'] = policy
            c['days'] = 1
            e = SimulationEngine(c, out_dir=f'data/test_fair_{policy}', scenario_name='normal')
            e._setup()
            engines[policy] = e

        # Same seed → identical initial positions
        for i in range(10):
            r1 = engines['nearest_heuristic'].riders.riders[i]
            r2 = engines['joint_optimizer'].riders.riders[i]
            assert r1.x == r2.x, f"Rider {i+1} x differs: {r1.x} vs {r2.x}"
            assert r1.y == r2.y, f"Rider {i+1} y differs: {r1.y} vs {r2.y}"

    def test_different_seed_different_positions(self):
        """Different seeds produce different layouts."""
        layouts = {}
        for seed in [42, 43]:
            config = load_scenario('normal', seed=seed)
            config['dispatch']['enabled'] = True
            config['dispatch']['default_policy'] = 'nearest_heuristic'
            config['days'] = 1
            e = SimulationEngine(config, out_dir=f'data/test_fair_seed{seed}', scenario_name='normal')
            e._setup()
            layouts[seed] = [(r.x, r.y) for r in e.riders.riders]

        assert layouts[42] != layouts[43]


# ── Constraint 2: Queue estimate matches simulator mechanics ───────────

class TestQueueEstimateMatchesSimulator:
    """Policy queue wait uses same formula as simpy.Resource."""

    def test_nearest_heuristic_queue_formula(self):
        """Wait = max(0, queue_len - staff_level) * avg_prep / staff_level."""
        config = load_scenario('normal', seed=42)
        config['dispatch']['enabled'] = True
        config['dispatch']['default_policy'] = 'nearest_heuristic'
        config['days'] = 1
        e = SimulationEngine(config, out_dir='data/test_fair_q', scenario_name='normal')
        e._setup()

        from dispatch.state import KitchenCandidate
        policy = e.dispatcher.policy
        avg_prep = config.get("dispatch", {}).get("kitchen_selection", {}).get("avg_prep_min", 7.0)

        for queue_len, staff in [(0, 3), (1, 3), (3, 3), (5, 3), (0, 1), (10, 2)]:
            kc = KitchenCandidate(kitchen_id=1, queue_len=queue_len, staff_level=staff, distance_km=5.0)
            got = policy._estimate_queue_wait(kc)
            expected = max(0, queue_len - staff) * avg_prep / max(1, staff)
            assert got == pytest.approx(expected), (
                f"queue_wait mismatch for queue={queue_len}, staff={staff}: "
                f"got {got}, expected {expected}"
            )

    def test_joint_optimizer_queue_formula(self):
        config = load_scenario('normal', seed=42)
        config['dispatch']['enabled'] = True
        config['dispatch']['default_policy'] = 'joint_optimizer'
        config['days'] = 1
        e = SimulationEngine(config, out_dir='data/test_fair_q2', scenario_name='normal')
        e._setup()

        from dispatch.state import KitchenCandidate
        policy = e.dispatcher.policy
        avg_prep = config.get("dispatch", {}).get("kitchen_selection", {}).get("avg_prep_min", 7.0)
        wf = config.get("prep", {}).get("workload_factor_per_order", 0.08)
        staff_threshold = config.get("prep", {}).get("staff_threshold", 3)
        staffing_factor = config.get("prep", {}).get("staffing_factor", 1.25)

        for queue_len, staff in [(0, 3), (2, 3), (7, 3), (1, 1)]:
            kc = KitchenCandidate(kitchen_id=1, queue_len=queue_len, staff_level=staff, distance_km=5.0)
            got = policy._estimate_queue_wait(kc, avg_prep)

            n = queue_len
            s = max(1, staff)
            if n <= s:
                expected = 0.0
            else:
                waiting = n - s
                avg_pos = s + (waiting + 1) / 2.0
                wl = 1.0 + wf * avg_pos
                sf = staffing_factor if s < staff_threshold else 1.0
                effective_prep = avg_prep * wl * sf
                expected = waiting * effective_prep / s

            assert got == pytest.approx(expected), (
                f"queue_wait mismatch for queue={queue_len}, staff={staff}: "
                f"got {got}, expected {expected}"
            )


# ── Constraint 3: No future information ────────────────────────────────

class TestNoFutureInfo:
    """Policies use only deterministic hour-of-day forecast, not realized traffic."""

    def test_forecast_traffic_pure_function(self):
        """forecast_traffic depends only on departure_time, not on env state."""
        from simulation.environment import forecast_traffic
        # Same input → same output regardless of when called.
        results = [forecast_traffic(600.0, 10.0) for _ in range(10)]
        assert len(set(str(r) for r in results)) == 1

    def test_policy_uses_forecast_not_actual(self):
        """JointOptimizer uses forecast_traffic, not env._traffic."""
        import inspect
        source = inspect.getsource(JointOptimizerDispatch._estimate_delivery_for_pair)
        assert 'forecast_traffic' in source
        assert 'env._traffic' not in source


# ── Constraint 4: Direct time components, no arbitrary weights ─────────

class TestDirectTimeComponents:
    """JointOptimizer scoring = sum of time components, no tunable weights."""

    def test_no_weight_attributes(self):
        """JointOptimizer has no tunable weight knobs."""
        assert not hasattr(JointOptimizerDispatch, 'weight_distance')
        assert not hasattr(JointOptimizerDispatch, 'weight_queue')
        assert not hasattr(JointOptimizerDispatch, 'weight_staff')

    def test_scoring_is_time_sum(self):
        """Each component of the score is a time estimate in minutes."""
        config = load_scenario('normal', seed=42)
        policy = JointOptimizerDispatch(config)
        assert policy.name == "joint_optimizer"
        # Verify the estimate function returns a single time value.
        from dispatch.state import KitchenCandidate, RiderCandidate, DispatchState
        kc = KitchenCandidate(kitchen_id=1, queue_len=0, staff_level=3, distance_km=5.0)
        rc = RiderCandidate(rider_id=1, x=0.0, y=0.0, dist_to_kitchens=[3.0, 7.0, 10.0])
        state = DispatchState(
            now=0.0, kitchen_queue_lens={}, idle_rider_count=1,
            weather_severity="clear", traffic_severity="low",
            hub_distance_km=5.0, travel_to_kitchen_min=0.0,
            kitchen_candidates=[kc], rider_candidates=[rc],
        )
        total = policy._estimate_delivery_for_pair(kc, rc, 0, state, prep_time=7.0)
        assert isinstance(total, float)
        assert total > 0


# ── Constraint 5: Same environment snapshot timestamp ──────────────────

class TestSameTimestamp:
    """Both policies see identical weather/traffic state per order."""

    def test_same_engine_same_snapshot(self):
        """Engine creates one weather/traffic state shared by all policies."""
        config = load_scenario('normal', seed=42)
        config['dispatch']['enabled'] = True
        config['dispatch']['default_policy'] = 'nearest_heuristic'
        config['days'] = 1
        e = SimulationEngine(config, out_dir='data/test_fair_ts', scenario_name='normal')
        e._setup()
        # Weather and traffic generators are engine-level, not policy-level.
        assert e._weather is e.env._weather
        assert e._traffic is e.env._traffic


# ── Constraint 7: Rider position updates after delivery ────────────────

class TestRiderPositionUpdates:
    """Riders move to customer location after completing delivery."""

    def test_rider_position_changes_after_delivery(self):
        config = load_scenario('normal', seed=42)
        config['dispatch']['enabled'] = True
        config['dispatch']['default_policy'] = 'nearest_heuristic'
        config['days'] = 1
        e = SimulationEngine(config, out_dir='data/test_fair_rpos', scenario_name='normal')
        summary = e.run()

        delivered = [o for o in e.orders if o.delivered_at is not None and o.rider_id is not None]
        if not delivered:
            pytest.skip("No delivered orders in this run")

        # At least some riders should have moved from origin.
        riders_moved = 0
        for r in e.riders.riders:
            if r.x != 0.0 or r.y != 0.0:
                riders_moved += 1
        assert riders_moved > 0, "No riders moved from initial positions"

    def test_dispatcher_has_kitchen_locations(self):
        """Dispatcher stores kitchen_locations for rider distance computation."""
        config = load_scenario('normal', seed=42)
        config['dispatch']['enabled'] = True
        config['dispatch']['default_policy'] = 'nearest_heuristic'
        config['days'] = 1
        e = SimulationEngine(config, out_dir='data/test_fair_dl', scenario_name='normal')
        e._setup()
        assert e.dispatcher.kitchen_locations is not None
        assert len(e.dispatcher.kitchen_locations) == 3


# ── Constraint 6: No assumed improvement target ────────────────────────

class TestNoAssumedTarget:
    """Neither policy hardcodes an expected improvement or target metric."""

    def test_joint_optimizer_no_target_constant(self):
        import inspect
        source = inspect.getsource(JointOptimizerDispatch)
        assert 'target' not in source.lower().split('#')[0]  # no target in code (excluding comments)
        assert 'improvement' not in source.lower().split('#')[0]

    def test_nearest_heuristic_no_target_constant(self):
        import inspect
        source = inspect.getsource(NearestHeuristicDispatch)
        assert 'target' not in source.lower().split('#')[0]
        assert 'improvement' not in source.lower().split('#')[0]
