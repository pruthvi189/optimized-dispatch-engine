import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pytest
import simpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import SimulationEngine, load_scenario  # noqa: E402
from simulation.engine import DISPATCH_STREAM_SALT  # noqa: E402
from simulation.cancellations import CancellationManager  # noqa: E402
from simulation.dispatcher import Dispatcher  # noqa: E402
from simulation.entities import Order, OrderComplexity, OrderStatus  # noqa: E402
from simulation.environment import TrafficGenerator, WeatherGenerator  # noqa: E402
from simulation.event_log import EventLog  # noqa: E402
from simulation.riders import RiderPool  # noqa: E402
from simulation.rng import spawn_streams  # noqa: E402
from dispatch.policies import (  # noqa: E402
    AdaptiveDispatch,
    ImmediateDispatch,
    make_policy,
)
from dispatch.state import DispatchState  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTOR_DIR = os.path.join(ROOT, "artifacts")
HAS_ARTIFACTS = os.path.isdir(PREDICTOR_DIR)

needs_artifacts = pytest.mark.skipif(not HAS_ARTIFACTS, reason="Phase 2 artifacts not trained")


class FakePredictor:
    def __init__(self, mean=6.0, low=4.5, high=8.5, uncertainty="medium"):
        self.mean = mean
        self.low = low
        self.high = high
        self.uncertainty = uncertainty

    def predict(self, features):
        return {
            "prep_mean": self.mean,
            "prep_low": self.low,
            "prep_high": self.high,
            "uncertainty": self.uncertainty,
        }


def _state(now=10.0, queue=2, weather="clear", traffic="low", travel=2.4):
    return DispatchState(
        now=now,
        kitchen_queue_lens={1: queue},
        idle_rider_count=5,
        weather_severity=weather,
        traffic_severity=traffic,
        hub_distance_km=1.0,
        travel_to_kitchen_min=travel,
    )


def _order():
    return Order(
        order_id=1, kitchen_id=1, placed_at=10.0, items=3,
        complexity=OrderComplexity.STANDARD, distance_km=3.0,
        staff_level=3, weather_severity="clear", traffic_severity="low",
    )


def _config(policy="immediate"):
    config = load_scenario("normal", seed=42)
    config["dispatch"]["enabled"] = True
    config["dispatch"]["default_policy"] = policy
    config["dispatch"]["predictor_dir"] = PREDICTOR_DIR
    return config


def _run(policy, seed=42, days=1, out="data/test_dispatch", scenario="normal"):
    config = _config(policy)
    config["seed"] = seed
    config["days"] = days
    engine = SimulationEngine(config, out_dir=str(out), scenario_name=scenario)
    return engine, engine.run()


# ---- Policy correctness (Task 2) -------------------------------------------


def test_immediate_dispatches_at_now():
    decision = ImmediateDispatch(_config()).decide(_order(), _state())
    assert decision.dispatch_at == 10.0
    assert decision.policy == "immediate"


def test_adaptive_dispatch_never_before_now():
    decision = AdaptiveDispatch(FakePredictor(), _config("adaptive")).decide(_order(), _state())
    assert decision.dispatch_at >= 10.0


def test_adaptive_buffer_larger_for_high_uncertainty():
    cfg = _config("adaptive")
    low_d = AdaptiveDispatch(FakePredictor(uncertainty="low"), cfg).decide(_order(), _state())
    high_d = AdaptiveDispatch(FakePredictor(uncertainty="high"), cfg).decide(_order(), _state())
    assert high_d.risk_buffer_min > low_d.risk_buffer_min
    assert high_d.risk_buffer_min > 0.0


def test_adaptive_records_prediction_and_rationale():
    decision = AdaptiveDispatch(FakePredictor(), _config("adaptive")).decide(_order(), _state())
    assert decision.predicted_prep_mean == 6.0
    assert decision.predicted_prep_low == 4.5
    assert decision.predicted_prep_high == 8.5
    assert decision.uncertainty == "medium"
    assert "buffer" in decision.rationale
    assert decision.eta > decision.dispatch_at


def test_make_policy_adaptive_without_predictor_raises():
    with pytest.raises(ValueError, match="predictor"):
        make_policy("adaptive", None, _config("adaptive"))
    assert make_policy("immediate", None, _config()).name == "immediate"


# ---- Delivery completes orders ---------------------------------------------


def test_delivery_completes_orders(tmp_path):
    engine, summary = _run("immediate", out=str(tmp_path / "run"))
    assert summary["orders_completed"] > 0
    delivered = [o for o in engine.orders if o.delivered_at is not None]
    assert delivered
    assert all(o.status == OrderStatus.COMPLETED for o in delivered)
    types = {e["event_type"] for e in engine.event_log.events}
    assert {"rider_dispatched", "rider_at_kitchen", "rider_pickup", "rider_delivered"} <= types


# ---- Determinism -------------------------------------------------------------


def test_same_seed_adaptive_is_deterministic(tmp_path):
    out1 = tmp_path / "a1"
    out2 = tmp_path / "a2"
    for out in (out1, out2):
        config = _config("adaptive")
        engine = SimulationEngine(config, out_dir=str(out), scenario_name="normal")
        engine.run()
    h1 = hashlib.sha256((out1 / "event_log.csv").read_bytes()).hexdigest()
    h2 = hashlib.sha256((out2 / "event_log.csv").read_bytes()).hexdigest()
    assert h1 == h2


# ---- Fair compare: policies share arrivals + prep + hub distances ----------


def test_fair_compare_identical_arrivals_prep_and_hub(tmp_path):
    out_i = str(tmp_path / "i")
    out_a = str(tmp_path / "a")
    _, _ = _run("immediate", seed=42, out=out_i)
    _, _ = _run("adaptive", seed=42, out=out_a)
    imp = pd.read_csv(os.path.join(out_i, "orders.csv")).sort_values("order_id")
    ada = pd.read_csv(os.path.join(out_a, "orders.csv")).sort_values("order_id")
    assert len(imp) == len(ada)
    for col in ("placed_at", "hub_distance_km", "workload_at_placement"):
        assert (imp[col].values == ada[col].values).all(), col
    prep_i = imp.dropna(subset=["actual_prep_duration_min"])
    prep_a = ada.dropna(subset=["actual_prep_duration_min"])
    assert (prep_i["actual_prep_duration_min"].values == prep_a["actual_prep_duration_min"].values).all()

    # Pairing + drain invariants: the same status and cancel_reason per order
    # (so the two policies observed identical upstream events), and no order
    # left in-flight after the drain.
    for col in ("status", "cancel_reason"):
        i_vals = imp[col].fillna("").values
        a_vals = ada[col].fillna("").values
        assert (i_vals == a_vals).all(), col
    assert set(imp["status"]).issubset({"completed", "cancelled"})
    assert set(ada["status"]).issubset({"completed", "cancelled"})


# ---- Cancellation: rider cancel redispatch ---------------------------------


def test_rider_cancel_redispatches(tmp_path):
    config = _config("immediate")
    config["cancellation_rates"]["customer_cancel_per_min"] = 0.0
    config["cancellation_rates"]["kitchen_failure_per_min"] = 0.0
    env = simpy.Environment()
    streams = spawn_streams(config["seed"])
    env._weather = WeatherGenerator(env, streams["weather"], config)
    env._traffic = TrafficGenerator(env, streams["traffic"], config)

    riders = RiderPool(config)
    event_log = EventLog(out_dir=str(tmp_path))
    dispatch_rng = np.random.default_rng(config["seed"] + DISPATCH_STREAM_SALT)
    dispatcher = Dispatcher(env, config, ImmediateDispatch(config), riders, event_log, dispatch_rng)
    dispatcher.bind_kitchens([])
    cancels = CancellationManager(
        env, streams["cancellations"], config, [], riders, event_log, dispatcher=dispatcher
    )

    order = Order(
        order_id=1, kitchen_id=1, placed_at=0.0, items=2,
        complexity=OrderComplexity.SIMPLE, distance_km=3.0,
        dispatch_at=0.0, hub_distance_km=1.0, status=OrderStatus.READY,
    )
    first = riders.assign_next_idle(at=0.0)
    order.rider_id = first.rider_id

    cancels.rider_cancel(order, first.rider_id)
    env.run(until=100)

    types = [e["event_type"] for e in event_log.events]
    assert "cancellation_rider" in types
    assert "order_redispatch" in types
    assert order.status == OrderStatus.COMPLETED
    assert order.delivered_at is not None


# ---- Metrics invariants -----------------------------------------------------


def test_metrics_invariants(tmp_path):
    _, summary = _run("immediate", out=str(tmp_path / "m"))
    assert 0.0 <= summary["on_time_rate"] <= 1.0
    for key in ("avg_delivery_min", "avg_late_min", "avg_order_wait_min",
                "avg_rider_wait_kitchen_min", "avg_rider_idle_min"):
        assert summary[key] >= 0.0
    assert summary["cost_score"] >= 0.0
    assert summary["orders_completed"] + summary["orders_cancelled"] <= summary["orders_placed"]
