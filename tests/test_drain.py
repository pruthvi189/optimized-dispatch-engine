"""Tests for the post-generation drain period.

The drain must (a) let in-flight orders finish for both policies using the
identical rule, and (b) fall back to a hard timeout safety net that still
marks the run finished."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import SimulationEngine, load_scenario  # noqa: E402
from simulation.entities import Order, OrderComplexity, OrderStatus  # noqa: E402
from simulation.kitchen import kitchen_process  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTOR_DIR = os.path.join(ROOT, "artifacts")


def _config(policy="immediate", days=1, drain_timeout=240, demand_multiplier=1.0):
    config = load_scenario("normal", seed=42)
    config["days"] = days
    config["drain_timeout_min"] = drain_timeout
    config["demand_multiplier"] = demand_multiplier
    config["dispatch"]["enabled"] = True
    config["dispatch"]["default_policy"] = policy
    config["dispatch"]["predictor_dir"] = PREDICTOR_DIR
    return config


def _run(config, out):
    engine = SimulationEngine(config, out_dir=str(out), scenario_name="normal")
    engine.run()
    return engine


def _all_terminal(engine):
    return all(o.status in (OrderStatus.COMPLETED, OrderStatus.CANCELLED, OrderStatus.FAILED)
               for o in engine.orders)


def test_run_drains_all_orders(tmp_path):
    engine = _run(_config(), tmp_path / "drain")

    assert engine.is_finished
    assert engine.env.now > engine.total_minutes
    assert _all_terminal(engine)
    s = engine.summary
    assert s["orders_placed"] == s["orders_completed"] + s["orders_cancelled"]
    assert s["orders_in_flight"] == 0


def test_both_policies_fully_drain(tmp_path):
    imp = _run(_config("immediate"), tmp_path / "i")
    ada = _run(_config("adaptive"), tmp_path / "a")

    assert imp.is_finished and ada.is_finished
    for engine in (imp, ada):
        assert engine.env.now > engine.total_minutes
        assert _all_terminal(engine)
        assert engine.summary["orders_in_flight"] == 0


@pytest.mark.skipif(
    not os.path.isdir(PREDICTOR_DIR),
    reason="Phase 2 artifacts not trained (needed for adaptive)",
)
def test_adaptive_drains_within_timeout(tmp_path):
    engine = _run(_config("adaptive"), tmp_path / "ada")
    assert engine.is_finished
    assert engine.env.now <= engine.hard_stop
    assert _all_terminal(engine)


def test_safety_timeout_marks_finished_with_inflight(tmp_path):
    # drain_timeout = 0 -> hard_stop == total_minutes -> finished even if an
    # order is still in flight (the safety net).
    config = _config(demand_multiplier=1.0, drain_timeout=0)
    engine = _run(config, tmp_path / "to")

    assert engine.is_finished
    assert engine.env.now == engine.total_minutes


def test_inflight_order_is_drained(tmp_path):
    """A single order placed near the end of the day, still in flight at the
    generation cutoff, must be completed by the drain."""
    config = _config(demand_multiplier=0.0, drain_timeout=240)
    # Disable cancellations so the order cannot be lost.
    config["cancellation_rates"]["customer_cancel_per_min"] = 0.0
    config["cancellation_rates"]["kitchen_failure_per_min"] = 0.0
    config["cancellation_rates"]["rider_cancel_per_order"] = 0.0

    engine = SimulationEngine(config, out_dir=str(tmp_path / "manual"), scenario_name="normal")
    engine._setup()

    # Advance to just before the generation window ends, then place a single
    # order whose (long-ish) prep cannot finish before the cutoff.
    placed_at = engine.total_minutes - 5.0
    engine.advance(placed_at)
    assert engine.env.now == placed_at

    kitchen = engine.kitchens[0]
    order_id = next(engine.order_counter)
    order = Order(
        order_id=order_id,
        kitchen_id=kitchen.kitchen_id,
        placed_at=engine.env.now,
        items=3,
        complexity=OrderComplexity.COMPLEX,
        distance_km=3.0,
        promised_delivery_min=config["dispatch"]["promised_delivery_min"],
        workload_at_placement=len(kitchen.current_orders),
        staff_level=kitchen.staff_level,
        weather_severity="clear",
        traffic_severity="low",
    )
    kitchen.current_orders.append(order)
    engine.all_orders.append(order)
    engine.env.process(
        kitchen_process(engine.env, engine.streams["prep"], config, order,
                        engine.kitchens, engine.event_log)
    )
    engine.dispatcher.dispatch(order)

    engine.advance(engine.total_minutes)
    # At the cutoff the order must not yet be terminal.
    assert order.status not in (OrderStatus.COMPLETED, OrderStatus.CANCELLED, OrderStatus.FAILED)
    assert not engine.is_finished

    engine.drain()
    engine.finalize()
    assert order.status == OrderStatus.COMPLETED
    assert order.delivered_at is not None
    assert engine.env.now > engine.total_minutes
    assert engine.is_finished
    assert engine.summary["orders_in_flight"] == 0
