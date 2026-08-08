import sys
import os

import simpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.cancellations import CancellationManager  # noqa: E402
from simulation.entities import Order, Kitchen, OrderStatus, OrderComplexity  # noqa: E402
from simulation.riders import RiderPool  # noqa: E402
from simulation.event_log import EventLog  # noqa: E402
from simulation.rng import spawn_streams  # noqa: E402


def _order(kitchen_id=1, order_id=1):
    return Order(
        order_id=order_id, kitchen_id=kitchen_id, placed_at=0.0, items=2,
        complexity=OrderComplexity.SIMPLE, distance_km=2.0,
    )


def test_customer_cancel(tmp_path):
    env = simpy.Environment()
    rng = spawn_streams(0)["cancellations"]
    kitchen = Kitchen(kitchen_id=1, staff_level=3)
    kitchen.resource = simpy.Resource(env, capacity=3)
    log = EventLog(out_dir=str(tmp_path))
    mgr = CancellationManager(
        env, rng, {"cancellation_rates": {}}, [kitchen], RiderPool({"riders": {"count": 2}}), log
    )
    order = _order()
    kitchen.current_orders.append(order)
    mgr.customer_cancel(order, wasted_prep=True)
    assert order.status == OrderStatus.CANCELLED
    assert order.cancel_reason == "customer"
    assert log.events[-1]["event_type"] == "cancellation_customer"


def test_rider_cancel(tmp_path):
    env = simpy.Environment()
    rng = spawn_streams(0)["cancellations"]
    kitchen = Kitchen(kitchen_id=1, staff_level=3)
    kitchen.resource = simpy.Resource(env, capacity=3)
    pool = RiderPool({"riders": {"count": 1}})
    log = EventLog(out_dir=str(tmp_path))
    mgr = CancellationManager(
        env, rng, {"cancellation_rates": {}}, [kitchen], pool, log
    )
    order = _order()
    mgr.rider_cancel(order, rider_id=1)
    assert order.status == OrderStatus.CANCELLED
    assert pool.riders[0].status.value == "unavailable"
    assert log.events[-1]["event_type"] == "cancellation_rider"


def test_kitchen_failure_event(tmp_path):
    env = simpy.Environment()
    rng = spawn_streams(0)["cancellations"]
    kitchen = Kitchen(kitchen_id=1, staff_level=3)
    kitchen.resource = simpy.Resource(env, capacity=3)
    log = EventLog(out_dir=str(tmp_path))
    mgr = CancellationManager(
        env, rng, {"cancellation_rates": {}}, [kitchen], RiderPool({"riders": {"count": 2}}), log
    )
    mgr.kitchen_failure(kitchen)
    assert any(e["event_type"] == "kitchen_failure_start" for e in log.events)


def test_customer_cancel_monitor_hits(tmp_path):
    """With a 100% per-minute hazard, a prepping order eventually gets cancelled."""
    env = simpy.Environment()
    rng = spawn_streams(0)["cancellations"]
    kitchen = Kitchen(kitchen_id=1, staff_level=3)
    kitchen.resource = simpy.Resource(env, capacity=3)
    log = EventLog(out_dir=str(tmp_path))
    mgr = CancellationManager(
        env, rng, {"cancellation_rates": {"customer_cancel_per_min": 1.0}}, [kitchen],
        RiderPool({"riders": {"count": 2}}), log,
    )
    order = _order()
    order.status = OrderStatus.PREPPING
    kitchen.current_orders.append(order)
    env.run(until=10)
    assert order.status == OrderStatus.CANCELLED
