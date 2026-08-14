def compute_metrics(orders, riders, sim_length, config):
    """Aggregate KPIs for one run. Only fully-delivered orders count toward
    wait/late metrics; orders still in-flight at the end of the run (after the
    drain period) are excluded.

    Primary objective is end-to-end delivery time (delivered_at - placed_at);
    percentiles (p50/p90/p95/p99) and lateness complement the average.

    - order_wait: food sits ready, waiting for a rider (pickup - ready).
    - rider_wait_kitchen: rider arrives early and waits for food (ready - rider
      arrival). Delivered only, so pickup/travel time is excluded by
      construction: the delivery process records rider_arrived_kitchen_at on
      kitchen arrival and pickup_at after READY + pickup_time_min.
    - orders_in_flight: placed orders that are neither delivered nor cancelled
      (a diagnostic; should be 0 after a drain).
    - rider idle: rider not assigned to anything (status IDLE), reported for
      transparency.
    - cost = idle_w * wasted_rider_min + wait_w * total_order_wait + late_w * total_late
      where wasted_rider = assigned-but-unproductive time only (rider waiting at the
      kitchen for food). Unassigned time is available capacity, not dispatch waste —
      including it would make the cost score reward an oversized fleet, not the policy.
    """
    import numpy as np

    delivered = [o for o in orders if o.delivered_at is not None]
    cancelled = [o for o in orders if o.status.value == "cancelled"]

    promised = config["dispatch"]["promised_delivery_min"]
    late = [max(0.0, o.delivered_at - (o.placed_at + promised)) for o in delivered]
    delivery_times = [d.delivered_at - d.placed_at for d in delivered]
    late_count = sum(1 for d in delivered if d.delivered_at > d.placed_at + promised)
    order_wait = [
        max(0.0, o.pickup_at - o.prep_finished_at)
        for o in delivered
        if o.pickup_at is not None and o.prep_finished_at is not None
    ]
    rider_wait_kitchen = [
        max(0.0, o.prep_finished_at - o.rider_arrived_kitchen_at)
        for o in delivered
        if o.rider_arrived_kitchen_at is not None
    ]

    busy_total = 0.0
    for r in riders:
        busy_total += r.busy_min
        if r.assigned_at is not None:
            busy_total += max(0.0, sim_length - r.assigned_at)
    idle_total = max(0.0, sim_length * len(riders) - busy_total)
    wasted_rider = sum(rider_wait_kitchen)

    w = config["dispatch"]["cost_weights"]
    cost_score = w["idle"] * wasted_rider + w["wait"] * sum(order_wait) + w["late"] * sum(late)

    def pct(p):
        return round(float(np.percentile(delivery_times, p)), 3) if delivery_times else 0.0

    return {
        "orders_placed": len(orders),
        "orders_completed": len(delivered),
        "orders_cancelled": len(cancelled),
        "on_time_rate": round(sum(1 for d in delivered if d.delivered_at <= d.placed_at + promised) / len(delivered), 4) if delivered else 0.0,
        "avg_delivery_min": round(sum(delivery_times) / len(delivery_times), 3) if delivery_times else 0.0,
        "p50_delivery_min": pct(50),
        "p90_delivery_min": pct(90),
        "p95_delivery_min": pct(95),
        "p99_delivery_min": pct(99),
        "max_delivery_min": round(max(delivery_times), 3) if delivery_times else 0.0,
        "avg_late_min": round(sum(late) / len(late), 3) if late else 0.0,
        "late_count": late_count,
        "avg_order_wait_min": round(sum(order_wait) / len(order_wait), 3) if order_wait else 0.0,
        "avg_rider_wait_kitchen_min": round(sum(rider_wait_kitchen) / len(rider_wait_kitchen), 3) if rider_wait_kitchen else 0.0,
        "avg_rider_idle_min": round(idle_total / len(riders), 3) if riders else 0.0,
        "cost_score": round(cost_score, 1),
        "orders_in_flight": max(0, len(orders) - len(delivered) - len(cancelled)),
    }


def format_metrics(m: dict) -> str:
    return (
        f"placed={m['orders_placed']} completed={m['orders_completed']} "
        f"cancelled={m['orders_cancelled']} on_time={m['on_time_rate']:.1%} "
        f"delivery={m['avg_delivery_min']}min p95={m.get('p95_delivery_min', m['avg_delivery_min'])}min "
        f"late={m.get('late_count', 0)} wait={m['avg_order_wait_min']}min "
        f"rider_kitchen_wait={m['avg_rider_wait_kitchen_min']}min "
        f"rider_idle={m['avg_rider_idle_min']}min cost={m['cost_score']}"
    )
