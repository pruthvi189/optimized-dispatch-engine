def compute_metrics(orders, riders, sim_length, config):
    """Aggregate KPIs for one run. Only fully-delivered orders count toward
    wait/late metrics; orders still in-flight at sim end are excluded.

    - order_wait: food sits ready, waiting for a rider (pickup - ready).
    - rider_wait_kitchen: rider arrives early and waits for food (ready - rider arrival).
    - rider idle: rider not assigned to anything (status IDLE), reported for transparency.
    - cost = idle_w * wasted_rider_min + wait_w * total_order_wait + late_w * total_late
      where wasted_rider = assigned-but-unproductive time only (rider waiting at the
      kitchen for food). Unassigned time is available capacity, not dispatch waste —
      including it would make the cost score reward an oversized fleet, not the policy.
    """
    delivered = [o for o in orders if o.delivered_at is not None]
    cancelled = [o for o in orders if o.status.value == "cancelled"]

    promised = config["dispatch"]["promised_delivery_min"]
    late = [max(0.0, o.delivered_at - (o.placed_at + promised)) for o in delivered]
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

    return {
        "orders_placed": len(orders),
        "orders_completed": len(delivered),
        "orders_cancelled": len(cancelled),
        "on_time_rate": round(sum(1 for d in delivered if d.delivered_at <= d.placed_at + promised) / len(delivered), 4) if delivered else 0.0,
        "avg_delivery_min": round(sum(d.delivered_at - d.placed_at for d in delivered) / len(delivered), 3) if delivered else 0.0,
        "avg_late_min": round(sum(late) / len(late), 3) if late else 0.0,
        "avg_order_wait_min": round(sum(order_wait) / len(order_wait), 3) if order_wait else 0.0,
        "avg_rider_wait_kitchen_min": round(sum(rider_wait_kitchen) / len(rider_wait_kitchen), 3) if rider_wait_kitchen else 0.0,
        "avg_rider_idle_min": round(idle_total / len(riders), 3) if riders else 0.0,
        "cost_score": round(cost_score, 1),
    }


def format_metrics(m: dict) -> str:
    return (
        f"placed={m['orders_placed']} completed={m['orders_completed']} "
        f"cancelled={m['orders_cancelled']} on_time={m['on_time_rate']:.1%} "
        f"delivery={m['avg_delivery_min']}min wait={m['avg_order_wait_min']}min "
        f"rider_kitchen_wait={m['avg_rider_wait_kitchen_min']}min "
        f"rider_idle={m['avg_rider_idle_min']}min cost={m['cost_score']}"
    )
