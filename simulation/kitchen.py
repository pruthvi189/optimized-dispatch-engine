import simpy

from .entities import Order, OrderStatus, OrderComplexity

# Base prep ranges (minutes) by complexity: (low, high)
BASE_PREP_RANGES = {
    OrderComplexity.SIMPLE: (3.0, 6.0),
    OrderComplexity.STANDARD: (5.0, 9.0),
    OrderComplexity.COMPLEX: (8.0, 15.0),
}


def sample_prep_time(rng, order: Order, kitchen, weather, traffic, config) -> float:
    """Prep duration conditioned on complexity, workload, staffing, weather."""
    prep = config["prep"]
    lo, hi = BASE_PREP_RANGES[order.complexity]
    base = rng.uniform(lo, hi)

    workload_factor = 1.0 + prep.get("workload_factor_per_order", 0.08) * max(0, len(kitchen.current_orders))
    if kitchen.staff_level < prep.get("staff_threshold", 3):
        workload_factor *= prep.get("staffing_factor", 1.25)

    weather_factor = weather.prep_factor()

    duration = base * workload_factor * weather_factor
    min_prep, max_prep = prep.get("clip", [2.0, 25.0])
    return float(min(max(duration, min_prep), max_prep))


def kitchen_process(env, rng, config, order: Order, kitchens, event_log):
    kitchen = next(k for k in kitchens if k.kitchen_id == order.kitchen_id)

    while kitchen.failed:
        yield env.timeout(1)

    order.status = OrderStatus.PREPPING
    order.prep_started_at = env.now

    with kitchen.resource.request() as req:
        yield req
        order.workload_at_placement = len(kitchen.current_orders)

        weather = getattr(env, "_weather")
        traffic = getattr(env, "_traffic")
        order.weather_severity = weather.current_severity().value
        order.traffic_severity = traffic.current_severity().value

        duration = sample_prep_time(rng, order, kitchen, weather, traffic, config)
        yield env.timeout(duration)

        if order.status == OrderStatus.CANCELLED:
            return

        order.prep_finished_at = env.now
        order.actual_prep_duration_min = duration
        order.status = OrderStatus.READY
        if order in kitchen.current_orders:
            kitchen.current_orders.remove(order)
        event_log.record(
            env.now, "order_completed",
            order_id=order.order_id,
            kitchen_id=order.kitchen_id,
            payload={
                "prep_duration_min": round(duration, 2),
                "items": order.items,
                "complexity": order.complexity.value,
                "workload_at_placement": order.workload_at_placement,
                "staff_level": order.staff_level,
                "weather": order.weather_severity,
                "traffic": order.traffic_severity,
                "hour": int(env.now // 60) % 24,
            },
        )
