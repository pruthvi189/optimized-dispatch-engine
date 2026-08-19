from .entities import Order, OrderStatus, OrderComplexity

# Base prep ranges (minutes) by complexity: (low, high)
BASE_PREP_RANGES = {
    OrderComplexity.SIMPLE: (3.0, 6.0),
    OrderComplexity.STANDARD: (5.0, 9.0),
    OrderComplexity.COMPLEX: (8.0, 15.0),
}

# Midpoints of base prep ranges, used for order-specific prep estimation.
_BASE_PREP_MIDPOINTS = {
    OrderComplexity.SIMPLE: 4.5,
    OrderComplexity.STANDARD: 7.0,
    OrderComplexity.COMPLEX: 11.5,
}


def estimate_prep_for_order(order: Order) -> dict:
    """Estimate prep time for a specific order based on its attributes.

    Uses the order's complexity and item count to produce an order-specific
    prep estimate with a prediction range. This is a transparent calculation
    using the same base prep ranges the simulator uses — not an ML prediction.

    Returns dict with keys: prep_mean, prep_low, prep_high, uncertainty.
    """
    base_mid = _BASE_PREP_MIDPOINTS[order.complexity]
    lo, hi = BASE_PREP_RANGES[order.complexity]

    # Scale by item count relative to complexity midpoint.
    # Simple = 1-2 items (midpoint 1.5), Standard = 3-5 (midpoint 4), Complex = 6+ (midpoint 7).
    item_scale = {
        OrderComplexity.SIMPLE: 1.0 + 0.05 * (order.items - 1.5),
        OrderComplexity.STANDARD: 1.0 + 0.03 * (order.items - 4.0),
        OrderComplexity.COMPLEX: 1.0 + 0.02 * (order.items - 7.0),
    }
    scale = item_scale.get(order.complexity, 1.0)
    prep_mean = base_mid * max(0.7, scale)

    # Prediction range: base range scaled by the same factor.
    prep_low = lo * max(0.7, scale)
    prep_high = hi * max(0.7, scale)

    # Uncertainty tier based on range width relative to mean.
    width = prep_high - prep_low
    if width < 2.0:
        tier = "low"
    elif width < 4.0:
        tier = "medium"
    else:
        tier = "high"

    return {
        "prep_mean": round(prep_mean, 2),
        "prep_low": round(prep_low, 2),
        "prep_high": round(prep_high, 2),
        "uncertainty": tier,
    }


def sample_prep_time(rng, order: Order, kitchen, weather, traffic, config) -> float:
    """Prep duration conditioned on complexity, workload, staffing, weather."""
    prep = config["prep"]
    lo, hi = BASE_PREP_RANGES[order.complexity]
    base = rng.uniform(lo, hi)

    workload_factor = 1.0 + prep.get("workload_factor_per_order", 0.027) * max(0, len(kitchen.current_orders))
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
    # Timestamp when the order enters the kitchen queue (before acquiring the
    # single resource), so queue wait is measurable separately from prep time.
    order.entered_kitchen_at = env.now

    with kitchen.resource.request() as req:
        yield req
        # Prep only starts once the kitchen capacity is acquired. This keeps
        # kitchen_queue (= prep_started_at - entered_kitchen_at) honest: it is
        # pure queue time, not prep time.
        order.prep_started_at = env.now
        # NOTE: workload/weather/traffic features are snapshotted at PLACEMENT
        # in OrderGenerator._create_order and must NOT be overwritten here
        # (prep-start state is not known at placement time). Only the actual
        # prep duration below uses prep-time conditions.
        weather = getattr(env, "_weather")
        traffic = getattr(env, "_traffic")

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
