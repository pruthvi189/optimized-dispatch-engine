from .entities import OrderStatus, RiderStatus


def delivery_process(env, config, order, rider, riders, event_log, kitchen_locations=None):
    """Rider lifecycle for one order: travel to kitchen -> wait for READY ->
    pick up -> travel to customer -> complete. Releases the rider on finish.

    Rider-to-kitchen travel uses the rider→kitchen distance matrix (generated
    once per seed, shared by policies and simulation). Falls back to
    hub_distance_km if no matrix is available.

    After delivery, the rider's position is updated to the customer location
    so dispatch policies can consider rider proximity for future orders.
    """
    d = config["dispatch"]

    weather = getattr(env, "_weather").current_severity().value
    traffic = getattr(env, "_traffic").current_severity().value
    speed_kmh = config["riders"]["speed_kmh"]
    wf = d["weather_speed_factor"][weather]
    tf = d["traffic_speed_factor"][traffic]

    # Rider-to-kitchen travel: use the matrix distance for the ACTUAL
    # assigned rider (which may differ from the pre-assigned rider if
    # the pre-assigned rider was taken). This is the single source of truth.
    rider_kitchen_dist = riders.get_rider_kitchen_distance(rider.rider_id, order.kitchen_id)
    travel_kitchen = riders.travel_to_kitchen_min(rider_kitchen_dist, weather, traffic)

    yield env.timeout(travel_kitchen)
    order.rider_arrived_kitchen_at = env.now
    event_log.record(
        env.now, "rider_at_kitchen",
        order_id=order.order_id, kitchen_id=order.kitchen_id, rider_id=rider.rider_id,
        payload={"travel_to_kitchen_min": round(travel_kitchen, 2),
                 "rider_kitchen_dist_km": round(rider_kitchen_dist, 4)},
    )

    while order.status != OrderStatus.READY:
        if order.status in (OrderStatus.CANCELLED, OrderStatus.FAILED):
            riders.release(rider.rider_id, at=env.now)
            return
        yield env.timeout(0.1)

    yield env.timeout(d["pickup_time_min"])
    order.pickup_at = env.now
    rider.status = RiderStatus.DELIVERING
    event_log.record(
        env.now, "rider_pickup",
        order_id=order.order_id, kitchen_id=order.kitchen_id, rider_id=rider.rider_id,
    )

    # Customer leg travel — always from kitchen to customer using spatial model.
    weather = getattr(env, "_weather").current_severity().value
    traffic = getattr(env, "_traffic").current_severity().value
    wf = d["weather_speed_factor"][weather]
    tf = d["traffic_speed_factor"][traffic]
    travel_customer_min = order.distance_km / speed_kmh * 60.0 * tf * wf
    yield env.timeout(travel_customer_min)

    order.delivered_at = env.now
    order.status = OrderStatus.COMPLETED
    event_log.record(
        env.now, "rider_delivered",
        order_id=order.order_id, kitchen_id=order.kitchen_id, rider_id=rider.rider_id,
        payload={"travel_customer_min": round(travel_customer_min, 2)},
    )

    # Update rider position to customer location for subsequent assignments.
    new_x = order.customer_x if order.customer_x is not None else rider.x
    new_y = order.customer_y if order.customer_y is not None else rider.y
    riders.release(rider.rider_id, at=env.now, new_x=new_x, new_y=new_y)
