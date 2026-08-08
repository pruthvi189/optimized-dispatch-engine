import simpy

from .entities import OrderStatus, RiderStatus


def delivery_process(env, config, order, rider, riders, event_log):
    """Rider lifecycle for one order: travel to kitchen -> wait for READY ->
    pick up -> travel to customer -> complete. Releases the rider on finish."""
    d = config["dispatch"]

    weather = getattr(env, "_weather").current_severity().value
    traffic = getattr(env, "_traffic").current_severity().value
    travel_kitchen = riders.travel_to_kitchen_min(order.hub_distance_km, weather, traffic)
    yield env.timeout(travel_kitchen)
    order.rider_arrived_kitchen_at = env.now
    event_log.record(
        env.now, "rider_at_kitchen",
        order_id=order.order_id, kitchen_id=order.kitchen_id, rider_id=rider.rider_id,
        payload={"travel_to_kitchen_min": round(travel_kitchen, 2)},
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

    weather = getattr(env, "_weather").current_severity().value
    traffic = getattr(env, "_traffic").current_severity().value
    travel_customer = riders.travel_to_customer_min(order.distance_km, weather, traffic)
    yield env.timeout(travel_customer)

    order.delivered_at = env.now
    order.status = OrderStatus.COMPLETED
    event_log.record(
        env.now, "rider_delivered",
        order_id=order.order_id, kitchen_id=order.kitchen_id, rider_id=rider.rider_id,
        payload={"travel_customer_min": round(travel_customer, 2)},
    )
    riders.release(rider.rider_id, at=env.now)
