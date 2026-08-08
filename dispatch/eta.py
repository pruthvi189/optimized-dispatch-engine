def compute_eta(now, prep_mean, dispatch_at, travel_to_kitchen, pickup_time,
                distance_km, weather, traffic, config) -> float:
    """Estimated delivery time (sim minutes) at decision time.

    rider_arrival = dispatch_at + travel_to_kitchen
    ready_est     = now + prep_mean
    food_wait     = rider arriving early waits for food (max 0).
    eta           = rider_arrival + food_wait + pickup + travel_to_customer.
    """
    rider_arrival = dispatch_at + travel_to_kitchen
    ready_est = now + prep_mean
    food_wait = max(0.0, ready_est - rider_arrival)

    d = config["dispatch"]
    wf = d["weather_speed_factor"][weather]
    tf = d["traffic_speed_factor"][traffic]
    speed_kmh = config["riders"]["speed_kmh"]
    travel_customer = distance_km / speed_kmh * 60.0 * tf * wf

    return rider_arrival + food_wait + pickup_time + travel_customer
