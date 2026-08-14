from simulation.environment import forecast_traffic


def compute_eta(now, prep_mean, dispatch_at, travel_to_kitchen, pickup_time,
                distance_km, weather, traffic, config) -> float:
    """Estimated delivery time (sim minutes) at decision time.

    rider_arrival = dispatch_at + travel_to_kitchen
    ready_est     = now + prep_mean
    food_wait     = rider arriving early waits for food (max 0).
    eta           = rider_arrival + food_wait + pickup + travel_to_customer.

    The customer leg uses forecast traffic at the departure time (not the
    traffic observed at dispatch time), so the ETA reflects the expected
    congestion when the rider is actually on the road toward the customer.
    """
    rider_arrival = dispatch_at + travel_to_kitchen
    ready_est = now + prep_mean
    food_wait = max(0.0, ready_est - rider_arrival)

    d = config["dispatch"]
    wf = d["weather_speed_factor"][weather]
    speed_kmh = config["riders"]["speed_kmh"]
    base_travel_customer = distance_km / speed_kmh * 60.0
    leg_start = rider_arrival + food_wait + pickup_time
    sev = forecast_traffic(leg_start, base_travel_customer)
    tf = d["traffic_speed_factor"][sev.value]
    travel_customer = base_travel_customer * tf * wf

    return rider_arrival + food_wait + pickup_time + travel_customer
