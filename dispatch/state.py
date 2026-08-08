from dataclasses import dataclass


@dataclass
class DispatchState:
    """Immutable snapshot of everything a policy may read. Built fresh per order
    at decision time so decisions are pure functions of (order, state, config)."""

    now: float
    kitchen_queue_lens: dict
    idle_rider_count: int
    weather_severity: str
    traffic_severity: str
    hub_distance_km: float
    travel_to_kitchen_min: float

    @classmethod
    def from_env(cls, env, kitchens, riders, hub_distance_km, config):
        weather = getattr(env, "_weather").current_severity().value
        traffic = getattr(env, "_traffic").current_severity().value
        return cls(
            now=env.now,
            kitchen_queue_lens={k.kitchen_id: len(k.current_orders) for k in kitchens},
            idle_rider_count=riders.idle_count(),
            weather_severity=weather,
            traffic_severity=traffic,
            hub_distance_km=hub_distance_km,
            travel_to_kitchen_min=riders.travel_to_kitchen_min(
                hub_distance_km, weather, traffic
            ),
        )
