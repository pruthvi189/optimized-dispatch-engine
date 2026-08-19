from dataclasses import dataclass, field


@dataclass
class KitchenCandidate:
    """Snapshot of one kitchen's state for policy evaluation."""
    kitchen_id: int
    queue_len: int
    staff_level: int
    distance_km: float  # customer-to-kitchen distance


@dataclass
class RiderCandidate:
    """Snapshot of one idle rider's state for policy evaluation."""
    rider_id: int
    x: float
    y: float
    dist_to_kitchens: list[float]  # [dist_to_k1, dist_to_k2, dist_to_k3]


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
    # Kitchen selection fields (Phase 9). None when spatial model is disabled.
    kitchen_candidates: list[KitchenCandidate] | None = None
    # Rider selection fields (Phase 10). None when spatial model is disabled.
    rider_candidates: list[RiderCandidate] | None = None

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

    @classmethod
    def from_env_with_spatial(cls, env, kitchens, riders, hub_distance_km, config,
                              distance_to_kitchens: list[float],
                              kitchen_locations: list[tuple[float, float]] | None = None):
        """Build state with kitchen candidates and (optionally) rider candidates.

        `distance_to_kitchens` is a list of distances (km) from the customer
        to each kitchen, in kitchen-ID order (index 0 → kitchen 1).

        When `kitchen_locations` is provided, rider candidates are populated
        with each idle rider's position and distance to each kitchen.
        """
        weather = getattr(env, "_weather").current_severity().value
        traffic = getattr(env, "_traffic").current_severity().value
        candidates = [
            KitchenCandidate(
                kitchen_id=k.kitchen_id,
                queue_len=len(k.current_orders),
                staff_level=k.staff_level,
                distance_km=distance_to_kitchens[k.kitchen_id - 1],
            )
            for k in kitchens
        ]

        rider_candidates = None
        if kitchen_locations is not None:
            raw = riders.idle_riders_info(kitchen_locations)
            rider_candidates = [
                RiderCandidate(
                    rider_id=r["rider_id"],
                    x=r["x"],
                    y=r["y"],
                    dist_to_kitchens=r["dist_to_kitchens"],
                )
                for r in raw
            ]

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
            kitchen_candidates=candidates,
            rider_candidates=rider_candidates,
        )
