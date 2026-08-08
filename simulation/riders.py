from .entities import Rider, RiderStatus


def travel_time_min(distance_km: float, speed_kmh: float, traffic_factor: float = 1.0, weather_factor: float = 1.0) -> float:
    """Travel duration = distance / speed, scaled by traffic and weather."""
    if speed_kmh <= 0:
        raise ValueError("rider speed must be positive")
    return distance_km / speed_kmh * 60.0 * traffic_factor * weather_factor


class RiderPool:
    """Holds riders, availability, and travel estimates. Dispatch timing lives
    in the policy; the pool only assigns riders and tracks busy time."""

    def __init__(self, config):
        self.config = config
        self.riders = [
            Rider(rider_id=i + 1, speed_kmh=config["riders"].get("speed_kmh", 25.0))
            for i in range(config["riders"]["count"])
        ]
        self._idle_index = 0

    def idle_count(self) -> int:
        return sum(1 for r in self.riders if r.status == RiderStatus.IDLE)

    def assign_next_idle(self, at: float | None = None) -> Rider | None:
        for r in self.riders:
            if r.status == RiderStatus.IDLE:
                r.status = RiderStatus.ASSIGNED
                r.assigned_at = at
                return r
        return None

    def mark_unavailable(self, rider_id: int):
        for r in self.riders:
            if r.rider_id == rider_id:
                r.status = RiderStatus.UNAVAILABLE
                return

    def release(self, rider_id: int, at: float | None = None):
        for r in self.riders:
            if r.rider_id == rider_id:
                if at is not None and r.assigned_at is not None:
                    r.busy_min += max(0.0, at - r.assigned_at)
                r.assigned_at = None
                r.status = RiderStatus.IDLE
                return

    def sample_hub_distance(self, rng) -> float:
        """Distance from the rider hub to the order's kitchen, per order."""
        lo, hi = self.config["dispatch"]["hub_distance_range_km"]
        return float(rng.uniform(lo, hi))

    def travel_to_kitchen_min(self, distance_km: float, weather: str, traffic: str) -> float:
        d = self.config["dispatch"]
        wf = d["weather_speed_factor"][weather]
        tf = d["traffic_speed_factor"][traffic]
        speed = self.config["riders"]["speed_kmh"]
        return travel_time_min(distance_km, speed, traffic_factor=tf, weather_factor=wf)

    def travel_to_customer_min(self, distance_km: float, weather: str, traffic: str) -> float:
        return self.travel_to_kitchen_min(distance_km, weather, traffic)
