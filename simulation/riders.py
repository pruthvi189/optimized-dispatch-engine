import numpy as np

from .entities import Rider, RiderStatus


def travel_time_min(distance_km: float, speed_kmh: float, traffic_factor: float = 1.0, weather_factor: float = 1.0) -> float:
    """Travel duration = distance / speed, scaled by traffic and weather."""
    if speed_kmh <= 0:
        raise ValueError("rider speed must be positive")
    return distance_km / speed_kmh * 60.0 * traffic_factor * weather_factor


class RiderPool:
    """Holds riders, availability, and travel estimates. Dispatch timing lives
    in the policy; the pool only assigns riders and tracks busy time."""

    def __init__(self, config, initial_positions=None, rider_to_kitchen_matrix=None):
        self.config = config
        self.riders = [
            Rider(rider_id=i + 1, speed_kmh=config["riders"].get("speed_kmh", 22.0))
            for i in range(config["riders"]["count"])
        ]
        # Assign initial positions. If provided (from engine), use those exact
        # positions so paired experiments share identical rider layouts.
        if initial_positions is not None:
            for r, (x, y) in zip(self.riders, initial_positions):
                r.x = x
                r.y = y

        # Rider→kitchen distance matrix. Shape: (n_riders, n_kitchens).
        # Generated once per seed by the engine. This is the SINGLE source
        # of truth for rider→kitchen travel distances used by both policies
        # (for scoring) and delivery_process (for simulation).
        self.rider_to_kitchen_matrix = rider_to_kitchen_matrix

    def idle_count(self) -> int:
        return sum(1 for r in self.riders if r.status == RiderStatus.IDLE)

    def assign_next_idle(self, at: float | None = None) -> Rider | None:
        for r in self.riders:
            if r.status == RiderStatus.IDLE:
                r.status = RiderStatus.ASSIGNED
                r.assigned_at = at
                return r
        return None

    def assign_by_id(self, rider_id: int, at: float | None = None) -> Rider | None:
        """Assign a specific rider by ID (for joint-optimized policies)."""
        for r in self.riders:
            if r.rider_id == rider_id and r.status == RiderStatus.IDLE:
                r.status = RiderStatus.ASSIGNED
                r.assigned_at = at
                return r
        return None

    def mark_unavailable(self, rider_id: int):
        for r in self.riders:
            if r.rider_id == rider_id:
                r.status = RiderStatus.UNAVAILABLE
                return

    def release(self, rider_id: int, at: float | None = None, new_x: float | None = None, new_y: float | None = None):
        """Release a rider back to idle. Optionally update position (after delivery)."""
        for r in self.riders:
            if r.rider_id == rider_id:
                if at is not None and r.assigned_at is not None:
                    r.busy_min += max(0.0, at - r.assigned_at)
                r.assigned_at = None
                r.status = RiderStatus.IDLE
                if new_x is not None and new_y is not None:
                    r.x = new_x
                    r.y = new_y
                return

    def sample_hub_distance(self, rng) -> float:
        """Distance from the rider hub to the order's kitchen, per order.

        Retained for backward compatibility with policies that use hub_distance_km.
        The new spatial model policies use rider positions directly instead.
        """
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

    def get_rider_kitchen_distance(self, rider_id: int, kitchen_id: int) -> float:
        """Look up the rider→kitchen distance from the matrix.

        rider_id and kitchen_id are 1-indexed. The matrix is 0-indexed.
        Returns the distance in km. Falls back to hub_distance_range_km
        if no matrix is available (legacy mode).
        """
        if self.rider_to_kitchen_matrix is not None:
            return float(self.rider_to_kitchen_matrix[rider_id - 1][kitchen_id - 1])
        # Fallback: sample from hub_distance_range_km (legacy mode).
        lo, hi = self.config["dispatch"]["hub_distance_range_km"]
        rng = np.random.default_rng()
        return float(rng.uniform(lo, hi))

    def idle_riders_info(self, kitchen_locations: list[tuple[float, float]]) -> list[dict]:
        """Return idle riders with their positions and distances to each kitchen.

        Each entry: {rider_id, x, y, dist_to_kitchens: [d1, d2, d3]}.
        Used by policies to evaluate rider-kitchen combinations.

        When the rider→kitchen matrix is available, dist_to_kitchens is
        read from the matrix (single source of truth). Otherwise falls
        back to spatial Euclidean distance.
        """
        result = []
        for r in self.riders:
            if r.status != RiderStatus.IDLE:
                continue
            if self.rider_to_kitchen_matrix is not None:
                dists = [
                    float(self.rider_to_kitchen_matrix[r.rider_id - 1][k])
                    for k in range(len(kitchen_locations))
                ]
            else:
                from .spatial import Point2D
                rider_pos = Point2D(r.x, r.y)
                dists = [
                    rider_pos.distance_to(Point2D(kx, ky))
                    for kx, ky in kitchen_locations
                ]
            result.append({
                "rider_id": r.rider_id,
                "x": r.x,
                "y": r.y,
                "dist_to_kitchens": dists,
            })
        return result
