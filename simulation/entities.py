from enum import Enum
from dataclasses import dataclass, field


class OrderStatus(str, Enum):
    PLACED = "placed"
    PREPPING = "prepping"
    READY = "ready"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RiderStatus(str, Enum):
    IDLE = "idle"
    ASSIGNED = "assigned"
    DELIVERING = "delivering"
    UNAVAILABLE = "unavailable"


class WeatherSeverity(str, Enum):
    CLEAR = "clear"
    RAIN = "rain"
    STORM = "storm"


class TrafficSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HEAVY = "heavy"


class OrderComplexity(str, Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"


def complexity_from_items(items: int) -> OrderComplexity:
    if items <= 2:
        return OrderComplexity.SIMPLE
    if items <= 5:
        return OrderComplexity.STANDARD
    return OrderComplexity.COMPLEX


@dataclass
class Order:
    order_id: int
    kitchen_id: int
    placed_at: float
    items: int
    complexity: OrderComplexity
    distance_km: float
    promised_delivery_min: float = 10.0
    status: OrderStatus = OrderStatus.PLACED
    workload_at_placement: int = 0
    staff_level: int = 0
    weather_severity: str = WeatherSeverity.CLEAR.value
    traffic_severity: str = TrafficSeverity.LOW.value
    prep_started_at: float | None = None
    prep_finished_at: float | None = None
    actual_prep_duration_min: float | None = None
    rider_id: int | None = None
    cancel_reason: str | None = None
    # Phase 3 dispatch/delivery fields.
    dispatch_policy: str | None = None
    dispatch_at: float | None = None
    hub_distance_km: float | None = None
    travel_to_kitchen_min: float | None = None
    rider_arrived_kitchen_at: float | None = None
    pickup_at: float | None = None
    delivered_at: float | None = None
    eta_min: float | None = None
    predicted_prep_mean: float | None = None
    predicted_prep_low: float | None = None
    predicted_prep_high: float | None = None
    uncertainty: str | None = None
    risk_buffer_min: float | None = None
    decision_rationale: str | None = None


@dataclass
class Kitchen:
    kitchen_id: int
    staff_level: int
    resource: object = None
    current_orders: list = field(default_factory=list)
    failed: bool = False


@dataclass
class Rider:
    rider_id: int
    status: RiderStatus = RiderStatus.IDLE
    speed_kmh: float = 22.0
    location_km: float = 0.0
    assigned_at: float | None = None
    busy_min: float = 0.0


@dataclass
class WeatherState:
    severity: WeatherSeverity = WeatherSeverity.CLEAR
    changed_at: float = 0.0


@dataclass
class TrafficState:
    severity: TrafficSeverity = TrafficSeverity.LOW
    changed_at: float = 0.0
