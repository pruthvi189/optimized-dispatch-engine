"""Pydantic request models for the Phase 4 API."""

from pydantic import BaseModel, Field


class FeaturesIn(BaseModel):
    """Features for a prep-time prediction (mirrors Predictor.predict).

    traffic_severity is not a prep-model feature (prep duration depends on
    weather/workload/staffing only), so it is intentionally absent here.
    """

    items_count: int = Field(ge=1, le=20)
    workload_at_placement: float = Field(ge=0, le=50)
    staff_level: int = Field(ge=1, le=5)
    hour_of_day: int = Field(ge=0, le=23)
    order_complexity: str = Field(pattern="simple|standard|complex")
    weather_severity: str = Field(pattern="clear|rain|storm")
    kitchen_id: int = Field(ge=1)


class DispatchIn(BaseModel):
    """Inputs for an offline dispatch decision (no simulation side effects).

    ``hour_of_day`` sets the simulated decision clock (the "now" the policy
    reasons about); weather/traffic are taken from the live engine when one is
    running, else clear/low defaults.
    """

    items_count: int = Field(default=2, ge=1, le=20)
    kitchen_id: int = Field(default=1, ge=1)
    distance_km: float = Field(default=3.0, gt=0, le=20)
    hour_of_day: int | None = Field(default=None, ge=0, le=23)


class StepIn(BaseModel):
    minutes: float = Field(default=5.0, gt=0)


class RunnerConfigIn(BaseModel):
    scenario: str = Field(default="normal", pattern="normal|lunch_rush|rain|low_staffing|traffic_spike")
    seed: int = 42
    policy: str = Field(default="nearest_heuristic", pattern="immediate|adaptive|nearest_heuristic|joint_optimizer|optimized_kitchen|nearest_kitchen")
    days: int = Field(default=1, ge=1, le=30)
    speed: float | None = Field(default=60.0, gt=0)
    step_minutes: float = Field(default=1.0, gt=0)
