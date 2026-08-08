from dataclasses import dataclass, field, asdict

from .eta import compute_eta
from .state import DispatchState


@dataclass
class DispatchDecision:
    """What a policy decided, plus everything needed to explain and audit it."""

    dispatch_at: float
    policy: str
    rationale: str
    eta: float = 0.0
    predicted_prep_mean: float | None = None
    predicted_prep_low: float | None = None
    predicted_prep_high: float | None = None
    uncertainty: str | None = None
    risk_buffer_min: float = 0.0
    travel_to_kitchen_min: float = 0.0
    hub_distance_km: float = 0.0
    inputs: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload.pop("inputs")
        payload["inputs"] = self.inputs
        return payload


class DispatchPolicy:
    """Base class. A policy decides WHEN to dispatch a rider for an order."""

    name = "base"

    def decide(self, order, state: DispatchState) -> DispatchDecision:
        raise NotImplementedError


class ImmediateDispatch(DispatchPolicy):
    """Anti-pattern baseline: dispatch a rider the moment the order is placed."""

    name = "immediate"

    def __init__(self, config):
        self.config = config

    def decide(self, order, state: DispatchState) -> DispatchDecision:
        eta = compute_eta(
            now=state.now,
            prep_mean=0.0,
            dispatch_at=state.now,
            travel_to_kitchen=state.travel_to_kitchen_min,
            pickup_time=self.config["dispatch"]["pickup_time_min"],
            distance_km=order.distance_km,
            weather=state.weather_severity,
            traffic=state.traffic_severity,
            config=self.config,
        )
        return DispatchDecision(
            dispatch_at=state.now,
            policy=self.name,
            rationale="immediate dispatch at placement",
            eta=eta,
            travel_to_kitchen_min=state.travel_to_kitchen_min,
            hub_distance_km=state.hub_distance_km,
        )


class AdaptiveDispatch(DispatchPolicy):
    """Times dispatch so the rider reaches the kitchen just as the order is
    predicted ready, plus a dynamic risk buffer from prediction uncertainty
    and live kitchen congestion."""

    name = "adaptive"

    def __init__(self, predictor, config):
        self.predictor = predictor
        self.config = config

    def decide(self, order, state: DispatchState) -> DispatchDecision:
        d = self.config["dispatch"]
        features = {
            "items_count": order.items,
            "workload_at_placement": state.kitchen_queue_lens.get(order.kitchen_id, 0),
            "staff_level": order.staff_level,
            "hour_of_day": int(state.now // 60) % 24,
            "order_complexity": order.complexity.value,
            "weather_severity": state.weather_severity,
            "traffic_severity": state.traffic_severity,
            "kitchen_id": order.kitchen_id,
        }
        pred = self.predictor.predict(features)

        d = self.config["dispatch"]
        buffer = d["risk_buffer_min"][pred["uncertainty"]]
        buffer += d["congestion_buffer_per_order"] * state.kitchen_queue_lens.get(order.kitchen_id, 0)

        prep_mean = pred["prep_mean"]
        quantile = d.get("dispatch_quantile", "low")
        prep_est = pred["prep_low"] if quantile == "low" else prep_mean
        dispatch_at = state.now + prep_est - state.travel_to_kitchen_min - buffer
        dispatch_at = max(state.now, dispatch_at)

        eta = compute_eta(
            now=state.now,
            prep_mean=prep_mean,
            dispatch_at=dispatch_at,
            travel_to_kitchen=state.travel_to_kitchen_min,
            pickup_time=d["pickup_time_min"],
            distance_km=order.distance_km,
            weather=state.weather_severity,
            traffic=state.traffic_severity,
            config=self.config,
        )
        rationale = (
            f"prep_{quantile}={prep_est:.2f} (mean={prep_mean:.2f}), "
            f"travel_to_kitchen={state.travel_to_kitchen_min:.2f}, "
            f"buffer={buffer:.2f} ({pred['uncertainty']})"
        )
        return DispatchDecision(
            dispatch_at=dispatch_at,
            policy=self.name,
            rationale=rationale,
            eta=eta,
            predicted_prep_mean=prep_mean,
            predicted_prep_low=pred["prep_low"],
            predicted_prep_high=pred["prep_high"],
            uncertainty=pred["uncertainty"],
            risk_buffer_min=buffer,
            travel_to_kitchen_min=state.travel_to_kitchen_min,
            hub_distance_km=state.hub_distance_km,
            inputs=features,
        )


def make_policy(name: str, predictor, config) -> DispatchPolicy:
    if name == "immediate":
        return ImmediateDispatch(config)
    if name == "adaptive":
        if predictor is None:
            raise ValueError(
                "adaptive policy requires a predictor; run Phase 2 training "
                "(python train_models.py) first and pass --predictor-dir"
            )
        return AdaptiveDispatch(predictor, config)
    raise ValueError(f"unknown dispatch policy: {name}")
