from dataclasses import dataclass, field, asdict

from .eta import compute_eta
from .state import DispatchState
from simulation.environment import forecast_traffic


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
    and live kitchen congestion. Dispatch is also promise-aware: if food-
    readiness timing would make the order miss its delivery SLA, the rider is
    dispatched earlier (toward the latest safe time) so the promise stays
    reachable."""

    name = "adaptive"

    def __init__(self, predictor, config):
        self.predictor = predictor
        self.config = config

    def _customer_leg(self, order, state, leg_start: float | None = None) -> float:
        """Predicted rider travel from kitchen to customer (minutes). Uses the
        forecast traffic at the leg's start time rather than dispatch-time
        traffic. `leg_start` is when the rider departs the kitchen (sim
        minutes); defaults to `now` when the start time is unknown."""
        d = self.config["dispatch"]
        wf = d["weather_speed_factor"][state.weather_severity]
        speed_kmh = self.config["riders"]["speed_kmh"]
        base = order.distance_km / speed_kmh * 60.0
        sev = forecast_traffic(leg_start if leg_start is not None else state.now, base)
        tf = d["traffic_speed_factor"][sev.value]
        return base * tf * wf

    def _clamp_to_promise(self, order, state, prep_mean, dispatch_at, travel) -> float:
        """Promise-aware clamp: never dispatch so late that the delivery SLA is
        already out of reach. Returns (dispatch_at, clamped_bool)."""
        d = self.config["dispatch"]
        promise = d["promised_delivery_min"]
        pickup = d["pickup_time_min"]
        # When does the rider actually start the customer leg? Arrive at the
        # kitchen, wait out any food-readiness gap, then pick up.
        rider_arrival = dispatch_at + travel
        food_wait = max(0.0, state.now + prep_mean - rider_arrival)
        leg_start = rider_arrival + food_wait + pickup
        cust_leg = self._customer_leg(order, state, leg_start=leg_start)

        # Latest kitchen arrival that still meets the SLA (assuming the rider
        # does not wait on food): budget = placed + promise - pickup - cust_leg.
        budget = order.placed_at + promise - pickup - cust_leg
        latest_dispatch = budget - travel

        # If even immediate dispatch cannot have the food ready in time, send
        # the rider now anyway (best effort).
        if state.now + prep_mean > budget:
            return state.now, True
        return min(dispatch_at, latest_dispatch), dispatch_at > latest_dispatch

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

        dispatch_at, clamped = self._clamp_to_promise(
            order, state, prep_mean, dispatch_at, state.travel_to_kitchen_min
        )

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
        if clamped:
            rationale += f", promise-clamped to {dispatch_at:.2f}"
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
