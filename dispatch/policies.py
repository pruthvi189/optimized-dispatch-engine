from dataclasses import dataclass, field, asdict

from .eta import compute_eta
from .state import DispatchState, KitchenCandidate, RiderCandidate
from simulation.environment import forecast_traffic
from simulation.kitchen import estimate_prep_for_order


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
    # Kitchen selection fields (Phase 9).
    selected_kitchen_id: int | None = None
    selected_kitchen_distance: float | None = None
    # Rider selection field (Phase 10).
    selected_rider_id: int | None = None
    # Order attributes for prep estimation (items/complexity).
    items: int | None = None
    complexity: str | None = None

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


class OptimizedKitchenDispatch(DispatchPolicy):
    """Selects the best kitchen for each order, then dispatches immediately.

    For each candidate kitchen, estimates the total delivery time:
        estimated_delivery = current_time
            + queue_wait (approximate from queue length * avg_prep / staff)
            + avg_prep_time
            + rider_travel_to_kitchen (from hub)
            + pickup_time
            + rider_travel_to_customer (from kitchen to customer)

    Picks the kitchen with the lowest estimated delivery time.
    Dispatch timing is immediate (same as baseline), but the kitchen
    selection is optimized.

    Weighting: the candidate evaluation uses configurable weights so that
    kitchen load balancing is explicit and auditable, not silently tuned.
    """

    name = "optimized_kitchen"

    def __init__(self, config):
        self.config = config
        ks = config.get("kitchen_selection", {})
        self.weight_distance = ks.get("weight_distance", 1.0)
        self.weight_queue = ks.get("weight_queue", 1.5)
        self.weight_staff = ks.get("weight_staff", 1.0)
        self.avg_prep_min = ks.get("avg_prep_min", 7.0)

    def _estimate_delivery_time(self, candidate: KitchenCandidate, state: DispatchState,
                                config: dict) -> float:
        """Estimate end-to-end delivery time if we send the order to this kitchen.

        Components:
          1. Queue wait: approximate as queue_len * avg_prep / staff_level
          2. Prep time: avg_prep_min (deterministic estimate)
          3. Rider travel to kitchen: hub_distance / speed (weather/traffic adjusted)
          4. Pickup: fixed pickup_time_min
          5. Rider travel to customer: distance / speed (weather/traffic adjusted)
        """
        d = config["dispatch"]
        speed_kmh = config["riders"]["speed_kmh"]

        # 1. Queue wait estimate.
        effective_staff = max(1, candidate.staff_level)
        queue_wait = candidate.queue_len * self.avg_prep_min / effective_staff

        # 2. Prep time estimate.
        prep_time = self.avg_prep_min

        # 3. Rider travel to kitchen.
        wf = d["weather_speed_factor"][state.weather_severity]
        tf = d["traffic_speed_factor"][state.traffic_severity]
        travel_to_kitchen = candidate.distance_km / speed_kmh * 60.0 * tf * wf
        # Also account for hub-to-kitchen travel (rider starts at hub).
        hub_travel = state.hub_distance_km / speed_kmh * 60.0 * tf * wf

        # 4. Pickup.
        pickup = d["pickup_time_min"]

        # 5. Rider travel to customer.
        #    Use forecast traffic at the estimated departure time.
        base_cust = candidate.distance_km / speed_kmh * 60.0
        depart_time = state.now + hub_travel + queue_wait + prep_time + pickup
        cust_sev = forecast_traffic(depart_time, base_cust)
        tf_cust = d["traffic_speed_factor"][cust_sev.value]
        travel_to_customer = base_cust * tf_cust * wf

        total = hub_travel + queue_wait + prep_time + pickup + travel_to_customer
        return total

    def decide(self, order, state: DispatchState) -> DispatchDecision:
        if not state.kitchen_candidates:
            raise ValueError("OptimizedKitchenDispatch requires kitchen_candidates in state")

        config = self.config
        d = config["dispatch"]
        wf = d["weather_speed_factor"][state.weather_severity]
        tf = d["traffic_speed_factor"][state.traffic_severity]
        speed_kmh = config["riders"]["speed_kmh"]

        # Evaluate each kitchen.
        best_candidate = None
        best_score = float("inf")
        evaluations = []

        for c in state.kitchen_candidates:
            delivery_est = self._estimate_delivery_time(c, state, config)
            # Apply weighting: distance has an explicit weight, queue has its own.
            # The score = weighted_sum used for ranking (not the raw estimate).
            score = (
                self.weight_distance * c.distance_km
                + self.weight_queue * c.queue_len
                + self.weight_staff * (1.0 / max(1, c.staff_level))
            )
            evaluations.append({
                "kitchen_id": c.kitchen_id,
                "distance_km": round(c.distance_km, 2),
                "queue_len": c.queue_len,
                "staff_level": c.staff_level,
                "delivery_est_min": round(delivery_est, 2),
                "score": round(score, 3),
            })
            if delivery_est < best_score:
                best_score = delivery_est
                best_candidate = c

        # Fallback: pick by score if delivery_est is tied.
        if best_candidate is None:
            best_candidate = min(state.kitchen_candidates, key=lambda c: (
                self.weight_distance * c.distance_km
                + self.weight_queue * c.queue_len
            ))

        # Compute ETA using the selected kitchen's distance.
        selected_distance = best_candidate.distance_km
        hub_travel_min = state.hub_distance_km / speed_kmh * 60.0 * tf * wf
        travel_to_kitchen_min = selected_distance / speed_kmh * 60.0 * tf * wf
        pickup_time = d["pickup_time_min"]
        prep_est = self.avg_prep_min

        # Customer leg with forecast traffic.
        base_cust = selected_distance / speed_kmh * 60.0
        depart_time = state.now + hub_travel_min + prep_est + pickup_time
        cust_sev = forecast_traffic(depart_time, base_cust)
        tf_cust = d["traffic_speed_factor"][cust_sev.value]
        travel_customer = base_cust * tf_cust * wf

        eta = state.now + hub_travel_min + prep_est + pickup_time + travel_customer

        # Dispatch immediately — kitchen selection is the optimization, not timing.
        dispatch_at = state.now

        rationale = (
            f"selected kitchen {best_candidate.kitchen_id} "
            f"(dist={best_candidate.distance_km:.2f}km, "
            f"queue={best_candidate.queue_len}, "
            f"staff={best_candidate.staff_level}), "
            f"delivery_est={best_score:.2f}min"
        )
        return DispatchDecision(
            dispatch_at=dispatch_at,
            policy=self.name,
            rationale=rationale,
            eta=eta,
            predicted_prep_mean=prep_est,
            travel_to_kitchen_min=travel_to_kitchen_min,
            hub_distance_km=state.hub_distance_km,
            selected_kitchen_id=best_candidate.kitchen_id,
            selected_kitchen_distance=best_candidate.distance_km,
            inputs={"evaluations": evaluations},
        )


class NearestKitchenDispatch(DispatchPolicy):
    """Selects the kitchen closest to the customer by straight-line distance.
    No queue awareness, no load balancing — purely geometric.
    Dispatch timing is immediate (same as baseline)."""

    name = "nearest_kitchen"

    def __init__(self, config):
        self.config = config

    def decide(self, order, state: DispatchState) -> DispatchDecision:
        if not state.kitchen_candidates:
            raise ValueError("NearestKitchenDispatch requires kitchen_candidates in state")

        d = self.config["dispatch"]
        speed_kmh = self.config["riders"]["speed_kmh"]
        wf = d["weather_speed_factor"][state.weather_severity]
        tf = d["traffic_speed_factor"][state.traffic_severity]

        best = min(state.kitchen_candidates, key=lambda c: c.distance_km)
        selected_distance = best.distance_km

        hub_travel_min = state.hub_distance_km / speed_kmh * 60.0 * tf * wf
        travel_to_kitchen_min = selected_distance / speed_kmh * 60.0 * tf * wf
        pickup_time = d["pickup_time_min"]

        base_cust = selected_distance / speed_kmh * 60.0
        depart_time = state.now + hub_travel_min + pickup_time
        cust_sev = forecast_traffic(depart_time, base_cust)
        tf_cust = d["traffic_speed_factor"][cust_sev.value]
        travel_customer = base_cust * tf_cust * wf

        eta = state.now + hub_travel_min + pickup_time + travel_customer

        return DispatchDecision(
            dispatch_at=state.now,
            policy=self.name,
            rationale=f"nearest kitchen {best.kitchen_id} (dist={selected_distance:.2f}km)",
            eta=eta,
            travel_to_kitchen_min=travel_to_kitchen_min,
            hub_distance_km=state.hub_distance_km,
            selected_kitchen_id=best.kitchen_id,
            selected_kitchen_distance=best.distance_km,
            inputs={"evaluations": [
                {"kitchen_id": c.kitchen_id, "distance_km": round(c.distance_km, 2)}
                for c in state.kitchen_candidates
            ]},
        )


class NearestHeuristicDispatch(DispatchPolicy):
    """Baseline: nearest kitchen + nearest rider.

    1. Select the kitchen with the shortest customer→kitchen distance.
    2. Select the idle rider with the shortest rider→chosen_kitchen distance.
    3. Dispatch immediately.

    No queue awareness, no predictive scoring — a simple, credible baseline.
    """

    name = "nearest_heuristic"

    def __init__(self, config):
        self.config = config

    def _estimate_queue_wait(self, kitchen_candidate: KitchenCandidate) -> float:
        """Estimate wait before this order can start prep.

        Uses the same resource model as the simulator: the kitchen has a
        Resource with capacity = staff_level. Orders already being processed
        (up to staff_level) don't cause wait. Only orders beyond that
        capacity contribute to queue wait.

        estimated_wait = max(0, n - staff) * avg_prep / staff
        """
        ks = self.config.get("kitchen_selection", {})
        avg_prep = ks.get("avg_prep_min", 7.0)
        n = kitchen_candidate.queue_len
        s = max(1, kitchen_candidate.staff_level)
        return max(0, n - s) * avg_prep / s

    def decide(self, order, state: DispatchState) -> DispatchDecision:
        if not state.kitchen_candidates:
            raise ValueError("NearestHeuristicDispatch requires kitchen_candidates in state")

        d = self.config["dispatch"]
        speed_kmh = self.config["riders"]["speed_kmh"]
        ks = self.config.get("kitchen_selection", {})
        wf = d["weather_speed_factor"][state.weather_severity]
        tf = d["traffic_speed_factor"][state.traffic_severity]

        # 1. Pick nearest kitchen to customer.
        best_kitchen = min(state.kitchen_candidates, key=lambda c: c.distance_km)
        kitchen_distance = best_kitchen.distance_km
        kitchen_idx = best_kitchen.kitchen_id - 1

        # 2. Pick nearest rider to that kitchen (if any idle).
        best_rider = None
        rider_to_kitchen_dist = 0.0
        if state.rider_candidates:
            best_rider = min(
                state.rider_candidates,
                key=lambda r: r.dist_to_kitchens[kitchen_idx],
            )
            rider_to_kitchen_dist = best_rider.dist_to_kitchens[kitchen_idx]

        # 3. Order-specific prep estimate based on complexity and item count.
        prep_est = estimate_prep_for_order(order)
        prep_time = prep_est["prep_mean"]

        # Compute time components for ETA and rationale.
        rider_to_kitchen_min = rider_to_kitchen_dist / speed_kmh * 60.0 * tf * wf
        queue_wait = self._estimate_queue_wait(best_kitchen)
        pickup_time = d["pickup_time_min"]
        base_cust = kitchen_distance / speed_kmh * 60.0
        depart_time = state.now + rider_to_kitchen_min + queue_wait + prep_time + pickup_time
        cust_sev = forecast_traffic(depart_time, base_cust)
        tf_cust = d["traffic_speed_factor"][cust_sev.value]
        travel_customer = base_cust * tf_cust * wf

        eta = state.now + rider_to_kitchen_min + queue_wait + prep_time + pickup_time + travel_customer

        rationale = (
            f"kitchen {best_kitchen.kitchen_id} (dist={kitchen_distance:.2f}km)"
            + (f", rider {best_rider.rider_id} (rider_to_kitchen={rider_to_kitchen_dist:.2f}km)"
               if best_rider else ", no idle riders (will wait)")
        )
        return DispatchDecision(
            dispatch_at=state.now,
            policy=self.name,
            rationale=rationale,
            eta=eta,
            predicted_prep_mean=prep_time,
            predicted_prep_low=prep_est["prep_low"],
            predicted_prep_high=prep_est["prep_high"],
            uncertainty=prep_est["uncertainty"],
            travel_to_kitchen_min=rider_to_kitchen_min,
            hub_distance_km=state.hub_distance_km,
            selected_kitchen_id=best_kitchen.kitchen_id,
            selected_kitchen_distance=kitchen_distance,
            selected_rider_id=best_rider.rider_id if best_rider else None,
            items=order.items,
            complexity=order.complexity.value,
            inputs={
                "kitchen_id": best_kitchen.kitchen_id,
                "rider_id": best_rider.rider_id if best_rider else None,
                "rider_to_kitchen_dist": round(rider_to_kitchen_dist, 2),
                "kitchen_distance": round(kitchen_distance, 2),
            },
        )


class JointOptimizerDispatch(DispatchPolicy):
    """Joint kitchen + rider optimizer.

    Evaluates all feasible (kitchen, rider) combinations and selects the pair
    with the lowest estimated end-to-end delivery time.

    For each combination:
        total = max(rider_to_kitchen_time, queue_wait + prep_time)
              + pickup_time
              + expected_customer_travel_time

    Traffic and weather sensitivity scale the corresponding speed factors:
        tf_adj = 1.0 + (tf_raw - 1.0) * traffic_sensitivity
        wf_adj = 1.0 + (wf_raw - 1.0) * weather_sensitivity

    Kitchen wait uses the same resource model as the simulator (accounts for
    workload_factor per queued order from config and staffing_factor 1.25x
    for staff < 3).

    Traffic forecast for the customer leg uses the deterministic hour-of-day
    baseline at the estimated departure time (same function the simulator uses
    for realized traffic — forecast_traffic from environment.py).
    """

    name = "joint_optimizer"

    def __init__(self, config):
        self.config = config
        ks = config.get("kitchen_selection", {})
        self.avg_prep_min = ks.get("avg_prep_min", 7.0)
        opt = ks.get("optimizer", {})
        self.weight_queue_wait = opt.get("weight_queue_wait", 1.0)
        self.weight_rider_to_kitchen = opt.get("weight_rider_to_kitchen", 1.0)
        self.weight_customer_travel = opt.get("weight_customer_travel", 1.0)
        self.traffic_sensitivity = opt.get("traffic_sensitivity", 1.0)
        self.weather_sensitivity = opt.get("weather_sensitivity", 1.0)
        prep = config.get("prep", {})
        self.workload_factor_per_order = prep.get("workload_factor_per_order", 0.027)
        self.staff_threshold = prep.get("staff_threshold", 3)
        self.staffing_factor = prep.get("staffing_factor", 1.25)

    def _estimate_queue_wait(self, kitchen_candidate: KitchenCandidate, prep_time: float) -> float:
        """Estimate wait before this order can start prep.

        Uses the simulator's actual resource model: the kitchen has a
        Resource with capacity = staff_level. Only orders beyond that
        capacity contribute to queue wait.

        Accounts for workload_factor (from config) and staffing_factor
        (1.25x for staff < staff_threshold) that the simulator applies.
        """
        n = kitchen_candidate.queue_len
        s = max(1, kitchen_candidate.staff_level)
        if n <= s:
            return 0.0
        waiting = n - s
        # Average queue position of orders this new order waits for:
        # positions s+1 through n, average = s + (waiting + 1) / 2
        avg_pos = s + (waiting + 1) / 2.0
        # Workload factor same as simulator: 1.0 + workload_factor_per_order * queue_position
        workload_factor = 1.0 + self.workload_factor_per_order * avg_pos
        # Staffing factor: 1.25x if staff < threshold
        staffing_factor = self.staffing_factor if s < self.staff_threshold else 1.0
        # Each waiting order's effective prep time
        effective_prep = prep_time * workload_factor * staffing_factor
        return waiting * effective_prep / s

    def _estimate_delivery_for_pair(
        self,
        kitchen: KitchenCandidate,
        rider: RiderCandidate,
        kitchen_idx: int,
        state: DispatchState,
        prep_time: float,
    ) -> float:
        """Estimate total delivery time for one (kitchen, rider) pair.

        Uses the corrected ETA formula:
            max(rider_to_kitchen, queue_wait + prep_time) + pickup + customer_travel

        Traffic and weather sensitivity scale the corresponding speed factors:
        - traffic_sensitivity amplifies/dampens traffic speed factors
        - weather_sensitivity amplifies/dampens weather speed factors

        weight_customer_travel scales the customer-leg travel component.
        """
        d = self.config["dispatch"]
        speed_kmh = self.config["riders"]["speed_kmh"]
        wf_raw = d["weather_speed_factor"][state.weather_severity]
        tf_raw = d["traffic_speed_factor"][state.traffic_severity]

        # Apply sensitivity scaling: base=1.0, scaled = 1.0 + (raw - 1.0) * sensitivity
        wf = 1.0 + (wf_raw - 1.0) * self.weather_sensitivity
        tf = 1.0 + (tf_raw - 1.0) * self.traffic_sensitivity

        # 1. Rider -> kitchen travel.
        rider_to_kitchen_dist = rider.dist_to_kitchens[kitchen_idx]
        rider_to_kitchen_min = rider_to_kitchen_dist / speed_kmh * 60.0 * tf * wf

        # 2. Expected kitchen wait (resource model).
        queue_wait = self._estimate_queue_wait(kitchen, prep_time)

        # 3. Pickup time.
        pickup_time = d["pickup_time_min"]

        # 4. Kitchen -> customer travel with forecast traffic at departure.
        kitchen_distance = kitchen.distance_km
        base_cust = kitchen_distance / speed_kmh * 60.0
        depart_time = state.now + max(rider_to_kitchen_min, queue_wait + prep_time) + pickup_time
        cust_sev = forecast_traffic(depart_time, base_cust)
        tf_cust_raw = d["traffic_speed_factor"][cust_sev.value]
        tf_cust = 1.0 + (tf_cust_raw - 1.0) * self.traffic_sensitivity
        travel_customer = base_cust * tf_cust * wf

        # Correct formula: max(rider_to_kitchen, queue_wait + prep_time) + pickup + customer_travel
        # This accounts for rider waiting for food OR food waiting for rider.
        return (
            max(rider_to_kitchen_min, queue_wait + prep_time)
            + pickup_time
            + self.weight_customer_travel * travel_customer
        )

    def decide(self, order, state: DispatchState) -> DispatchDecision:
        if not state.kitchen_candidates:
            raise ValueError("JointOptimizerDispatch requires kitchen_candidates in state")

        # Order-specific prep estimate.
        prep_est = estimate_prep_for_order(order)
        prep_time = prep_est["prep_mean"]

        best_total = float("inf")
        best_kitchen = None
        best_rider = None
        evaluations = []

        if state.rider_candidates:
            for kitchen in state.kitchen_candidates:
                kitchen_idx = kitchen.kitchen_id - 1
                for rider in state.rider_candidates:
                    total = self._estimate_delivery_for_pair(
                        kitchen, rider, kitchen_idx, state, prep_time,
                    )
                    evaluations.append({
                        "kitchen_id": kitchen.kitchen_id,
                        "rider_id": rider.rider_id,
                        "rider_to_kitchen_km": round(rider.dist_to_kitchens[kitchen_idx], 2),
                        "kitchen_distance_km": round(kitchen.distance_km, 2),
                        "total_est_min": round(total, 2),
                    })
                    if total < best_total:
                        best_total = total
                        best_kitchen = kitchen
                        best_rider = rider

        # Fallback: pick nearest kitchen when no riders are idle.
        if best_kitchen is None:
            best_kitchen = min(state.kitchen_candidates, key=lambda c: c.distance_km)
            best_total = 0.0

        d = self.config["dispatch"]
        speed_kmh = self.config["riders"]["speed_kmh"]
        wf_raw = d["weather_speed_factor"][state.weather_severity]
        tf_raw = d["traffic_speed_factor"][state.traffic_severity]
        wf = 1.0 + (wf_raw - 1.0) * self.weather_sensitivity
        tf = 1.0 + (tf_raw - 1.0) * self.traffic_sensitivity

        kitchen_idx = best_kitchen.kitchen_id - 1
        rider_to_kitchen_dist = best_rider.dist_to_kitchens[kitchen_idx] if best_rider else 0.0
        rider_to_kitchen_min = rider_to_kitchen_dist / speed_kmh * 60.0 * tf * wf
        queue_wait = self._estimate_queue_wait(best_kitchen, prep_time)
        pickup_time = d["pickup_time_min"]
        base_cust = best_kitchen.distance_km / speed_kmh * 60.0
        depart_time = state.now + max(rider_to_kitchen_min, queue_wait + prep_time) + pickup_time
        cust_sev = forecast_traffic(depart_time, base_cust)
        tf_cust_raw = d["traffic_speed_factor"][cust_sev.value]
        tf_cust = 1.0 + (tf_cust_raw - 1.0) * self.traffic_sensitivity
        travel_customer = base_cust * tf_cust * wf
        eta = state.now + max(rider_to_kitchen_min, queue_wait + prep_time) + pickup_time + travel_customer

        rationale = (
            f"kitchen {best_kitchen.kitchen_id} (dist={best_kitchen.distance_km:.2f}km, "
            f"queue={best_kitchen.queue_len})"
            + (f", rider {best_rider.rider_id} (rider_to_kitchen={rider_to_kitchen_dist:.2f}km), "
               f"total_est={best_total:.2f}min"
               if best_rider else ", no idle riders (will wait)")
        )
        return DispatchDecision(
            dispatch_at=state.now,
            policy=self.name,
            rationale=rationale,
            eta=eta,
            predicted_prep_mean=prep_time,
            predicted_prep_low=prep_est["prep_low"],
            predicted_prep_high=prep_est["prep_high"],
            uncertainty=prep_est["uncertainty"],
            travel_to_kitchen_min=rider_to_kitchen_min,
            hub_distance_km=state.hub_distance_km,
            selected_kitchen_id=best_kitchen.kitchen_id,
            selected_kitchen_distance=best_kitchen.distance_km,
            selected_rider_id=best_rider.rider_id if best_rider else None,
            items=order.items,
            complexity=order.complexity.value,
            inputs={"evaluations": evaluations},
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
    if name == "nearest_kitchen":
        return NearestKitchenDispatch(config)
    if name == "nearest_heuristic":
        return NearestHeuristicDispatch(config)
    if name == "joint_optimizer":
        return JointOptimizerDispatch(config)
    if name == "optimized_kitchen":
        return OptimizedKitchenDispatch(config)
    raise ValueError(f"unknown dispatch policy: {name}")
