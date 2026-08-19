from dispatch.state import DispatchState
from .delivery import delivery_process


class Dispatcher:
    """Decides when to dispatch a rider for each order (via the active policy),
    schedules the dispatch at that time, acquires a rider, and runs delivery."""

    def __init__(self, env, config, policy, riders, event_log, dispatch_rng, prep_rng=None,
                 kitchen_locations=None):
        self.env = env
        self.config = config
        self.policy = policy
        self.riders = riders
        self.event_log = event_log
        self.rng = dispatch_rng
        self.prep_rng = prep_rng
        self.kitchens = None
        self.kitchen_locations = kitchen_locations

    def bind_kitchens(self, kitchens):
        self.kitchens = kitchens

    def dispatch(self, order):
        """Called at order placement. Decides, records, and schedules.

        For policies that select a kitchen, this also assigns the kitchen and
        starts the kitchen process. For policies that select a rider, the
        rider is pre-assigned and passed to the delivery process.
        """
        hub_distance = self.riders.sample_hub_distance(self.rng)

        # Build state with or without spatial candidates.
        if order.distance_to_kitchens is not None:
            state = DispatchState.from_env_with_spatial(
                self.env, self.kitchens, self.riders, hub_distance, self.config,
                order.distance_to_kitchens,
                kitchen_locations=self.kitchen_locations,
            )
        else:
            state = DispatchState.from_env(
                self.env, self.kitchens, self.riders, hub_distance, self.config
            )

        decision = self.policy.decide(order, state)

        order.dispatch_policy = decision.policy
        order.dispatch_at = decision.dispatch_at
        order.hub_distance_km = hub_distance
        order.travel_to_kitchen_min = decision.travel_to_kitchen_min
        order.eta_min = decision.eta
        order.predicted_prep_mean = decision.predicted_prep_mean
        order.predicted_prep_low = decision.predicted_prep_low
        order.predicted_prep_high = decision.predicted_prep_high
        order.uncertainty = decision.uncertainty
        order.risk_buffer_min = decision.risk_buffer_min
        order.decision_rationale = decision.rationale

        # Kitchen assignment from policy (optimized_kitchen, nearest_kitchen,
        # nearest_heuristic, joint_optimizer).
        if decision.selected_kitchen_id is not None:
            self._assign_kitchen(order, decision.selected_kitchen_id, decision.selected_kitchen_distance)

        self.event_log.record(
            self.env.now, "dispatch_decision",
            order_id=order.order_id, kitchen_id=order.kitchen_id,
            payload=decision.to_payload(),
        )
        self.env.process(self._dispatch_process(order, preassigned_rider_id=decision.selected_rider_id))

    def _assign_kitchen(self, order, kitchen_id: int, distance: float | None = None):
        """Assign a kitchen to an order (for deferred-assignment policies).

        - Sets order.kitchen_id and order.staff_level
        - Appends order to kitchen.current_orders
        - Sets workload_at_placement and weather/traffic features
        - Starts the kitchen process
        """
        from .kitchen import kitchen_process

        kitchen = next(k for k in self.kitchens if k.kitchen_id == kitchen_id)
        order.kitchen_id = kitchen.kitchen_id
        order.staff_level = kitchen.staff_level
        order.distance_km = distance if distance is not None else order.distance_km
        order.selected_kitchen_distance = distance
        kitchen.current_orders.append(order)
        order.workload_at_placement = len(kitchen.current_orders)
        order.weather_severity = self.env._weather.current_severity().value
        order.traffic_severity = self.env._traffic.current_severity().value

        self.event_log.record(
            self.env.now, "kitchen_assigned",
            order_id=order.order_id, kitchen_id=kitchen.kitchen_id,
            payload={
                "distance_km": round(distance, 4) if distance is not None else None,
                "queue_len": len(kitchen.current_orders),
            },
        )
        self.env.process(
            kitchen_process(self.env, self.prep_rng, self.config, order, self.kitchens, self.event_log)
        )

    def redispatch(self, order):
        """Re-queue a rider for an order after a rider cancel. Food is already
        being prepared or ready; dispatch as soon as a rider is free."""
        if order.rider_id is not None:
            order.rider_id = None
        order.dispatch_at = self.env.now
        self.event_log.record(
            self.env.now, "order_redispatch",
            order_id=order.order_id, kitchen_id=order.kitchen_id,
        )
        self.env.process(self._dispatch_process(order))

    def _dispatch_process(self, order, preassigned_rider_id=None):
        delay = max(0.0, order.dispatch_at - self.env.now)
        if delay > 0:
            yield self.env.timeout(delay)

        # If a specific rider was selected by the policy, assign that rider.
        if preassigned_rider_id is not None:
            rider = self.riders.assign_by_id(preassigned_rider_id, at=self.env.now)
            if rider is None:
                # Preassigned rider was taken — fall back to any idle rider.
                while True:
                    if order.status.value in ("cancelled", "failed"):
                        return
                    rider = self.riders.assign_next_idle(at=self.env.now)
                    if rider is not None:
                        break
                    yield self.env.timeout(0.1)
        else:
            while True:
                if order.status.value in ("cancelled", "failed"):
                    return
                rider = self.riders.assign_next_idle(at=self.env.now)
                if rider is not None:
                    break
                yield self.env.timeout(0.1)

        if order.status.value in ("cancelled", "failed"):
            self.riders.release(rider.rider_id, at=self.env.now)
            return
        order.rider_id = rider.rider_id
        self.event_log.record(
            self.env.now, "rider_dispatched",
            order_id=order.order_id, kitchen_id=order.kitchen_id, rider_id=rider.rider_id,
            payload={"dispatch_at": round(order.dispatch_at, 2), "policy": order.dispatch_policy},
        )
        self.env.process(
            delivery_process(self.env, self.config, order, rider, self.riders, self.event_log,
                             kitchen_locations=self.kitchen_locations)
        )
