import simpy

from dispatch.state import DispatchState
from .delivery import delivery_process


class Dispatcher:
    """Decides when to dispatch a rider for each order (via the active policy),
    schedules the dispatch at that time, acquires a rider, and runs delivery."""

    def __init__(self, env, config, policy, riders, event_log, dispatch_rng):
        self.env = env
        self.config = config
        self.policy = policy
        self.riders = riders
        self.event_log = event_log
        self.rng = dispatch_rng
        self.kitchens = None

    def bind_kitchens(self, kitchens):
        self.kitchens = kitchens

    def dispatch(self, order):
        """Called at order placement. Decides, records, and schedules."""
        hub_distance = self.riders.sample_hub_distance(self.rng)
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

        self.event_log.record(
            self.env.now, "dispatch_decision",
            order_id=order.order_id, kitchen_id=order.kitchen_id,
            payload=decision.to_payload(),
        )
        self.env.process(self._dispatch_process(order))

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

    def _dispatch_process(self, order):
        delay = max(0.0, order.dispatch_at - self.env.now)
        if delay > 0:
            yield self.env.timeout(delay)
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
            delivery_process(self.env, self.config, order, rider, self.riders, self.event_log)
        )
