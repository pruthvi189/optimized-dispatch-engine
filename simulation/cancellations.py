from .entities import Order, OrderStatus


class CancellationManager:
    """Emits cancellation events and performs state transitions, including
    rider release, redispatch, and kitchen-failure cancellation."""

    def __init__(self, env, rng, config, kitchens, riders, event_log, dispatcher=None):
        self.env = env
        self.rng = rng
        self.config = config
        self.kitchens = kitchens
        self.riders = riders
        self.event_log = event_log
        self.dispatcher = dispatcher
        self.cancelled_orders = []
        self.env.process(self._kitchen_failures())
        self.env.process(self._customer_cancel_monitor())

    def customer_cancel(self, order: Order, wasted_prep: bool):
        order.status = OrderStatus.CANCELLED
        order.cancel_reason = "customer"
        if order.rider_id is not None:
            self.riders.release(order.rider_id, at=self.env.now)
            order.rider_id = None
        self.cancelled_orders.append(order)
        self.event_log.record(
            self.env.now, "cancellation_customer",
            order_id=order.order_id,
            kitchen_id=order.kitchen_id,
            payload={"wasted_prep": wasted_prep},
        )

    def rider_cancel(self, order: Order, rider_id: int):
        """Phase 3: the rider drops the delivery -> penalty for the rider,
        redispatch the order to the next available rider. Without a dispatcher
        (Phase 1/2 mode) the order is cancelled instead."""
        self.riders.mark_unavailable(rider_id)
        self.event_log.record(
            self.env.now, "cancellation_rider",
            order_id=order.order_id,
            kitchen_id=order.kitchen_id,
            rider_id=rider_id,
        )
        self.env.process(self._rider_penalty(rider_id))
        if self.dispatcher is None:
            order.status = OrderStatus.CANCELLED
            order.cancel_reason = "rider"
            self.cancelled_orders.append(order)
            return
        if order.rider_id == rider_id:
            order.rider_id = None
        self.dispatcher.redispatch(order)

    def _rider_penalty(self, rider_id: int):
        penalty = self.config.get("dispatch", {}).get("penalty_min", 10.0)
        yield self.env.timeout(penalty)
        self.riders.release(rider_id, at=self.env.now)

    def kitchen_failure(self, kitchen):
        self.event_log.record(
            self.env.now, "kitchen_failure_start",
            kitchen_id=kitchen.kitchen_id,
        )
        for order in list(kitchen.current_orders):
            if order.status in (OrderStatus.PLACED, OrderStatus.PREPPING):
                order.status = OrderStatus.CANCELLED
                order.cancel_reason = "kitchen"
                if order.rider_id is not None:
                    self.riders.release(order.rider_id, at=self.env.now)
                    order.rider_id = None
                self.cancelled_orders.append(order)
                self.event_log.record(
                    self.env.now, "cancellation_kitchen",
                    order_id=order.order_id,
                    kitchen_id=order.kitchen_id,
                )
        kitchen.current_orders = [o for o in kitchen.current_orders if o.status != OrderStatus.CANCELLED]

    def _kitchen_failures(self):
        rate = self.config["cancellation_rates"].get("kitchen_failure_per_min", 0.0002)
        while True:
            if self.rng.random() < rate:
                kitchen = self.rng.choice(self.kitchens)
                if not kitchen.failed:
                    kitchen.failed = True
                    self.kitchen_failure(kitchen)
                    duration = self.rng.uniform(5, 20)
                    yield self.env.timeout(duration)
                    kitchen.failed = False
                    self.event_log.record(
                        self.env.now, "kitchen_failure_end",
                        kitchen_id=kitchen.kitchen_id,
                    )
            yield self.env.timeout(1)

    def _customer_cancel_monitor(self):
        """Per-minute hazard that a prepping order is cancelled by the customer."""
        rate = self.config["cancellation_rates"].get("customer_cancel_per_min", 0.0005)
        while True:
            for kitchen in self.kitchens:
                for order in list(kitchen.current_orders):
                    if order.status == OrderStatus.PREPPING and self.rng.random() < rate:
                        wasted_prep = order.prep_started_at is not None
                        self.customer_cancel(order, wasted_prep=wasted_prep)
                        if order in kitchen.current_orders:
                            kitchen.current_orders.remove(order)
            yield self.env.timeout(1)
