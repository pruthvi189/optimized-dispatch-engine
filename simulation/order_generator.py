import simpy

from .entities import Order, OrderComplexity, complexity_from_items

# Default hourly demand (orders/hour) by hour of day, 24 entries.
DEFAULT_DEMAND_CURVE = (
    2,   # 00
    1,   # 01
    1,   # 02
    1,   # 03
    2,   # 04
    3,   # 05
    8,   # 06
    15,  # 07
    18,  # 08
    12,  # 09
    10,  # 10
    22,  # 11
    30,  # 12
    28,  # 13
    18,  # 14
    14,  # 15
    14,  # 16
    18,  # 17
    22,  # 18
    30,  # 19
    35,  # 20
    32,  # 21
    20,  # 22
    8,   # 23
)


def hour_of_day(sim_time: float) -> int:
    return int(sim_time // 60) % 24


def sample_items(rng, config) -> int:
    weights = config["orders"].get("items_weights", [0.5, 0.3, 0.2])
    options = [1, 2, 3]
    if len(weights) == 4:
        options = [1, 2, 3, 4]
    return int(rng.choice(options, p=weights))


def sample_distance_km(rng, config) -> float:
    lo, hi = config["orders"].get("distance_range_km", [1.0, 3.5])
    return float(rng.uniform(lo, hi))


class OrderGenerator:
    """Non-homogeneous Poisson arrivals driven by a per-hour demand curve."""

    def __init__(self, env, rng, prep_rng, config, kitchens, event_log, order_counter, dispatcher=None):
        self.env = env
        self.rng = rng
        self.prep_rng = prep_rng
        self.config = config
        self.kitchens = kitchens
        self.event_log = event_log
        self.order_counter = order_counter
        self.dispatcher = dispatcher
        self.all_orders = []
        self.demand_multiplier = config["demand_multiplier"]
        curve = config.get("demand_curve")
        self.curve = tuple(curve) if curve else DEFAULT_DEMAND_CURVE
        self.env.process(self._run())

    def _rate(self) -> float:
        hour = hour_of_day(self.env.now)
        return self.curve[hour % 24] * self.demand_multiplier

    def _run(self):
        from .kitchen import kitchen_process

        while True:
            rate = self._rate()
            interarrival = self.rng.exponential(60.0 / max(rate, 1e-9))
            yield self.env.timeout(interarrival)
            order = self._create_order()
            self.event_log.record(
                self.env.now, "order_placed",
                order_id=order.order_id,
                kitchen_id=order.kitchen_id,
                payload={
                    "items": order.items,
                    "complexity": order.complexity.value,
                    "distance_km": round(order.distance_km, 2),
                    "hour": hour_of_day(self.env.now),
                },
            )
            self.env.process(kitchen_process(self.env, self.prep_rng, self.config, order, self.kitchens, self.event_log))
            if self.dispatcher is not None:
                self.dispatcher.dispatch(order)

    def _create_order(self) -> Order:
        order_id = next(self.order_counter)
        kitchen = self.rng.choice(self.kitchens)
        items = sample_items(self.rng, self.config)
        distance = sample_distance_km(self.rng, self.config)
        workload = len(kitchen.current_orders)
        order = Order(
            order_id=order_id,
            kitchen_id=kitchen.kitchen_id,
            placed_at=self.env.now,
            items=items,
            complexity=complexity_from_items(items),
            distance_km=distance,
            workload_at_placement=workload,
            staff_level=kitchen.staff_level,
        )
        kitchen.current_orders.append(order)
        self.all_orders.append(order)
        return order
