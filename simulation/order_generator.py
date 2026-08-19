import numpy as np

from .entities import Order, complexity_from_items

# Default hourly demand (orders/hour) by hour of day, 24 entries.
# Scaled from the original Bangalore bimodal shape to ~420 orders/day.
# Evening peak (17-23) remains the dominant period.
# Late-night orders (00-03) included at low rates.
# Total: sum = 422.
DEMAND_TOTAL_ORDERS = 422
DEFAULT_DEMAND_CURVE = (
    3,   # 00  late-night
    3,   # 01  late-night
    3,   # 02  late-night
    2,   # 03  late-night (tapering)
    0,   # 04  dead zone
    0,   # 05  dead zone
    0,   # 06  dead zone
    4,   # 07  early morning
    19,  # 08  morning ramp
    20,  # 09  morning peak
    20,  # 10  morning peak
    16,  # 11  pre-lunch
    12,  # 12  lunch
    12,  # 13  lunch
    8,   # 14  afternoon dip
    10,  # 15  afternoon
    11,  # 16  pre-evening
    38,  # 17  evening peak
    39,  # 18  evening peak
    45,  # 19  evening peak (highest)
    40,  # 20  evening peak
    45,  # 21  evening peak
    43,  # 22  late evening
    29,  # 23  late evening
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

    def __init__(self, env, rng, prep_rng, config, kitchens, event_log, order_counter, dispatcher=None,
                 generation_end_min: float = float("inf")):
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
        # No orders are placed at or after this sim time (end of the generation
        # window). The drain period that follows only lets existing orders finish.
        self.generation_end_min = generation_end_min
        curve = config.get("demand_curve")
        self.curve = tuple(curve) if curve else DEFAULT_DEMAND_CURVE
        # Spatial model for kitchen-selection experiments (optional).
        self.spatial_model = None
        self.env.process(self._run())

    def _rate(self) -> float:
        hour = hour_of_day(self.env.now)
        return self.curve[hour % 24] * self.demand_multiplier

    def _run(self):
        from .kitchen import kitchen_process

        while True:
            rate = self._rate()
            if rate <= 0:
                # Skip to the next hour with non-zero demand.
                hour = hour_of_day(self.env.now)
                next_hour = (hour + 1) % 24
                skip_min = ((next_hour - hour) % 24) * 60.0
                # Also check if next hour is also zero; walk forward up to 24h.
                for _ in range(24):
                    if self.curve[next_hour % 24] * self.demand_multiplier > 0:
                        break
                    next_hour += 1
                    skip_min += 60.0
                skip_min = max(skip_min, 1.0)
                if self.env.now + skip_min > self.generation_end_min:
                    return
                yield self.env.timeout(skip_min)
                continue
            interarrival = self.rng.exponential(60.0 / max(rate, 1e-9))
            if self.env.now + interarrival > self.generation_end_min:
                return
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
            # Start kitchen process for non-deferred orders (baseline path).
            # OptimizedKitchen-deferred orders (kitchen_id=None) get their
            # kitchen process started later by Dispatcher._assign_kitchen().
            if order.kitchen_id is not None:
                self.env.process(kitchen_process(
                    self.env, self.prep_rng, self.config, order,
                    self.kitchens, self.event_log,
                ))
            if self.dispatcher is not None:
                self.dispatcher.dispatch(order)

    def _create_order(self) -> Order:
        order_id = next(self.order_counter)
        items = sample_items(self.rng, self.config)

        # Spatial model path: both baseline and optimized use the same 2D
        # geometry.  The only difference is how the kitchen is chosen.
        if self.spatial_model is not None:
            from .spatial import generate_customer_location, compute_distances_to_kitchens

            customer_loc = generate_customer_location(self.rng)
            kitchen_dists = compute_distances_to_kitchens(
                customer_loc, self.spatial_model.kitchen_locations
            )
            distance_to_kitchens = [round(d, 4) for d in kitchen_dists]

            use_deferred = (
                self.dispatcher is not None
                and getattr(self.dispatcher.policy, "name", "") in (
                    "optimized_kitchen", "nearest_kitchen",
                    "nearest_heuristic", "joint_optimizer",
                )
            )

            # Always draw kitchen choice to keep RNG aligned across policies.
            # Deferred policies ignore the result (set kitchen_id=None).
            kitchen = self.rng.choice(self.kitchens)
            kitchen_idx = kitchen.kitchen_id - 1

            if use_deferred:
                # OptimizedKitchen / Nearest: defer — policy picks the best kitchen.
                distance = kitchen_dists[kitchen_idx]
                order = Order(
                    order_id=order_id,
                    kitchen_id=None,          # deferred
                    placed_at=self.env.now,
                    items=items,
                    complexity=complexity_from_items(items),
                    distance_km=distance,
                    staff_level=0,            # set after kitchen assignment
                    customer_x=customer_loc.x,
                    customer_y=customer_loc.y,
                    distance_to_kitchens=distance_to_kitchens,
                )
                self.all_orders.append(order)
                return order

            # Baseline (immediate / adaptive): use the drawn kitchen.
            distance = kitchen_dists[kitchen_idx]
            order = Order(
                order_id=order_id,
                kitchen_id=kitchen.kitchen_id,
                placed_at=self.env.now,
                items=items,
                complexity=complexity_from_items(items),
                distance_km=distance,
                staff_level=kitchen.staff_level,
                customer_x=customer_loc.x,
                customer_y=customer_loc.y,
                distance_to_kitchens=distance_to_kitchens,
                selected_kitchen_distance=distance,
            )
            kitchen.current_orders.append(order)
            order.workload_at_placement = len(kitchen.current_orders)
            order.weather_severity = self.env._weather.current_severity().value
            order.traffic_severity = self.env._traffic.current_severity().value
            self.all_orders.append(order)
            return order

        # No spatial model: original path (U(3,17) distance, random kitchen).
        kitchen = self.rng.choice(self.kitchens)
        distance = sample_distance_km(self.rng, self.config)
        order = Order(
            order_id=order_id,
            kitchen_id=kitchen.kitchen_id,
            placed_at=self.env.now,
            items=items,
            complexity=complexity_from_items(items),
            distance_km=distance,
            staff_level=kitchen.staff_level,
        )
        kitchen.current_orders.append(order)
        order.workload_at_placement = len(kitchen.current_orders)
        order.weather_severity = self.env._weather.current_severity().value
        order.traffic_severity = self.env._traffic.current_severity().value
        self.all_orders.append(order)
        return order
