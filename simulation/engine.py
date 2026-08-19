import itertools

import numpy as np
import simpy

from .entities import Kitchen, OrderStatus
from .event_log import EventLog
from .environment import WeatherGenerator, TrafficGenerator
from .order_generator import OrderGenerator
from .riders import RiderPool
from .cancellations import CancellationManager
from .dispatcher import Dispatcher
from .rng import spawn_streams

DISPATCH_STREAM_SALT = 1_000_000
RIDER_POSITION_STREAM_SALT = 2_000_000
RIDER_KITCHEN_MATRIX_STREAM_SALT = 3_000_000

# Step size for the post-generation drain. Small enough that the finish time of
# the last order is reported to within one step.
DRAIN_STEP_MIN = 1.0

# Statuses that mean an order will never finish on its own.
_TERMINAL_STATUS = (OrderStatus.COMPLETED, OrderStatus.CANCELLED, OrderStatus.FAILED)


class SimulationEngine:
    """Single public entry point. Runs a SimPy simulation from a config dict."""

    def __init__(self, config: dict, out_dir: str, scenario_name: str = "unknown", save_outputs: bool = True):
        self.config = config
        self.out_dir = out_dir
        self.scenario_name = scenario_name
        self.save_outputs = save_outputs
        self.total_minutes = config["days"] * 1440
        self.env = simpy.Environment()
        self.streams = spawn_streams(config["seed"])
        # Dedicated dispatch stream: derived, NOT appended to rng.STREAMS, so the
        # five Phase 1/2 streams keep their exact seeds (byte-identical determinism).
        self.streams["dispatch"] = np.random.default_rng(config["seed"] + DISPATCH_STREAM_SALT)
        # Deterministic per-run identity so pooled training CSVs from different
        # runs (order_id resets to 1 every run) never collapse onto each other.
        self.run_id = f"{scenario_name}_seed{config['seed']}"
        self.event_log = EventLog(out_dir=self.out_dir if save_outputs else None, run_id=self.run_id)
        self.order_counter = itertools.count(1)
        self.all_orders = []
        self.summary = {}
        self.dispatcher = None
        self.is_finished = False

    def _build_predictor(self):
        d = self.config.get("dispatch", {})
        if not d.get("enabled", False) or d.get("default_policy", "immediate") != "adaptive":
            return None
        try:
            from models.predict import Predictor
        except ImportError as exc:
            raise RuntimeError("adaptive dispatch needs models.predict (Phase 2)") from exc
        return Predictor.load(d.get("predictor_dir", "artifacts"))

    def _generate_rider_positions(self, num_riders: int, kitchen_locations: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Generate initial rider positions once per seed.

        Positions are uniformly distributed within the service area.
        These are passed to RiderPool so both baseline and optimized
        policies share identical rider layouts for fair comparison.
        """
        from .spatial import SERVICE_AREA_HALF
        pos_rng = np.random.default_rng(self.config["seed"] + RIDER_POSITION_STREAM_SALT)
        positions = []
        for _ in range(num_riders):
            x = float(pos_rng.uniform(-SERVICE_AREA_HALF, SERVICE_AREA_HALF))
            y = float(pos_rng.uniform(-SERVICE_AREA_HALF, SERVICE_AREA_HALF))
            positions.append((x, y))
        return positions

    def _generate_rider_kitchen_matrix(self, num_riders: int, num_kitchens: int) -> np.ndarray:
        """Generate a per-rider, per-kitchen distance matrix once per seed.

        Returns an (num_riders x num_kitchens) ndarray with values in
        [hub_distance_range_km[0], hub_distance_range_km[1]].

        This matrix is the SINGLE source of truth for rider→kitchen travel
        distances. Both policies (for scoring) and delivery_process (for
        simulation) read from this matrix, ensuring the distance the optimizer
        evaluates equals the distance the simulator applies.

        Values are intentionally synthetic (no Bangalore GPS data). Documented
        as synthetic in the paper.
        """
        lo, hi = self.config["dispatch"]["hub_distance_range_km"]
        mat_rng = np.random.default_rng(self.config["seed"] + RIDER_KITCHEN_MATRIX_STREAM_SALT)
        return mat_rng.uniform(lo, hi, size=(num_riders, num_kitchens))

    def _setup(self):
        config = self.config

        kitchen_count = config["kitchens"]["count"]
        staff_level = config["kitchens"]["staff_level"]
        # Support per-kitchen staff levels: staff_levels list overrides staff_level
        staff_levels = config["kitchens"].get("staff_levels")
        if staff_levels is None:
            staff_levels = [staff_level] * kitchen_count
        elif len(staff_levels) != kitchen_count:
            raise ValueError(f"staff_levels length {len(staff_levels)} != kitchen_count {kitchen_count}")
        self.kitchens = [
            Kitchen(kitchen_id=i + 1, staff_level=staff_levels[i])
            for i in range(kitchen_count)
        ]
        for k in self.kitchens:
            k.resource = simpy.Resource(self.env, capacity=max(1, k.staff_level))

        # Generate kitchen locations for spatial model.
        from .spatial import DEFAULT_KITCHEN_LOCATIONS
        dispatch_cfg_full = config.get("dispatch", {})
        spatial_cfg = dispatch_cfg_full.get("kitchen_selection", {})
        kitchen_locations = spatial_cfg.get("kitchen_locations", DEFAULT_KITCHEN_LOCATIONS)

        # Generate rider positions ONCE per seed, so both policies see
        # identical initial layouts.
        rider_count = config["riders"]["count"]
        rider_positions = self._generate_rider_positions(rider_count, kitchen_locations)

        # Generate rider→kitchen distance matrix ONCE per seed.
        # Both policies and delivery_process use this single matrix,
        # ensuring the distance the optimizer evaluates equals the distance
        # the simulator applies (fair comparison requirement).
        rider_kitchen_matrix = self._generate_rider_kitchen_matrix(
            rider_count, len(kitchen_locations)
        )

        self.riders = RiderPool(config, initial_positions=rider_positions,
                                rider_to_kitchen_matrix=rider_kitchen_matrix)

        self._weather = WeatherGenerator(self.env, self.streams["weather"], config)
        self._traffic = TrafficGenerator(self.env, self.streams["traffic"], config)
        self.env._weather = self._weather
        self.env._traffic = self._traffic

        dispatch_cfg = config.get("dispatch", {})
        if dispatch_cfg.get("enabled", False):
            from dispatch.policies import make_policy

            predictor = self._build_predictor()
            policy = make_policy(dispatch_cfg.get("default_policy", "immediate"), predictor, config)
            self.dispatcher = Dispatcher(
                self.env, config, policy, self.riders, self.event_log,
                self.streams["dispatch"], prep_rng=self.streams["prep"],
                kitchen_locations=kitchen_locations,
            )
            self.dispatcher.bind_kitchens(self.kitchens)

        self.cancellations = CancellationManager(
            self.env, self.streams["cancellations"], config, self.kitchens, self.riders,
            self.event_log, dispatcher=self.dispatcher,
        )
        self.order_generator = OrderGenerator(
            self.env, self.streams["arrivals"], self.streams["prep"], config,
            self.kitchens, self.event_log, self.order_counter, dispatcher=self.dispatcher,
            generation_end_min=self.total_minutes,
        )
        self.order_generator.all_orders = self.all_orders

        # Wire up spatial model for kitchen-selection experiments.
        # Always attach when dispatch is enabled so both baseline and optimized
        # policies use the same 2D geometry (fair comparison).
        if dispatch_cfg.get("enabled", False):
            class _SpatialModel:
                pass
            sm = _SpatialModel()
            sm.kitchen_locations = kitchen_locations
            self.order_generator.spatial_model = sm

    @property
    def hard_stop(self) -> float:
        """Latest sim time this run will reach: the generation window plus the
        drain timeout. Both policies use the identical rule."""
        return self.total_minutes + self.config.get("drain_timeout_min", 240)

    def _update_finished(self):
        if self.env.now < self.total_minutes:
            self.is_finished = False
            return
        all_terminal = all(o.status in _TERMINAL_STATUS for o in self.all_orders) if self.all_orders else True
        self.is_finished = self.env.now >= self.hard_stop or all_terminal

    def advance(self, until: float):
        """Advance the simulation to `until` (sim minutes). Callable repeatedly;
        equivalent to a single env.run(until) over the same window. The run is
        only finished once the generation window AND the drain are complete."""
        self.env.run(until=until)
        self._update_finished()

    def drain(self):
        """Advance in small steps until every order is terminal (completed,
        cancelled or failed) or the drain timeout expires. Called after the
        generation window so late-arriving orders still finish. The rule is
        identical for both policies, so it adds no comparison bias."""
        while not self.is_finished:
            self.advance(min(self.env.now + DRAIN_STEP_MIN, self.hard_stop))

    def finalize(self):
        self._collect_orders()
        if self.save_outputs:
            self.event_log.write()
            self.event_log.write_orders_csv(self.orders)
        self._summarize()
        return self.summary

    def run(self):
        self._setup()
        self.advance(self.total_minutes)
        self.drain()
        return self.finalize()

    def _collect_orders(self):
        self.orders = list(self.all_orders)

    def _summarize(self):
        ready = [o for o in self.orders if o.status in (OrderStatus.READY, OrderStatus.COMPLETED)]
        cancelled = [o for o in self.orders if o.status == OrderStatus.CANCELLED]
        prep_times = [o.actual_prep_duration_min for o in ready if o.actual_prep_duration_min is not None]
        self.summary = {
            "seed": self.config["seed"],
            "scenario": self.scenario_name,
            "days": self.config["days"],
            "orders_placed": len(self.orders),
            "orders_prepared": len(ready),
            "orders_cancelled": len(cancelled),
            "avg_prep_min": round(sum(prep_times) / len(prep_times), 2) if prep_times else None,
            "avg_workload_at_placement": round(sum(o.workload_at_placement for o in ready) / len(ready), 2) if ready else None,
            "policy": self.dispatcher.policy.name if self.dispatcher else None,
        }
        if self.dispatcher is not None:
            from dispatch.metrics import compute_metrics
            # sim_length = actual end of simulation (generation + drain), so
            # riders who were still carrying an order at the generation cutoff
            # are credited their full busy time.
            self.summary.update(
                compute_metrics(self.orders, self.riders.riders, self.env.now, self.config)
            )
