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


class SimulationEngine:
    """Single public entry point. Runs a SimPy simulation from a config dict."""

    def __init__(self, config: dict, out_dir: str, scenario_name: str = "unknown"):
        self.config = config
        self.out_dir = out_dir
        self.scenario_name = scenario_name
        self.total_minutes = config["days"] * 1440
        self.env = simpy.Environment()
        self.streams = spawn_streams(config["seed"])
        # Dedicated dispatch stream: derived, NOT appended to rng.STREAMS, so the
        # five Phase 1/2 streams keep their exact seeds (byte-identical determinism).
        self.streams["dispatch"] = np.random.default_rng(config["seed"] + DISPATCH_STREAM_SALT)
        self.event_log = EventLog(out_dir=self.out_dir)
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

    def _setup(self):
        config = self.config

        kitchen_count = config["kitchens"]["count"]
        staff_level = config["kitchens"]["staff_level"]
        self.kitchens = [
            Kitchen(kitchen_id=i + 1, staff_level=staff_level)
            for i in range(kitchen_count)
        ]
        for k in self.kitchens:
            k.resource = simpy.Resource(self.env, capacity=max(1, k.staff_level))

        self.riders = RiderPool(config)

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
                self.env, config, policy, self.riders, self.event_log, self.streams["dispatch"]
            )
            self.dispatcher.bind_kitchens(self.kitchens)

        self.cancellations = CancellationManager(
            self.env, self.streams["cancellations"], config, self.kitchens, self.riders,
            self.event_log, dispatcher=self.dispatcher,
        )
        self.order_generator = OrderGenerator(
            self.env, self.streams["arrivals"], self.streams["prep"], config,
            self.kitchens, self.event_log, self.order_counter, dispatcher=self.dispatcher,
        )
        self.order_generator.all_orders = self.all_orders

    def advance(self, until: float):
        """Advance the simulation to `until` (sim minutes). Callable repeatedly;
        equivalent to a single env.run(until) over the same window."""
        self.env.run(until=min(until, self.total_minutes))
        if self.env.now >= self.total_minutes:
            self.is_finished = True

    def finalize(self):
        self._collect_orders()
        self.event_log.write()
        self.event_log.write_orders_csv(self.orders)
        self._summarize()
        return self.summary

    def run(self):
        self._setup()
        self.advance(self.total_minutes)
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
            self.summary.update(
                compute_metrics(self.orders, self.riders.riders, self.config["days"] * 1440, self.config)
            )
