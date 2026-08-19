"""Background-thread simulation runner. Owns the SimPy engine, advances it in
small increments on a configurable clock, and pushes snapshots to the hub."""

import os
import threading
import time

from simulation import SimulationEngine

from .snapshot import build_snapshot


class SimulationRunner:
    """Advances one SimulationEngine on a worker thread.

    - `start()` (re)builds the engine if needed and begins advancing.
    - `pause()` / `start()` pause and resume; `step(minutes)` advances exactly
      `minutes` while paused (or before start).
    - `reset(config)` swaps in a new config and rebuilds the engine.
    - `snapshot()` / `status()` return the latest throttled snapshot.
    """

    def __init__(self, config, hub=None, speed=60.0, step_minutes=1.0,
                 scenario_name="unknown", out_dir=None):
        self.config = config
        self.hub = hub
        self.speed = speed
        self.step_minutes = step_minutes
        self.scenario_name = scenario_name
        self.out_dir = out_dir or os.path.join(
            "data", "api_runs", f"{scenario_name}_{config['seed']}"
        )
        os.makedirs(self.out_dir, exist_ok=True)

        self._engine = None
        self._thread = None
        self._latest = None
        self._finalized = False
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._step_available = threading.Event()
        self._step_request = None
        self._lock = threading.Lock()
        self.running = False
        self.paused = False
        self.finished = False

    # ---- control (callable from any thread) --------------------------------

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if not self._pause.is_set():
                    return
                self._pause.clear()
                self.paused = False
                return
            if self._engine is None or self.finished:
                self._engine = SimulationEngine(
                    self.config, out_dir=self.out_dir, scenario_name=self.scenario_name
                )
                self._engine._setup()
                self._finalized = False
            self._stop.clear()
            self.running = True
            self.paused = False
            self.finished = False
            self._thread = threading.Thread(target=self._loop, daemon=True, name="sim-runner")
            self._thread.start()

    def pause(self):
        self._pause.set()
        with self._lock:
            self.paused = True

    def step(self, minutes=None):
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and not self._pause.is_set():
                raise RuntimeError("step only valid while paused (or before start)")
            self._step_request = minutes or self.step_minutes
            self._step_available.set()

    def reset(self, config, scenario_name=None):
        with self._lock:
            was_running = self._thread is not None and self._thread.is_alive()
            if was_running:
                self._stop.set()
        if was_running:
            self._thread.join(timeout=10)
        with self._lock:
            self._stop.clear()
            self._pause.clear()
            self._step_available.clear()
            self._step_request = None
            self.config = config
            if scenario_name:
                self.scenario_name = scenario_name
            self._engine = None
            self._latest = None
            self._finalized = False
            self.running = False
            self.paused = False
            self.finished = False

    def stop(self):
        with self._lock:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        with self._lock:
            self.running = False
            self._thread = None

    # ---- read (callable from any thread) -----------------------------------

    @property
    def engine(self):
        with self._lock:
            return self._engine

    def snapshot(self):
        with self._lock:
            return self._latest

    def status(self):
        with self._lock:
            snap = self._latest
            running, paused, finished = self.running, self.paused, self.finished
        if snap is not None:
            snap = dict(snap)
            snap["running"] = running
            snap["paused"] = paused
            snap["finished"] = finished
            return snap
        return {
            "sim_time_min": 0.0,
            "scenario": self.scenario_name,
            "policy": self.config.get("dispatch", {}).get("default_policy"),
            "days": self.config.get("days"),
            "seed": self.config.get("seed"),
            "total_minutes": self.config.get("days", 1) * 1440,
            "speed": self.speed,
            "running": self.running,
            "paused": self.paused,
            "finished": self.finished,
            "weather": None,
            "traffic": None,
            "kitchens": [],
            "riders": [],
            "recent_decisions": [],
            "metrics": {},
            "events": [],
        }

    # ---- worker thread -----------------------------------------------------

    def _loop(self):
        engine = self._engine
        while not self._stop.is_set():
            if self._step_available.is_set():
                with self._lock:
                    minutes = self._step_request or self.step_minutes
                    self._step_request = None
                self._step_available.clear()
                self._advance(engine, minutes)
                self._emit()
                continue
            if self._pause.is_set() or engine.is_finished:
                time.sleep(0.02)
                continue
            self._advance(engine, self.step_minutes)
            self._emit()
            if self.speed:
                time.sleep(self.step_minutes / self.speed)
        if not self._finalized and engine.env.now > 0:
            engine.finalize()
            self._finalized = True

    def _advance(self, engine, minutes):
        end = min(engine.env.now + minutes, engine.hard_stop)
        engine.advance(end)
        if engine.is_finished and not self._finalized:
            engine.finalize()
            self._finalized = True
            with self._lock:
                self.finished = True
                self.running = False

    def _emit(self):
        with self._lock:
            running = self.running
            paused = self.paused
            finished = self.finished
        snap = build_snapshot(
            self._engine,
            meta={"speed": self.speed, "running": running, "paused": paused, "finished": finished},
        )
        with self._lock:
            self._latest = snap
        if self.hub is not None:
            self.hub.publish(snap)
