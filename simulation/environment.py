import simpy

from .entities import WeatherSeverity, TrafficSeverity, WeatherState, TrafficState

WEATHER_FACTORS = {
    WeatherSeverity.CLEAR: 1.0,
    WeatherSeverity.RAIN: 1.1,
    WeatherSeverity.STORM: 1.2,
}

TRAFFIC_FACTORS = {
    TrafficSeverity.LOW: 1.0,
    TrafficSeverity.MODERATE: 1.2,
    TrafficSeverity.HEAVY: 1.55,
}

# Hour-of-day baseline traffic severity index: 0=low, 1=moderate, 2=heavy.
# Bangalore profile (TomTom 2025): night is clear, morning peak 08-11, evening
# peak 17-21, and midday stays congested (never fully clears).
TRAFFIC_BASELINE_BY_HOUR = (
    (0, 6, 0),
    (6, 8, 1),
    (8, 11, 2),
    (11, 17, 1),
    (17, 21, 2),
    (21, 24, 1),
)


def traffic_baseline(hour: int) -> TrafficSeverity:
    for start, end, idx in TRAFFIC_BASELINE_BY_HOUR:
        if start <= hour < end:
            return [TrafficSeverity.LOW, TrafficSeverity.MODERATE, TrafficSeverity.HEAVY][idx]
    return TrafficSeverity.LOW


def forecast_traffic(now: float, travel_min: float) -> TrafficSeverity:
    """Expected traffic severity during a trip that starts at `now` (sim minutes)
    and lasts ~travel_min minutes. Resolved from the deterministic hour-of-day
    baseline at the trip's midpoint; random spike events are unknown a priori,
    so the forecast is the expected (baseline) condition, not the realized one."""
    midpoint = now + travel_min / 2.0
    return traffic_baseline(int(midpoint // 60) % 24)


class WeatherGenerator:
    """State process that cycles weather severities with random durations."""

    def __init__(self, env, rng, config):
        self.env = env
        self.rng = rng
        self.config = config
        start = config["weather"].get("start_severity", "clear")
        self.state = WeatherState(severity=WeatherSeverity(start), changed_at=0.0)
        self.duration_range = config["weather"].get("duration_min", [30, 90])
        self.env.process(self._run())

    def _run(self):
        while True:
            min_dur, max_dur = self.duration_range
            duration = self.rng.uniform(min_dur, max_dur)
            yield self.env.timeout(duration)
            self._transition()

    def _transition(self):
        severity = self.state.severity
        transitions = self.config["weather"].get("transitions", {})
        weights = transitions.get(severity.value)
        if weights:
            options = list(weights.keys())
            probs = [weights[o] for o in options]
            self.state.severity = WeatherSeverity(self.rng.choice(options, p=probs))
        else:
            self.state.severity = WeatherSeverity.CLEAR
        self.state.changed_at = self.env.now

    def current_severity(self) -> WeatherSeverity:
        return self.state.severity

    def prep_factor(self) -> float:
        return WEATHER_FACTORS[self.state.severity]

    def travel_factor(self) -> float:
        return WEATHER_FACTORS[self.state.severity]


class TrafficGenerator:
    """State process with a time-of-day baseline plus random spike events."""

    def __init__(self, env, rng, config):
        self.env = env
        self.rng = rng
        self.config = config
        self.state = TrafficState(severity=TrafficSeverity.LOW, changed_at=0.0)
        self.env.process(self._run())

    def _run(self):
        while True:
            spike_prob = self.config["traffic"].get("spike_prob_per_min", 0.0005)
            if self.rng.random() < spike_prob:
                self.state.severity = TrafficSeverity.HEAVY
                self.state.changed_at = self.env.now
                yield self.env.timeout(self.rng.uniform(10, 30))
                self.state.severity = traffic_baseline(self._hour())
                self.state.changed_at = self.env.now
            yield self.env.timeout(1)

    def _hour(self) -> int:
        return int(self.env.now // 60) % 24

    def current_severity(self) -> TrafficSeverity:
        return self.state.severity

    def travel_factor(self) -> float:
        return TRAFFIC_FACTORS[self.state.severity]
