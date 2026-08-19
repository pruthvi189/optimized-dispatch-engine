"""Experiment runner for paired Adaptive vs Immediate dispatch simulations.

Runs 10,000+ randomized environments with paired experiments to determine
if Adaptive dispatch provides statistically significant improvements.
"""

import copy
import csv
import json
import os
import statistics
import time
from dataclasses import dataclass, asdict, field
from typing import Literal

import numpy as np

from simulation import SimulationEngine, load_scenario
from dispatch.metrics import compute_metrics
from dispatch.statistics import (
    METRIC_DIRECTIONS,
    paired_difference_stats,
    wilson_ci,
)


ScenarioType = Literal["normal", "lunch_rush", "rain", "low_staffing", "traffic_spike"]


DELIVERY_WIN_THRESHOLD = 0.1  # minutes: min avg-delivery gap for a non-tie


@dataclass
class ExperimentConfig:
    """Configuration for the experiment runner."""
    num_experiments: int = 10000
    base_seed: int = 42
    days: int = 1
    scenario: ScenarioType = "normal"
    predictor_dir: str = "artifacts"
    out_dir: str = "data/experiments"
    save_individual: bool = False  # Whether to save each run's orders.csv
    save_distributions: bool = False  # Whether to accumulate delivery-time histograms


@dataclass
class PairedResult:
    """Result of a single paired experiment (one seed, both policies)."""
    seed: int
    scenario: str
    days: int
    immediate: dict
    adaptive: dict
    differences: dict  # adaptive - immediate for each metric


@dataclass
class ExperimentSummary:
    """Aggregate summary of all paired experiments."""
    num_experiments: int
    scenario: str
    days: int
    base_seed: int

    # Win counts
    adaptive_wins: int
    immediate_wins: int
    ties: int

    # Metric differences (adaptive - immediate)
    # Negative means adaptive is better (lower is better for these metrics)
    on_time_pct_diff_mean: float
    on_time_pct_diff_median: float
    on_time_pct_diff_std: float
    avg_delivery_min_diff_mean: float
    avg_delivery_min_diff_median: float
    avg_delivery_min_diff_std: float
    p50_delivery_min_diff_mean: float
    p90_delivery_min_diff_mean: float
    p95_delivery_min_diff_mean: float
    avg_late_min_diff_mean: float
    avg_late_min_diff_median: float
    avg_late_min_diff_std: float
    avg_order_wait_min_diff_mean: float
    avg_rider_wait_kitchen_min_diff_mean: float
    avg_rider_wait_kitchen_min_diff_median: float
    cost_score_diff_mean: float
    cost_score_diff_median: float
    cost_score_diff_std: float

    # Per-scenario breakdown (if multiple scenarios tested)
    scenario_breakdown: dict

    # Added Phase 2 (delivery time primary objective) - defaults keep older
    # experiment_summary.json files loadable.
    p99_delivery_min_diff_mean: float = 0.0
    max_delivery_min_diff_mean: float = 0.0
    late_count_diff_mean: float = 0.0

    # Added Task 5 (statistical comparison + win-rate reporting). Defaults keep
    # older experiment_summary.json files loadable.
    adaptive_win_rate: float = 0.0
    adaptive_win_rate_ci_low: float = 0.0
    adaptive_win_rate_ci_high: float = 0.0
    immediate_win_rate: float = 0.0
    tie_rate: float = 0.0
    win_method: str = "avg_delivery_min"
    paired_stats: dict = field(default_factory=dict)


class DistributionAccumulator:
    """Incrementally aggregates per-order delivery-time histograms for both
    policies across paired runs.

    Raw delivery times are retained (not just histogram bins) so that true
    pooled percentiles can be emitted for the dashboard; the histogram cap
    (``max_min``) only affects the binned charts, never the percentiles.
    """

    def __init__(self, bin_min: float = 1.0, max_min: float = 240.0):
        self.bin_min = bin_min
        self.max_min = max_min
        self.edges = np.arange(0.0, max_min + bin_min, bin_min)
        self.adaptive_counts = np.zeros(len(self.edges) - 1, dtype=np.int64)
        self.immediate_counts = np.zeros(len(self.edges) - 1, dtype=np.int64)
        self.adaptive_times: list[float] = []
        self.immediate_times: list[float] = []
        self.adaptive_total = 0
        self.immediate_total = 0
        self.adaptive_delivery_sum = 0.0
        self.immediate_delivery_sum = 0.0
        self.num_runs = 0

    def _expand(self, new_max: float):
        """Expand edges and re-bin existing counts when data exceeds current range."""
        new_edges = np.arange(0.0, new_max + self.bin_min, self.bin_min)
        new_adaptive = np.zeros(len(new_edges) - 1, dtype=np.int64)
        new_immediate = np.zeros(len(new_edges) - 1, dtype=np.int64)
        if self.adaptive_times:
            new_adaptive += np.histogram(self.adaptive_times, bins=new_edges)[0]
        if self.immediate_times:
            new_immediate += np.histogram(self.immediate_times, bins=new_edges)[0]
        self.edges = new_edges
        self.max_min = new_max
        self.adaptive_counts = new_adaptive
        self.immediate_counts = new_immediate

    def update(self, adaptive_times, immediate_times):
        # Auto-expand if any delivery time exceeds current range.
        all_times = list(adaptive_times) + list(immediate_times)
        if all_times:
            data_max = max(all_times)
            if data_max > self.max_min:
                self._expand(float(np.ceil(data_max / 10.0) * 10 + 10))
        self.adaptive_counts += np.histogram(adaptive_times, bins=self.edges)[0]
        self.immediate_counts += np.histogram(immediate_times, bins=self.edges)[0]
        self.adaptive_total += len(adaptive_times)
        self.immediate_total += len(immediate_times)
        self.adaptive_delivery_sum += sum(adaptive_times)
        self.immediate_delivery_sum += sum(immediate_times)
        self.adaptive_times.extend(adaptive_times)
        self.immediate_times.extend(immediate_times)
        self.num_runs += 1

    def _percentiles(self, times) -> dict[int, float]:
        if not times:
            return {50: 0.0, 90: 0.0, 95: 0.0, 99: 0.0}
        values = np.percentile(times, [50, 90, 95, 99])
        return {q: round(float(v), 3) for q, v in zip((50, 90, 95, 99), values)}

    def _series(self, counts, total, delivery_sum, times):
        cdf = np.cumsum(counts).astype(float)
        if total > 0:
            cdf /= total
        return {
            "bin_counts": counts.tolist(),
            "edges": self.edges.tolist(),
            "cdf": [round(x, 5) for x in cdf.tolist()],
            "total_orders": int(total),
            "avg_delivery_min": round(delivery_sum / total, 3) if total else 0.0,
            "percentiles": self._percentiles(times),
        }

    def to_dict(self, scenario: str) -> dict:
        return {
            "scenario": scenario,
            "num_paired_runs": self.num_runs,
            "max_min": self.max_min,
            "adaptive": self._series(self.adaptive_counts, self.adaptive_total, self.adaptive_delivery_sum, self.adaptive_times),
            "immediate": self._series(self.immediate_counts, self.immediate_total, self.immediate_delivery_sum, self.immediate_times),
        }


def _run_single_simulation(config: dict, out_dir: str, scenario_name: str, save_outputs: bool):
    """Run one simulation; returns the engine (for per-order data) and metrics."""
    engine = SimulationEngine(config, out_dir=out_dir, scenario_name=scenario_name, save_outputs=save_outputs)
    engine.run()
    metrics = compute_metrics(engine.orders, engine.riders.riders, engine.env.now, config)
    return engine, metrics


def run_paired_experiment(
    seed: int,
    base_config: dict,
    predictor_dir: str,
    out_root: str,
    scenario_name: str,
    save_individual: bool = False,
    dist_accumulator: DistributionAccumulator | None = None,
) -> PairedResult:
    """Run a paired experiment: Immediate and Adaptive on the same seed."""
    # Create configs for both policies (deep copy to avoid shared nested dicts)
    config_immediate = copy.deepcopy(base_config)
    config_immediate["seed"] = seed
    config_immediate["dispatch"]["default_policy"] = "immediate"
    config_immediate["dispatch"]["predictor_dir"] = predictor_dir

    config_adaptive = copy.deepcopy(base_config)
    config_adaptive["seed"] = seed
    config_adaptive["dispatch"]["default_policy"] = "adaptive"
    config_adaptive["dispatch"]["predictor_dir"] = predictor_dir

    # Output directories
    out_i = os.path.join(out_root, f"{scenario_name}_seed{seed}_day{base_config['days']}_immediate")
    out_a = os.path.join(out_root, f"{scenario_name}_seed{seed}_day{base_config['days']}_adaptive")

    if save_individual:
        os.makedirs(out_i, exist_ok=True)
        os.makedirs(out_a, exist_ok=True)

    # Run Immediate
    immediate_engine, immediate_metrics = _run_single_simulation(config_immediate, out_i, scenario_name, save_outputs=save_individual)

    # Run Adaptive
    adaptive_engine, adaptive_metrics = _run_single_simulation(config_adaptive, out_a, scenario_name, save_outputs=save_individual)

    if dist_accumulator is not None:
        dist_accumulator.update(
            [o.delivered_at - o.placed_at for o in adaptive_engine.orders if o.delivered_at is not None],
            [o.delivered_at - o.placed_at for o in immediate_engine.orders if o.delivered_at is not None],
        )

    # Compute differences (adaptive - immediate)
    # For metrics where lower is better, negative diff means adaptive wins
    differences = {
        "on_time_rate": adaptive_metrics["on_time_rate"] - immediate_metrics["on_time_rate"],
        "avg_delivery_min": adaptive_metrics["avg_delivery_min"] - immediate_metrics["avg_delivery_min"],
        "p50_delivery_min": adaptive_metrics["p50_delivery_min"] - immediate_metrics["p50_delivery_min"],
        "p90_delivery_min": adaptive_metrics["p90_delivery_min"] - immediate_metrics["p90_delivery_min"],
        "p95_delivery_min": adaptive_metrics["p95_delivery_min"] - immediate_metrics["p95_delivery_min"],
        "p99_delivery_min": adaptive_metrics["p99_delivery_min"] - immediate_metrics["p99_delivery_min"],
        "max_delivery_min": adaptive_metrics["max_delivery_min"] - immediate_metrics["max_delivery_min"],
        "late_count": adaptive_metrics["late_count"] - immediate_metrics["late_count"],
        "avg_late_min": adaptive_metrics["avg_late_min"] - immediate_metrics["avg_late_min"],
        "avg_order_wait_min": adaptive_metrics["avg_order_wait_min"] - immediate_metrics["avg_order_wait_min"],
        "avg_rider_wait_kitchen_min": adaptive_metrics["avg_rider_wait_kitchen_min"] - immediate_metrics["avg_rider_wait_kitchen_min"],
        "avg_rider_idle_min": adaptive_metrics["avg_rider_idle_min"] - immediate_metrics["avg_rider_idle_min"],
        "cost_score": adaptive_metrics["cost_score"] - immediate_metrics["cost_score"],
        "orders_in_flight": adaptive_metrics["orders_in_flight"] - immediate_metrics["orders_in_flight"],
    }

    return PairedResult(
        seed=seed,
        scenario=scenario_name,
        days=base_config["days"],
        immediate=immediate_metrics,
        adaptive=adaptive_metrics,
        differences=differences,
    )


def run_experiment(config: ExperimentConfig, progress_callback=None) -> tuple[list[PairedResult], ExperimentSummary]:
    """Run the full experiment suite."""
    base_config = load_scenario(config.scenario, seed=config.base_seed)

    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)

    dist_accumulator = DistributionAccumulator() if config.save_distributions else None
    results = []
    start_time = time.time()

    for i in range(config.num_experiments):
        seed = config.base_seed + i
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (config.num_experiments - i) / rate if rate > 0 else 0
            print(f"  Progress: {i}/{config.num_experiments} ({i/config.num_experiments*100:.1f}%) - {rate:.1f} exp/s - ETA: {eta:.0f}s")

        try:
            result = run_paired_experiment(
                seed=seed,
                base_config=base_config,
                predictor_dir=config.predictor_dir,
                out_root=config.out_dir,
                scenario_name=config.scenario,
                save_individual=config.save_individual,
                dist_accumulator=dist_accumulator,
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR at seed {seed}: {e}")
            continue

        if progress_callback:
            progress_callback(i + 1, config.num_experiments)

    # Compute summary
    summary = compute_summary(results, config)

    if dist_accumulator is not None:
        dist_path = os.path.join(config.out_dir, "delivery_distribution.json")
        with open(dist_path, "w") as f:
            json.dump(dist_accumulator.to_dict(config.scenario), f, indent=2)
        print(f"Delivery-time distribution saved to {dist_path}")

    return results, summary


# Metrics whose paired difference (adaptive - immediate) is statistically
# summarised by compute_summary.
_SUMMARY_STATS_KEYS = (
    "avg_delivery_min",
    "on_time_rate",
    "avg_late_min",
    "p95_delivery_min",
    "avg_order_wait_min",
    "avg_rider_wait_kitchen_min",
    "cost_score",
)


def _paired_stats_for(results: list[PairedResult]) -> dict:
    """Compute bootstrap-CI + sign-flip permutation statistics for each
    summary metric. Missing per-run differences default to 0.0."""
    out: dict[str, dict] = {}
    for key in _SUMMARY_STATS_KEYS:
        diffs = [r.differences.get(key, 0.0) for r in results]
        try:
            out[key] = paired_difference_stats(
                diffs,
                direction=METRIC_DIRECTIONS[key],
            ).to_dict()
        except ValueError:
            out[key] = None
    return out


def compute_summary(results: list[PairedResult], config: ExperimentConfig) -> ExperimentSummary:
    if not results:
        return ExperimentSummary(
            num_experiments=0,
            scenario=config.scenario,
            days=config.days,
            base_seed=config.base_seed,
            adaptive_wins=0,
            immediate_wins=0,
            ties=0,
            on_time_pct_diff_mean=0,
            on_time_pct_diff_median=0,
            on_time_pct_diff_std=0,
            avg_delivery_min_diff_mean=0,
            avg_delivery_min_diff_median=0,
            avg_delivery_min_diff_std=0,
            p50_delivery_min_diff_mean=0,
            p90_delivery_min_diff_mean=0,
            p95_delivery_min_diff_mean=0,
            avg_late_min_diff_mean=0,
            avg_late_min_diff_median=0,
            avg_late_min_diff_std=0,
            avg_order_wait_min_diff_mean=0,
            avg_rider_wait_kitchen_min_diff_mean=0,
            avg_rider_wait_kitchen_min_diff_median=0,
            cost_score_diff_mean=0,
            cost_score_diff_median=0,
            cost_score_diff_std=0,
            scenario_breakdown={},
        )

    # Count wins
    adaptive_wins = 0
    immediate_wins = 0
    ties = 0

    for r in results:
        # Win defined by end-to-end delivery time (primary objective).
        # A run is a tie unless average delivery differs by the threshold.
        if r.adaptive["avg_delivery_min"] < r.immediate["avg_delivery_min"] - DELIVERY_WIN_THRESHOLD:
            adaptive_wins += 1
        elif r.immediate["avg_delivery_min"] < r.adaptive["avg_delivery_min"] - DELIVERY_WIN_THRESHOLD:
            immediate_wins += 1
        else:
            ties += 1

    # Collect differences
    on_time_diffs = [r.differences["on_time_rate"] for r in results]
    delivery_diffs = [r.differences["avg_delivery_min"] for r in results]
    p50_diffs = [r.differences.get("p50_delivery_min", 0) for r in results]
    p90_diffs = [r.differences.get("p90_delivery_min", 0) for r in results]
    p95_diffs = [r.differences.get("p95_delivery_min", 0) for r in results]
    p99_diffs = [r.differences.get("p99_delivery_min", 0) for r in results]
    max_diffs = [r.differences.get("max_delivery_min", 0) for r in results]
    late_diffs = [r.differences["avg_late_min"] for r in results]
    late_count_diffs = [r.differences.get("late_count", 0) for r in results]
    order_wait_diffs = [r.differences["avg_order_wait_min"] for r in results]
    rider_wait_diffs = [r.differences["avg_rider_wait_kitchen_min"] for r in results]
    cost_diffs = [r.differences["cost_score"] for r in results]

    return ExperimentSummary(
        num_experiments=len(results),
        scenario=config.scenario,
        days=config.days,
        base_seed=config.base_seed,
        adaptive_wins=adaptive_wins,
        immediate_wins=immediate_wins,
        ties=ties,
        on_time_pct_diff_mean=statistics.mean(on_time_diffs) if on_time_diffs else 0,
        on_time_pct_diff_median=statistics.median(on_time_diffs) if on_time_diffs else 0,
        on_time_pct_diff_std=statistics.stdev(on_time_diffs) if len(on_time_diffs) > 1 else 0,
        avg_delivery_min_diff_mean=statistics.mean(delivery_diffs) if delivery_diffs else 0,
        avg_delivery_min_diff_median=statistics.median(delivery_diffs) if delivery_diffs else 0,
        avg_delivery_min_diff_std=statistics.stdev(delivery_diffs) if len(delivery_diffs) > 1 else 0,
        p50_delivery_min_diff_mean=statistics.mean(p50_diffs) if p50_diffs else 0,
        p90_delivery_min_diff_mean=statistics.mean(p90_diffs) if p90_diffs else 0,
        p95_delivery_min_diff_mean=statistics.mean(p95_diffs) if p95_diffs else 0,
        avg_late_min_diff_mean=statistics.mean(late_diffs) if late_diffs else 0,
        avg_late_min_diff_median=statistics.median(late_diffs) if late_diffs else 0,
        avg_late_min_diff_std=statistics.stdev(late_diffs) if len(late_diffs) > 1 else 0,
        avg_order_wait_min_diff_mean=statistics.mean(order_wait_diffs) if order_wait_diffs else 0,
        avg_rider_wait_kitchen_min_diff_mean=statistics.mean(rider_wait_diffs) if rider_wait_diffs else 0,
        avg_rider_wait_kitchen_min_diff_median=statistics.median(rider_wait_diffs) if rider_wait_diffs else 0,
        cost_score_diff_mean=statistics.mean(cost_diffs) if cost_diffs else 0,
        cost_score_diff_median=statistics.median(cost_diffs) if cost_diffs else 0,
        cost_score_diff_std=statistics.stdev(cost_diffs) if len(cost_diffs) > 1 else 0,
        scenario_breakdown={},
        p99_delivery_min_diff_mean=statistics.mean(p99_diffs) if p99_diffs else 0,
        max_delivery_min_diff_mean=statistics.mean(max_diffs) if max_diffs else 0,
        late_count_diff_mean=statistics.mean(late_count_diffs) if late_count_diffs else 0,

        # Win-rate reporting (win criterion: avg_delivery_min). 95% Wilson
        # interval on the adaptive win proportion.
        adaptive_win_rate=adaptive_wins / len(results),
        immediate_win_rate=immediate_wins / len(results),
        tie_rate=ties / len(results),
        **_win_rate_ci(adaptive_wins, len(results)),
        win_method="avg_delivery_min",
        paired_stats=_paired_stats_for(results),
    )


def _win_rate_ci(wins, n):
    lo, hi = wilson_ci(wins, n) if n else (0.0, 0.0)
    return {"adaptive_win_rate_ci_low": lo, "adaptive_win_rate_ci_high": hi}


def save_results(results: list[PairedResult], summary: ExperimentSummary, out_dir: str):
    """Save experiment results to CSV and JSON."""
    # Save paired results CSV
    csv_path = os.path.join(out_dir, "experiments.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "scenario", "days",
            "immediate_on_time_rate", "immediate_avg_delivery_min",
            "immediate_p50_delivery_min", "immediate_p90_delivery_min", "immediate_p95_delivery_min",
            "immediate_p99_delivery_min", "immediate_max_delivery_min", "immediate_late_count",
            "immediate_avg_late_min",
            "immediate_avg_order_wait_min", "immediate_avg_rider_wait_kitchen_min",
            "immediate_avg_rider_idle_min", "immediate_cost_score",
            "adaptive_on_time_rate", "adaptive_avg_delivery_min",
            "adaptive_p50_delivery_min", "adaptive_p90_delivery_min", "adaptive_p95_delivery_min",
            "adaptive_p99_delivery_min", "adaptive_max_delivery_min", "adaptive_late_count",
            "adaptive_avg_late_min",
            "adaptive_avg_order_wait_min", "adaptive_avg_rider_wait_kitchen_min",
            "adaptive_avg_rider_idle_min", "adaptive_cost_score",
            "diff_on_time_rate", "diff_avg_delivery_min",
            "diff_p50_delivery_min", "diff_p90_delivery_min", "diff_p95_delivery_min",
            "diff_p99_delivery_min", "diff_max_delivery_min", "diff_late_count",
            "diff_avg_late_min",
            "diff_avg_order_wait_min", "diff_avg_rider_wait_kitchen_min",
            "diff_avg_rider_idle_min", "diff_cost_score",
            "immediate_orders_in_flight", "adaptive_orders_in_flight", "diff_orders_in_flight",
        ])
        for r in results:
            def _g(d, key):
                return d.get(key, 0.0)

            writer.writerow([
                r.seed, r.scenario, r.days,
                r.immediate["on_time_rate"], r.immediate["avg_delivery_min"],
                _g(r.immediate, "p50_delivery_min"), _g(r.immediate, "p90_delivery_min"), _g(r.immediate, "p95_delivery_min"),
                _g(r.immediate, "p99_delivery_min"), _g(r.immediate, "max_delivery_min"), _g(r.immediate, "late_count"),
                r.immediate["avg_late_min"],
                r.immediate["avg_order_wait_min"], r.immediate["avg_rider_wait_kitchen_min"],
                r.immediate["avg_rider_idle_min"], r.immediate["cost_score"],
                r.adaptive["on_time_rate"], r.adaptive["avg_delivery_min"],
                _g(r.adaptive, "p50_delivery_min"), _g(r.adaptive, "p90_delivery_min"), _g(r.adaptive, "p95_delivery_min"),
                _g(r.adaptive, "p99_delivery_min"), _g(r.adaptive, "max_delivery_min"), _g(r.adaptive, "late_count"),
                r.adaptive["avg_late_min"],
                r.adaptive["avg_order_wait_min"], r.adaptive["avg_rider_wait_kitchen_min"],
                r.adaptive["avg_rider_idle_min"], r.adaptive["cost_score"],
                r.differences["on_time_rate"], r.differences["avg_delivery_min"],
                _g(r.differences, "p50_delivery_min"), _g(r.differences, "p90_delivery_min"), _g(r.differences, "p95_delivery_min"),
                _g(r.differences, "p99_delivery_min"), _g(r.differences, "max_delivery_min"), _g(r.differences, "late_count"),
                r.differences["avg_late_min"],
                r.differences["avg_order_wait_min"],                 r.differences["avg_rider_wait_kitchen_min"],
                r.differences["avg_rider_idle_min"], r.differences["cost_score"],
                _g(r.immediate, "orders_in_flight"), _g(r.adaptive, "orders_in_flight"),
                _g(r.differences, "orders_in_flight"),
            ])

    # Save summary JSON
    summary_path = os.path.join(out_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)

    print(f"\nResults saved to {out_dir}")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


def load_results(out_dir: str) -> tuple[list[PairedResult], ExperimentSummary]:
    """Load experiment results from CSV and JSON."""
    csv_path = os.path.join(out_dir, "experiments.csv")
    summary_path = os.path.join(out_dir, "experiment_summary.json")

    results = []
    if os.path.exists(csv_path):

        def _f(row, key):
            return float(row[key]) if key in row and row[key] != "" else 0.0

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(PairedResult(
                    seed=int(row["seed"]),
                    scenario=row["scenario"],
                    days=int(row["days"]),
                    immediate={
                        "on_time_rate": _f(row, "immediate_on_time_rate"),
                        "avg_delivery_min": _f(row, "immediate_avg_delivery_min"),
                        "p50_delivery_min": _f(row, "immediate_p50_delivery_min"),
                        "p90_delivery_min": _f(row, "immediate_p90_delivery_min"),
                        "p95_delivery_min": _f(row, "immediate_p95_delivery_min"),
                        "p99_delivery_min": _f(row, "immediate_p99_delivery_min"),
                        "max_delivery_min": _f(row, "immediate_max_delivery_min"),
                        "late_count": _f(row, "immediate_late_count"),
                        "avg_late_min": _f(row, "immediate_avg_late_min"),
                        "avg_order_wait_min": _f(row, "immediate_avg_order_wait_min"),
                        "avg_rider_wait_kitchen_min": _f(row, "immediate_avg_rider_wait_kitchen_min"),
                        "avg_rider_idle_min": _f(row, "immediate_avg_rider_idle_min"),
                        "cost_score": _f(row, "immediate_cost_score"),
                        "orders_in_flight": _f(row, "immediate_orders_in_flight"),
                    },
                    adaptive={
                        "on_time_rate": _f(row, "adaptive_on_time_rate"),
                        "avg_delivery_min": _f(row, "adaptive_avg_delivery_min"),
                        "p50_delivery_min": _f(row, "adaptive_p50_delivery_min"),
                        "p90_delivery_min": _f(row, "adaptive_p90_delivery_min"),
                        "p95_delivery_min": _f(row, "adaptive_p95_delivery_min"),
                        "p99_delivery_min": _f(row, "adaptive_p99_delivery_min"),
                        "max_delivery_min": _f(row, "adaptive_max_delivery_min"),
                        "late_count": _f(row, "adaptive_late_count"),
                        "avg_late_min": _f(row, "adaptive_avg_late_min"),
                        "avg_order_wait_min": _f(row, "adaptive_avg_order_wait_min"),
                        "avg_rider_wait_kitchen_min": _f(row, "adaptive_avg_rider_wait_kitchen_min"),
                        "avg_rider_idle_min": _f(row, "adaptive_avg_rider_idle_min"),
                        "cost_score": _f(row, "adaptive_cost_score"),
                        "orders_in_flight": _f(row, "adaptive_orders_in_flight"),
                    },
                    differences={
                        "on_time_rate": _f(row, "diff_on_time_rate"),
                        "avg_delivery_min": _f(row, "diff_avg_delivery_min"),
                        "p50_delivery_min": _f(row, "diff_p50_delivery_min"),
                        "p90_delivery_min": _f(row, "diff_p90_delivery_min"),
                        "p95_delivery_min": _f(row, "diff_p95_delivery_min"),
                        "p99_delivery_min": _f(row, "diff_p99_delivery_min"),
                        "max_delivery_min": _f(row, "diff_max_delivery_min"),
                        "late_count": _f(row, "diff_late_count"),
                        "avg_late_min": _f(row, "diff_avg_late_min"),
                        "avg_order_wait_min": _f(row, "diff_avg_order_wait_min"),
                        "avg_rider_wait_kitchen_min": _f(row, "diff_avg_rider_wait_kitchen_min"),
                        "avg_rider_idle_min": _f(row, "diff_avg_rider_idle_min"),
                        "cost_score": _f(row, "diff_cost_score"),
                        "orders_in_flight": _f(row, "diff_orders_in_flight"),
                    },
                ))

    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
            summary = ExperimentSummary(**data)

    return results, summary


def print_summary(summary: ExperimentSummary):
    """Print a formatted summary of experiment results."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT SUMMARY: {summary.scenario} ({summary.days} day(s), {summary.num_experiments} paired runs)")
    print("=" * 70)
    print(f"\nWIN CRITERION: average end-to-end delivery time (min) "
          f"(tie unless |diff| > {DELIVERY_WIN_THRESHOLD:.2f} min)")
    print("\nWIN COUNTS:")
    print(f"  Adaptive wins:  {summary.adaptive_wins}")
    print(f"  Immediate wins: {summary.immediate_wins}")
    print(f"  Ties:           {summary.ties}")
    print(f"  Adaptive win rate: {summary.adaptive_win_rate*100:.1f}% "
          f"(95% CI {summary.adaptive_win_rate_ci_low*100:.1f}% - {summary.adaptive_win_rate_ci_high*100:.1f}%)")

    print("\nMETRIC DIFFERENCES (Adaptive - Immediate):")
    print(f"  {'Metric':<40} {'Mean':>10} {'Median':>10} {'Std':>10}")
    print(f"  {'-'*70}")
    print(f"  {'On-time % (pp)':<40} {summary.on_time_pct_diff_mean*100:>10.2f} {summary.on_time_pct_diff_median*100:>10.2f} {summary.on_time_pct_diff_std*100:>10.2f}")
    print(f"  {'Avg delivery (min)':<40} {summary.avg_delivery_min_diff_mean:>10.3f} {summary.avg_delivery_min_diff_median:>10.3f} {summary.avg_delivery_min_diff_std:>10.3f}")
    print(f"  {'P50 delivery (min)':<40} {summary.p50_delivery_min_diff_mean:>10.3f}")
    print(f"  {'P90 delivery (min)':<40} {summary.p90_delivery_min_diff_mean:>10.3f}")
    print(f"  {'P95 delivery (min)':<40} {summary.p95_delivery_min_diff_mean:>10.3f}")
    print(f"  {'P99 delivery (min)':<40} {summary.p99_delivery_min_diff_mean:>10.3f}")
    print(f"  {'Max delivery (min)':<40} {summary.max_delivery_min_diff_mean:>10.3f}")
    print(f"  {'Late orders (#)':<40} {summary.late_count_diff_mean:>10.3f}")
    print(f"  {'Avg lateness (min)':<40} {summary.avg_late_min_diff_mean:>10.3f} {summary.avg_late_min_diff_median:>10.3f} {summary.avg_late_min_diff_std:>10.3f}")
    print(f"  {'Avg order wait (min)':<40} {summary.avg_order_wait_min_diff_mean:>10.3f}")
    print(f"  {'Avg rider kitchen wait (min)':<40} {summary.avg_rider_wait_kitchen_min_diff_mean:>10.3f} {summary.avg_rider_wait_kitchen_min_diff_median:>10.3f}")

    print("\nNOTE: Negative values mean Adaptive is better (lower is better for all metrics "
          "except on-time %)")
    print("      Positive on-time % means Adaptive has higher on-time rate")
    print("      Cost score differences are reported for transparency only; wins use delivery time.")

    print(f"\nPAIRED STATISTICS (method: {summary.paired_stats[next(iter(summary.paired_stats))]['method'] if summary.paired_stats else 'n/a'})")
    print(f"  {'Metric':<32} {'Mean':>9} {'95% CI':>22} {'p(perm)':>9} {'p(t)':>9} {'d_z':>7} {'sig':>4}")
    print(f"  {'-'*90}")
    for key in _SUMMARY_STATS_KEYS:
        st = summary.paired_stats.get(key)
        if st is None:
            continue
        ci = f"[{st['ci95_low']:+.3f}, {st['ci95_high']:+.3f}]"
        print(f"  {key:<32} {st['mean_diff']:>+9.3f} {ci:>22} {st['p_value_permutation']:>9.4f} "
              f"{st['p_value_ttest']:>9.4f} {st['cohens_dz']:>+7.3f} {'yes' if st['significant'] else 'no':>4}")
    print("=" * 70)


def run_multi_scenario_experiment(
    scenarios: list[ScenarioType],
    num_experiments_per_scenario: int = 2000,
    base_seed: int = 42,
    days: int = 1,
    predictor_dir: str = "artifacts",
    out_dir: str = "data/experiments",
    progress_callback=None,
    save_distributions: bool = True,
) -> dict[str, tuple[list[PairedResult], ExperimentSummary]]:
    """Run experiments across multiple scenarios."""
    all_results = {}

    total = num_experiments_per_scenario * len(scenarios)
    done = 0

    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*70}")

        config = ExperimentConfig(
            num_experiments=num_experiments_per_scenario,
            base_seed=base_seed,
            days=days,
            scenario=scenario,
            predictor_dir=predictor_dir,
            out_dir=os.path.join(out_dir, scenario),
            save_individual=False,
            save_distributions=save_distributions,
        )

        def _cb(current: int, _total: int, done: int = done) -> None:
            if progress_callback:
                progress_callback(done + current, total)

        results, summary = run_experiment(config, progress_callback=_cb)
        save_results(results, summary, config.out_dir)
        print_summary(summary)

        done += num_experiments_per_scenario
        all_results[scenario] = (results, summary)

    # Print cross-scenario comparison
    print("\n" + "=" * 70)
    print("CROSS-SCENARIO COMPARISON")
    print("=" * 70)
    print(f"\n{'Scenario':<20} {'Adaptive Wins':>12} {'Immediate Wins':>14} {'Ties':>8} {'Win Rate':>10}")
    print(f"  {'-'*64}")
    for scenario, (_, summary) in all_results.items():
        win_rate = summary.adaptive_wins / max(summary.num_experiments, 1) * 100
        print(f"  {scenario:<20} {summary.adaptive_wins:>12} {summary.immediate_wins:>14} {summary.ties:>8} {win_rate:>9.1f}%")

    return all_results