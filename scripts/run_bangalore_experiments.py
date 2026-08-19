"""Run Bangalore-calibrated experiments across all scenarios.

Uses the existing paired experiment framework with the Bangalore-calibrated
simulator. Runs all 5 scenarios, 2000 paired runs each.

This script does NOT tune any adaptive policy parameters. It only applies
real-data calibration to the simulator distributions.

Usage:
    python scripts/run_bangalore_experiments.py
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dispatch.experiment import (
    ExperimentConfig,
    run_experiment,
    save_results,
    print_summary,
    run_multi_scenario_experiment,
)

SCENARIOS = ["normal", "lunch_rush", "rain", "low_staffing", "traffic_spike"]
EXPERIMENTS_PER_SCENARIO = 2000
BASE_SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "experiments")


def main():
    print("=" * 70)
    print("Bangalore-Calibrated Experiment Suite")
    print("=" * 70)
    print(f"Scenarios: {SCENARIOS}")
    print(f"Experiments per scenario: {EXPERIMENTS_PER_SCENARIO}")
    print(f"Base seed: {BASE_SEED}")
    print(f"Output: {OUT_DIR}")
    print(f"Predictor: artifacts (synthetic-data-trained, predicts prep time)")
    print()

    start = time.time()

    all_results = run_multi_scenario_experiment(
        scenarios=SCENARIOS,
        num_experiments_per_scenario=EXPERIMENTS_PER_SCENARIO,
        base_seed=BASE_SEED,
        days=1,
        predictor_dir="artifacts",
        out_dir=OUT_DIR,
        save_distributions=True,
    )

    elapsed = time.time() - start
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save cross-scenario comparison
    comparison = {}
    for scenario, (_, summary) in all_results.items():
        comparison[scenario] = {
            "adaptive_wins": summary.adaptive_wins,
            "immediate_wins": summary.immediate_wins,
            "ties": summary.ties,
            "win_rate": round(summary.adaptive_wins / max(summary.num_experiments, 1) * 100, 1),
            "avg_delivery_diff": round(summary.avg_delivery_min_diff_mean, 4),
            "p50_diff": round(summary.p50_delivery_min_diff_mean, 4),
            "p90_diff": round(summary.p90_delivery_min_diff_mean, 4),
            "p95_diff": round(summary.p95_delivery_min_diff_mean, 4),
            "p99_diff": round(summary.p99_delivery_min_diff_mean, 4),
            "on_time_diff": round(summary.on_time_pct_diff_mean * 100, 4),
            "late_count_diff": round(summary.late_count_diff_mean, 4),
            "cost_diff": round(summary.cost_score_diff_mean, 2),
        }

    comparison_path = os.path.join(OUT_DIR, "cross_scenario_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Cross-scenario comparison saved to {comparison_path}")


if __name__ == "__main__":
    main()
