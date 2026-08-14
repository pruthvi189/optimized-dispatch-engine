#!/usr/bin/env python
"""CLI for running paired Adaptive vs Immediate dispatch experiments."""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dispatch.experiment import (
    ExperimentConfig,
    run_experiment,
    run_multi_scenario_experiment,
    save_results,
    print_summary,
)


def main():
    parser = argparse.ArgumentParser(description="Run paired Adaptive vs Immediate dispatch experiments.")
    parser.add_argument("--experiments", type=int, default=10000,
                        help="Number of paired experiments to run (default: 10000)")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days to simulate per experiment (default: 1)")
    parser.add_argument("--scenario", default="normal",
                        choices=["normal", "lunch_rush", "rain", "low_staffing", "traffic_spike"],
                        help="Scenario to test (default: normal)")
    parser.add_argument("--predictor-dir", default="artifacts",
                        help="Phase 2 predictor artifacts directory (default: artifacts)")
    parser.add_argument("--out", default="data/experiments",
                        help="Output directory for results (default: data/experiments)")
    parser.add_argument("--multi-scenario", action="store_true",
                        help="Run experiments across all scenarios")
    parser.add_argument("--experiments-per-scenario", type=int, default=2000,
                        help="Number of experiments per scenario in multi-scenario mode (default: 2000)")

    args = parser.parse_args()

    if args.multi_scenario:
        scenarios = ["normal", "lunch_rush", "rain", "low_staffing", "traffic_spike"]
        run_multi_scenario_experiment(
            scenarios=scenarios,
            num_experiments_per_scenario=args.experiments_per_scenario,
            base_seed=args.base_seed,
            days=args.days,
            predictor_dir=args.predictor_dir,
            out_dir=args.out,
        )
    else:
        config = ExperimentConfig(
            num_experiments=args.experiments,
            base_seed=args.base_seed,
            days=args.days,
            scenario=args.scenario,
            predictor_dir=args.predictor_dir,
            out_dir=args.out,
            save_individual=False,
            save_distributions=True,
        )

        print(f"Running {config.num_experiments} paired experiments...")
        print(f"  Scenario: {config.scenario}")
        print(f"  Base seed: {config.base_seed}")
        print(f"  Days: {config.days}")
        print(f"  Output: {config.out_dir}")

        results, summary = run_experiment(config)
        save_results(results, summary, config.out_dir)
        print_summary(summary)


if __name__ == "__main__":
    main()