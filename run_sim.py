import argparse
import os

from simulation import SimulationEngine
from simulation.scenarios import load_scenario


def main():
    parser = argparse.ArgumentParser(description="Run an adaptive dispatch simulation.")
    parser.add_argument("--scenario", default="normal", help="Scenario name (config/scenarios/<name>.yaml)")
    parser.add_argument("--seed", type=int, default=None, help="Override the scenario seed")
    parser.add_argument("--days", type=int, default=None, help="Override the number of days to simulate")
    parser.add_argument("--out", default=None, help="Output directory override")
    args = parser.parse_args()

    config = load_scenario(args.scenario, seed=args.seed)
    if args.days is not None:
        config["days"] = args.days

    if args.out:
        out_dir = args.out
    else:
        out_dir = os.path.join(
            "data", f"{args.scenario}_seed{config['seed']}_day{config['days']}"
        )

    engine = SimulationEngine(config, out_dir=out_dir, scenario_name=args.scenario)
    summary = engine.run()

    print("=== Simulation complete ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nOutputs written to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
