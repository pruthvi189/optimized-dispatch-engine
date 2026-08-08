import argparse
import os
import shutil

from simulation import SimulationEngine, load_scenario


def main():
    parser = argparse.ArgumentParser(description="Generate pooled training data from scenario runs.")
    parser.add_argument("--scenarios", default="normal,rain,low_staffing", help="Comma-separated scenario names")
    parser.add_argument("--seeds", default="1,2,3", help="Comma-separated seeds")
    parser.add_argument("--days", type=int, default=2, help="Days per run")
    parser.add_argument("--out", default="data/train", help="Output directory for pooled CSVs")
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    os.makedirs(args.out, exist_ok=True)

    for scenario in scenarios:
        for seed in seeds:
            config = load_scenario(scenario, seed=seed)
            config["days"] = args.days
            out_dir = os.path.join(args.out, f"tmp_{scenario}_seed{seed}")
            engine = SimulationEngine(config, out_dir=out_dir, scenario_name=scenario)
            summary = engine.run()
            dest = os.path.join(args.out, f"{scenario}_seed{seed}.csv")
            shutil.copy(os.path.join(out_dir, "orders.csv"), dest)
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"{scenario} seed={seed}: {summary['orders_placed']} orders -> {dest}")


if __name__ == "__main__":
    main()
