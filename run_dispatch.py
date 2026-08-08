import argparse
import os
import tempfile

from simulation import SimulationEngine, load_scenario
from dispatch.metrics import format_metrics


def run_policy(policy: str, scenario: str, seed: int, days: int, predictor_dir: str, out_root: str):
    config = load_scenario(scenario, seed=seed)
    config["days"] = days
    config["dispatch"]["default_policy"] = policy
    config["dispatch"]["predictor_dir"] = predictor_dir
    out_dir = os.path.join(out_root, f"{scenario}_seed{seed}_day{days}_{policy}")
    engine = SimulationEngine(config, out_dir=out_dir, scenario_name=scenario)
    summary = engine.run()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the Phase 3 dispatch engine.")
    parser.add_argument("--policy", default=None, choices=["immediate", "adaptive"],
                        help="Policy to run (default: config/dispatch.yaml default_policy)")
    parser.add_argument("--scenario", default="normal", help="Scenario preset name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--predictor-dir", default="artifacts",
                        help="Phase 2 artifacts dir (required for adaptive)")
    parser.add_argument("--compare", action="store_true",
                        help="Run immediate + adaptive on the same scenario/seed and compare")
    parser.add_argument("--out", default="data/runs", help="Output root for run artifacts")
    args = parser.parse_args()

    if args.compare:
        rows = []
        for policy in ("immediate", "adaptive"):
            summary = run_policy(policy, args.scenario, args.seed, args.days,
                                 args.predictor_dir, args.out)
            rows.append((policy, summary))
            print(f"\n[{policy}] {format_metrics(summary)}")
        print("\n=== Comparison ===")
        header = ("policy", "placed", "completed", "cancelled", "on_time", "avg_wait", "late", "rider_kitchen", "cost")
        print(f"{header[0]:<10} {header[1]:>6} {header[2]:>9} {header[3]:>9} {header[4]:>8} "
              f"{header[5]:>8} {header[6]:>6} {header[7]:>12} {header[8]:>8}")
        for policy, s in rows:
            print(f"{policy:<10} {s['orders_placed']:>6} {s['orders_completed']:>9} "
                  f"{s['orders_cancelled']:>9} {s['on_time_rate']:>7.1%} "
                  f"{s['avg_order_wait_min']:>8.3f} {s['avg_late_min']:>6.3f} "
                  f"{s['avg_rider_wait_kitchen_min']:>12.3f} {s['cost_score']:>8.1f}")
        return

    policy = args.policy or load_scenario(args.scenario, args.seed)["dispatch"]["default_policy"]
    summary = run_policy(policy, args.scenario, args.seed, args.days, args.predictor_dir, args.out)
    print(f"policy={policy} {format_metrics(summary)}")


if __name__ == "__main__":
    main()
