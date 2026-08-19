"""Small validation experiment: Immediate vs OptimizedKitchen (30 seeds × 5 scenarios).

Verifies correctness of the kitchen-selection pipeline. Results are saved
to data/experiments/validation_kitchen_selection/.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.engine import SimulationEngine
from simulation.scenarios import load_scenario

SEEDS_PER_SCENARIO = 30
DAYS = 1
SCENARIOS = ["normal", "low_staffing", "lunch_rush", "rain", "traffic_spike"]
OUT_BASE = "data/experiments/validation_kitchen_selection"


def run_pair(scenario, seed):
    """Run one seed with both policies. Returns (immediate_summary, optimized_summary)."""
    # Immediate (baseline).
    cfg_i = load_scenario(scenario, seed=seed)
    cfg_i["dispatch"]["enabled"] = True
    cfg_i["dispatch"]["default_policy"] = "immediate"
    cfg_i["days"] = DAYS
    out_i = os.path.join(OUT_BASE, scenario, f"seed_{seed}_immediate")
    engine_i = SimulationEngine(cfg_i, out_dir=out_i, scenario_name=scenario)
    summary_i = engine_i.run()

    # OptimizedKitchen.
    cfg_o = load_scenario(scenario, seed=seed)
    cfg_o["dispatch"]["enabled"] = True
    cfg_o["dispatch"]["default_policy"] = "optimized_kitchen"
    cfg_o["days"] = DAYS
    out_o = os.path.join(OUT_BASE, scenario, f"seed_{seed}_optimized")
    engine_o = SimulationEngine(cfg_o, out_dir=out_o, scenario_name=scenario)
    summary_o = engine_o.run()

    return summary_i, summary_o


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    rows = []
    t0 = time.time()
    total_pairs = len(SCENARIOS) * SEEDS_PER_SCENARIO
    done = 0

    for scenario in SCENARIOS:
        for seed in range(1, SEEDS_PER_SCENARIO + 1):
            try:
                si, so = run_pair(scenario, seed)
                delta_delivery = so["avg_delivery_min"] - si["avg_delivery_min"]
                delta_on_time = so["on_time_rate"] - si["on_time_rate"]
                delta_cost = so["cost_score"] - si["cost_score"]
                rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    # Immediate (baseline).
                    "imm_delivery": si["avg_delivery_min"],
                    "imm_on_time": si["on_time_rate"],
                    "imm_cost": si["cost_score"],
                    "imm_completed": si["orders_completed"],
                    # OptimizedKitchen.
                    "opt_delivery": so["avg_delivery_min"],
                    "opt_on_time": so["on_time_rate"],
                    "opt_cost": so["cost_score"],
                    "opt_completed": so["orders_completed"],
                    # Kitchen selection metrics.
                    "opt_avg_kitchen_dist": so.get("avg_selected_kitchen_distance_km"),
                    "opt_kitchen_load_std": so.get("kitchen_load_std"),
                    # Deltas.
                    "delta_delivery": delta_delivery,
                    "delta_on_time": delta_on_time,
                    "delta_cost": delta_cost,
                })
                done += 1
                if done % 5 == 0 or done == total_pairs:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total_pairs - done) / rate if rate > 0 else 0
                    print(f"  [{done}/{total_pairs}] {scenario} seed={seed} "
                          f"delivery_delta={delta_delivery:+.2f}min "
                          f"on_time_delta={delta_on_time:+.1%} "
                          f"({rate:.1f} runs/s, ETA {eta:.0f}s)")
            except Exception as e:
                print(f"  ERROR {scenario} seed={seed}: {e}")
                rows.append({"scenario": scenario, "seed": seed, "error": str(e)})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_BASE, "validation_results.csv")
    df.to_csv(csv_path, index=False)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Results: {csv_path}")

    # Summary by scenario.
    valid = df[df.get("error", pd.Series(dtype=str)).isna() | ~df["error"].notna()]
    if "delta_delivery" in valid.columns:
        print("\n=== SUMMARY BY SCENARIO ===")
        for sc in SCENARIOS:
            sub = valid[valid["scenario"] == sc]
            if sub.empty:
                continue
            imm_avg = sub["imm_delivery"].mean()
            opt_avg = sub["opt_delivery"].mean()
            delta = opt_avg - imm_avg
            on_time_imm = sub["imm_on_time"].mean()
            on_time_opt = sub["opt_on_time"].mean()
            kitchen_dist = sub["opt_avg_kitchen_dist"].dropna().mean()
            print(f"  {sc:15s}: imm={imm_avg:.2f}min opt={opt_avg:.2f}min "
                  f"delta={delta:+.2f}min | on_time_imm={on_time_imm:.1%} "
                  f"on_time_opt={on_time_opt:.1%} | avg_kitchen_dist={kitchen_dist:.2f}km")

        imm_all = valid["imm_delivery"].mean()
        opt_all = valid["opt_delivery"].mean()
        print(f"\n  ALL SCENARIOS : imm={imm_all:.2f}min opt={opt_all:.2f}min "
              f"delta={opt_all - imm_all:+.2f}min")


if __name__ == "__main__":
    main()
