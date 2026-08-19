"""Bounded parameter search for JointOptimizerDispatch.

Splits: train = seeds 1-10, val = seeds 11-20.
Searches over queue_wait_weight, traffic_sensitivity, weather_sensitivity,
rider_to_kitchen_weight. Locks best params, validates on held-out set.
Reports honest results.
"""

import os
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\pruth\Desktop\projects\extra projects\adaptive-dispatch-engine")

from simulation.engine import SimulationEngine
from simulation.scenarios import load_scenario

SCENARIOS = ["normal", "low_staffing", "lunch_rush", "rain", "traffic_spike"]
TRAIN_SEEDS = list(range(1, 11))
VAL_SEEDS = list(range(11, 21))
OUT_DIR = "data/param_search"

# Parameter grid (bounded, compact for speed).
PARAM_GRID = {
    "weight_queue_wait": [1.0, 2.0, 5.0, 10.0],
    "traffic_sensitivity": [1.0, 2.0, 3.0, 5.0],
    "weather_sensitivity": [1.0, 2.0, 3.0],
    "weight_rider_to_kitchen": [1.0, 2.0, 3.0],
}


def run_one(scenario, seed, policy="joint_optimizer", params=None):
    cfg = load_scenario(scenario, seed=seed)
    cfg["dispatch"]["enabled"] = True
    cfg["dispatch"]["default_policy"] = policy
    cfg["days"] = 1
    if params:
        cfg["dispatch"].setdefault("kitchen_selection", {})["optimizer"] = params
    engine = SimulationEngine(cfg, out_dir=os.devnull, scenario_name=scenario,
                              save_outputs=False)
    summary = engine.run()
    return engine, summary


def avg_delivery_time(seeds, scenarios, params=None):
    """Run all (seed, scenario) combos and return mean delivery time."""
    times = []
    for scenario in scenarios:
        for seed in seeds:
            try:
                engine, _ = run_one(scenario, seed, params=params)
                delivered = [o for o in engine.orders if o.delivered_at is not None]
                for o in delivered:
                    times.append(o.delivered_at - o.placed_at)
            except Exception as e:
                print(f"  ERROR seed={seed} {scenario}: {e}")
    return np.mean(times) if times else float("inf")


def grid_search():
    """Exhaustive grid search over parameter combinations on train set."""
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total = len(combos)
    print(f"Grid search: {total} combinations x {len(TRAIN_SEEDS)} seeds x "
          f"{len(SCENARIOS)} scenarios = {total * len(TRAIN_SEEDS) * len(SCENARIOS)} runs",
          flush=True)

    best_avg = float("inf")
    best_params = None
    results = []
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        avg = avg_delivery_time(TRAIN_SEEDS, SCENARIOS, params=params)
        results.append({"params": params, "avg_delivery": round(avg, 4)})
        if avg < best_avg:
            best_avg = avg
            best_params = params
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] best={best_avg:.2f} ETA={eta:.0f}s  "
                  f"current={avg:.2f} params={params}", flush=True)

    elapsed = time.time() - t0
    print(f"\nGrid search done in {elapsed:.1f}s")
    print(f"Best train avg: {best_avg:.2f} min")
    print(f"Best params: {best_params}")

    # Sort results and print top 10.
    results.sort(key=lambda x: x["avg_delivery"])
    print("\nTop 10 parameter combinations:")
    for r in results[:10]:
        print(f"  avg={r['avg_delivery']:.4f}  {r['params']}")

    return best_params, best_avg, results


def validate(seeds, scenarios, params, label="VAL"):
    """Run full validation and return detailed metrics."""
    all_rows = []
    for scenario in scenarios:
        for seed in seeds:
            try:
                engine, _ = run_one(scenario, seed, params=params)
                delivered = [o for o in engine.orders if o.delivered_at is not None]
                for o in delivered:
                    dt = o.delivered_at - o.placed_at
                    rtk = None
                    if o.rider_arrived_kitchen_at and o.dispatch_at:
                        rtk = o.rider_arrived_kitchen_at - o.dispatch_at
                    kw = None
                    if o.pickup_at and o.rider_arrived_kitchen_at:
                        kw = o.pickup_at - o.rider_arrived_kitchen_at
                    ktc = None
                    if o.delivered_at and o.pickup_at:
                        ktc = o.delivered_at - o.pickup_at
                    all_rows.append({
                        "scenario": scenario, "seed": seed,
                        "delivery_time": dt, "on_time": dt <= 40.0,
                        "rider_to_kitchen": rtk, "kitchen_wait": kw,
                        "kitchen_to_customer": ktc,
                        "kitchen_id": o.kitchen_id,
                    })
            except Exception as e:
                print(f"  ERROR {label} seed={seed} {scenario}: {e}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return None, df

    pct = lambda p, arr: float(np.percentile(arr, p)) if len(arr) else 0
    summary = {
        "n_orders": len(df),
        "avg_delivery": round(df["delivery_time"].mean(), 4),
        "median_delivery": round(df["delivery_time"].median(), 4),
        "p95_delivery": round(pct(95, df["delivery_time"].values), 4),
        "on_time_rate": round(df["on_time"].mean(), 4),
        "rider_to_kitchen": round(df["rider_to_kitchen"].dropna().mean(), 4) if df["rider_to_kitchen"].notna().any() else None,
        "kitchen_wait": round(df["kitchen_wait"].dropna().mean(), 4) if df["kitchen_wait"].notna().any() else None,
        "kitchen_to_customer": round(df["kitchen_to_customer"].dropna().mean(), 4) if df["kitchen_to_customer"].notna().any() else None,
    }

    # Per-scenario breakdown.
    per_scenario = {}
    for sc in SCENARIOS:
        sub = df[df["scenario"] == sc]
        if sub.empty:
            continue
        per_scenario[sc] = {
            "n": len(sub),
            "avg": round(sub["delivery_time"].mean(), 4),
            "median": round(sub["delivery_time"].median(), 4),
            "p95": round(pct(95, sub["delivery_time"].values), 4),
            "on_time": round(sub["on_time"].mean(), 4),
        }
    summary["per_scenario"] = per_scenario
    return summary, df


def run_baseline(seeds, scenarios, label="BASELINE"):
    """Run nearest_heuristic baseline for comparison."""
    all_rows = []
    for scenario in scenarios:
        for seed in seeds:
            try:
                engine, _ = run_one(scenario, seed, policy="nearest_heuristic")
                delivered = [o for o in engine.orders if o.delivered_at is not None]
                for o in delivered:
                    dt = o.delivered_at - o.placed_at
                    rtk = None
                    if o.rider_arrived_kitchen_at and o.dispatch_at:
                        rtk = o.rider_arrived_kitchen_at - o.dispatch_at
                    kw = None
                    if o.pickup_at and o.rider_arrived_kitchen_at:
                        kw = o.pickup_at - o.rider_arrived_kitchen_at
                    ktc = None
                    if o.delivered_at and o.pickup_at:
                        ktc = o.delivered_at - o.pickup_at
                    all_rows.append({
                        "scenario": scenario, "seed": seed,
                        "delivery_time": dt, "on_time": dt <= 40.0,
                        "rider_to_kitchen": rtk, "kitchen_wait": kw,
                        "kitchen_to_customer": ktc,
                        "kitchen_id": o.kitchen_id,
                    })
            except Exception as e:
                print(f"  ERROR {label} seed={seed} {scenario}: {e}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return None, df

    pct = lambda p, arr: float(np.percentile(arr, p)) if len(arr) else 0
    summary = {
        "n_orders": len(df),
        "avg_delivery": round(df["delivery_time"].mean(), 4),
        "median_delivery": round(df["delivery_time"].median(), 4),
        "p95_delivery": round(pct(95, df["delivery_time"].values), 4),
        "on_time_rate": round(df["on_time"].mean(), 4),
        "rider_to_kitchen": round(df["rider_to_kitchen"].dropna().mean(), 4) if df["rider_to_kitchen"].notna().any() else None,
        "kitchen_wait": round(df["kitchen_wait"].dropna().mean(), 4) if df["kitchen_wait"].notna().any() else None,
        "kitchen_to_customer": round(df["kitchen_to_customer"].dropna().mean(), 4) if df["kitchen_to_customer"].notna().any() else None,
    }
    per_scenario = {}
    for sc in SCENARIOS:
        sub = df[df["scenario"] == sc]
        if sub.empty:
            continue
        per_scenario[sc] = {
            "n": len(sub),
            "avg": round(sub["delivery_time"].mean(), 4),
            "median": round(sub["delivery_time"].median(), 4),
            "p95": round(pct(95, sub["delivery_time"].values), 4),
            "on_time": round(sub["on_time"].mean(), 4),
        }
    summary["per_scenario"] = per_scenario
    return summary, df


def bootstrap_significance(opt_df, base_df, n_boot=10000):
    """Bootstrap test: is optimizer significantly better than baseline?
    Returns (mean_delta, p_value, ci_lo, ci_hi)."""
    rng = np.random.default_rng(42)
    opt_times = opt_df["delivery_time"].values
    base_times = base_df["delivery_time"].values
    observed_delta = opt_times.mean() - base_times.mean()

    # Pool residuals under null (shift opt to match base mean).
    opt_centered = opt_times - opt_times.mean()
    base_centered = base_times - base_times.mean()

    count = 0
    deltas = []
    for _ in range(n_boot):
        opt_sample = rng.choice(opt_centered, size=len(opt_times), replace=True) + base_times.mean()
        base_sample = rng.choice(base_centered, size=len(base_times), replace=True) + base_times.mean()
        delta = opt_sample.mean() - base_sample.mean()
        deltas.append(delta)
        if delta >= observed_delta:
            count += 1

    p_value = count / n_boot
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    return observed_delta, p_value, ci_lo, ci_hi


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # Step 1: Baseline on BOTH train and val.
    print("=" * 70)
    print("STEP 1: Running baseline (nearest_heuristic)")
    print("=" * 70)
    base_train_summary, base_train_df = run_baseline(TRAIN_SEEDS, SCENARIOS, "TRAIN")
    base_val_summary, base_val_df = run_baseline(VAL_SEEDS, SCENARIOS, "VAL")
    print(f"  Train baseline avg: {base_train_summary['avg_delivery']:.2f} min")
    print(f"  Val baseline avg:   {base_val_summary['avg_delivery']:.2f} min")

    # Step 2: Grid search on train set.
    print("\n" + "=" * 70)
    print("STEP 2: Grid search on train set (seeds 1-10)")
    print("=" * 70)
    best_params, best_train_avg, all_results = grid_search()

    # Step 3: Validate best params on val set.
    print("\n" + "=" * 70)
    print("STEP 3: Validation on held-out set (seeds 11-20)")
    print("=" * 70)
    print(f"Locked parameters: {best_params}")
    opt_val_summary, opt_val_df = validate(VAL_SEEDS, SCENARIOS, best_params, "VAL")

    # Also run optimizer on train set with locked params for reference.
    opt_train_summary, opt_train_df = validate(TRAIN_SEEDS, SCENARIOS, best_params, "TRAIN")

    # Step 4: Report results.
    print("\n" + "=" * 70)
    print("FINAL RESULTS (Locked Parameters)")
    print("=" * 70)

    print(f"\nLocked parameters: {json.dumps(best_params, indent=2)}")

    print(f"\n--- TRAIN SET (seeds 1-10) ---")
    print(f"  Baseline avg:  {base_train_summary['avg_delivery']:.2f} min")
    print(f"  Optimizer avg: {opt_train_summary['avg_delivery']:.2f} min")
    delta_train = opt_train_summary['avg_delivery'] - base_train_summary['avg_delivery']
    print(f"  Delta:         {delta_train:+.2f} min ({delta_train/base_train_summary['avg_delivery']*100:+.2f}%)")

    print(f"\n--- VAL SET (seeds 11-20) ---")
    print(f"  Baseline avg:  {base_val_summary['avg_delivery']:.2f} min")
    print(f"  Optimizer avg: {opt_val_summary['avg_delivery']:.2f} min")
    delta_val = opt_val_summary['avg_delivery'] - base_val_summary['avg_delivery']
    print(f"  Delta:         {delta_val:+.2f} min ({delta_val/base_val_summary['avg_delivery']*100:+.2f}%)")

    print(f"\n  Baseline median: {base_val_summary['median_delivery']:.2f}  "
          f"P95: {base_val_summary['p95_delivery']:.2f}  on_time: {base_val_summary['on_time_rate']:.1%}")
    print(f"  Optimizer median: {opt_val_summary['median_delivery']:.2f}  "
          f"P95: {opt_val_summary['p95_delivery']:.2f}  on_time: {opt_val_summary['on_time_rate']:.1%}")

    # Per-scenario on val set.
    print(f"\n--- PER-SCENARIO (VAL SET) ---")
    for sc in SCENARIOS:
        b = base_val_summary["per_scenario"].get(sc, {})
        o = opt_val_summary["per_scenario"].get(sc, {})
        if not b or not o:
            continue
        d = o["avg"] - b["avg"]
        print(f"  {sc:15s}: base={b['avg']:.2f}  opt={o['avg']:.2f}  delta={d:+.2f}  "
              f"on_time: {b['on_time']:.1%} -> {o['on_time']:.1%}")

    # Statistical significance.
    print(f"\n--- STATISTICAL SIGNIFICANCE (VAL SET) ---")
    delta, p_val, ci_lo, ci_hi = bootstrap_significance(opt_val_df, base_val_df)
    print(f"  Mean delta:  {delta:+.4f} min")
    print(f"  p-value:     {p_val:.4f}")
    print(f"  95% CI:      [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    if p_val < 0.05:
        print(f"  SIGNIFICANT at alpha=0.05")
    else:
        print(f"  NOT significant at alpha=0.05")

    # Component breakdown.
    print(f"\n--- COMPONENT BREAKDOWN (VAL SET) ---")
    for comp in ["rider_to_kitchen", "kitchen_wait", "kitchen_to_customer"]:
        b = base_val_summary.get(comp)
        o = opt_val_summary.get(comp)
        if b is not None and o is not None:
            d = o - b
            print(f"  {comp:25s}: base={b:.2f}  opt={o:.2f}  delta={d:+.2f}")

    # Save results.
    results = {
        "locked_params": best_params,
        "train": {"baseline": base_train_summary, "optimizer": opt_train_summary},
        "val": {"baseline": base_val_summary, "optimizer": opt_val_summary},
        "significance": {"delta": delta, "p_value": p_val, "ci_95": [ci_lo, ci_hi]},
    }
    with open(os.path.join(OUT_DIR, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}/final_results.json")

    # Save all grid search results.
    with open(os.path.join(OUT_DIR, "grid_search_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Grid search results saved to {OUT_DIR}/grid_search_results.json")

    print("\n" + "=" * 70)
    print("STOPPING. Locked validation results above.")
    print("=" * 70)
