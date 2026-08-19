import os, sys, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dispatch.experiment import run_paired_experiment, DistributionAccumulator
from simulation import load_scenario

SCENARIOS = ["normal", "lunch_rush", "rain", "low_staffing", "traffic_spike"]
NUM = 2000
BASE_SEED = 42
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "experiments")
PREDICTOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _run_batch(args):
    scenario, seeds = args
    base_config = load_scenario(scenario, seed=BASE_SEED)
    acc = DistributionAccumulator()
    for seed in seeds:
        run_paired_experiment(
            seed=seed,
            base_config=base_config,
            predictor_dir=PREDICTOR,
            out_root=OUT_ROOT,
            scenario_name=scenario,
            save_individual=False,
            dist_accumulator=acc,
        )
    return scenario, acc


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    for scenario in SCENARIOS:
        seeds = list(range(BASE_SEED, BASE_SEED + NUM))
        n_workers = 8
        chunk = max(1, len(seeds) // n_workers)
        batches = [seeds[i:i + chunk] for i in range(0, len(seeds), chunk)]
        t0 = time.time()
        accs = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_run_batch, (scenario, b)): i for i, b in enumerate(batches)}
            for fut in as_completed(futs):
                s, acc = fut.result()
                if s not in accs:
                    accs[s] = DistributionAccumulator()
                accs[s].adaptive_counts += acc.adaptive_counts
                accs[s].immediate_counts += acc.immediate_counts
                accs[s].adaptive_times += acc.adaptive_times
                accs[s].immediate_times += acc.immediate_times
                accs[s].adaptive_total += acc.adaptive_total
                accs[s].immediate_total += acc.immediate_total
                accs[s].adaptive_delivery_sum += acc.adaptive_delivery_sum
                accs[s].immediate_delivery_sum += acc.immediate_delivery_sum
                accs[s].num_runs += acc.num_runs
        merged = accs[scenario]
        out_dir = os.path.join(OUT_ROOT, scenario)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "delivery_distribution.json"), "w") as f:
            json.dump(merged.to_dict(scenario), f, indent=2)
        elapsed = time.time() - t0
        print(f"[{scenario}] {merged.num_runs} paired runs in {elapsed:.0f}s -> saved", flush=True)


if __name__ == "__main__":
    main()
