"""Experiment runner API endpoints."""

import json
import os
import threading
from dataclasses import asdict
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from dispatch.experiment import (
    ExperimentConfig,
    run_experiment,
    run_multi_scenario_experiment,
    load_results,
    print_summary,
)

router = APIRouter(prefix="/experiments", tags=["experiments"])

# Global experiment state
_experiment_thread = None
_experiment_running = False
_experiment_progress = {"current": 0, "total": 0, "status": "idle"}


class ExperimentRequest(BaseModel):
    num_experiments: int = Field(default=10000, ge=1, le=50000)
    base_seed: int = Field(default=42, ge=0)
    days: int = Field(default=1, ge=1, le=30)
    scenario: str = Field(default="normal")
    predictor_dir: str = Field(default="artifacts")
    out_dir: str = Field(default="data/experiments")
    multi_scenario: bool = Field(default=False)
    experiments_per_scenario: int = Field(default=2000, ge=1, le=20000)
    save_distributions: bool = Field(default=True)


class ExperimentStatusResponse(BaseModel):
    running: bool
    progress: dict
    results: dict | None = None


def _run_experiment_background(config: ExperimentConfig, multi_scenario: bool, experiments_per_scenario: int):
    """Background task to run experiments."""
    global _experiment_running, _experiment_progress

    try:
        _experiment_running = True

        if multi_scenario:
            scenarios = ["normal", "lunch_rush", "rain", "low_staffing", "traffic_spike"]
            total = experiments_per_scenario * len(scenarios)
            _experiment_progress = {"current": 0, "total": total, "status": "running"}

            def _cb(current: int, _total: int) -> None:
                _experiment_progress["current"] = current

            run_multi_scenario_experiment(
                scenarios=scenarios,
                num_experiments_per_scenario=experiments_per_scenario,
                base_seed=config.base_seed,
                days=config.days,
                predictor_dir=config.predictor_dir,
                out_dir=config.out_dir,
                progress_callback=_cb,
                save_distributions=config.save_distributions,
            )
        else:
            _experiment_progress = {"current": 0, "total": config.num_experiments, "status": "running"}

            def _cb(current: int, _total: int) -> None:
                _experiment_progress["current"] = current

            results, summary = run_experiment(config, progress_callback=_cb)
            from dispatch.experiment import save_results
            save_results(results, summary, config.out_dir)

        _experiment_progress["status"] = "completed"
        _experiment_progress["current"] = _experiment_progress["total"]
    except Exception as e:
        _experiment_progress["status"] = f"error: {str(e)}"
    finally:
        _experiment_running = False


@router.post("/run")
def run_experiment_endpoint(request: ExperimentRequest, background_tasks: BackgroundTasks):
    """Start a new experiment run in the background."""
    global _experiment_thread, _experiment_running

    if _experiment_running:
        raise HTTPException(status_code=409, detail="Experiment already running")

    config = ExperimentConfig(
        num_experiments=request.num_experiments,
        base_seed=request.base_seed,
        days=request.days,
        scenario=request.scenario,
        predictor_dir=request.predictor_dir,
        out_dir=request.out_dir,
        save_individual=False,
        save_distributions=request.save_distributions,
    )

    _experiment_thread = threading.Thread(
        target=_run_experiment_background,
        args=(config, request.multi_scenario, request.experiments_per_scenario),
        daemon=True,
    )
    _experiment_thread.start()

    return {"status": "started", "config": asdict(config)}


@router.get("/status")
def get_experiment_status() -> ExperimentStatusResponse:
    """Get current experiment status."""
    return ExperimentStatusResponse(
        running=_experiment_running,
        progress=_experiment_progress,
    )


@router.get("/results")
def get_experiment_results(out_dir: str = "data/experiments"):
    """Load and return the latest experiment results.

    Falls back to aggregating per-scenario subdirectories when no
    top-level summary exists (multi-scenario runs).
    """
    results, summary = load_results(out_dir)
    distribution = _load_distribution(out_dir)
    if summary is not None:
        return {
            "summary": asdict(summary),
            "num_results": len(results),
            "multi_scenario": False,
            "scenarios": {},
            "distributions": {"default": distribution} if distribution else {},
        }

    scenarios = _load_scenario_results(out_dir)
    if scenarios:
        return {
            "summary": _aggregate_scenario_summaries(scenarios),
            "num_results": sum(len(r) for r in scenarios.values()),
            "multi_scenario": True,
            "scenarios": {name: asdict(s) for name, (_, s) in scenarios.items()},
            "distributions": {
                name: _load_distribution(os.path.join(out_dir, name))
                for name in scenarios
            },
        }

    raise HTTPException(status_code=404, detail="No experiment results found")


def _load_distribution(scenario_dir: str) -> dict | None:
    """Load the per-scenario delivery-time distribution JSON if present."""
    path = os.path.join(scenario_dir, "delivery_distribution.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _load_scenario_results(out_dir: str) -> dict:
    """Load per-scenario summaries from subdirectories of out_dir."""
    found = {}
    if not os.path.isdir(out_dir):
        return found
    for name in sorted(os.listdir(out_dir)):
        sub = os.path.join(out_dir, name)
        if not os.path.isdir(sub):
            continue
        _, summary = load_results(sub)
        if summary is not None:
            found[name] = (os.path.join(sub, "experiments.csv"), summary)
    return found


def _aggregate_scenario_summaries(scenarios: dict) -> dict:
    """Build a top-line aggregate summary across scenarios."""
    summary_fields = [
        "adaptive_wins", "immediate_wins", "ties", "num_experiments",
        "on_time_pct_diff_mean", "on_time_pct_diff_median", "on_time_pct_diff_std",
        "avg_delivery_min_diff_mean", "avg_delivery_min_diff_median", "avg_delivery_min_diff_std",
        "p50_delivery_min_diff_mean", "p90_delivery_min_diff_mean", "p95_delivery_min_diff_mean",
        "avg_late_min_diff_mean", "avg_late_min_diff_median", "avg_late_min_diff_std",
        "avg_order_wait_min_diff_mean",
        "avg_rider_wait_kitchen_min_diff_mean", "avg_rider_wait_kitchen_min_diff_median",
        "cost_score_diff_mean", "cost_score_diff_median", "cost_score_diff_std",
        "p99_delivery_min_diff_mean", "max_delivery_min_diff_mean", "late_count_diff_mean",
    ]

    agg = {
        "num_experiments": sum(s.num_experiments for _, s in scenarios.values()),
        "scenario": "multi",
        "days": next(iter(scenarios.values()))[1].days,
        "base_seed": next(iter(scenarios.values()))[1].base_seed,
        "adaptive_wins": sum(s.adaptive_wins for _, s in scenarios.values()),
        "immediate_wins": sum(s.immediate_wins for _, s in scenarios.values()),
        "ties": sum(s.ties for _, s in scenarios.values()),
        "scenario_breakdown": {},
    }
    for field in summary_fields[4:]:
        values = [getattr(s, field) for _, s in scenarios.values()]
        agg[field] = sum(values) / len(values) if values else 0

    return agg


@router.get("/results/{scenario}")
def get_scenario_results(scenario: str, out_dir: str = "data/experiments"):
    """Load results for a specific scenario."""
    scenario_dir = os.path.join(out_dir, scenario)
    results, summary = load_results(scenario_dir)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"No results found for scenario {scenario}")

    return {
        "scenario": scenario,
        "summary": asdict(summary),
        "num_results": len(results),
    }