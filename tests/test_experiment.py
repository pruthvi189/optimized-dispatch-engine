"""Tests for experiment runner."""

import json
import os
import tempfile
import shutil

import pytest

from dispatch.experiment import (
    DistributionAccumulator,
    ExperimentConfig,
    run_paired_experiment,
    run_experiment,
    compute_summary,
    save_results,
    load_results,
    PairedResult,
    ExperimentSummary,
)


def _base_config():
    """Load a minimal config for testing."""
    from simulation import load_scenario
    return load_scenario("normal", seed=42)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_paired_experiment_runs(temp_dir):
    """Test that a single paired experiment runs without errors."""
    config = _base_config()
    config["days"] = 1
    config["dispatch"]["enabled"] = True
    config["dispatch"]["predictor_dir"] = "artifacts"

    # Use a fake predictor dir for testing - will fall back to baseline
    result = run_paired_experiment(
        seed=100,
        base_config=config,
        predictor_dir="artifacts",
        out_root=temp_dir,
        scenario_name="normal",
        save_individual=False,
    )

    assert isinstance(result, PairedResult)
    assert result.seed == 100
    assert result.scenario == "normal"
    assert result.days == 1

    # Check metrics exist in both policies
    for policy_dict in (result.immediate, result.adaptive):
        assert "on_time_rate" in policy_dict
        assert "avg_delivery_min" in policy_dict
        assert "avg_late_min" in policy_dict
        assert "cost_score" in policy_dict

    # Check differences
    assert "on_time_rate" in result.differences
    assert "avg_delivery_min" in result.differences
    assert "cost_score" in result.differences


def test_compute_summary():
    """Test summary computation from paired results."""
    results = [
        PairedResult(
            seed=1, scenario="normal", days=1,
            immediate={"on_time_rate": 0.8, "avg_delivery_min": 14.0, "avg_late_min": 2.0,
                       "avg_order_wait_min": 1.0, "avg_rider_wait_kitchen_min": 3.0,
                       "avg_rider_idle_min": 50.0, "cost_score": 100.0},
            adaptive={"on_time_rate": 0.85, "avg_delivery_min": 13.5, "avg_late_min": 1.5,
                      "avg_order_wait_min": 0.8, "avg_rider_wait_kitchen_min": 2.5,
                      "avg_rider_idle_min": 45.0, "cost_score": 90.0},
            differences={"on_time_rate": 0.05, "avg_delivery_min": -0.5, "avg_late_min": -0.5,
                         "p50_delivery_min": -0.5, "p90_delivery_min": -0.5, "p95_delivery_min": -0.5,
                         "p99_delivery_min": -0.5, "max_delivery_min": -0.5, "late_count": -1,
                         "avg_order_wait_min": -0.2, "avg_rider_wait_kitchen_min": -0.5,
                         "avg_rider_idle_min": -5.0, "cost_score": -10.0},
        ),
        PairedResult(
            seed=2, scenario="normal", days=1,
            immediate={"on_time_rate": 0.75, "avg_delivery_min": 15.0, "avg_late_min": 3.0,
                       "avg_order_wait_min": 1.5, "avg_rider_wait_kitchen_min": 4.0,
                       "avg_rider_idle_min": 60.0, "cost_score": 120.0},
            adaptive={"on_time_rate": 0.78, "avg_delivery_min": 14.5, "avg_late_min": 2.5,
                      "avg_order_wait_min": 1.2, "avg_rider_wait_kitchen_min": 3.5,
                      "avg_rider_idle_min": 55.0, "cost_score": 110.0},
            differences={"on_time_rate": 0.03, "avg_delivery_min": -0.5, "avg_late_min": -0.5,
                         "p50_delivery_min": -0.5, "p90_delivery_min": -0.5, "p95_delivery_min": -0.5,
                         "p99_delivery_min": -0.5, "max_delivery_min": -0.5, "late_count": -1,
                         "avg_order_wait_min": -0.3, "avg_rider_wait_kitchen_min": -0.5,
                         "avg_rider_idle_min": -5.0, "cost_score": -10.0},
        ),
    ]

    config = ExperimentConfig(num_experiments=2, scenario="normal", days=1)
    summary = compute_summary(results, config)

    assert summary.num_experiments == 2
    assert summary.adaptive_wins == 2  # Both have lower average delivery time
    assert summary.immediate_wins == 0
    assert summary.ties == 0
    assert summary.on_time_pct_diff_mean == 0.04  # (0.05 + 0.03) / 2
    assert summary.avg_delivery_min_diff_mean == -0.5
    assert summary.p50_delivery_min_diff_mean == -0.5
    assert summary.p95_delivery_min_diff_mean == -0.5
    assert summary.late_count_diff_mean == -1.0
    assert summary.cost_score_diff_mean == -10.0


def test_save_and_load_results(temp_dir):
    """Test saving and loading experiment results."""
    results = [
        PairedResult(
            seed=1, scenario="normal", days=1,
            immediate={"on_time_rate": 0.8, "avg_delivery_min": 14.0, "avg_late_min": 2.0,
                       "avg_order_wait_min": 1.0, "avg_rider_wait_kitchen_min": 3.0,
                       "avg_rider_idle_min": 50.0, "cost_score": 100.0},
            adaptive={"on_time_rate": 0.85, "avg_delivery_min": 13.5, "avg_late_min": 1.5,
                      "avg_order_wait_min": 0.8, "avg_rider_wait_kitchen_min": 2.5,
                      "avg_rider_idle_min": 45.0, "cost_score": 90.0},
            differences={"on_time_rate": 0.05, "avg_delivery_min": -0.5, "avg_late_min": -0.5,
                         "p50_delivery_min": -0.5, "p90_delivery_min": -0.5, "p95_delivery_min": -0.5,
                         "p99_delivery_min": -0.5, "max_delivery_min": -0.5, "late_count": -1,
                         "avg_order_wait_min": -0.2, "avg_rider_wait_kitchen_min": -0.5,
                         "avg_rider_idle_min": -5.0, "cost_score": -10.0},
        ),
    ]

    config = ExperimentConfig(num_experiments=1, scenario="normal", days=1)
    summary = compute_summary(results, config)

    save_results(results, summary, temp_dir)

    # Verify files exist
    assert os.path.exists(os.path.join(temp_dir, "experiments.csv"))
    assert os.path.exists(os.path.join(temp_dir, "experiment_summary.json"))

    # Load and verify
    loaded_results, loaded_summary = load_results(temp_dir)

    assert len(loaded_results) == 1
    assert loaded_results[0].seed == 1
    assert loaded_summary.num_experiments == 1
    assert loaded_summary.adaptive_wins == 1
    assert loaded_results[0].differences["p95_delivery_min"] == -0.5
    assert loaded_results[0].adaptive["late_count"] == 0.0

    # Round-trip preserves the new statistical fields.
    assert loaded_summary.adaptive_win_rate == 1.0
    assert loaded_summary.paired_stats["avg_delivery_min"]["mean_diff"] == -0.5
    assert "orders_in_flight" in loaded_results[0].differences


# ---- Statistical comparison (Task 5) ------------------------------------------


def test_summary_has_paired_stats_and_win_rates():
    results = [
        PairedResult(
            seed=1, scenario="normal", days=1,
            immediate={"on_time_rate": 0.8, "avg_delivery_min": 14.0, "avg_late_min": 2.0,
                       "avg_order_wait_min": 1.0, "avg_rider_wait_kitchen_min": 3.0,
                       "avg_rider_idle_min": 50.0, "cost_score": 100.0, "orders_in_flight": 0},
            adaptive={"on_time_rate": 0.85, "avg_delivery_min": 13.5, "avg_late_min": 1.5,
                      "avg_order_wait_min": 0.8, "avg_rider_wait_kitchen_min": 2.5,
                      "avg_rider_idle_min": 45.0, "cost_score": 90.0, "orders_in_flight": 0},
            differences={"on_time_rate": 0.05, "avg_delivery_min": -0.5, "avg_late_min": -0.5,
                         "p50_delivery_min": -0.5, "p90_delivery_min": -0.5, "p95_delivery_min": -0.5,
                         "p99_delivery_min": -0.5, "max_delivery_min": -0.5, "late_count": -1,
                         "avg_order_wait_min": -0.2, "avg_rider_wait_kitchen_min": -0.5,
                         "avg_rider_idle_min": -5.0, "cost_score": -10.0, "orders_in_flight": 0.0},
        ),
        PairedResult(
            seed=2, scenario="normal", days=1,
            immediate={"on_time_rate": 0.75, "avg_delivery_min": 15.0, "avg_late_min": 3.0,
                       "avg_order_wait_min": 1.5, "avg_rider_wait_kitchen_min": 4.0,
                       "avg_rider_idle_min": 60.0, "cost_score": 120.0, "orders_in_flight": 0},
            adaptive={"on_time_rate": 0.78, "avg_delivery_min": 14.5, "avg_late_min": 2.5,
                      "avg_order_wait_min": 1.2, "avg_rider_wait_kitchen_min": 3.5,
                      "avg_rider_idle_min": 55.0, "cost_score": 110.0, "orders_in_flight": 0},
            differences={"on_time_rate": 0.03, "avg_delivery_min": -0.5, "avg_late_min": -0.5,
                         "p50_delivery_min": -0.5, "p90_delivery_min": -0.5, "p95_delivery_min": -0.5,
                         "p99_delivery_min": -0.5, "max_delivery_min": -0.5, "late_count": -1,
                         "avg_order_wait_min": -0.3, "avg_rider_wait_kitchen_min": -0.5,
                         "avg_rider_idle_min": -5.0, "cost_score": -10.0, "orders_in_flight": 0.0},
        ),
    ]
    config = ExperimentConfig(num_experiments=2, scenario="normal", days=1)
    summary = compute_summary(results, config)

    assert summary.num_experiments == 2
    assert summary.adaptive_win_rate == pytest.approx(1.0)
    assert summary.immediate_win_rate == 0.0
    assert summary.tie_rate == 0.0
    assert summary.win_method == "avg_delivery_min"
    assert summary.adaptive_win_rate_ci_low > 0.0
    assert summary.adaptive_win_rate_ci_high > 0.0
    assert summary.immediate_win_rate == 0.0

    for key in ("avg_delivery_min", "on_time_rate", "avg_late_min", "p95_delivery_min",
                "avg_order_wait_min", "avg_rider_wait_kitchen_min", "cost_score"):
        assert key in summary.paired_stats
        st = summary.paired_stats[key]
        assert st["n"] == 2
        assert st["ci95_low"] <= st["mean_diff"] <= st["ci95_high"]
        assert 0.0 <= st["p_value_permutation"] <= 1.0
        assert 0.0 <= st["p_value_ttest"] <= 1.0
    # Identical diffs of -0.5 on a lower-is-better metric: CI excludes 0.
    assert summary.paired_stats["avg_delivery_min"]["mean_diff"] == -0.5
    assert summary.paired_stats["avg_delivery_min"]["ci95_high"] < 0.0


def test_paired_experiment_reproducibility(temp_dir):
    """Test that paired experiments are reproducible with the same seed."""
    config = _base_config()
    config["days"] = 1
    config["dispatch"]["enabled"] = True
    config["dispatch"]["predictor_dir"] = "artifacts"

    # Run twice with same seed
    result1 = run_paired_experiment(
        seed=999,
        base_config=config,
        predictor_dir="artifacts",
        out_root=os.path.join(temp_dir, "run1"),
        scenario_name="normal",
        save_individual=False,
    )

    result2 = run_paired_experiment(
        seed=999,
        base_config=config,
        predictor_dir="artifacts",
        out_root=os.path.join(temp_dir, "run2"),
        scenario_name="normal",
        save_individual=False,
    )

    # Results should be identical
    assert result1.immediate["cost_score"] == result2.immediate["cost_score"]
    assert result1.adaptive["cost_score"] == result2.adaptive["cost_score"]
    assert result1.immediate["on_time_rate"] == result2.immediate["on_time_rate"]
    assert result1.adaptive["on_time_rate"] == result2.adaptive["on_time_rate"]


def test_distribution_accumulator_to_dict_contract():
    """DistributionAccumulator.to_dict() must emit the exact key layout the
    dashboard's DistributionData / DistributionSeries types expect."""
    acc = DistributionAccumulator()
    acc.update([10.0, 20.0, 30.0], [15.0, 25.0])
    acc.update([5.0, 35.0, 40.0, 45.0], [8.0])
    d = acc.to_dict("normal")

    assert set(d.keys()) == {"scenario", "num_paired_runs", "max_min", "adaptive", "immediate"}
    assert d["scenario"] == "normal"
    assert d["num_paired_runs"] == 2
    assert d["max_min"] == 240.0

    for series in (d["adaptive"], d["immediate"]):
        assert set(series.keys()) == {"bin_counts", "edges", "cdf", "total_orders", "avg_delivery_min", "percentiles"}
        assert len(series["bin_counts"]) == 240
        assert len(series["edges"]) == 241
        assert len(series["cdf"]) == 240
        assert isinstance(series["total_orders"], int)
        assert series["avg_delivery_min"] >= 0.0
        # CDF is monotonically non-decreasing and reaches 1.0
        assert all(b >= a for a, b in zip(series["cdf"], series["cdf"][1:]))
        assert series["cdf"][-1] == pytest.approx(1.0)
        # Percentiles are real numbers within the delivery-time range
        assert {str(k) for k in series["percentiles"].keys()} == {"50", "90", "95", "99"}
        assert all(0.0 <= pct <= 600.0 for pct in series["percentiles"].values())


def test_run_experiment_saves_distribution_contract(temp_dir):
    """run_experiment with save_distributions=True writes a delivery_distribution.json
    matching the frontend DistributionData contract."""
    config = ExperimentConfig(
        num_experiments=1,
        base_seed=42,
        days=1,
        scenario="normal",
        predictor_dir="artifacts",
        out_dir=temp_dir,
        save_individual=False,
        save_distributions=True,
    )
    results, summary = run_experiment(config)
    assert len(results) == 1
    assert summary.num_experiments == 1

    dist_path = os.path.join(temp_dir, "delivery_distribution.json")
    assert os.path.exists(dist_path)
    with open(dist_path) as f:
        d = json.load(f)

    assert set(d.keys()) == {"scenario", "num_paired_runs", "max_min", "adaptive", "immediate"}
    assert d["scenario"] == "normal"
    assert d["num_paired_runs"] == 1
    for series in (d["adaptive"], d["immediate"]):
        assert set(series.keys()) == {"bin_counts", "edges", "cdf", "total_orders", "avg_delivery_min", "percentiles"}
        assert len(series["bin_counts"]) == len(series["edges"]) - 1 == len(series["cdf"])
        assert series["total_orders"] > 0
        assert series["cdf"][-1] == pytest.approx(1.0)
        assert {str(k) for k in series["percentiles"].keys()} == {"50", "90", "95", "99"}
        assert all(0.0 <= pct <= 600.0 for pct in series["percentiles"].values())


def test_experiment_results_distributions_wrapper(temp_dir):
    """GET /experiments/results maps delivery_distribution.json into the frontend
    contract: {'default': ...} for single-scenario, per-scenario keys for multi."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers.experiment import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # Empty dir -> no results -> 200 with empty data (graceful)
    empty = os.path.join(temp_dir, "empty")
    os.makedirs(empty)
    resp = client.get(f"/experiments/results?out_dir={empty}")
    assert resp.status_code == 200
    assert resp.json()["num_results"] == 0

    # Single-scenario: summary + delivery_distribution.json -> wrapped under "default"
    config = ExperimentConfig(
        num_experiments=1,
        base_seed=42,
        days=1,
        scenario="normal",
        predictor_dir="artifacts",
        out_dir=temp_dir,
        save_distributions=True,
    )
    results, summary = run_experiment(config)
    save_results(results, summary, temp_dir)

    resp = client.get(f"/experiments/results?out_dir={temp_dir}")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["multi_scenario"] is False
    dist = payload["distributions"]["default"]
    assert dist["scenario"] == "normal"
    assert "bin_counts" in dist["adaptive"]
    assert "cdf" in dist["immediate"]

    # Multi-scenario: per-scenario subdirs -> distributions keyed by scenario name
    multi = os.path.join(temp_dir, "multi")
    for sc in ("normal", "rain"):
        sub = os.path.join(multi, sc)
        os.makedirs(sub, exist_ok=True)
        acc = DistributionAccumulator()
        acc.update([10.0, 20.0], [15.0])
        with open(os.path.join(sub, "delivery_distribution.json"), "w") as f:
            json.dump(acc.to_dict(sc), f)
        r = PairedResult(
            seed=1, scenario=sc, days=1,
            immediate={"on_time_rate": 0.8, "avg_delivery_min": 14.0, "avg_late_min": 2.0,
                       "avg_order_wait_min": 1.0, "avg_rider_wait_kitchen_min": 3.0,
                       "avg_rider_idle_min": 50.0, "cost_score": 100.0},
            adaptive={"on_time_rate": 0.85, "avg_delivery_min": 13.5, "avg_late_min": 1.5,
                      "avg_order_wait_min": 0.8, "avg_rider_wait_kitchen_min": 2.5,
                      "avg_rider_idle_min": 45.0, "cost_score": 90.0},
            differences={"on_time_rate": 0.05, "avg_delivery_min": -0.5, "avg_late_min": -0.5,
                         "p50_delivery_min": -0.5, "p90_delivery_min": -0.5, "p95_delivery_min": -0.5,
                         "p99_delivery_min": -0.5, "max_delivery_min": -0.5, "late_count": -1,
                         "avg_order_wait_min": -0.2, "avg_rider_wait_kitchen_min": -0.5,
                         "avg_rider_idle_min": -5.0, "cost_score": -10.0},
        )
        save_results([r], compute_summary([r], ExperimentConfig(num_experiments=1, scenario=sc, days=1)), sub)

    resp = client.get(f"/experiments/results?out_dir={multi}")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["multi_scenario"] is True
    assert set(payload["distributions"].keys()) == {"normal", "rain"}
    assert payload["distributions"]["normal"]["scenario"] == "normal"
    assert "adaptive" in payload["distributions"]["rain"]