"""Tests for the Phase 2 delivery-time primary objective:
metric percentiles, the delivery-time win criterion, and distribution
accumulation."""

from dispatch.experiment import (
    ExperimentConfig,
    PairedResult,
    compute_summary,
    DistributionAccumulator,
)
from dispatch.metrics import compute_metrics
from simulation.entities import Order, OrderComplexity, OrderStatus

import pytest


def _order(order_id: int, delivered_at: float, placed_at: float = 0.0):
    return Order(
        order_id=order_id,
        kitchen_id=1,
        placed_at=placed_at,
        items=2,
        complexity=OrderComplexity.SIMPLE,
        distance_km=2.0,
        promised_delivery_min=15.0,
        status=OrderStatus.COMPLETED,
        delivered_at=delivered_at,
    )


def _config():
    return {
        "dispatch": {
            "promised_delivery_min": 15.0,
            "cost_weights": {"idle": 1, "wait": 5, "late": 10},
        }
    }


def test_compute_metrics_delivery_percentiles():
    orders = [
        _order(1, 10.0),
        _order(2, 12.0),
        _order(3, 14.0),
        _order(4, 16.0),
        _order(5, 20.0),
    ]
    m = compute_metrics(orders, [], 1440, _config())

    assert m["avg_delivery_min"] == 14.4
    assert m["p50_delivery_min"] == 14.0
    assert m["p90_delivery_min"] == 18.4
    assert m["p95_delivery_min"] == 19.2
    assert m["p99_delivery_min"] == 19.84
    assert m["max_delivery_min"] == 20.0
    assert m["late_count"] == 2
    assert m["on_time_rate"] == 0.6
    assert m["orders_completed"] == 5


def test_win_criterion_uses_delivery_time_not_cost():
    results = [
        PairedResult(
            seed=1, scenario="normal", days=1,
            immediate={"on_time_rate": 0.9, "avg_delivery_min": 15.0, "avg_late_min": 1.0,
                       "avg_order_wait_min": 0.5, "avg_rider_wait_kitchen_min": 1.0,
                       "avg_rider_idle_min": 10.0, "cost_score": 50.0},
            adaptive={"on_time_rate": 0.9, "avg_delivery_min": 13.5, "avg_late_min": 1.0,
                      "avg_order_wait_min": 1.5, "avg_rider_wait_kitchen_min": 2.0,
                      "avg_rider_idle_min": 20.0, "cost_score": 90.0},
            differences={"on_time_rate": 0.0, "avg_delivery_min": -1.5, "avg_late_min": 0.0,
                         "p50_delivery_min": -1.5, "p90_delivery_min": -1.5, "p95_delivery_min": -1.5,
                         "p99_delivery_min": -1.5, "max_delivery_min": -1.5, "late_count": 0,
                         "avg_order_wait_min": 1.0, "avg_rider_wait_kitchen_min": 1.0,
                         "avg_rider_idle_min": 10.0, "cost_score": 40.0},
        ),
    ]
    config = ExperimentConfig(num_experiments=1, scenario="normal", days=1)
    summary = compute_summary(results, config)

    assert summary.adaptive_wins == 1
    assert summary.immediate_wins == 0
    assert summary.ties == 0
    assert summary.avg_delivery_min_diff_mean == -1.5


def test_win_ties_within_delivery_threshold():
    results = [
        PairedResult(
            seed=1, scenario="normal", days=1,
            immediate={"on_time_rate": 0.9, "avg_delivery_min": 15.0, "avg_late_min": 1.0,
                       "avg_order_wait_min": 0.5, "avg_rider_wait_kitchen_min": 1.0,
                       "avg_rider_idle_min": 10.0, "cost_score": 50.0},
            adaptive={"on_time_rate": 0.9, "avg_delivery_min": 14.95, "avg_late_min": 1.0,
                      "avg_order_wait_min": 0.6, "avg_rider_wait_kitchen_min": 1.0,
                      "avg_rider_idle_min": 10.0, "cost_score": 51.0},
            differences={"on_time_rate": 0.0, "avg_delivery_min": -0.05, "avg_late_min": 0.0,
                         "avg_order_wait_min": 0.1, "avg_rider_wait_kitchen_min": 0.0,
                         "avg_rider_idle_min": 0.0, "cost_score": 1.0},
        ),
    ]
    config = ExperimentConfig(num_experiments=1, scenario="normal", days=1)
    summary = compute_summary(results, config)

    assert summary.adaptive_wins == 0
    assert summary.immediate_wins == 0
    assert summary.ties == 1


def test_distribution_accumulator():
    acc = DistributionAccumulator()
    acc.update([10.0, 11.0, 12.0], [14.0, 15.0])
    acc.update([10.0], [13.0, 16.0])

    d = acc.to_dict("normal")

    assert d["num_paired_runs"] == 2
    assert d["adaptive"]["total_orders"] == 4
    assert d["immediate"]["total_orders"] == 4
    assert d["adaptive"]["avg_delivery_min"] == 10.75
    assert d["immediate"]["avg_delivery_min"] == 14.5
    assert sum(d["adaptive"]["bin_counts"]) == 4
    assert sum(d["immediate"]["bin_counts"]) == 4
    assert abs(d["adaptive"]["cdf"][-1] - 1.0) < 1e-6
    assert abs(d["immediate"]["cdf"][-1] - 1.0) < 1e-6


def test_rider_wait_kitchen_measures_food_wait():
    cfg = _config()
    # Rider arrives BEFORE food ready -> wait = ready - rider arrival.
    early = Order(
        order_id=1, kitchen_id=1, placed_at=0.0, items=2,
        complexity=OrderComplexity.SIMPLE, distance_km=2.0,
        promised_delivery_min=15.0, status=OrderStatus.COMPLETED,
        delivered_at=20.0, pickup_at=16.0,
        prep_finished_at=15.0, rider_arrived_kitchen_at=12.0,
    )
    # Rider arrives AFTER food ready -> wait is clamped to 0 (food wait, not rider wait).
    late = Order(
        order_id=2, kitchen_id=1, placed_at=0.0, items=2,
        complexity=OrderComplexity.SIMPLE, distance_km=2.0,
        promised_delivery_min=15.0, status=OrderStatus.COMPLETED,
        delivered_at=22.0, pickup_at=18.0,
        prep_finished_at=12.0, rider_arrived_kitchen_at=16.0,
    )
    m = compute_metrics([early, late], [], 1440, cfg)
    assert m["avg_rider_wait_kitchen_min"] == pytest.approx((15.0 - 12.0 + 0.0) / 2)
    assert m["avg_order_wait_min"] == pytest.approx((16.0 - 15.0 + 18.0 - 12.0) / 2)


def test_orders_in_flight_counts_incomplete():
    cfg = _config()
    done = _order(1, 10.0)
    cancelled = Order(
        order_id=2, kitchen_id=1, placed_at=0.0, items=2,
        complexity=OrderComplexity.SIMPLE, distance_km=2.0,
        promised_delivery_min=15.0, status=OrderStatus.CANCELLED,
    )
    in_flight = Order(
        order_id=3, kitchen_id=1, placed_at=0.0, items=2,
        complexity=OrderComplexity.SIMPLE, distance_km=2.0,
        promised_delivery_min=15.0, status=OrderStatus.PREPPING,
    )
    m = compute_metrics([done, cancelled, in_flight], [], 1440, cfg)
    assert m["orders_in_flight"] == 1
    assert m["orders_completed"] == 1
    assert m["orders_cancelled"] == 1
