"""Tests for root-cause analysis."""

import pytest
from simulation.entities import Order, OrderComplexity, OrderStatus
from dispatch.root_cause import (
    analyze_order,
    analyze_orders,
    aggregate_root_causes,
    RootCauseAnalysis,
    StageDurations,
)


def _make_order(
    placed_at: float = 0.0,
    prep_started_at: float = 1.0,
    prep_finished_at: float = 8.0,
    dispatch_at: float = 2.0,
    rider_arrived_kitchen_at: float = 6.0,
    pickup_at: float = 8.5,
    delivered_at: float = 18.0,
    actual_prep_duration_min: float = 7.0,
    status: OrderStatus = OrderStatus.COMPLETED,
) -> Order:
    return Order(
        order_id=1,
        kitchen_id=1,
        placed_at=placed_at,
        items=3,
        complexity=OrderComplexity.STANDARD,
        distance_km=3.0,
        staff_level=3,
        prep_started_at=prep_started_at,
        prep_finished_at=prep_finished_at,
        dispatch_at=dispatch_at,
        rider_arrived_kitchen_at=rider_arrived_kitchen_at,
        pickup_at=pickup_at,
        delivered_at=delivered_at,
        actual_prep_duration_min=actual_prep_duration_min,
        status=status,
    )


def test_on_time_order_returns_analysis():
    """On-time orders should return analysis with is_late=False."""
    order = _make_order(delivered_at=12.0)  # 12 min delivery, 15 min promise
    result = analyze_order(order, promised_delivery_min=15.0)

    assert result is not None
    assert result.is_late is False
    assert result.lateness_min == 0.0
    assert result.delivery_time_min == 12.0


def test_late_order_kitchen_prep_bottleneck():
    """Order with long kitchen prep should identify KITCHEN_PREP as primary cause."""
    # Kitchen prep took 12 min (started at 1, finished at 13)
    # Rider arrived at 8, had to wait 5 min for food
    # Customer travel 5 min -> delivered at 18
    # Promise 15, late by 3
    order = _make_order(
        prep_started_at=1.0,
        prep_finished_at=13.0,
        dispatch_at=2.0,
        rider_arrived_kitchen_at=8.0,
        pickup_at=13.5,  # rider waited 5.5 min
        delivered_at=18.5,
        actual_prep_duration_min=12.0,
    )
    result = analyze_order(order, promised_delivery_min=15.0)

    assert result is not None
    assert result.is_late is True
    assert result.primary_root_cause == "KITCHEN_PREP"
    assert "KITCHEN_PREP" in result.contributing_factors
    assert result.lateness_min == 3.5


def test_late_order_customer_travel_bottleneck():
    """Order with long customer travel should identify CUSTOMER_TRAVEL as primary cause."""
    # Quick prep (5 min), quick dispatch, but customer far away (10 min travel)
    order = _make_order(
        prep_started_at=0.5,
        prep_finished_at=5.5,
        dispatch_at=1.0,
        rider_arrived_kitchen_at=4.0,
        pickup_at=6.0,  # food ready, pickup quick
        delivered_at=16.0,  # 10 min customer travel
        actual_prep_duration_min=5.0,
    )
    result = analyze_order(order, promised_delivery_min=15.0)

    assert result is not None
    assert result.is_late is True
    assert result.primary_root_cause == "CUSTOMER_TRAVEL"
    assert "CUSTOMER_TRAVEL" in result.contributing_factors


def test_late_order_kitchen_queue_bottleneck():
    """Order with long kitchen queue should identify KITCHEN_QUEUE."""
    # Order sat in queue for 5 min before prep started
    order = _make_order(
        placed_at=0.0,
        prep_started_at=5.0,  # 5 min queue
        prep_finished_at=10.0,
        dispatch_at=6.0,
        rider_arrived_kitchen_at=9.0,
        pickup_at=10.5,
        delivered_at=16.0,
        actual_prep_duration_min=5.0,
    )
    result = analyze_order(order, promised_delivery_min=15.0)

    assert result is not None
    assert result.is_late is True
    assert "KITCHEN_QUEUE" in result.contributing_factors


def test_late_order_dispatch_delay_bottleneck():
    """Order with long dispatch delay should identify DISPATCH_DELAY."""
    # Dispatch happened 8 min after order placed
    order = _make_order(
        placed_at=0.0,
        prep_started_at=1.0,
        prep_finished_at=6.0,
        dispatch_at=8.0,  # 8 min dispatch delay
        rider_arrived_kitchen_at=11.0,
        pickup_at=11.5,
        delivered_at=17.0,
        actual_prep_duration_min=5.0,
    )
    result = analyze_order(order, promised_delivery_min=15.0)

    assert result is not None
    assert result.is_late is True
    assert "DISPATCH_DELAY" in result.contributing_factors


def test_late_order_rider_travel_bottleneck():
    """Order with long rider travel to kitchen should identify RIDER_TRAVEL_TO_KITCHEN."""
    # Rider took 8 min to reach kitchen (far hub)
    order = _make_order(
        prep_started_at=1.0,
        prep_finished_at=6.0,
        dispatch_at=2.0,
        rider_arrived_kitchen_at=10.0,  # 8 min travel
        pickup_at=10.5,
        delivered_at=16.0,
        actual_prep_duration_min=5.0,
    )
    result = analyze_order(order, promised_delivery_min=15.0)

    assert result is not None
    assert result.is_late is True
    assert "RIDER_TRAVEL_TO_KITCHEN" in result.contributing_factors


def test_late_order_rider_wait_at_kitchen():
    """Order where rider waits at kitchen should flag KITCHEN_PREP as primary."""
    # Rider arrived early, waited 5 min for food
    order = _make_order(
        prep_started_at=1.0,
        prep_finished_at=10.0,
        dispatch_at=2.0,
        rider_arrived_kitchen_at=5.0,
        pickup_at=10.5,  # rider waited 5.5 min
        delivered_at=16.0,
        actual_prep_duration_min=9.0,
    )
    result = analyze_order(order, promised_delivery_min=15.0)

    assert result is not None
    assert result.is_late is True
    # When rider waits, kitchen prep is the bottleneck
    assert result.primary_root_cause == "KITCHEN_PREP"
    assert "RIDER_WAIT_AT_KITCHEN" in result.contributing_factors


def test_multiple_contributing_factors():
    """Order with multiple issues should list multiple contributing factors."""
    order = _make_order(
        prep_started_at=4.0,  # queue delay
        prep_finished_at=12.0,  # long prep
        dispatch_at=5.0,
        rider_arrived_kitchen_at=9.0,
        pickup_at=12.5,  # rider wait
        delivered_at=20.0,  # long customer travel
        actual_prep_duration_min=8.0,
    )
    result = analyze_order(order, promised_delivery_min=15.0)

    assert result is not None
    assert result.is_late is True
    assert len(result.contributing_factors) >= 2


def test_undelivered_order_returns_none():
    """Orders that aren't completed should return None."""
    order = _make_order(
        status=OrderStatus.PLACED,
        delivered_at=None,
    )
    result = analyze_order(order, promised_delivery_min=15.0)
    assert result is None


def test_analyze_orders_filters_completed():
    """analyze_orders should only analyze completed orders."""
    orders = [
        _make_order(placed_at=0.0, delivered_at=12.0),  # on-time
        _make_order(placed_at=1.0, delivered_at=18.0),  # late
        _make_order(placed_at=2.0, status=OrderStatus.PLACED, delivered_at=None),  # not delivered
        _make_order(placed_at=3.0, status=OrderStatus.CANCELLED, delivered_at=None),  # cancelled
    ]
    # Manually set order_ids
    for i, o in enumerate(orders):
        o.order_id = i + 1
    results = analyze_orders(orders, promised_delivery_min=15.0)

    assert len(results) == 2  # only completed orders
    assert results[0].order_id == 1
    assert results[1].order_id == 2


def test_aggregate_root_causes():
    """Aggregate should correctly count and calculate percentages."""
    analyses = [
        RootCauseAnalysis(
            order_id=1, is_late=True, delivery_time_min=18.0, promise_time_min=15.0,
            lateness_min=3.0, primary_root_cause="KITCHEN_PREP",
            contributing_factors=["KITCHEN_PREP", "RIDER_WAIT_AT_KITCHEN"],
            stage_durations=StageDurations()
        ),
        RootCauseAnalysis(
            order_id=2, is_late=True, delivery_time_min=20.0, promise_time_min=15.0,
            lateness_min=5.0, primary_root_cause="KITCHEN_PREP",
            contributing_factors=["KITCHEN_PREP"],
            stage_durations=StageDurations()
        ),
        RootCauseAnalysis(
            order_id=3, is_late=True, delivery_time_min=17.0, promise_time_min=15.0,
            lateness_min=2.0, primary_root_cause="CUSTOMER_TRAVEL",
            contributing_factors=["CUSTOMER_TRAVEL"],
            stage_durations=StageDurations()
        ),
        RootCauseAnalysis(
            order_id=4, is_late=False, delivery_time_min=12.0, promise_time_min=15.0,
            lateness_min=0.0, primary_root_cause="MULTIPLE_FACTORS",
            contributing_factors=[],
            stage_durations=StageDurations()
        ),
    ]

    agg = aggregate_root_causes(analyses)

    assert agg["total_orders"] == 4
    assert agg["late_orders"] == 3
    assert agg["on_time_rate"] == 0.25
    assert agg["root_cause_distribution"]["KITCHEN_PREP"] == 2
    assert agg["root_cause_distribution"]["CUSTOMER_TRAVEL"] == 1
    assert agg["primary_cause_percentages"]["KITCHEN_PREP"] == 66.7
    assert agg["primary_cause_percentages"]["CUSTOMER_TRAVEL"] == 33.3
    assert agg["contributing_factor_percentages"]["KITCHEN_PREP"] == 66.7
    assert agg["contributing_factor_percentages"]["RIDER_WAIT_AT_KITCHEN"] == 33.3
    assert agg["contributing_factor_percentages"]["CUSTOMER_TRAVEL"] == 33.3


def test_stage_durations_calculation():
    """Stage durations should be calculated correctly."""
    order = _make_order(
        placed_at=0.0,
        prep_started_at=2.0,      # 2 min queue
        prep_finished_at=9.0,     # 7 min prep
        dispatch_at=3.0,          # 3 min dispatch delay
        rider_arrived_kitchen_at=7.0,  # 4 min rider travel
        pickup_at=9.5,            # 2.5 min rider wait
        delivered_at=16.0,        # 6.5 min customer travel
        actual_prep_duration_min=7.0,
    )
    result = analyze_order(order, promised_delivery_min=15.0)

    assert result is not None
    sd = result.stage_durations
    assert sd.kitchen_queue == 2.0
    assert sd.kitchen_prep == 7.0
    assert sd.dispatch_delay == 3.0
    assert sd.rider_to_kitchen == 4.0
    assert sd.rider_wait == 2.5
    assert sd.customer_travel == 6.5