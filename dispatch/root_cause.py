"""Root-cause analysis for late orders.

Analyzes order lifecycle timestamps to determine why orders were late.
"""

from dataclasses import dataclass
from typing import Literal

from simulation.entities import Order, OrderStatus


RootCauseCategory = Literal[
    "KITCHEN_PREP",
    "KITCHEN_QUEUE",
    "DISPATCH_DELAY",
    "RIDER_AVAILABILITY",
    "RIDER_TRAVEL_TO_KITCHEN",
    "RIDER_WAIT_AT_KITCHEN",
    "CUSTOMER_TRAVEL",
    "MULTIPLE_FACTORS",
]


@dataclass
class StageDurations:
    """Duration of each stage in the order lifecycle (minutes)."""
    kitchen_queue: float = 0.0
    kitchen_prep: float = 0.0
    dispatch_delay: float = 0.0
    rider_to_kitchen: float = 0.0
    rider_wait: float = 0.0
    customer_travel: float = 0.0

    def to_dict(self) -> dict:
        return {
            "kitchen_queue": round(self.kitchen_queue, 2),
            "kitchen_prep": round(self.kitchen_prep, 2),
            "dispatch_delay": round(self.dispatch_delay, 2),
            "rider_to_kitchen": round(self.rider_to_kitchen, 2),
            "rider_wait": round(self.rider_wait, 2),
            "customer_travel": round(self.customer_travel, 2),
        }


@dataclass
class RootCauseAnalysis:
    """Root-cause analysis result for a single order."""
    order_id: int
    is_late: bool
    delivery_time_min: float
    promise_time_min: float
    lateness_min: float
    primary_root_cause: RootCauseCategory
    contributing_factors: list[RootCauseCategory]
    stage_durations: StageDurations


def analyze_order(order: Order, promised_delivery_min: float) -> RootCauseAnalysis | None:
    """Analyze a single order to determine root cause of lateness.

    Returns None if order is not completed (delivered).
    """
    if order.delivered_at is None or order.placed_at is None:
        return None

    delivery_time = order.delivered_at - order.placed_at
    is_late = delivery_time > promised_delivery_min
    lateness = max(0.0, delivery_time - promised_delivery_min)

    # Calculate stage durations
    stages = StageDurations()

    # Kitchen queue: time from order placed to prep started
    if order.prep_started_at is not None and order.placed_at is not None:
        stages.kitchen_queue = max(0.0, order.prep_started_at - order.placed_at)

    # Kitchen prep: actual prep duration
    if order.actual_prep_duration_min is not None:
        stages.kitchen_prep = order.actual_prep_duration_min
    elif order.prep_finished_at is not None and order.prep_started_at is not None:
        stages.kitchen_prep = max(0.0, order.prep_finished_at - order.prep_started_at)

    # Dispatch delay: time from order placed to dispatch decision
    if order.dispatch_at is not None and order.placed_at is not None:
        stages.dispatch_delay = max(0.0, order.dispatch_at - order.placed_at)

    # Rider travel to kitchen: time from dispatch to rider arrival
    if order.rider_arrived_kitchen_at is not None and order.dispatch_at is not None:
        stages.rider_to_kitchen = max(0.0, order.rider_arrived_kitchen_at - order.dispatch_at)

    # Rider wait at kitchen: time from rider arrival to food ready (pickup)
    if order.pickup_at is not None and order.rider_arrived_kitchen_at is not None:
        stages.rider_wait = max(0.0, order.pickup_at - order.rider_arrived_kitchen_at)

    # Customer travel: time from pickup to delivery
    if order.delivered_at is not None and order.pickup_at is not None:
        stages.customer_travel = max(0.0, order.delivered_at - order.pickup_at)

    if not is_late:
        return RootCauseAnalysis(
            order_id=order.order_id,
            is_late=False,
            delivery_time_min=round(delivery_time, 2),
            promise_time_min=promised_delivery_min,
            lateness_min=0.0,
            primary_root_cause="MULTIPLE_FACTORS",
            contributing_factors=[],
            stage_durations=stages,
        )

    # Determine root cause for late orders
    primary, contributing = _determine_root_cause(order, stages, promised_delivery_min)

    return RootCauseAnalysis(
        order_id=order.order_id,
        is_late=True,
        delivery_time_min=round(delivery_time, 2),
        promise_time_min=promised_delivery_min,
        lateness_min=round(lateness, 2),
        primary_root_cause=primary,
        contributing_factors=contributing,
        stage_durations=stages,
    )


def _determine_root_cause(
    order: Order,
    stages: StageDurations,
    promised_delivery_min: float
) -> tuple[RootCauseCategory, list[RootCauseCategory]]:
    """Determine the primary root cause and contributing factors.

    This uses a heuristic based on the order's timeline and the
    concept of bottleneck/limiting factor rather than just "longest duration".
    """
    # For adaptive dispatch, the dispatch delay might be negative (dispatched before order)
    # In that case, dispatch_delay is 0
    dispatch_delay = stages.dispatch_delay

    # Calculate the critical path durations
    # The actual path: queue -> prep -> (dispatch + rider_to_kitchen) -> pickup -> customer_travel
    # But prep and rider_to_kitchen can overlap depending on when dispatch happens

    # Let's determine which stages actually contributed to the lateness
    # by checking if each stage pushed the delivery past the promise

    # We know: delivery_time = sum of all stages that actually happened sequentially
    # But some stages overlap. The true sequential path is:
    # 1. Kitchen queue (always sequential)
    # 2. Kitchen prep (sequential after queue)
    # 3. Max(rider_to_kitchen, remaining_prep_after_dispatch) - these can overlap
    # 4. Pickup (sequential after rider arrives AND food ready)
    # 5. Customer travel (sequential after pickup)

    # A more practical approach: identify which stages have the most "excess" time
    # beyond what would be needed to meet the SLA

    # The promised delivery time breakdown (ideal case):
    # We can work backwards from the promise to see where the slack was consumed

    # Calculate the theoretical minimum time for each stage
    # (these are lower bounds based on distance, etc.)

    # For root cause, we look at which stages had actual duration significantly
    # above what was "budgeted" in the ideal path

    # Since we don't have per-stage budgets, we use a relative approach:
    # Find which stage(s) contributed most to the lateness

    # Approach: reconstruct the timeline and find the bottleneck

    # The key insight: some stages happen concurrently
    # - Kitchen prep happens while rider is traveling (if dispatched early enough)
    # - Rider wait at kitchen happens if rider arrives before food is ready

    # Let's check if the rider had to wait for food (rider_wait > 0)
    # If so, the kitchen prep was the bottleneck (or dispatch was too early)

    # If rider_wait == 0 but customer_travel is high, then travel is the bottleneck

    # If dispatch_delay is high (order waited for dispatch), that's a factor

    contributing = []

    # Check each potential cause
    if stages.kitchen_queue > 2.0:  # More than 2 min in queue
        contributing.append("KITCHEN_QUEUE")

    if stages.kitchen_prep > 10.0:  # Prep took > 10 min
        contributing.append("KITCHEN_PREP")

    if dispatch_delay > 5.0:  # Dispatch delayed > 5 min
        contributing.append("DISPATCH_DELAY")

    if stages.rider_to_kitchen > 5.0:  # Travel to kitchen > 5 min
        contributing.append("RIDER_TRAVEL_TO_KITCHEN")

    if stages.rider_wait > 3.0:  # Rider waited > 3 min for food
        contributing.append("RIDER_WAIT_AT_KITCHEN")
        # If rider waited, kitchen prep was likely the bottleneck
        if "KITCHEN_PREP" not in contributing and stages.kitchen_prep > 5.0:
            contributing.append("KITCHEN_PREP")

    if stages.customer_travel > 8.0:  # Customer travel > 8 min
        contributing.append("CUSTOMER_TRAVEL")

    # If no specific cause found, check general categories
    if not contributing:
        # Find the stage with the largest duration
        durations = {
            "KITCHEN_QUEUE": stages.kitchen_queue,
            "KITCHEN_PREP": stages.kitchen_prep,
            "DISPATCH_DELAY": dispatch_delay,
            "RIDER_TRAVEL_TO_KITCHEN": stages.rider_to_kitchen,
            "RIDER_WAIT_AT_KITCHEN": stages.rider_wait,
            "CUSTOMER_TRAVEL": stages.customer_travel,
        }
        max_stage = max(durations, key=durations.get)
        if durations[max_stage] > 1.0:
            contributing.append(max_stage)

    # Determine primary cause
    # Priority order based on operational significance
    primary_priority = [
        "KITCHEN_PREP",
        "RIDER_AVAILABILITY",  # Would need to check if no rider was available
        "CUSTOMER_TRAVEL",
        "KITCHEN_QUEUE",
        "DISPATCH_DELAY",
        "RIDER_TRAVEL_TO_KITCHEN",
        "RIDER_WAIT_AT_KITCHEN",
    ]

    primary = "MULTIPLE_FACTORS"
    for p in primary_priority:
        if p in contributing:
            primary = p
            break

    # Special case: if rider waited at kitchen, kitchen prep is the true bottleneck
    if primary == "RIDER_WAIT_AT_KITCHEN":
        primary = "KITCHEN_PREP"

    # If multiple factors, mark as such
    if len(contributing) > 1 and primary != "MULTIPLE_FACTORS":
        # Check if there's a clear dominant factor
        pass

    return primary, contributing


def analyze_orders(orders: list[Order], promised_delivery_min: float) -> list[RootCauseAnalysis]:
    """Analyze all completed orders."""
    results = []
    for order in orders:
        if order.status == OrderStatus.COMPLETED and order.delivered_at is not None:
            analysis = analyze_order(order, promised_delivery_min)
            if analysis is not None:
                results.append(analysis)
    return results


def aggregate_root_causes(analyses: list[RootCauseAnalysis]) -> dict:
    """Aggregate root-cause statistics from multiple order analyses."""
    total = len(analyses)
    late_orders = [a for a in analyses if a.is_late]
    late_count = len(late_orders)

    if late_count == 0:
        return {
            "total_orders": total,
            "late_orders": 0,
            "on_time_rate": 1.0,
            "root_cause_distribution": {},
            "primary_cause_percentages": {},
        }

    # Count primary causes
    primary_counts: dict[str, int] = {}
    for a in late_orders:
        primary_counts[a.primary_root_cause] = primary_counts.get(a.primary_root_cause, 0) + 1

    # Calculate percentages
    primary_percentages = {
        cause: round(count / late_count * 100, 1)
        for cause, count in primary_counts.items()
    }

    # Also count contributing factors
    contributing_counts: dict[str, int] = {}
    for a in late_orders:
        for factor in a.contributing_factors:
            contributing_counts[factor] = contributing_counts.get(factor, 0) + 1

    contributing_percentages = {
        factor: round(count / late_count * 100, 1)
        for factor, count in contributing_counts.items()
    }

    return {
        "total_orders": total,
        "late_orders": late_count,
        "on_time_rate": round((total - late_count) / total, 4),
        "root_cause_distribution": primary_counts,
        "primary_cause_percentages": primary_percentages,
        "contributing_factor_counts": contributing_counts,
        "contributing_factor_percentages": contributing_percentages,
    }


def format_root_cause_analysis(analysis: RootCauseAnalysis) -> str:
    """Format a single order's root-cause analysis for display."""
    lines = [
        f"ORDER #{analysis.order_id}",
        "",
        f"Delivery Time: {analysis.delivery_time_min:.1f} min",
        f"Promise:       {analysis.promise_time_min:.1f} min",
        f"Lateness:      {analysis.lateness_min:.1f} min",
        "",
        f"Primary Cause: {analysis.primary_root_cause}",
    ]

    if analysis.contributing_factors:
        lines.append(f"Contributing:  {', '.join(analysis.contributing_factors)}")

    lines.append("")
    lines.append("Stage Durations:")
    sd = analysis.stage_durations
    lines.append(f"  Kitchen Queue:          {sd.kitchen_queue:.1f} min")
    lines.append(f"  Kitchen Prep:           {sd.kitchen_prep:.1f} min")
    lines.append(f"  Dispatch Delay:         {sd.dispatch_delay:.1f} min")
    lines.append(f"  Rider -> Kitchen:       {sd.rider_to_kitchen:.1f} min")
    lines.append(f"  Rider Wait at Kitchen:  {sd.rider_wait:.1f} min")
    lines.append(f"  Kitchen -> Customer:    {sd.customer_travel:.1f} min")

    return "\n".join(lines)


def format_aggregate_analysis(aggregate: dict) -> str:
    """Format aggregate root-cause statistics for display."""
    lines = [
        f"Total Orders:  {aggregate['total_orders']}",
        f"Late Orders:   {aggregate['late_orders']}",
        f"On-Time Rate:  {aggregate['on_time_rate']:.1%}",
        "",
        "ROOT CAUSE DISTRIBUTION (Primary):",
    ]

    for cause, pct in sorted(
        aggregate.get("primary_cause_percentages", {}).items(),
        key=lambda x: x[1],
        reverse=True
    ):
        count = aggregate["root_cause_distribution"].get(cause, 0)
        bar = "█" * int(pct / 2)
        lines.append(f"  {cause:<30} {bar} {pct:.1f}% ({count})")

    lines.append("")
    lines.append("CONTRIBUTING FACTORS:")
    for factor, pct in sorted(
        aggregate.get("contributing_factor_percentages", {}).items(),
        key=lambda x: x[1],
        reverse=True
    ):
        count = aggregate["contributing_factor_counts"].get(factor, 0)
        bar = "█" * int(pct / 2)
        lines.append(f"  {factor:<30} {bar} {pct:.1f}% ({count})")

    return "\n".join(lines)