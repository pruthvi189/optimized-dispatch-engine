"""Root-cause analysis endpoints."""

from fastapi import APIRouter, HTTPException, Request

from dispatch.root_cause import (
    analyze_order,
    analyze_orders,
    aggregate_root_causes,
    RootCauseAnalysis,
)

router = APIRouter(prefix="/analysis", tags=["analysis"])


def _to_dict(analysis: RootCauseAnalysis) -> dict:
    return {
        "order_id": analysis.order_id,
        "is_late": analysis.is_late,
        "delivery_time_min": analysis.delivery_time_min,
        "promise_time_min": analysis.promise_time_min,
        "lateness_min": analysis.lateness_min,
        "primary_root_cause": analysis.primary_root_cause,
        "contributing_factors": analysis.contributing_factors,
        "stage_durations": analysis.stage_durations.to_dict(),
    }


@router.get("/root-causes")
def get_root_causes(request: Request):
    """Get aggregate root-cause analysis for all completed orders."""
    engine = request.app.state.runner.engine
    if engine is None:
        raise HTTPException(status_code=404, detail="No simulation running")

    orders = engine.all_orders
    promised = engine.config["dispatch"]["promised_delivery_min"]

    analyses = analyze_orders(orders, promised)
    aggregate = aggregate_root_causes(analyses)

    # Include individual order analyses (late orders only, limited)
    late_analyses = [a for a in analyses if a.is_late]
    late_analyses.sort(key=lambda x: x.lateness_min, reverse=True)

    return {
        "aggregate": aggregate,
        "late_orders": [_to_dict(a) for a in late_analyses[:100]],
        "total_analyzed": len(analyses),
    }


@router.get("/root-causes/orders/{order_id}")
def get_order_root_cause(order_id: int, request: Request):
    """Get root-cause analysis for a specific order."""
    engine = request.app.state.runner.engine
    if engine is None:
        raise HTTPException(status_code=404, detail="No simulation running")

    orders = engine.all_orders
    order = next((o for o in orders if o.order_id == order_id), None)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

    promised = engine.config["dispatch"]["promised_delivery_min"]
    analysis = analyze_order(order, promised)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not delivered")

    return _to_dict(analysis)