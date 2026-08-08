"""Order lifecycle endpoints."""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/orders", tags=["orders"])


def _summary(o):
    return {
        "order_id": o.order_id,
        "kitchen_id": o.kitchen_id,
        "placed_at": round(o.placed_at, 2),
        "status": o.status.value,
        "items": o.items,
        "complexity": o.complexity.value,
        "distance_km": o.distance_km,
        "staff_level": o.staff_level,
        "workload_at_placement": o.workload_at_placement,
        "weather_severity": o.weather_severity,
        "traffic_severity": o.traffic_severity,
        "prep_duration_min": o.actual_prep_duration_min,
        "dispatch_policy": o.dispatch_policy,
        "dispatch_at": o.dispatch_at,
        "rider_id": o.rider_id,
        "eta_min": o.eta_min,
        "predicted_prep_mean": o.predicted_prep_mean,
        "uncertainty": o.uncertainty,
        "risk_buffer_min": o.risk_buffer_min,
        "pickup_at": o.pickup_at,
        "delivered_at": o.delivered_at,
        "cancel_reason": o.cancel_reason,
    }


@router.get("")
def list_orders(request: Request, status: str | None = None, limit: int = 100):
    engine = request.app.state.runner.engine
    orders = engine.all_orders if engine is not None else []
    if status:
        orders = [o for o in orders if o.status.value == status]
    return [_summary(o) for o in orders[-limit:]]


@router.get("/{order_id}")
def get_order(order_id: int, request: Request):
    engine = request.app.state.runner.engine
    orders = engine.all_orders if engine is not None else []
    for o in orders:
        if o.order_id == order_id:
            return _summary(o)
    raise HTTPException(status_code=404, detail=f"order {order_id} not found")
