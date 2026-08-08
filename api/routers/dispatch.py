"""Dispatch decision endpoints: live decision history + offline what-if."""

import json

from fastapi import APIRouter, Request

from ..schemas import DispatchIn
from dispatch.state import DispatchState
from simulation.entities import Order, OrderComplexity, complexity_from_items
from simulation.riders import travel_time_min

router = APIRouter(prefix="/dispatch", tags=["dispatch"])


def _active_policy(request):
    engine = request.app.state.runner.engine
    if engine is not None and engine.dispatcher is not None:
        return engine.dispatcher.policy
    from dispatch.policies import make_policy
    from .prediction import _predictor

    config = request.app.state.config
    return make_policy(config["dispatch"]["default_policy"], _predictor(request), config)


def _live_state(request, kitchen_id, hour_of_day):
    config = request.app.state.config
    engine = request.app.state.runner.engine
    if engine is not None:
        now = engine.env.now
        weather = engine.env._weather.current_severity().value
        traffic = engine.env._traffic.current_severity().value
        idle = engine.riders.idle_count()
        queue_lens = {k.kitchen_id: len(k.current_orders) for k in engine.kitchens}
        hub = engine.riders.sample_hub_distance(engine.streams["dispatch"])
    else:
        now = 0.0
        weather, traffic = "clear", "low"
        idle = config["riders"]["count"]
        queue_lens = {}
        lo, hi = config["dispatch"]["hub_distance_range_km"]
        hub = round((lo + hi) / 2, 3)

    d = config["dispatch"]
    travel = travel_time_min(
        hub, config["riders"]["speed_kmh"],
        traffic_factor=d["traffic_speed_factor"][traffic],
        weather_factor=d["weather_speed_factor"][weather],
    )
    return now, DispatchState(
        now=now,
        kitchen_queue_lens=queue_lens,
        idle_rider_count=idle,
        weather_severity=weather,
        traffic_severity=traffic,
        hub_distance_km=hub,
        travel_to_kitchen_min=travel,
    )


@router.post("")
def decide(body: DispatchIn, request: Request):
    """What the active policy would decide for a synthetic order, right now."""
    config = request.app.state.config
    now, state = _live_state(request, body.kitchen_id, body.hour_of_day)
    order = Order(
        order_id=None,
        kitchen_id=body.kitchen_id,
        placed_at=now,
        items=body.items_count,
        complexity=complexity_from_items(body.items_count),
        distance_km=body.distance_km,
        staff_level=config["kitchens"]["staff_level"],
    )
    decision = _active_policy(request).decide(order, state)
    return decision.to_payload()


@router.get("/decisions")
def recent_decisions(request: Request, limit: int = 50):
    engine = request.app.state.runner.engine
    events = engine.event_log.events if engine is not None else []
    out = []
    for e in reversed(events):
        if e.get("event_type") == "dispatch_decision":
            payload = json.loads(e.get("payload_json") or "{}")
            out.append({
                "sim_time": e.get("sim_time"),
                "order_id": e.get("order_id"),
                "kitchen_id": e.get("kitchen_id"),
                **{k: payload[k] for k in (
                    "policy", "dispatch_at", "predicted_prep_mean", "predicted_prep_low",
                    "predicted_prep_high", "uncertainty", "risk_buffer_min",
                    "travel_to_kitchen_min", "hub_distance_km", "eta", "rationale",
                ) if k in payload},
            })
            if len(out) >= limit:
                break
    return out
