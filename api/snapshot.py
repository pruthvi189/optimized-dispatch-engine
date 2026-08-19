"""JSON-serializable snapshot of live engine state. This schema is the
Phase 5 dashboard contract — the WebSocket sends exactly these shapes."""

import json

from dispatch.metrics import compute_metrics


def _order_row(order):
    return {
        "id": order.order_id,
        "status": order.status.value,
        "items": order.items,
        "complexity": order.complexity.value,
        "placed_at": round(order.placed_at, 2),
        "dispatch_at": order.dispatch_at,
    }


def _decision_row(record):
    p = json.loads(record.get("payload_json") or "{}")
    return {
        "order_id": record.get("order_id"),
        "policy": p.get("policy"),
        "dispatch_at": p.get("dispatch_at"),
        "prep_mean": p.get("predicted_prep_mean"),
        "prep_low": p.get("predicted_prep_low"),
        "prep_high": p.get("predicted_prep_high"),
        "uncertainty": p.get("uncertainty"),
        "risk_buffer_min": p.get("risk_buffer_min"),
        "travel_to_kitchen_min": p.get("travel_to_kitchen_min"),
        "rationale": p.get("rationale"),
        "items": p.get("items"),
        "complexity": p.get("complexity"),
        "selected_kitchen_id": p.get("selected_kitchen_id"),
        "selected_rider_id": p.get("selected_rider_id"),
        "selected_kitchen_distance": p.get("selected_kitchen_distance"),
        "inputs": p.get("inputs"),
    }


def build_snapshot(engine, meta=None):
    """Build a plain-dict snapshot. Safe before _setup, mid-run, and after."""
    meta = meta or {}

    kitchens = []
    for k in getattr(engine, "kitchens", []) or []:
        kitchens.append({
            "id": k.kitchen_id,
            "queue_len": len(k.current_orders),
            "orders": [_order_row(o) for o in list(k.current_orders)],
        })

    assigned = {}
    for o in getattr(engine, "all_orders", []) or []:
        if o.rider_id is not None and o.delivered_at is None and o.status.value not in ("cancelled", "failed"):
            assigned[o.rider_id] = o.order_id

    riders = []
    rider_pool = getattr(engine, "riders", None)
    rider_objs = rider_pool.riders if rider_pool is not None else []
    for r in rider_objs:
        riders.append({
            "id": r.rider_id,
            "status": r.status.value,
            "busy_min": round(r.busy_min, 2),
            "assigned_to": assigned.get(r.rider_id),
        })

    weather = getattr(engine.env, "_weather", None)
    traffic = getattr(engine.env, "_traffic", None)
    weather_sev = weather.current_severity().value if weather else None
    traffic_sev = traffic.current_severity().value if traffic else None

    metrics = {}
    if rider_pool is not None:
        metrics = compute_metrics(
            getattr(engine, "all_orders", []) or [],
            rider_pool.riders,
            engine.env.now,
            engine.config,
        )

    events = []
    if getattr(engine, "event_log", None) is not None:
        events = engine.event_log.events
    recent = events[-50:]
    decisions = [
        _decision_row(e) for e in reversed(events)
        if e.get("event_type") == "dispatch_decision"
    ][:20]

    return {
        "sim_time_min": round(engine.env.now, 2),
        "scenario": getattr(engine, "scenario_name", None),
        "policy": engine.dispatcher.policy.name if getattr(engine, "dispatcher", None) else None,
        "days": engine.config.get("days"),
        "seed": engine.config.get("seed"),
        "total_minutes": getattr(engine, "total_minutes", 0),
        "speed": meta.get("speed"),
        "running": bool(meta.get("running", False)),
        "paused": bool(meta.get("paused", False)),
        "finished": bool(getattr(engine, "is_finished", False)),
        "weather": weather_sev,
        "traffic": traffic_sev,
        "kitchens": kitchens,
        "riders": riders,
        "recent_decisions": decisions,
        "metrics": metrics,
        "events": [{
            "sim_time": e.get("sim_time"),
            "event_type": e.get("event_type"),
            "order_id": e.get("order_id"),
            "rider_id": e.get("rider_id"),
            "payload": e.get("payload_json"),
        } for e in recent],
    }
