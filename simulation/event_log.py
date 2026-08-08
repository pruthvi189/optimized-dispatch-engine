import json
import os
import csv

from .entities import OrderStatus

EVENT_COLUMNS = ["sim_time", "event_type", "order_id", "kitchen_id", "rider_id", "payload_json"]

ORDER_COLUMNS = [
    "order_id", "kitchen_id", "placed_at", "hour_of_day", "day_of_week",
    "order_complexity", "items_count", "workload_at_placement", "staff_level",
    "weather_severity", "traffic_severity", "actual_prep_duration_min",
    "status", "cancel_reason",
    "dispatch_policy", "dispatch_at", "rider_id", "hub_distance_km",
    "travel_to_kitchen_min", "rider_arrived_kitchen_at", "pickup_at", "delivered_at",
    "eta_min", "predicted_prep_mean", "predicted_prep_low", "predicted_prep_high",
    "uncertainty", "risk_buffer_min", "decision_rationale",
]


class EventLog:
    """Buffered writer for the raw event log; builder for the orders table."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.events = []

    def record(self, sim_time, event_type, order_id=None, kitchen_id=None, rider_id=None, payload=None):
        self.events.append({
            "sim_time": round(sim_time, 2),
            "event_type": event_type,
            "order_id": order_id,
            "kitchen_id": kitchen_id,
            "rider_id": rider_id,
            "payload_json": json.dumps(payload or {}),
        })

    def write(self):
        path = os.path.join(self.out_dir, "event_log.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EVENT_COLUMNS)
            writer.writeheader()
            writer.writerows(self.events)

    def write_orders_csv(self, orders):
        path = os.path.join(self.out_dir, "orders.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ORDER_COLUMNS)
            writer.writeheader()
            for o in orders:
                writer.writerow({
                    "order_id": o.order_id,
                    "kitchen_id": o.kitchen_id,
                    "placed_at": round(o.placed_at, 2),
                    "hour_of_day": int(o.placed_at // 60) % 24,
                    "day_of_week": int(o.placed_at // 1440) % 7,
                    "order_complexity": o.complexity.value,
                    "items_count": o.items,
                    "workload_at_placement": o.workload_at_placement,
                    "staff_level": o.staff_level,
                    "weather_severity": o.weather_severity,
                    "traffic_severity": o.traffic_severity,
                    "actual_prep_duration_min": round(o.actual_prep_duration_min, 2) if o.actual_prep_duration_min is not None else "",
                    "status": o.status.value,
                    "cancel_reason": o.cancel_reason or "",
                    "dispatch_policy": o.dispatch_policy or "",
                    "dispatch_at": round(o.dispatch_at, 2) if o.dispatch_at is not None else "",
                    "rider_id": o.rider_id or "",
                    "hub_distance_km": round(o.hub_distance_km, 3) if o.hub_distance_km is not None else "",
                    "travel_to_kitchen_min": round(o.travel_to_kitchen_min, 2) if o.travel_to_kitchen_min is not None else "",
                    "rider_arrived_kitchen_at": round(o.rider_arrived_kitchen_at, 2) if o.rider_arrived_kitchen_at is not None else "",
                    "pickup_at": round(o.pickup_at, 2) if o.pickup_at is not None else "",
                    "delivered_at": round(o.delivered_at, 2) if o.delivered_at is not None else "",
                    "eta_min": round(o.eta_min, 2) if o.eta_min is not None else "",
                    "predicted_prep_mean": round(o.predicted_prep_mean, 2) if o.predicted_prep_mean is not None else "",
                    "predicted_prep_low": round(o.predicted_prep_low, 2) if o.predicted_prep_low is not None else "",
                    "predicted_prep_high": round(o.predicted_prep_high, 2) if o.predicted_prep_high is not None else "",
                    "uncertainty": o.uncertainty or "",
                    "risk_buffer_min": round(o.risk_buffer_min, 2) if o.risk_buffer_min is not None else "",
                    "decision_rationale": o.decision_rationale or "",
                })
