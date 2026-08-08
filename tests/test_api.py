"""Phase 4 API tests: REST routers, runner control flow, and /ws streaming."""

import time

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.schemas import RunnerConfigIn

SNAPSHOT_KEYS = {
    "sim_time_min", "scenario", "policy", "days", "seed", "total_minutes",
    "speed", "running", "paused", "finished", "weather", "traffic",
    "kitchens", "riders", "recent_decisions", "metrics", "events",
}


def _app(**overrides):
    overrides.setdefault("seed", 101)
    overrides.setdefault("speed", 1000.0)
    overrides.setdefault("step_minutes", 1.0)
    return create_app(initial=RunnerConfigIn(**overrides))


def _wait_until(predicate, timeout=20.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---- REST: status / config ------------------------------------------------


def test_status_before_start():
    with TestClient(_app()) as c:
        s = c.get("/sim/status").json()
    assert s["sim_time_min"] == 0.0
    assert s["scenario"] == "normal"
    assert s["policy"] == "adaptive"
    assert SNAPSHOT_KEYS.issubset(s.keys())


def test_config_endpoint():
    with TestClient(_app(scenario="rain", seed=7, policy="immediate")) as c:
        cfg = c.get("/sim/config").json()
    assert cfg["scenario"] == "rain"
    assert cfg["seed"] == 7
    assert cfg["policy"] == "immediate"
    assert cfg["days"] == 1
    assert cfg["speed"] == 1000.0


# ---- REST: orders ----------------------------------------------------------


def test_orders_after_completed_run():
    with TestClient(_app()) as c:
        c.post("/sim/start")
        assert _wait_until(lambda: c.get("/sim/status").json()["finished"])
        orders = c.get("/orders").json()
        assert orders
        o = orders[0]
        for key in ("order_id", "kitchen_id", "placed_at", "status", "items", "distance_km"):
            assert key in o
        assert c.get(f"/orders/{o['order_id']}").json()["order_id"] == o["order_id"]
        statuses = {x["status"] for x in c.get("/orders?limit=1000").json()}
        assert statuses.issubset({"placed", "prepping", "ready", "completed", "cancelled", "failed"})


# ---- REST: prediction -----------------------------------------------------


def test_prediction_returns_interval():
    with TestClient(_app()) as c:
        r = c.post("/prediction", json={
            "items_count": 3,
            "workload_at_placement": 2,
            "staff_level": 3,
            "hour_of_day": 12,
            "order_complexity": "standard",
            "weather_severity": "clear",
            "traffic_severity": "low",
            "kitchen_id": 1,
        })
    assert r.status_code == 200
    body = r.json()
    for key in ("prep_mean", "prep_low", "prep_high", "uncertainty"):
        assert key in body
    assert body["prep_low"] <= body["prep_mean"] <= body["prep_high"]


# ---- REST: dispatch decisions ---------------------------------------------


def test_dispatch_decision_offline():
    with TestClient(_app()) as c:
        r = c.post("/dispatch", json={
            "items_count": 4,
            "kitchen_id": 1,
            "distance_km": 3.0,
            "hour_of_day": 12,
        })
    assert r.status_code == 200
    body = r.json()
    assert body["policy"] == "adaptive"
    assert body["dispatch_at"] is not None
    assert body["eta"] > 0
    assert "rationale" in body


def test_dispatch_decisions_history_empty_before_run():
    with TestClient(_app()) as c:
        assert c.get("/dispatch/decisions").json() == []


# ---- runner control: pause / step / reset ---------------------------------


def test_pause_step_reset_flow():
    with TestClient(_app()) as c:
        c.post("/sim/start")
        assert _wait_until(lambda: c.get("/sim/status").json()["sim_time_min"] > 0)
        c.post("/sim/pause")
        assert _wait_until(lambda: c.get("/sim/status").json()["paused"] is True)

        # Wait until an in-flight auto-advance settles before sampling.
        def _settled():
            a = c.get("/sim/status").json()
            if not a["paused"]:
                return False
            time.sleep(0.1)
            b = c.get("/sim/status").json()
            return b["sim_time_min"] == a["sim_time_min"] > 0

        assert _wait_until(_settled)
        before = c.get("/sim/status").json()["sim_time_min"]
        c.post("/sim/step", json={"minutes": 5})
        assert _wait_until(lambda: c.get("/sim/status").json()["sim_time_min"] - before >= 5.0)
        after = c.get("/sim/status").json()["sim_time_min"]
        assert 5.0 <= after - before <= 6.0

        c.post("/sim/reset", json={"scenario": "rain", "seed": 7, "policy": "immediate"})
        s = c.get("/sim/status").json()
        assert s["scenario"] == "rain"
        assert s["policy"] == "immediate"
        assert s["seed"] == 7
        assert s["sim_time_min"] == 0.0


# ---- websocket streaming ---------------------------------------------------


def test_ws_streams_monotonic_snapshots():
    with TestClient(_app()) as c:
        with c.websocket_connect("/ws?autostart=true&speed=1000&scenario=normal&seed=101") as ws:
            seen = []
            for _ in range(4):
                seen.append(ws.receive_json())
        times = [m["sim_time_min"] for m in seen]
        assert times == sorted(times)
        assert seen[-1]["sim_time_min"] > 0
        assert seen[-1]["seed"] == 101
        assert SNAPSHOT_KEYS.issubset(seen[-1].keys())


def test_ws_honors_paused_state():
    with TestClient(_app()) as c:
        with c.websocket_connect("/ws?autostart=false&scenario=normal&seed=101") as ws:
            first = ws.receive_json()
        assert first["paused"] is False
        assert first["running"] is False
        assert first["sim_time_min"] == 0.0
