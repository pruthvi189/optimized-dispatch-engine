"""Run the Phase 5 API server.

Serves the built React dashboard (dashboard/dist) at / when present,
plus the Phase 4 API under /api and the /ws snapshot stream.

Example:
    python run_api.py --port 8000
    python run_api.py --scenario rain --seed 7 --speed 60 --port 8000
"""

import argparse
from pathlib import Path

import uvicorn

from api.app import create_app
from api.schemas import RunnerConfigIn


def main():
    parser = argparse.ArgumentParser(description="Run the Adaptive Dispatch Engine API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scenario", default="normal", choices=["normal", "lunch_rush", "rain", "low_staffing", "traffic_spike"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", default="adaptive", choices=["immediate", "adaptive"])
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--speed", type=float, default=None,
                        help="sim minutes per wall second; None = fast-as-possible")
    parser.add_argument("--step-minutes", type=float, default=1.0)
    parser.add_argument("--autostart", action="store_true", help="start the simulation on boot")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="do not mount the built dashboard even if dashboard/dist exists")
    args = parser.parse_args()

    initial = RunnerConfigIn(
        scenario=args.scenario, seed=args.seed, policy=args.policy,
        days=args.days, speed=args.speed, step_minutes=args.step_minutes,
    )

    dashboard_dir = None
    if not args.no_dashboard:
        candidate = Path(__file__).resolve().parent / "dashboard" / "dist"
        if (candidate / "index.html").exists():
            dashboard_dir = candidate

    app = create_app(initial=initial, autostart=args.autostart, dashboard_dir=dashboard_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
