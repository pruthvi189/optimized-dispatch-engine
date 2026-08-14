"""FastAPI app factory. The snapshot schema in `snapshot.py` is the Phase 5
dashboard contract — the /ws endpoint streams exactly those shapes."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .broadcast import WebSocketHub
from .config import build_config
from .routers import analysis, dispatch, experiment, orders, prediction, sim
from .runner import SimulationRunner
from .schemas import RunnerConfigIn


def create_app(initial: RunnerConfigIn | None = None, autostart: bool = False,
               dashboard_dir: str | Path | None = None):
    if initial is None:
        initial = RunnerConfigIn()
    config = build_config(scenario=initial.scenario, seed=initial.seed,
                          policy=initial.policy, days=initial.days)

    hub = WebSocketHub()
    runner = SimulationRunner(
        config, hub=hub, speed=initial.speed, step_minutes=initial.step_minutes,
        scenario_name=initial.scenario,
    )

    @asynccontextmanager
    async def lifespan(app):
        hub.start(asyncio.get_running_loop())
        if autostart:
            runner.start()
        yield
        runner.stop()
        await hub.stop()

    app = FastAPI(title="Adaptive Dispatch Engine API", version="0.4.0", lifespan=lifespan)
    app.state.hub = hub
    app.state.runner = runner
    app.state.config = config
    app.state.predictor = None
    app.state.predictor_dir = config["dispatch"].get("predictor_dir")

    app.include_router(orders.router)
    app.include_router(prediction.router)
    app.include_router(dispatch.router)
    app.include_router(sim.router)
    app.include_router(analysis.router)
    app.include_router(experiment.router)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        params = websocket.query_params
        scenario = params.get("scenario", initial.scenario)
        policy = params.get("policy", initial.policy)
        try:
            seed = int(params.get("seed", initial.seed))
            speed = float(params.get("speed", initial.speed or 60))
        except ValueError:
            await websocket.close(code=1008, reason="invalid seed or speed")
            return
        autostart_ws = params.get("autostart", "true").lower() != "false"

        current = app.state.config
        if (runner.scenario_name != scenario or current.get("seed") != seed
                or current["dispatch"]["default_policy"] != policy):
            new_config = build_config(scenario=scenario, seed=seed, policy=policy)
            runner.reset(new_config, scenario_name=scenario)
            app.state.config = new_config
        if runner.speed != speed:
            runner.speed = speed
        if autostart_ws:
            runner.start()

        hub.subscribe(websocket)
        try:
            await websocket.send_json(runner.status())
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.unsubscribe(websocket)

    if dashboard_dir is not None:
        dashboard_dir = Path(dashboard_dir)
        if (dashboard_dir / "index.html").exists():
            # Mounted last so /ws and /api/* routes win; "/" still serves the SPA
            # via html=True (index.html fallback for client-side routes).
            app.mount("/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")
        else:
            raise FileNotFoundError(f"dashboard/dist missing index.html (run `npm run build`): {dashboard_dir}")

    return app
