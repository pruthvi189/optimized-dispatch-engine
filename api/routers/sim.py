"""Simulation control endpoints: status, config, start/pause/resume/step/reset."""

from fastapi import APIRouter, HTTPException, Request

from ..config import build_config
from ..schemas import RunnerConfigIn, StepIn

router = APIRouter(prefix="/sim", tags=["sim"])


@router.get("/status")
def status(request: Request):
    return request.app.state.runner.status()


@router.get("/config")
def config(request: Request):
    runner = request.app.state.runner
    c = request.app.state.config
    return {
        "scenario": runner.scenario_name,
        "seed": c.get("seed"),
        "days": c.get("days"),
        "policy": c["dispatch"]["default_policy"],
        "speed": runner.speed,
        "step_minutes": runner.step_minutes,
        "predictor_dir": c["dispatch"].get("predictor_dir"),
        "dispatch_enabled": bool(c["dispatch"].get("enabled")),
    }


def _apply_config(request, body: RunnerConfigIn):
    config = build_config(scenario=body.scenario, seed=body.seed, policy=body.policy, days=body.days)
    request.app.state.config = config
    runner = request.app.state.runner
    runner.reset(config, scenario_name=body.scenario)
    runner.speed = body.speed
    runner.step_minutes = body.step_minutes
    return runner


def _publish(request, runner):
    hub = request.app.state.hub
    if hub is not None:
        hub.publish(runner.status())


@router.post("/start")
def start(request: Request):
    runner = request.app.state.runner
    runner.start()
    _publish(request, runner)
    return runner.status()


@router.post("/pause")
def pause(request: Request):
    runner = request.app.state.runner
    runner.pause()
    _publish(request, runner)
    return runner.status()


@router.post("/resume")
def resume(request: Request):
    runner = request.app.state.runner
    runner.start()
    _publish(request, runner)
    return runner.status()


@router.post("/step")
def step(body: StepIn, request: Request):
    runner = request.app.state.runner
    try:
        runner.step(body.minutes)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _publish(request, runner)
    return runner.status()


@router.post("/reset")
def reset(body: RunnerConfigIn, request: Request):
    _apply_config(request, body)
    _publish(request, request.app.state.runner)
    return request.app.state.runner.status()


@router.post("/config")
def set_config(body: RunnerConfigIn, request: Request):
    return _apply_config(request, body).status()
