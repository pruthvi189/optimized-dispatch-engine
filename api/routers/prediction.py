"""Prep-time prediction endpoint (no simulation side effects)."""

from fastapi import APIRouter, HTTPException, Request

from ..schemas import FeaturesIn

router = APIRouter(prefix="/prediction", tags=["prediction"])


def _predictor(request):
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        from models.predict import Predictor

        try:
            predictor = Predictor.load(request.app.state.predictor_dir)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="Phase 2 artifacts not found; run `python train_models.py --out artifacts` first",
            ) from exc
        request.app.state.predictor = predictor
    return predictor


@router.post("")
def predict(features: FeaturesIn, request: Request):
    return _predictor(request).predict(features.model_dump())
