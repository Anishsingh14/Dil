import time
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from core.auth import verify_api_key_and_get_developer
from db.models import Developer
from models.tabular_inference import get_tabular_engine


router = APIRouter(prefix="/api/v1", tags=["predict-tabular"])


class TabularRequest(BaseModel):
    age: int = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: int = Field(ge=0, le=300)
    chol: int = Field(ge=0, le=600)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=0, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0.0, le=10.0)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)

    def to_features_array(self) -> list:
        return [
            self.age, self.sex, self.cp, self.trestbps, self.chol,
            self.fbs, self.restecg, self.thalach, self.exang,
            self.oldpeak, self.slope, self.ca, self.thal
        ]


class TabularResponse(BaseModel):
    status: str = "success"
    risk_score: float
    risk_level: str
    model_version: str
    latency_ms: int
    timestamp: str


class ErrorResponse(BaseModel):
    status: str = "error"
    error_code: str
    detail: str | list
    retry_after_seconds: int | None = None
    timestamp: str


class ValidationErrorResponse(BaseModel):
    status: str = "error"
    error_code: str = "VALIDATION_ERROR"
    detail: list
    timestamp: str


async def get_current_developer(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> Developer:
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "error_code": "INVALID_API_KEY",
                "detail": "The provided X-API-Key is missing, malformed, or inactive.",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
    developer = await verify_api_key_and_get_developer(x_api_key, db)
    if not developer:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "error_code": "INVALID_API_KEY",
                "detail": "The provided X-API-Key is missing, malformed, or inactive.",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
    return developer


@router.post(
    "/predict-tabular",
    response_model=TabularResponse,
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ValidationErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def predict_tabular(
    request: Request,
    payload: TabularRequest,
    developer: Developer = Depends(get_current_developer),
) -> TabularResponse:
    engine = get_tabular_engine()
    if not engine.is_loaded():
        engine.load()

    import numpy as np
    features = np.array(payload.to_features_array(), dtype=np.float32)

    risk_score, risk_level, latency_ms = await engine.predict(features)

    return TabularResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        model_version=engine.model_version,
        latency_ms=latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )