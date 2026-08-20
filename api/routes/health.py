import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from core.config import settings
from models.tabular_inference import get_tabular_engine


router = APIRouter(tags=["health"])

START_TIME = time.time()


class HealthResponse(BaseModel):
    status: str
    database: str
    tabular_model: str
    imaging_model: str
    device: str
    uptime_seconds: int


@router.get("/healthz", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    db_status = "disconnected"
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    tabular_engine = get_tabular_engine()
    tabular_model_status = "loaded" if tabular_engine.is_loaded() else "not_loaded"
    device = tabular_engine.get_device()

    imaging_model_status = "not_loaded"

    overall_status = "ok"
    if db_status != "connected" or tabular_model_status != "loaded":
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        database=db_status,
        tabular_model=tabular_model_status,
        imaging_model=imaging_model_status,
        device=device,
        uptime_seconds=int(time.time() - START_TIME),
    )