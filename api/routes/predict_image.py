import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from core.auth import verify_api_key_and_get_developer
from db.models import Developer
from models.imaging_inference import get_imaging_engine


router = APIRouter(prefix="/api/v1", tags=["predict-image"])


class ImageResponse(BaseModel):
    status: str = "success"
    risk_score: float
    risk_level: str
    modality: str
    findings: dict
    model_version: str
    latency_ms: int
    timestamp: str


class ImageSeriesResponse(BaseModel):
    status: str = "success"
    risk_score: float
    risk_level: str
    modality: str
    findings: dict
    model_version: str
    latency_ms: int
    timestamp: str
    num_images: int
    individual_scores: List[float]


class ErrorResponse(BaseModel):
    status: str = "error"
    error_code: str
    detail: str
    retry_after_seconds: int | None = None
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


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dicom", ".dcm"}


def validate_file_extension(filename: str) -> bool:
    if not filename:
        return False
    ext = filename.lower()
    return any(ext.endswith(allowed) for allowed in ALLOWED_EXTENSIONS)


@router.post(
    "/predict-image",
    response_model=ImageResponse,
    responses={
        401: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    modality: str = Form(..., pattern="^(chest_xray|cardiac_mri)$"),
    patient_ref: str | None = Form(None),
    developer: Developer = Depends(get_current_developer),
) -> ImageResponse:
    """Single image prediction (backward compatible)."""
    if not validate_file_extension(file.filename or ""):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "status": "error",
                "error_code": "UNSUPPORTED_FILE_TYPE",
                "detail": f"File extension '{Path(file.filename).suffix}' is not supported. Accepted types: .png, .jpg, .dicom",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )

    file_bytes = await file.read()
    
    engine = get_imaging_engine()
    if not engine.is_loaded():
        engine.load()

    risk_score, risk_level, findings, latency_ms = await engine.predict(file_bytes, file.filename)

    return ImageResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        modality=modality,
        findings=findings,
        model_version=engine.model_version,
        latency_ms=latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


@router.post(
    "/predict-image-series",
    response_model=ImageSeriesResponse,
    responses={
        401: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def predict_image_series(
    request: Request,
    files: List[UploadFile] = File(...),
    modality: str = Form(..., pattern="^(chest_xray|cardiac_mri)$"),
    patient_ref: str | None = Form(None),
    developer: Developer = Depends(get_current_developer),
) -> ImageSeriesResponse:
    """Multi-image series prediction (for cardiac MRI or multi-view chest X-rays)."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "error_code": "NO_FILES_PROVIDED",
                "detail": "At least one image file must be provided.",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
    
    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "error_code": "TOO_MANY_FILES",
                "detail": "Maximum 50 images per series allowed.",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
    
    # Validate all files
    for file in files:
        if not validate_file_extension(file.filename or ""):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail={
                    "status": "error",
                    "error_code": "UNSUPPORTED_FILE_TYPE",
                    "detail": f"File extension '{Path(file.filename).suffix}' is not supported. Accepted types: .png, .jpg, .dicom",
                    "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                },
            )
    
    # Read all files
    files_data = []
    for file in files:
        file_bytes = await file.read()
        files_data.append((file_bytes, file.filename))
    
    engine = get_imaging_engine()
    if not engine.is_loaded():
        engine.load()
    
    # Process as series
    risk_score, risk_level, findings, latency_ms, individual_scores = await engine.predict_series(files_data)

    return ImageSeriesResponse(
        risk_score=risk_score,
        risk_level=risk_level,
        modality=modality,
        findings=findings,
        model_version=engine.model_version,
        latency_ms=latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        num_images=len(files_data),
        individual_scores=individual_scores,
    )


from pathlib import Path