from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from core.auth import verify_api_key_and_get_developer
from db.models import Developer


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
                "timestamp": "2026-08-20T03:31:12Z",
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
                "timestamp": "2026-08-20T03:31:12Z",
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
    if not validate_file_extension(file.filename or ""):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "status": "error",
                "error_code": "UNSUPPORTED_FILE_TYPE",
                "detail": f"File extension is not supported. Accepted types: .png, .jpg, .dicom",
                "timestamp": "2026-08-20T03:31:12Z",
            },
        )

    return ImageResponse(
        risk_score=0.0,
        risk_level="Low Risk",
        modality=modality,
        findings={},
        model_version="imaging-densenet121-lstm-v1.1.0",
        latency_ms=0,
        timestamp="2026-08-20T03:31:12Z",
    )